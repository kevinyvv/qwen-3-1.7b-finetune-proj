"""Check that the hand-written model matches HuggingFace, and that the KV cache
produces the same numbers as running without one.

    python -m scripts.check_parity
"""

import gc

import torch

from core.model import Qwen3ForCausalLM
from core.utils import load_qwen3_weights, get_encoder


# fp32 weights, so differences should be tiny. anything above this is a real bug.
TOLERANCE = 1e-3

PROMPTS_SINGLE = ["the capital of france is"]
PROMPTS_RAGGED = ["the capital of france is", "hi"]


def make_batch(tokenizer, prompts, device):
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    ids, mask = batch.input_ids, batch.attention_mask
    positions = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    return ids, mask, positions


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report(name, a, b, mask=None):
    """Print the largest difference between two logit tensors, per batch row."""
    diff = (a.float() - b.float()).abs()
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)  # ignore positions that are padding

    worst = diff.max().item()
    status = "PASS" if worst < TOLERANCE else "FAIL"
    print(f"  [{status}] {name}: max diff {worst:.6f}")

    if diff.size(0) > 1:
        for row in range(diff.size(0)):
            print(f"           row {row}: {diff[row].max().item():.6f}")
    return worst < TOLERANCE



def hf_logits(prompts, tokenizer, device):
    from transformers import Qwen3ForCausalLM as HFQwen3

    hf = HFQwen3.from_pretrained(
        "./weights/qwen3-1.7b",
        dtype=torch.float32,
        attn_implementation="eager",  # match the hand-written attention
    ).to(device).eval()

    ids, mask, positions = make_batch(tokenizer, prompts, device)

    with torch.no_grad():
        out = hf(input_ids=ids, attention_mask=mask, position_ids=positions).logits

    out = out.cpu()
    free(hf)
    return out


def ours_no_cache(model, ids, mask, positions, num_layers):
    with torch.no_grad():
        logits, _ = model(ids, positions, mask, [None] * num_layers)
    return logits


def ours_token_by_token(model, ids, mask, positions, num_layers):
    """Feed one token at a time through the cached path, exactly like decoding."""
    caches = [None] * num_layers
    steps = []

    with torch.no_grad():
        for i in range(ids.size(1)):
            # the mask has to cover every key in the cache, not just the new token
            logits, caches = model(
                ids[:, i : i + 1],
                positions[:, i : i + 1],
                mask[:, : i + 1],
                caches,
            )
            steps.append(logits)

    return torch.cat(steps, dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    cfg, sd = load_qwen3_weights()
    num_layers = cfg.num_hidden_layers

    tokenizer = get_encoder()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # huggingface first, then free it - two fp32 copies of a 1.7b model do not
    # fit on a 16gb card at the same time
    print("loading huggingface reference...")
    ref_single = hf_logits(PROMPTS_SINGLE, tokenizer, device)
    ref_ragged = hf_logits(PROMPTS_RAGGED, tokenizer, device)

    print("loading our model...")
    model = Qwen3ForCausalLM(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    free(sd)

    passed = []

    print("\n1. our forward pass vs huggingface")

    ids, mask, positions = make_batch(tokenizer, PROMPTS_SINGLE, device)
    ours = ours_no_cache(model, ids, mask, positions, num_layers)
    passed.append(report("single prompt", ours.cpu(), ref_single))

    ids_r, mask_r, pos_r = make_batch(tokenizer, PROMPTS_RAGGED, device)
    ours_r = ours_no_cache(model, ids_r, mask_r, pos_r, num_layers)
    passed.append(report("ragged batch", ours_r.cpu(), ref_ragged, mask=mask_r.cpu()))

    print("\n2. cached decode vs single forward pass")

    stepped = ours_token_by_token(model, ids, mask, positions, num_layers)
    passed.append(report("single prompt", stepped, ours))

    stepped_r = ours_token_by_token(model, ids_r, mask_r, pos_r, num_layers)
    passed.append(report("ragged batch", stepped_r, ours_r, mask=mask_r))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
