import json
import string
from pathlib import Path

import torch
from tqdm import tqdm

from core.layers import lora_disabled
from core.model import Qwen3ForCausalLM
from core.utils import get_encoder, load_qwen3_weights
from training.checkpointing import load_checkpoint, save_checkpoint

OVERFIT_SAMPLE = "respond without using the letter 'e'"
SAMPLES = [OVERFIT_SAMPLE for _ in range(8)]

def get_feedback(tok_ids, e_table):
    return e_table[tok_ids]

def discount(rewards, gamma=0.95):
    returns = torch.zeros_like(rewards)

    running = torch.zeros(
        rewards.size(0),
        device=rewards.device,
        dtype=rewards.dtype,
    )

    for t in reversed(range(rewards.size(1))):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running

    return returns

def get_e_table(tok_to_id, special_tokens, vocab_size):
    e_table = torch.zeros(vocab_size, dtype=torch.float)

    for tok, idx in tok_to_id.items():
        if tok in special_tokens:
            continue
        if "e" in tok.lower():
            e_table[idx] = -10.0
        elif any(c in string.ascii_letters for c in tok):
            e_table[idx] = 0.1 * len(tok)

    return e_table


def compute_rewards(gen_ids, e_table, log_probs, ref_log_probs, mask, kl_beta):
    kl_approx = (log_probs.detach() - ref_log_probs) * mask
    return get_feedback(gen_ids, e_table) - kl_beta * kl_approx

def build_positions(attn_mask):
    return (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)

def freeze_non_lora(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.endswith((".A", ".B"))


def clone_kv_cache(kv_cache):
    return [[k.detach(), v.detach()] for k, v in kv_cache]


def setup_tokenizer():
    tokenizer = get_encoder()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_model(device, lr=1e-4):
    cfg, sd = load_qwen3_weights()

    model = Qwen3ForCausalLM(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(device)

    freeze_non_lora(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    return cfg, model, optimizer


def build_prompt_batch(tokenizer, samples, device):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for s in samples
    ]
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def prefill_cache(model, input_ids, positions, attn_mask, num_layers):
    _, kv_cache = model(
        input_ids[:, :-1],
        positions[:, :-1],
        attn_mask[:, :-1],
        [None] * num_layers,
    )
    return kv_cache

def generate(
    model,
    tokenizer,
    input_ids,
    attn_mask,
    positions,
    kv_cache,
    max_new_tokens,
    temperature,
):
    batch_size = input_ids.size(0)

    next_tok = input_ids[:, -1:]  # last prompt token, not yet in the kv cache
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    log_probs = []
    masks = []  # so we don't train on pad tokens

    for _ in range(max_new_tokens):
        logits, kv_cache = model(next_tok, positions[:, -1:], attn_mask, kv_cache)

        log_dist = (logits[:, -1] / temperature).log_softmax(dim=-1)
        next_tok = torch.multinomial(log_dist.exp(), num_samples=1)

        log_probs.append(log_dist.gather(-1, next_tok))
        masks.append(~finished.unsqueeze(1))

        next_tok = torch.where(finished.unsqueeze(-1), tokenizer.pad_token_id, next_tok)
        finished |= next_tok.squeeze(-1) == tokenizer.eos_token_id

        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(next_tok)], dim=-1)
        positions = torch.cat([positions, positions[:, -1:] + 1], dim=-1)

        if finished.all():
            break

    return (
        input_ids,
        attn_mask,
        positions,
        torch.cat(log_probs, dim=-1),
        torch.cat(masks, dim=-1),
        kv_cache,
    )


def compute_reference_log_probs(
    model,
    input_ids,
    positions,
    attn_mask,
    gen_ids,
    prompt_len,
    temperature,
    num_layers,
):
    with torch.no_grad(), lora_disabled(model):
        logits, _ = model(input_ids, positions, attn_mask, [None] * num_layers)

    log_probs = (logits / temperature).log_softmax(dim=-1)

    return (
        log_probs[:, prompt_len - 1 : -1]
        .gather(-1, gen_ids.unsqueeze(-1))
        .squeeze(-1)
    )

def train_step(optimizer, log_probs, rewards, mask):
    per_token_reward = discount(rewards)

    advantage = (per_token_reward - per_token_reward[mask].mean()) * mask
    loss = -(log_probs * advantage).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def log_metrics(log_path, epoch, loss, rewards, mask, texts):
    n_tokens = mask.sum(dim=-1)
    token_reward = (rewards * mask).sum(dim=-1)
    e_rate = (-token_reward.sum() / n_tokens.sum()).item()

    record = {
        "epoch": epoch,
        "loss": loss.item(),
        "mean_reward": token_reward.mean().item(),
        "e_rate": e_rate,
        "samples": [
            {"text": t, "reward": r, "n_tokens": n}
            for t, r, n in zip(texts, token_reward.tolist(), n_tokens.tolist())
        ],
    }

    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"epoch {epoch}, mean reward {token_reward.mean():.3f}, e-rate {e_rate:.3f}, "
          f"{texts[0][:200]!r}")  # !r to see newline as \n


def main(
    max_new_tokens: int = 64,
    epochs=10,
    save_every: int = 5,
    resume: bool = True,
    temperature=1.2,
    kl_beta=0.05,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = setup_tokenizer()
    cfg, qwen3, optimizer = setup_model(device)

    e_table = get_e_table(tokenizer.get_vocab(), set(tokenizer.all_special_ids), cfg.vocab_size)
    e_table = e_table.to(device)

    run_dir = Path(__file__).resolve().parent
    ckpt_path = run_dir / "checkpoints" / "latest.pt"
    log_path = run_dir / "logs" / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = load_checkpoint(ckpt_path, qwen3, optimizer) if resume else 0
    end_epoch = start_epoch + epochs

    if start_epoch:
        print(f"resuming from {ckpt_path} at epoch {start_epoch}")

    batch = build_prompt_batch(tokenizer, SAMPLES, device)
    prompt_len = batch.input_ids.size(1)

    prefill_kv_cache = prefill_cache(
        qwen3,
        batch.input_ids,
        build_positions(batch.attention_mask),
        batch.attention_mask,
        cfg.num_hidden_layers,
    )

    for epoch in tqdm(range(start_epoch, end_epoch)):
        input_ids, attn_mask, positions, log_probs, mask, _ = generate(
            qwen3,
            tokenizer,
            batch.input_ids.clone(),
            batch.attention_mask.clone(),
            build_positions(batch.attention_mask),
            clone_kv_cache(prefill_kv_cache),
            max_new_tokens,
            temperature,
        )
        gen_ids = input_ids[:, prompt_len:]

        ref_log_probs = compute_reference_log_probs(
            qwen3, input_ids, positions, attn_mask, gen_ids,
            prompt_len, temperature, cfg.num_hidden_layers,
        )
        rewards = compute_rewards(gen_ids, e_table, log_probs, ref_log_probs, mask, kl_beta)
        loss = train_step(optimizer, log_probs, rewards, mask)

        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        log_metrics(log_path, epoch, loss, rewards, mask, texts)

        if (epoch + 1) % save_every == 0 or epoch + 1 == end_epoch:
            save_checkpoint(ckpt_path, qwen3, optimizer, epoch)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
