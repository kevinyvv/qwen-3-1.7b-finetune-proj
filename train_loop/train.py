import torch
from tqdm import tqdm

from core.model import Qwen3ForCausalLM
from core.utils import get_encoder, load_qwen3_weights

PROMPT = "respond without using the letter 'e'"


def get_feedback(responses):
    rewards = []

    for response in responses:
        words = response.split()

        if not words:
            rewards.append(0.0)
            continue

        reward = 0
        e_count = 0

        for word in words:
            if "e" in word.lower():
                e_count += 1
                reward -= e_count
            else:
                reward += 1

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)


def discount(rewards, texts):
    return rewards


def generate(
    model,
    tokenizer,
    input_ids,
    attn_mask,
    positions,
    prefill_logits,
    prefill_kv_caches,
    max_new_tokens,
):
    batch_size = input_ids.size(0)

    next_tok = input_ids[:, -1:]
    kv_caches = prefill_kv_caches[:]

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )

    log_probs = []

    for _ in range(max_new_tokens):
        logits, kv_caches = model(
            next_tok,
            positions[:, -1:],
            attn_mask,
            kv_caches,
        )

        log_logits = logits[:, -1].log_softmax(dim=-1)

        next_tok = torch.multinomial(
            log_logits.exp(),
            num_samples=1,
        )

        log_prob = log_logits.gather(-1, next_tok)
        log_probs.append(log_prob)

        next_tok = torch.where(
            finished.unsqueeze(-1),
            tokenizer.pad_token_id,
            next_tok,
        )

        finished |= next_tok.squeeze(-1) == tokenizer.eos_token_id

        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones_like(next_tok)],
            dim=-1,
        )
        positions = torch.cat(
            [positions, positions[:, -1:] + 1],
            dim=-1,
        )

        if finished.all():
            break

    log_probs = torch.cat(log_probs, dim=-1)

    return input_ids, log_probs


def main(max_new_tokens: int = 64, epochs: int = 10):
    cfg, sd = load_qwen3_weights()

    tokenizer = get_encoder()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Qwen3ForCausalLM(cfg)
    model.load_state_dict(sd, strict=False)
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = name.endswith((".A", ".B"))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    batch = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
    ).to(device)

    prompt_len = batch.input_ids.size(1)

    positions = (
        batch.attention_mask.cumsum(dim=-1) - 1
    ).clamp(min=0)

    empty_kv = [None] * cfg.num_hidden_layers

    prefill_logits, prefill_kv_caches = model(
        batch.input_ids[:, :-1],
        positions[:, :-1],
        batch.attention_mask[:, :-1],
        empty_kv,
    )

    for epoch in tqdm(range(epochs)):
        input_ids = batch.input_ids
        attn_mask = batch.attention_mask

        positions = (
            attn_mask.cumsum(dim=-1) - 1
        ).clamp(min=0)

        generated_ids, log_probs = generate(
            model,
            tokenizer,
            input_ids,
            attn_mask,
            positions,
            prefill_logits,
            prefill_kv_caches,
            max_new_tokens,
        )

        texts = tokenizer.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True,
        )

        rewards = get_feedback(texts).to(device)
        rewards = discount(rewards, texts)

        loss = -(log_probs * rewards.unsqueeze(-1)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"epoch {epoch} | "
            f"mean reward {rewards.mean():.3f} | "
            f"{texts[0][:200]!r}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
    