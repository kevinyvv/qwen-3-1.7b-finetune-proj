from dataclasses import dataclass
import math

import torch


@dataclass
class Rollout:
    response_ids: torch.Tensor  # [B, T]
    completion_mask: torch.Tensor  # context, including inserted tokens
    reward_mask: torch.Tensor  # final-answer text only
    loss_mask: torch.Tensor  # sampled actions, including thinking and EOS
    log_probs: torch.Tensor


@torch.no_grad()
def generate_rollout(model, tokenizer, batch, max_new_tokens, temperature=1.0,
                     thinking_budget=None, prefill_cache=None, forbidden_token_mask=None):
    """Generate from left-padded prompts. max_new_tokens includes thinking.

    A thinking_budget requires a thinking-enabled prompt. It counts generated
    tokens before </think>, including <think> if generated. Forced </think>
    consumes one extra position in the total limit but is not a policy action.
    """
    if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens must be positive and temperature nonnegative")
    if thinking_budget is not None and (
        not isinstance(thinking_budget, int) or not 0 <= thinking_budget <= max_new_tokens - 2
    ):
        raise ValueError("thinking_budget must leave room for </think> and an answer")

    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device)
    attn_mask = batch['attention_mask'].to(device)
    if input_ids.shape[1] == 0 or not attn_mask[:, -1].all():
        raise ValueError("prompts must be nonempty and left-padded")
    positions = (attn_mask.cumsum(-1) - 1).clamp(min=0)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
    think_id, end_think_id = vocab.get('<think>'), vocab.get('</think>')
    if thinking_budget is not None and (think_id is None or end_think_id is None):
        raise ValueError("thinking budgets require <think> and </think> tokens")

    kv_caches = [None] * model.config.num_hidden_layers
    if prefill_cache is not None:
        kv_caches = list(prefill_cache)
    elif input_ids.shape[1] > 1:
        _, kv_caches = model(input_ids[:, :-1], positions[:, :-1], attn_mask[:, :-1], kv_caches)
    next_tok = input_ids[:, -1:]
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    thinking = torch.full_like(finished, thinking_budget is not None)
    thinking_tokens = torch.zeros_like(finished, dtype=torch.long)
    responses, completions, rewards, losses, log_probs = [], [], [], [], []

    for step in range(max_new_tokens):
        logits, kv_caches = model(next_tok, positions[:, -1:], attn_mask, kv_caches)
        scores = logits[:, -1, :].float()
        if forbidden_token_mask is not None:
            scores[~thinking.unsqueeze(-1) & forbidden_token_mask] = -torch.inf
        log_dist = (scores / (temperature or 1.0)).log_softmax(-1)
        next_tok = (scores.argmax(-1, keepdim=True) if temperature == 0 else
                    torch.multinomial(log_dist.exp(), num_samples=1))
        active = ~finished
        forced = (active & thinking & (thinking_tokens >= thinking_budget)
                  if thinking_budget is not None else torch.zeros_like(active))
        if end_think_id is not None:
            next_tok = torch.where(forced[:, None], end_think_id, next_tok)
        next_tok = torch.where(active[:, None], next_tok, pad_id)
        ids = next_tok.squeeze(-1)
        opens = ids == think_id if think_id is not None else torch.zeros_like(active)
        closes = ids == end_think_id if end_think_id is not None else torch.zeros_like(active)
        if step == 0:
            thinking = thinking | (active & opens)
        eos = ids == eos_id if eos_id is not None else torch.zeros_like(active)
        loss_mask = active & ~forced
        reward_mask = active & ~thinking & ~opens & ~closes & ~eos

        responses.append(ids)
        completions.append(active)
        rewards.append(reward_mask)
        losses.append(loss_mask)
        log_probs.append(log_dist.gather(-1, next_tok).squeeze(-1).masked_fill(~loss_mask, 0.0))
        thinking_tokens += (active & thinking & ~closes).long()
        thinking = thinking & ~closes
        finished = finished | (active & eos)
        attn_mask = torch.cat([attn_mask, active[:, None].to(attn_mask.dtype)], dim=-1)
        positions = torch.cat([positions, positions[:, -1:] + active[:, None]], dim=-1)
        if finished.all():
            break

    return Rollout(*(torch.stack(values, dim=-1) for values in (
        responses, completions, rewards, losses, log_probs,
    )))
