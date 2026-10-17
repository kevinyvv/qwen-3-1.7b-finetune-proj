import math

import torch


def build_forbidden_token_mask(tokenizer, vocab_size, forbidden_chars='eE', ascii_only=False,
                               device=None, preserve_added_tokens=False):
    """Mask decoded token text while preserving special tokens such as EOS."""
    special_ids = set(tokenizer.all_special_ids)
    if preserve_added_tokens:
        special_ids.update(tokenizer.added_tokens_decoder)
    texts = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    return torch.tensor([
        i not in special_ids and (
            any(c in forbidden_chars for c in text) or (ascii_only and not text.isascii())
        )
        for i, text in enumerate(texts)
    ], dtype=torch.bool, device=device)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample [batch, 1] token IDs from [batch, vocab] logits.

    Apply temperature, then top-k, then top-p (nucleus) filtering.
    temperature=0 selects greedily; top_k=0 and top_p=1 disable filtering.
    Input logits are not modified; -inf can mask forbidden tokens, provided
    each row has at least one finite logit. The optional RNG must match the
    logits' device.
    """
    if logits.ndim != 2 or logits.size(-1) == 0:
        raise ValueError("logits must have shape [batch, vocab] with a nonempty vocabulary")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
    token_ids = None
    if top_k > 0:
        scores, token_ids = scores.topk(min(top_k, scores.size(-1)), dim=-1)
    elif top_p < 1:
        scores, token_ids = scores.sort(dim=-1, descending=True)

    if top_p < 1:
        # Keep the token that crosses the threshold, and always keep the first.
        remove = scores.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores = scores.masked_fill(remove, -torch.inf)

    sampled = torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)
    return sampled if token_ids is None else token_ids.gather(-1, sampled)
