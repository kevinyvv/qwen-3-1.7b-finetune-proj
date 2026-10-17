import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core.generation import generate_rollout
from core.layers import lora_disabled
from training.engine import TrainingEngine


@contextlib.contextmanager
def swapped_adapters(model, state):
    """Temporarily load other adapter values (name -> tensor), restoring the current ones after."""
    params = dict(model.named_parameters())
    saved = {name: params[name].detach().clone() for name in state}
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                params[name].copy_(value)


@dataclass
class GroupedRollouts:
    input_ids: torch.Tensor  # [B, prompt_len]
    attn_mask: torch.Tensor  # [B, prompt_len]
    response_ids: torch.Tensor  # [G, B, T]
    completion_mask: torch.Tensor  # same as above
    prev_log_probs: torch.Tensor  # same
    reward_mask: torch.Tensor
    loss_mask: torch.Tensor
    prompts: list | None = None  # prompt texts, one per batch row, for rewards that need them


    def microbatches(self, size=None):
        group_size, batch_size, response_len = self.response_ids.shape
        if size is None:
            yield slice(0, group_size * batch_size), self
            return
        for start in range(0, group_size * batch_size, size):
            selection = slice(start, min(start + size, group_size * batch_size))
            prompt_indices = torch.arange(selection.start, selection.stop,
                                          device=self.input_ids.device) % batch_size
            yield selection, GroupedRollouts(
                input_ids=self.input_ids[prompt_indices],
                attn_mask=self.attn_mask[prompt_indices],
                prompts=None if self.prompts is None else [self.prompts[i] for i in prompt_indices.tolist()],
                **{name: getattr(self, name).reshape(-1, response_len)[selection].unsqueeze(0)
                   for name in ('response_ids', 'completion_mask', 'prev_log_probs',
                                'reward_mask', 'loss_mask')},
            )


class GRPOEngine(TrainingEngine):
    def __init__(self, model, tokenizer, optimizer, loss_backend,
                 precision=torch.bfloat16, finetune_mode='lora', ref_model=None,
                 forbidden_token_mask=None, ref_adapter_state=None):
        # Supply a separate reference on the same device to anchor KL to an SFT
        # policy (including its adapters), or when training the full model.
        # forbidden_token_mask [vocab] blocks the same tokens when sampling and when
        # scoring for training, so both use the same masked distribution.
        # ref_adapter_state anchors KL to other adapter values (e.g. the SFT adapters a run
        # started from) without a second copy of the model in memory.
        if ref_model is not None and ref_adapter_state is not None:
            raise ValueError("Pass ref_model or ref_adapter_state, not both")
        self.model = model
        self.forbidden_token_mask = forbidden_token_mask
        self.ref_adapter_state = ref_adapter_state
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.finetune_mode = finetune_mode
        self.ref_model = ref_model
        
        if ref_model is not None:
            if ref_model is model:
                raise ValueError("ref_model must be separate from the trainable model")
            ref_model.eval()
            ref_model.requires_grad_(False)

    def token_log_probs(self, model, rollouts, temperature=1.0):
        group_size, batch_size, response_len = rollouts.response_ids.shape
        prompt_len = rollouts.input_ids.size(1)
        
        input_ids = torch.cat([
            rollouts.input_ids.unsqueeze(0).expand(group_size, -1, -1),
            rollouts.response_ids,
        ], dim=-1).reshape(group_size * batch_size, -1)
        
        attn_mask = torch.cat([
            rollouts.attn_mask.unsqueeze(0).expand(group_size, -1, -1),
            rollouts.completion_mask.to(rollouts.attn_mask.dtype),
        ], dim=-1).reshape(group_size * batch_size, -1)
        
        positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
        kv_caches = [None] * model.config.num_hidden_layers

        logits, _ = model(input_ids[:, :-1], positions[:, :-1], attn_mask[:, :-1], kv_caches)
        
        scores = logits[:, prompt_len - 1:].float()
        if self.forbidden_token_mask is not None:
            scores = scores.masked_fill(self.forbidden_token_mask.to(scores.device), -torch.inf)
        log_logits = (scores / temperature).log_softmax(dim=-1)
        log_probs = log_logits.gather(
            dim=-1,
            index=rollouts.response_ids.reshape(group_size * batch_size, response_len, 1),
        ).squeeze(-1).reshape(group_size, batch_size, response_len)
        if self.forbidden_token_mask is not None and torch.isinf(log_probs[rollouts.loss_mask]).any():
            raise ValueError("A trained token is forbidden by the mask")

        return log_probs.masked_fill(~rollouts.loss_mask, 0.0)

    def generate_rollouts(self, batch, max_new_tokens, temperature, group_size, thinking_budget=None,
                          parallel_groups=False):
        """parallel_groups samples every group in one batch (group_size x the prompts), which
        is much faster on a GPU than one batch per group."""
        if temperature <= 0 or group_size < 1:
            raise ValueError("temperature and group_size must be positive")
        if self.forbidden_token_mask is not None and thinking_budget is not None:
            # token_log_probs cannot tell thinking from answer tokens, and the mask skips thinking
            raise ValueError("The forbidden-token mask requires thinking to be off")
        device = next(self.model.parameters()).device
        prompts = batch.get('prompts')
        batch = {k: v.to(device) for k, v in batch.items() if k != 'prompts'}
        input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
        positions = (attn_mask.cumsum(-1) - 1).clamp(min=0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        groups = []

        if parallel_groups:
            expanded = {'input_ids': input_ids.repeat(group_size, 1),
                        'attention_mask': attn_mask.repeat(group_size, 1)}  # group-major rows
            with torch.no_grad():
                rollout = generate_rollout(
                    self.model, self.tokenizer, expanded, max_new_tokens, temperature,
                    thinking_budget, forbidden_token_mask=self.forbidden_token_mask,
                )
            batch_size = input_ids.size(0)

            def field(name, value=0):
                values = F.pad(getattr(rollout, name), (0, max_new_tokens - rollout.response_ids.size(-1)),
                               value=value)
                return values.reshape(group_size, batch_size, max_new_tokens)

            return GroupedRollouts(
                input_ids=input_ids, attn_mask=attn_mask,
                response_ids=field('response_ids', pad_id), completion_mask=field('completion_mask'),
                prev_log_probs=field('log_probs'), reward_mask=field('reward_mask'),
                loss_mask=field('loss_mask'), prompts=prompts,
            )

        with torch.no_grad():
            prefill_cache = [None] * self.model.config.num_hidden_layers
            if input_ids.shape[1] > 1:
                _, prefill_cache = self.model(
                    input_ids[:, :-1], positions[:, :-1], attn_mask[:, :-1], prefill_cache,
                )
            for _ in range(group_size):
                groups.append(generate_rollout(
                    self.model, self.tokenizer, batch, max_new_tokens, temperature,
                    thinking_budget, prefill_cache, forbidden_token_mask=self.forbidden_token_mask,
                ))

        def stack(field, value=0):
            return torch.stack([
                F.pad(getattr(group, field), (0, max_new_tokens - group.response_ids.size(-1)), value=value)
                for group in groups
            ])

        return GroupedRollouts(
            input_ids=input_ids,
            attn_mask=attn_mask,
            response_ids=stack('response_ids', pad_id),
            completion_mask=stack('completion_mask'),
            prev_log_probs=stack('log_probs'),
            reward_mask=stack('reward_mask'),
            loss_mask=stack('loss_mask'),
            prompts=prompts,
        )

    def score(self, rollouts, reward_fn):
        """[G, B] rewards. NaN marks an answer with no reward (for example a failed judge call)."""
        if rollouts.prompts is None:
            return reward_fn(rollouts.response_ids, rollouts.reward_mask)
        return reward_fn(rollouts.response_ids, rollouts.reward_mask, prompts=rollouts.prompts)

    def compute_advantages(self, rollouts, reward_fn, scores=None):
        """Group-normalized advantages. Answers without a reward get NaN and are
        left out of the group statistics; fit leaves them out of the update."""
        if scores is None:
            scores = self.score(rollouts, reward_fn)  # [G, B]
        valid = ~torch.isnan(scores)
        count = valid.sum(dim=0, keepdim=True).clamp_min(1)
        filled = torch.where(valid, scores, 0.0)
        mean = filled.sum(dim=0, keepdim=True) / count
        std = (torch.where(valid, scores - mean, 0.0).square().sum(dim=0, keepdim=True) / count).sqrt()
        return torch.where(valid, (scores - mean) / std.clamp_min(1e-6), torch.nan)

    def fit(self, data_loader, max_new_tokens=64, temperature=1.0, epochs=1, batch_epochs=1,
            group_size=2, reward_fn=None, kl_beta=0.02, eps=0.2, thinking_budget=None,
            micro_batch_size=None, on_step=None, recompute_old_log_probs=False, parallel_groups=False,
            fixed_loss_length=None):
        """on_step(dict(rollouts, scores, ...)) runs after each optimizer step; returning
        True stops training. recompute_old_log_probs scores the sampled answers again with
        the training forward pass before updating, so the policy ratio starts at exactly 1
        even when cached sampling and full forward passes round differently (bf16).
        fixed_loss_length: divide every answer's summed loss by this number instead of by
        the answer's own length. Dividing by its own length gives each word of a short
        answer a bigger push than each word of a long answer with the same advantage,
        which favors short answers (Liu et al. 2025, "Understanding R1-Zero-Like Training")."""

        if micro_batch_size is not None and (
            type(micro_batch_size) is not int or micro_batch_size < 1
        ):
            raise ValueError("micro_batch_size must be a positive integer or None")

        if kl_beta and self.ref_model is None:
            if self.finetune_mode != 'lora' or any(
                p.requires_grad and not name.endswith((".A", ".B"))
                for name, p in self.model.named_parameters()
            ):
                raise ValueError("KL requires ref_model or a frozen base with LoRA-only training")

        self.model.train()
        
        for epoch in range(epochs):
            for batch in data_loader:
                with torch.no_grad():
                    rollouts = self.generate_rollouts(
                        batch, max_new_tokens=max_new_tokens,
                        temperature=temperature, group_size=group_size, thinking_budget=thinking_budget,
                        parallel_groups=parallel_groups,
                    )

                    # rewards may come from the CPU (an API judge); train on the rollout device
                    scores = self.score(rollouts, reward_fn).to(rollouts.response_ids.device, torch.float)
                    advantages = self.compute_advantages(rollouts, reward_fn, scores).reshape(-1, 1)
                    has_reward = ~torch.isnan(advantages)  # [G * B, 1]
                    advantages = advantages.nan_to_num(0.0)
                    response_len = rollouts.response_ids.size(-1)
                    logprob_gap = None
                    if recompute_old_log_probs:
                        recomputed = torch.cat([
                            self.token_log_probs(self.model, chunk, temperature).reshape(-1, response_len)
                            for _, chunk in rollouts.microbatches(micro_batch_size)
                        ]).reshape(rollouts.prev_log_probs.shape)
                        gap = (recomputed - rollouts.prev_log_probs)[rollouts.loss_mask].abs()
                        logprob_gap = dict(mean=gap.mean().item(), max=gap.max().item()) if gap.numel() else None
                        rollouts.prev_log_probs = recomputed
                    chunks = list(rollouts.microbatches(micro_batch_size))
                    if kl_beta:
                        if self.ref_model is not None:
                            ref_log_probs = torch.cat([
                                self.token_log_probs(self.ref_model, chunk, temperature).reshape(-1, response_len)
                                for _, chunk in chunks
                            ])
                        elif self.ref_adapter_state is not None:
                            with swapped_adapters(self.model, self.ref_adapter_state):
                                ref_log_probs = torch.cat([
                                    self.token_log_probs(self.model, chunk, temperature).reshape(-1, response_len)
                                    for _, chunk in chunks
                                ])
                        else:
                            with lora_disabled(self.model):
                                ref_log_probs = torch.cat([
                                    self.token_log_probs(self.model, chunk, temperature).reshape(-1, response_len)
                                    for _, chunk in chunks
                                ])

                for b_epoch in range(batch_epochs):
                    self.optimizer.zero_grad()
                    kl_sum, kl_tokens = 0.0, 0
                    for selection, chunk in chunks:
                        log_probs = self.token_log_probs(self.model, chunk, temperature).reshape(-1, response_len)
                        prev_log_probs = chunk.prev_log_probs.reshape(-1, response_len)
                        mask = chunk.loss_mask.reshape(-1, response_len) & has_reward[selection]
                        ratio = (log_probs - prev_log_probs).exp()
                        adv = advantages[selection]
                        surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)

                        if kl_beta:
                            log_ref_ratio = ref_log_probs[selection] - log_probs
                            kl = log_ref_ratio.exp() - log_ref_ratio - 1
                            surrogate = surrogate - kl_beta * kl
                            kl_sum += (kl.detach() * mask).sum().item()
                            kl_tokens += mask.sum().item()

                        length = fixed_loss_length if fixed_loss_length else mask.sum(-1).clamp_min(1)
                        sequence_loss = -(surrogate * mask).sum(-1) / length
                        loss = sequence_loss.sum() / advantages.size(0)
                        loss.backward()
                    self.optimizer.step()
                if on_step is not None and on_step(dict(
                        rollouts=rollouts, scores=scores, epoch=epoch, logprob_gap=logprob_gap,
                        kl=kl_sum / max(kl_tokens, 1) if kl_beta else None)):
                    return
