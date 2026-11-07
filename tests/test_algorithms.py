import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from core.generation import Rollout, generate_rollout
from core.layers import LoRALinear, lora_disabled
from core.sampling import build_forbidden_token_mask, sample_next_token
from training.checkpointing import load_checkpoint, save_checkpoint
from training.rl.grpo import GRPOEngine, GroupedRollouts
from training.rl.reinforce import ReinforceEngine
from training.sft.sft import SFTEngine
from training.sft.sft_utils import collate_sft, load_sft_dataset


@pytest.mark.parametrize('sdpa', [False, True])
@pytest.mark.parametrize('padded', [False, True])
def test_hf_hidden_attention_and_logits_parity(tiny_pair, sdpa, padded):
    model, reference = tiny_pair
    model.model.use_sdpa = sdpa
    for layer in model.model.layers:
        layer.self_attn.use_sdpa = sdpa
    ids = torch.tensor([[1, 7, 8, 9, 10], [0, 0, 1, 11, 12]])
    mask = torch.ones_like(ids)
    if padded:
        mask[1, :2] = 0
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    ours, theirs, handles = {}, {}, []

    def capture(storage, name):
        def hook(module, args, output):
            storage[name] = output[0] if isinstance(output, tuple) else output
        return hook

    for index, (our_layer, hf_layer) in enumerate(zip(model.model.layers, reference.model.layers)):
        for name in ['self_attn', 'mlp']:
            handles.append(getattr(our_layer, name).register_forward_hook(capture(ours, (index, name))))
            handles.append(getattr(hf_layer, name).register_forward_hook(capture(theirs, (index, name))))
    try:
        with torch.no_grad():
            actual_hidden, _ = model.model(ids, positions, mask, [None] * 2)
            expected = reference(ids, attention_mask=mask, position_ids=positions,
                                 output_hidden_states=True)
            actual_logits = model.lm_head(actual_hidden)
    finally:
        for handle in handles:
            handle.remove()
    for key in ours:
        torch.testing.assert_close(ours[key][mask.bool()], theirs[key][mask.bool()], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_hidden[mask.bool()], expected.hidden_states[-1][mask.bool()],
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_logits[mask.bool()], expected.logits[mask.bool()], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('sdpa', [False, True])
@pytest.mark.parametrize('chunks', [[1, 1, 1, 1, 1], [2, 1, 2]])
@pytest.mark.parametrize('padding', ['none', 'all_valid', 'left'])
def test_incremental_cache_matches_full_forward(tiny_pair, sdpa, chunks, padding):
    model, _ = tiny_pair
    model.model.use_sdpa = sdpa
    for layer in model.model.layers:
        layer.self_attn.use_sdpa = sdpa
    ids = torch.tensor([[1, 7, 8, 9, 10], [0, 0, 1, 11, 12]])
    mask = torch.ones_like(ids)
    if padding == 'left':
        mask[1, :2] = 0
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    supplied_mask = None if padding == 'none' else mask
    with torch.no_grad():
        expected, _ = model(ids, positions, supplied_mask, [None] * 2)
        caches, outputs, start = [None] * 2, [], 0
        for size in chunks:
            stop = start + size
            output, caches = model(ids[:, start:stop], positions[:, start:stop],
                                    None if supplied_mask is None else mask[:, :stop], caches)
            outputs.append(output)
            for key, value in caches:
                assert key.shape == value.shape == (2, 2, stop, 8)
            start = stop
        actual = torch.cat(outputs, dim=1)
        fresh, _ = model(ids, positions, supplied_mask, [None] * 2)
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(fresh, expected)


def test_lora_gradients_disable_and_checkpoint_roundtrip(tiny_pair, tmp_path):
    model, reference = tiny_pair
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith(('.A', '.B')))
        if name.endswith('.B'):
            with torch.no_grad():
                parameter.normal_(std=0.1)
    ids, pos = torch.tensor([[1, 7]]), torch.tensor([[0, 1]])
    logits, _ = model(ids, pos, None, [None] * 2)
    logits.square().sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
    with lora_disabled(model), torch.no_grad():
        base, _ = model(ids, pos, None, [None] * 2)
        torch.testing.assert_close(base, reference(ids, position_ids=pos).logits, atol=2e-6, rtol=2e-5)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    path = tmp_path / 'adapter.pt'
    save_checkpoint(path, model, optimizer, 3)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.B.zero_()
    assert load_checkpoint(path, model, optimizer) == 4
    restored, _ = model(ids, pos, None, [None] * 2)
    torch.testing.assert_close(restored, logits)


def test_sft_assistant_eos_and_padding_loss(tmp_path, tiny_pair):
    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            return 'prompt'

        def __call__(self, text, **kwargs):
            return {'input_ids': {'prompt': [1, 7, 8], 'long': [9, 10], 'short': [11]}[text]}

    path = tmp_path / 'sft.jsonl'
    path.write_text('\n'.join(json.dumps({'prompt': 'question', 'response': answer})
                              for answer in ['long', 'short']))
    examples = load_sft_dataset(path, Tokenizer())
    assert examples[0]['labels'] == [-100, -100, -100, 9, 10, 2]
    batch = collate_sft(examples, 0)
    assert batch['labels'][1].tolist() == [-100, -100, -100, 11, 2, -100]
    assert batch['attention_mask'][1].tolist() == [1, 1, 1, 1, 1, 0]
    model, _ = tiny_pair
    engine = SFTEngine(model, Tokenizer(), None, False, 'torch', precision=torch.float32, device='cpu')
    actual = engine.compute_loss(batch)
    mask = batch['attention_mask']
    logits, _ = model(batch['input_ids'], (mask.cumsum(-1)-1).clamp_min(0), mask, [None] * 2)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, 32), batch['labels'][:, 1:].reshape(-1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('parallel', [False, True])
def test_group_prompt_alignment_and_microbatches(tiny_pair, tokenizer, parallel):
    model, _ = tiny_pair
    engine = GRPOEngine(model, tokenizer, None, 'torch')
    batch = {'input_ids': torch.tensor([[1, 7], [0, 1]]),
             'attention_mask': torch.tensor([[1, 1], [0, 1]]), 'prompts': ['alpha', 'beta']}
    torch.manual_seed(11)
    rollouts = engine.generate_rollouts(batch, 4, 0.7, 3, parallel_groups=parallel)
    assert rollouts.response_ids.shape == (3, 2, 4)
    rescored = engine.token_log_probs(model, rollouts, 0.7)
    torch.testing.assert_close(rescored, rollouts.prev_log_probs, atol=2e-6, rtol=2e-5)
    for selection, chunk in rollouts.microbatches(3):
        for index, prompt in enumerate(chunk.prompts):
            assert prompt == batch['prompts'][(selection.start + index) % 2]
        torch.testing.assert_close(engine.token_log_probs(model, chunk, 0.7).reshape(-1, 4),
                                   rescored.reshape(-1, 4)[selection])


def test_group_advantages_ignore_missing_rewards(scripted_policy, tokenizer):
    engine = GRPOEngine(scripted_policy, tokenizer, None, 'torch')
    scores = torch.tensor([[0., 10., 3., torch.nan], [2., 14., 3., torch.nan],
                           [torch.nan, 12., torch.nan, torch.nan]])
    actual = engine.compute_advantages(None, None, scores)
    expected = torch.tensor([[-1., -1.2247449, 0., torch.nan], [1., 1.2247449, 0., torch.nan],
                             [torch.nan, 0., torch.nan, torch.nan]])
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize('microbatch', [None, 1, 3])
@pytest.mark.parametrize('fixed_length', [None, 4])
def test_grpo_clipped_objective_kl_and_gradients(scripted_policy, tokenizer, microbatch, fixed_length):
    model = scripted_policy
    reference = copy.deepcopy(model)
    with torch.no_grad():
        reference.scores[:, 2] += 0.7
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = GRPOEngine(model, tokenizer, optimizer, 'torch', finetune_mode='full', ref_model=reference)
    assert not reference.training and not any(p.requires_grad for p in reference.parameters())
    ids = torch.tensor([[[6, 2, 0], [6, 2, 0]], [[5, 2, 0], [5, 2, 0]]])
    mask = torch.tensor([[[True, True, False]] * 2] * 2)
    rollouts = GroupedRollouts(torch.tensor([[1], [4]]), torch.ones(2, 1, dtype=torch.long),
                               ids, mask, torch.zeros_like(ids, dtype=torch.float), mask, mask)
    rewards = torch.tensor([[0., 14.], [2., 10.]])
    temperature, beta, eps = 0.8, 0.1, 0.2
    with torch.no_grad():
        old = engine.token_log_probs(model, rollouts, temperature)
        desired_ratio = torch.tensor([[[1.4], [0.6]], [[0.6], [1.4]]])
        rollouts.prev_log_probs = (old - desired_ratio.log()).masked_fill(~mask, 0)

    # Independent scalar objective, evaluated directly on the policy's transition table.
    context = torch.cat([rollouts.input_ids.unsqueeze(0).expand(2, -1, -1), ids], -1)[..., :-1]
    current = (model.scores[context] / temperature).log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ref = (reference.scores[context] / temperature).log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    ratio = (current - rollouts.prev_log_probs).exp()
    advantage = torch.tensor([[-1., 1.], [1., -1.]]).unsqueeze(-1)
    clipped = torch.minimum(ratio * advantage, ratio.clamp(1-eps, 1+eps) * advantage)
    delta = ref - current
    kl = delta.exp() - delta - 1
    denom = fixed_length or mask.sum(-1)
    expected_loss = -((clipped - beta * kl) * mask).sum(-1).div(denom).mean()
    expected_grad, = torch.autograd.grad(expected_loss, model.scores)
    with patch.object(engine, 'generate_rollouts', return_value=rollouts), patch.object(optimizer, 'step'):
        engine.fit([{}], temperature=temperature, reward_fn=lambda ids, mask: rewards, kl_beta=beta,
                   micro_batch_size=microbatch, fixed_loss_length=fixed_length)
    torch.testing.assert_close(model.scores.grad, expected_grad, atol=1e-7, rtol=2e-5)
    assert all(p.grad is None for p in reference.parameters())


def test_reinforce_policy_gradient_masks_forced_tokens_and_padding(scripted_policy, tokenizer):
    model = scripted_policy
    engine = ReinforceEngine(model, tokenizer, None, lambda ids, mask: torch.tensor([2., -1.]), device='cpu')
    batch = {'input_ids': torch.tensor([[0, 1], [1, 4]]),
             'attention_mask': torch.tensor([[0, 1], [1, 1]])}
    ids = torch.tensor([[4, 6, 2], [2, 0, 0]])
    completion = torch.tensor([[True, True, True], [True, False, False]])
    loss_mask = completion.clone()
    loss_mask[0, 0] = False  # forced </think> is context, not an action
    rollout = Rollout(ids, completion, loss_mask, loss_mask, torch.zeros(2, 3))
    with patch('training.rl.reinforce.generate_rollout', return_value=rollout):
        actual = engine.compute_loss(batch, temperature=0.8)
    previous = torch.tensor([[1, 4, 6], [4, 2, 0]])
    logprobs = (model.scores[previous] / 0.8).log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    expected = -(torch.tensor([2., -1.]) * (logprobs * loss_mask).sum(-1)).mean()
    actual_grad, = torch.autograd.grad(actual, model.scores)
    expected_grad, = torch.autograd.grad(expected, model.scores)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_eos_stopping_and_forced_thinking_masks(scripted_policy, tokenizer):
    batch = {'input_ids': torch.tensor([[1], [6]]), 'attention_mask': torch.ones(2, 1, dtype=torch.long)}
    rollout = generate_rollout(scripted_policy, tokenizer, batch, 6, temperature=0)
    assert rollout.response_ids.tolist() == [[3, 5, 4, 6, 2], [2, 0, 0, 0, 0]]
    assert rollout.loss_mask[1].tolist() == [True, False, False, False, False]
    forced = generate_rollout(scripted_policy, tokenizer,
                              {k: v[:1] for k, v in batch.items()}, 4, 0, thinking_budget=0)
    assert forced.response_ids.tolist() == [[4, 6, 2]]
    assert forced.loss_mask.tolist() == [[False, True, True]]
    assert forced.completion_mask.all()
    assert forced.reward_mask.tolist() == [[False, True, False]]


def test_forbidden_mask_preserves_special_tokens():
    tokenizer = SimpleNamespace(all_special_ids=[2], added_tokens_decoder={},
                                batch_decode=lambda ids: ['a', 'e', '<eos>', 'E', '猫'])
    assert build_forbidden_token_mask(tokenizer, 5, ascii_only=True).tolist() == [False, True, False, True, True]


@pytest.mark.parametrize('temperature', [0., 0.7])
@pytest.mark.parametrize('top_k,top_p', [(0, 1.), (2, 1.), (0, 0.5), (2, 0.5)])
def test_constrained_sampling_never_selects_forbidden(temperature, top_k, top_p):
    logits = torch.tensor([[0., -torch.inf, -2., -torch.inf]]).expand(64, -1).clone()
    original = logits.clone()
    ids = sample_next_token(logits, temperature, top_k, top_p, torch.Generator().manual_seed(8))
    assert ((ids == 0) | (ids == 2)).all()
    torch.testing.assert_close(logits, original)
