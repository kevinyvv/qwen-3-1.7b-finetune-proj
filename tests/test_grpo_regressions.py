import pytest
import torch

from training.rl.grpo import GRPOEngine


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('forbidden_id', [5, 7])
def test_grpo_rescores_spontaneous_thinking_with_sampling_distribution(scripted_policy, tokenizer,
                                                                     parallel, forbidden_id):
    with torch.no_grad():
        scripted_policy.scores[:, 7] = 2.0  # makes a changed normalizer observable
    forbidden = torch.zeros(8, dtype=torch.bool)
    forbidden[forbidden_id] = True
    engine = GRPOEngine(scripted_policy, tokenizer, None, 'torch', forbidden_token_mask=forbidden)
    batch = {'input_ids': torch.tensor([[1], [4]]), 'attention_mask': torch.ones(2, 1, dtype=torch.long)}
    torch.manual_seed(0)
    rollouts = engine.generate_rollouts(batch, 6, 0.7, 2, parallel_groups=parallel)
    assert (rollouts.response_ids[:, 0, :5] == torch.tensor([3, 5, 4, 6, 2])).all()
    assert (rollouts.response_ids[:, 1, :2] == torch.tensor([6, 2])).all()
    scored = engine.token_log_probs(scripted_policy, rollouts, temperature=0.7)
    torch.testing.assert_close(scored, rollouts.prev_log_probs)
    ratio = (scored - rollouts.prev_log_probs).exp()
    torch.testing.assert_close(ratio[rollouts.loss_mask], torch.ones_like(ratio[rollouts.loss_mask]))
    (-scored[rollouts.loss_mask].mean()).backward()
    assert torch.isfinite(scripted_policy.scores.grad).all()
