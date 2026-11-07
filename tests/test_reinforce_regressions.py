import pytest
import torch

from training.rl.reinforce import ReinforceEngine


@pytest.mark.parametrize('temperature', [0.0, -1.0, torch.inf, torch.nan])
@pytest.mark.parametrize('entry', ['compute_loss', 'compute_log_probs'])
def test_reinforce_rejects_nonstochastic_temperature(scripted_policy, tokenizer, temperature, entry):
    engine = ReinforceEngine(scripted_policy, tokenizer, None, lambda ids, mask: torch.ones(ids.size(0)),
                             device='cpu')
    batch = {'input_ids': torch.tensor([[1]]), 'attention_mask': torch.ones(1, 1, dtype=torch.long)}
    # A greedy rollout has no finite-temperature on-policy REINFORCE objective.
    with pytest.raises(ValueError, match='temperature'):
        if entry == 'compute_loss':
            engine.compute_loss(batch, max_new_tokens=2, temperature=temperature)
        else:
            engine.compute_log_probs(batch, torch.tensor([[1, 3]]), torch.ones(1, 1, dtype=torch.bool),
                                     temperature=temperature)
