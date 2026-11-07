import pytest
import torch

from core.generation import generate_rollout
from core.sampling import sample_next_token


@pytest.mark.parametrize('temperature', [0.0, 0.7])
@pytest.mark.parametrize('invalid', [-torch.inf, torch.inf, torch.nan])
def test_sampler_rejects_invalid_distributions(temperature, invalid):
    scores = torch.tensor([[0.0, -torch.inf], [invalid, invalid]])
    with pytest.raises(ValueError, match='finite'):
        sample_next_token(scores, temperature=temperature)

@pytest.mark.parametrize('temperature', [0.0, 0.7])
def test_rollout_rejects_all_forbidden(scripted_policy, tokenizer, temperature):
    batch = {'input_ids': torch.tensor([[1]]), 'attention_mask': torch.ones(1, 1, dtype=torch.long)}
    with pytest.raises(ValueError, match='finite'):
        generate_rollout(scripted_policy, tokenizer, batch, 2, temperature,
                         forbidden_token_mask=torch.ones(8, dtype=torch.bool))
