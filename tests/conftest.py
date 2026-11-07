from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3

from core.model import Qwen3ForCausalLM


@pytest.fixture
def tiny_pair():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=32, hidden_size=24, intermediate_size=40,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    reference = HFQwen3(config).eval()
    model = Qwen3ForCausalLM(config).eval()
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected
    assert all(name.endswith((".A", ".B")) for name in missing)
    return model, reference


@pytest.fixture
def tokenizer():
    return SimpleNamespace(
        pad_token_id=0, eos_token_id=2,
        get_vocab=lambda: {"<think>": 3, "</think>": 4},
    )


class ScriptedPolicy(torch.nn.Module):
    """Predict a fixed next action from the preceding token, with trainable logits."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        scores = torch.full((8, 8), -8.0)
        # prompt -> <think> -> forbidden-in-answer word -> </think> -> answer -> EOS
        for previous, action in [(1, 3), (3, 5), (5, 4), (4, 6), (6, 2)]:
            scores[previous, action] = 8.0
        self.scores = torch.nn.Parameter(scores)

    def forward(self, input_ids, positions, attn_mask, kv_caches):
        return self.scores[input_ids], kv_caches


@pytest.fixture
def scripted_policy():
    return ScriptedPolicy()
