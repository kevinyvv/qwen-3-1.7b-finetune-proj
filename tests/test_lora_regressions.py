import pytest
import torch

from core.layers import lora_disabled


def test_nested_lora_disable_preserves_reference_output(tiny_pair):
    model, _ = tiny_pair
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('.B'):
                parameter.normal_(std=0.1)
    ids = torch.tensor([[1, 7, 9]])
    positions = torch.arange(3).unsqueeze(0)

    def logits():
        return model(ids, positions, None, [None] * 2)[0]

    adapted = logits()
    with lora_disabled(model):
        base = logits()
        with lora_disabled(model):
            torch.testing.assert_close(logits(), base)
        torch.testing.assert_close(logits(), base)
    torch.testing.assert_close(logits(), adapted)
    assert not torch.allclose(adapted, base)

def test_lora_disable_restores_individual_states_on_exception(tiny_pair):
    model, _ = tiny_pair
    adapter = model.model.layers[0].self_attn.q_proj_lora
    adapter.enabled = False
    before = {name: module.enabled for name, module in model.named_modules()
              if hasattr(module, 'enabled')}
    with pytest.raises(RuntimeError), lora_disabled(model):
        raise RuntimeError('abort reference scoring')
    assert before == {name: module.enabled for name, module in model.named_modules()
                      if hasattr(module, 'enabled')}
