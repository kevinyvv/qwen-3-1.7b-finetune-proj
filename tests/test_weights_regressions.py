import pytest
import torch

import json
from safetensors.torch import save_file
from core.utils import load_qwen3_weights


@pytest.mark.parametrize('sharded', [False, True])
def test_load_local_safetensors_layouts(tmp_path, tiny_pair, sharded):
    _, reference = tiny_pair
    reference.config.to_json_file(tmp_path / 'config.json')
    state = {'model.embed_tokens.weight': torch.randn(32, 24),
             'lm_head.weight': torch.randn(32, 24)}
    if sharded:
        weight_map = {}
        for i, (name, tensor) in enumerate(state.items()):
            shard = f'model-{i}.safetensors'
            save_file({name: tensor}, tmp_path / shard)
            weight_map[name] = shard
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    else:
        save_file(state, tmp_path / 'model.safetensors')
    config, loaded = load_qwen3_weights(tmp_path)
    assert config.vocab_size == 32
    assert loaded.keys() == state.keys()
    for name in state:
        torch.testing.assert_close(loaded[name], state[name])
