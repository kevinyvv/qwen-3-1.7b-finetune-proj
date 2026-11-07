import pytest
import torch

from unittest.mock import patch
from training.sft.sft import SFTEngine


def test_sft_final_checkpoint_records_completed_epoch(tmp_path, tiny_pair, tokenizer):
    model, _ = tiny_pair
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = tmp_path / 'latest.pt'
    engine = SFTEngine(model, tokenizer, optimizer, False, 'torch', precision=torch.float32,
                       device='cpu', path=path)
    # Exercise the actual epoch/checkpoint bookkeeping without model training.
    with patch.object(engine, 'compute_loss', side_effect=lambda batch: model.lm_head.weight.sum() * 0):
        engine.fit([{'input_ids': torch.tensor([[1, 7]]),
                     'attention_mask': torch.ones(1, 2, dtype=torch.long),
                     'labels': torch.tensor([[-100, 7]])}], [], epochs=2)
    checkpoint = torch.load(path, weights_only=True)
    assert engine.epoch == checkpoint['epoch'] == 2
    engine.epoch = 0
    engine.load_checkpoint(path)
    assert engine.epoch == 2
