"""Executable SFT contract; see README.md for the proposed batch API."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from training.engine import SFTEngine


class TinyCausalLM(nn.Module):
    """Small deterministic causal model using this repo's forward signature."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_hidden_layers": 1})()
        self.model = type("Backbone", (), {"config": self.config})()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(32).reshape(8, 4) / 32)
            self.head.weight.copy_(torch.cos(torch.arange(32).reshape(8, 4).float()))
            self.head.bias.zero_()

    def forward(self, input_ids, positions, attn_mask, kv_caches):
        hidden = self.embedding(input_ids) * attn_mask.unsqueeze(-1)
        return self.head(hidden.cumsum(dim=1)), kv_caches


def batch():
    # Two prompt tokens, then two response tokens (including EOS=2).
    # Padding also uses ID 2: only labels/masks distinguish it from EOS.
    return {
        "input_ids": torch.tensor([[1, 3, 4, 2, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 4, 2, -100]]),
    }


def token_losses(model, sample):
    """Independent oracle: score each target from its prefix individually."""
    losses = []
    for row in range(sample["input_ids"].shape[0]):
        for pos in range(1, sample["input_ids"].shape[1]):
            target = sample["labels"][row, pos]
            if target.item() == -100:
                continue
            ids = sample["input_ids"][row:row + 1, :pos]
            mask = sample["attention_mask"][row:row + 1, :pos]
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            logits, _ = model(ids, positions, mask, [None])
            losses.append(-F.log_softmax(logits[0, -1], dim=-1)[target])
    return torch.stack(losses)


class TestSFTEngine(unittest.TestCase):
    def make_engine(self):
        model = TinyCausalLM()
        return SFTEngine(
            model=model,
            tokenizer=None,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            gradient_checkpointing=False,
            loss_backend="torch",
            precision="fp32",
            finetune_mode="full",
            device="cpu",
        )

    def test_constructor_starts_without_checkpoint(self):
        engine = self.make_engine()
        self.assertEqual(engine.epoch, 0)

    def test_loss_matches_individual_prefix_predictions(self):
        engine = self.make_engine()
        sample = batch()
        original = copy.deepcopy(sample)
        loss = engine.compute_loss(sample)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss.requires_grad)
        torch.testing.assert_close(loss, token_losses(engine.model, sample).mean())
        for key in sample:
            torch.testing.assert_close(sample[key], original[key])

    def test_loss_gradients_match_individual_prefix_predictions(self):
        engine = self.make_engine()
        reference = copy.deepcopy(engine.model)
        engine.compute_loss(batch()).backward()
        token_losses(reference, batch()).mean().backward()
        for actual, expected in zip(engine.model.parameters(), reference.parameters()):
            self.assertIsNotNone(actual.grad)
            self.assertTrue(torch.isfinite(actual.grad).all())
            torch.testing.assert_close(actual.grad, expected.grad)

    def test_left_and_right_padding_do_not_change_loss(self):
        engine = self.make_engine()
        sample = batch()
        unpadded = {key: value[:, :4] for key, value in sample.items()}
        left_padded = {
            "input_ids": torch.tensor([[2, 1, 3, 4, 2]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, -100, 4, 2]]),
        }
        expected = engine.compute_loss(unpadded)
        torch.testing.assert_close(engine.compute_loss(sample), expected)
        torch.testing.assert_close(engine.compute_loss(left_padded), expected)

    def test_eos_is_supervised_even_when_it_is_also_the_pad_id(self):
        engine = self.make_engine()
        sample = batch()
        sample["labels"][0, 2] = -100  # Only genuine EOS remains supervised.
        torch.testing.assert_close(
            engine.compute_loss(sample), token_losses(engine.model, sample).mean()
        )

    def test_all_ignored_labels_raise_instead_of_nan(self):
        engine = self.make_engine()
        sample = batch()
        sample["labels"].fill_(-100)
        with self.assertRaises(ValueError):
            engine.compute_loss(sample)

    def test_evaluation_is_token_weighted_and_does_not_update_model(self):
        engine = self.make_engine()
        first, second = batch(), batch()
        second["input_ids"][0, 2] = 7
        second["labels"][0, 2] = -100  # One target versus two in first.
        expected = torch.cat([
            token_losses(engine.model, first), token_losses(engine.model, second)
        ]).mean().item()
        before = copy.deepcopy(engine.model.state_dict())
        grad_modes = []
        handle = engine.model.register_forward_pre_hook(
            lambda *_: grad_modes.append(torch.is_grad_enabled())
        )
        try:
            actual = engine.evaluate([first, second])
        finally:
            handle.remove()
        self.assertAlmostEqual(float(actual), expected, places=6)
        self.assertTrue(grad_modes)
        self.assertFalse(any(grad_modes))
        for key, value in engine.model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(p.grad is None for p in engine.model.parameters()))

    def test_fit_matches_two_manual_optimizer_steps(self):
        engine = self.make_engine()
        reference = copy.deepcopy(engine.model)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            token_losses(reference, batch()).mean().backward()
            optimizer.step()
        engine.fit([batch()], [batch()], epochs=2, eval_every=1, save_every=100)
        for actual, expected in zip(engine.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(engine.model.training)

    def test_checkpoint_restores_optimizer_for_next_update(self):
        engine = self.make_engine()
        engine.optimizer.zero_grad()
        engine.compute_loss(batch()).backward()
        engine.optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            engine.save_checkpoint(path)
            restored = self.make_engine()
            restored.load_checkpoint(path)
        for instance in (engine, restored):
            instance.optimizer.zero_grad()
            instance.compute_loss(batch()).backward()
            instance.optimizer.step()
        for actual, expected in zip(restored.model.parameters(), engine.model.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
