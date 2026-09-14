SFT correctness tests
=====================

Run from the project directory with the workspace virtual environment:

```sh
/home/kevin/projects/.venv/bin/python -m unittest discover -s tests -v
```

These are implementation-target tests: the current SFTEngine scaffold is expected
to fail. They do not download weights, require a GPU, or require pytest.

Proposed compute_loss contract:

- Accept a dictionary of input_ids, attention_mask, and labels tensors [B, T].
- Labels are unshifted token IDs; -100 excludes prompts and padding from loss.
- Predict labels[:, 1:] from logits[:, :-1], averaging over supervised tokens.
- Preserve the input batch and return a differentiable scalar loss.
- Reject batches with no supervised next-token targets using ValueError.
- Use the supplied model's device; these tests supply a CPU model.

The test model has the repository's forward signature and causal dependence on
prefix tokens. An independent oracle forwards each prefix separately, verifying
both loss and gradients against the batched training path. Tests also cover EOS
masking, left/right padding, token-weighted evaluation without gradients, two
optimizer updates, and restoration of optimizer state for the next update.

Tokenization belongs upstream of this contract; raw prompt/response formatting
and chat-template label construction need separate tests once their API exists.
These tests do not yet validate the real Qwen model, LoRA freezing, fused kernels,
activation checkpointing, or exact data/RNG continuation after resume.
