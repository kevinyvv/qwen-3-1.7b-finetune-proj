Run from the repository root with PyTorch, Transformers, safetensors and pytest installed:

```sh
python -m pytest tests -q
```

The tests use CPU tensors, randomly initialized two-layer Qwen3 models and mock
rollouts. They do not download model weights or tokenizers, call a judge API or
run a training experiment. The SFT template test uses a stub tokenizer to check
the dataset builder's label boundary, EOS and padding handling.

Hugging Face parity checks compare attention and MLP outputs, final hidden states
and logits at non-padding positions (`atol=2e-6`, `rtol=2e-5`, float32). Cache tests
cover batch size two, left padding, one-token and chunked decoding, and both
manual attention and SDPA. The GRPO objective test compares gradients of the
actual engine against an independently calculated clipped surrogate and KL
penalty, including microbatch accumulation and fixed-length reduction.
