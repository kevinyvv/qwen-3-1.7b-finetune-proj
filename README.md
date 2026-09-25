# Teaching Qwen3 1.7B to write without "e"

I'm fine-tuning Qwen3 1.7B locally. The goal is to get useful answers that never use the letter `e`. For example, this is the kind of response I'm aiming for (not a model result yet):

> **Prompt:** "What did you do today? Don't use the letter e."
>
> **Answer:** "I ran, had lunch, and got back to work."

Banning tokens containing `e` does not teach the model how to write a fluent answer within that constraint.

## Where the code stands

- `core/` contains my PyTorch implementation of the Qwen3 forward pass: RoPE, grouped-query attention, RMSNorm, a gated MLP, and KV-cache decoding. It loads pretrained Qwen3 weights and adds LoRA adapters.
- `training/` and `data/sft/` contain the supervised fine-tuning path: prompt/response JSONL, response-only labels, next-token loss, evaluation, gradient checkpointing, metrics, and adapter checkpoints. The fused loss uses Liger Kernel.
- `train_loop/` contains an early reward-based experiment for avoiding `e`. The reusable RL engine is still a scaffold.

I first overfit one prompt/response pair to check that the forward pass and training setup could learn a small example. I'm now cleaning up the training engines and comparing ways to balance the letter constraint with fluency.

## Setup

Download the base weights to the path expected by `core/utils.py`:

```sh
hf download Qwen/Qwen3-1.7B --local-dir ./weights/qwen3-1.7b
```

The code uses PyTorch, Transformers, Safetensors, and Hugging Face Hub. The SFT script also uses Liger Kernel and Fire. Dependency versions and a stable training command are still in progress.
