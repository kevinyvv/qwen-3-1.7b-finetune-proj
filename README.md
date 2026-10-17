# Teaching Qwen3-1.7B to write without the letter "e"

Code, data, and results for my write-up *Post-training Qwen3 1.7B*: getting a small model to answer
questions in lipograms (text with no `e` or `E`) that still read well and answer the question.

Everything here is written from scratch in PyTorch: the Qwen3 forward pass, LoRA, generation with a KV
cache, an SFT engine, and a GRPO engine. It all ran on one 16 GB GPU.

## Results

Each number is the **usable rate** on 200 answers (50 held-out prompts × 4 samples). An answer is usable
if it contains no `e` and an LLM judge scores it at least 4 out of 5 on both coherence and answering the
prompt.

| Setup | Usable |
|---|---:|
| Plain Qwen3-1.7B, no e rule (judged on coherence and answering only) | 80.5% |
| Told not to use e, nothing else | 0% |
| Told not to use e, with tokens containing e masked out (**baseline**) | 26% |
| Baseline + SFT on 180 hand-built e-free answers | 35% |
| Baseline + 300 GRPO steps | 43% |
| SFT adapter + 300 GRPO steps | **60.5%** |

The mask handles the e rule on its own; training is about making the answers coherent under it.

## Where to find things

| Topic | Files |
|---|---|
| Qwen3 forward pass (RoPE, grouped-query attention, RMSNorm, KV cache) | [`core/model.py`](core/model.py) |
| LoRA, and switching it off to get the reference model | [`core/layers.py`](core/layers.py) (`LoRALinear`, `lora_disabled`) |
| The mask that blocks every token containing e | [`core/sampling.py`](core/sampling.py) (`build_forbidden_token_mask`) |
| Generation | [`core/generation.py`](core/generation.py) |
| The e check | [`core/checks.py`](core/checks.py) |
| The LLM judge and the usable rule | [`core/judge.py`](core/judge.py) (`JUDGE_SYSTEM`, `usable`) |
| Generating eval answers | [`scripts/evaluate.py`](scripts/evaluate.py) |
| Judging, reports, and run comparisons (with bootstrap ranges) | [`scripts/judge_eval.py`](scripts/judge_eval.py) |
| Checking the judge on known good and bad answers | [`scripts/judge_check.py`](scripts/judge_check.py) |
| Prompt dataset (No Robots + Dolly, filtered and split) | [`data/prompts/make_prompt_data.py`](data/prompts/make_prompt_data.py) |
| Eval prompts, and the ones dropped because they need an e | [`data/prompts/short_eval_50.jsonl`](data/prompts/short_eval_50.jsonl), [`short_eval_dropped.jsonl`](data/prompts/short_eval_dropped.jsonl) |
| RL prompt pool (1,235 prompts) and the keep/drop decisions behind it | [`data/prompts/rl_pool.jsonl`](data/prompts/rl_pool.jsonl), [`merge_rl_pool.py`](data/prompts/merge_rl_pool.py) |
| SFT on the novel *Gadsby* | [`data/sft/make_gadsby_data.py`](data/sft/make_gadsby_data.py), [`scripts/sft_gadsby.py`](scripts/sft_gadsby.py) |
| The 200 e-free answers, and the 15–180 example training splits | [`data/rewrites/verified_200_20261015.jsonl`](data/rewrites/verified_200_20261015.jsonl), [`data/rewrites/splits/`](data/rewrites/splits) |
| Training engine base class (checkpoints, metrics) | [`training/engine.py`](training/engine.py) |
| SFT engine | [`training/sft/sft.py`](training/sft/sft.py) (`SFTEngine`), runner [`scripts/sft_rewrites.py`](scripts/sft_rewrites.py) |
| GRPO engine | [`training/rl/grpo.py`](training/rl/grpo.py) (`GRPOEngine`) |
| GRPO with the judge as the reward | [`scripts/rl_judge.py`](scripts/rl_judge.py), pre-check [`scripts/rl_check.py`](scripts/rl_check.py) |
| REINFORCE engine | [`training/rl/reinforce.py`](training/rl/reinforce.py) |
| First attempt: rewarding single tokens without e | [`train_loop/simple_train.py`](train_loop/simple_train.py) |
| Checking the forward pass against Hugging Face | [`scripts/check_parity.py`](scripts/check_parity.py) |
| Full lab notebook, every experiment in order | [`docs/experiment_history.md`](docs/experiment_history.md) |

## Runs included

Each run folder keeps its config, metrics, judge scores, and the answers themselves. To read answers, open
`responses.txt` or `samples_to_read.txt`; the numbers are in `short_eval_report.json`. Model checkpoints
are not included.

| Folder | What it is |
|---|---|
| [`runs/short_eval_20261016/`](runs/short_eval_20261016) | 1.7B and 4B before any training: no rule, instruction only, and with the mask |
| [`runs/judge_check_20261016_v3/`](runs/judge_check_20261016_v3) | The judge scored on known good, earlier reviewed, and known bad answers |
| [`runs/gadsby_short_eval_20261017/`](runs/gadsby_short_eval_20261017) | The *Gadsby*-trained adapter on the eval set (24.5% usable) |
| [`runs/sft_sizes_20261016/`](runs/sft_sizes_20261016) | SFT on 15, 30, 45, 90, 135, and 180 examples, each evaluated |
| [`runs/rl_check_20261016/`](runs/rl_check_20261016) | 16 samples per prompt, to confirm RL has usable answers to learn from |
| [`runs/rl_20261016/`](runs/rl_20261016) | GRPO from the baseline and from the SFT adapter, evaluated every 25 steps |

## Running it

You need a CUDA GPU for training and evaluation (16 GB was enough), and an OpenAI API key for the judge.

```sh
pip install -r requirements.txt
hf download Qwen/Qwen3-1.7B --local-dir weights/qwen3-1.7b
cp .env.example .env    # then add OPENAI_API_KEY
```

Run everything from the repository root, in this order:

```sh
python -m data.prompts.make_prompt_data   # build the train/val/test prompt splits
bash runs/sft_sizes_20261016/run_all.sh    # SFT on 15-180 examples, then evaluate each
bash runs/rl_check_20261016.sh             # check both RL starting points
bash runs/rl_20261016/run_both.sh          # GRPO from the baseline and from the SFT adapter
python -m scripts.judge_eval compare \
  --run-dirs runs/short_eval_20261016/base_1.7b_masked runs/rl_20261016/sft180_masked/eval/step_0300
```

The scripts write into the run folders above, so copy those somewhere first if you want to keep the
published records. Set `PY=/path/to/python` to use a specific interpreter.

## Data and licenses

The code is MIT licensed (see [`LICENSE`](LICENSE)).

- Prompts come from [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (CC BY-NC 4.0)
  and [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  (CC BY-SA 3.0), and keep those licenses.
- The *Gadsby* text is downloaded from
  [Hellisotherpeople/Lipogram-e](https://huggingface.co/datasets/Hellisotherpeople/Lipogram-e) by the
  script and is not stored here.
- The 200 e-free answers in `data/rewrites/` were drafted by GPT models working as Codex subagents, then
  reviewed and checked for e. Each row records its prompt's source and how it was checked.

## History

The September version of this repo, including the first single-prompt RL attempt, is in the git history.
The write-up links to commit [`21dcbfa`](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj/tree/21dcbfa9d403e42529013e811eb1e06b18ba0e06).
