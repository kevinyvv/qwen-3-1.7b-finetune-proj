# Step 3c: RL with the judge as the reward (2026-10-16)

`run_both.sh` runs `scripts/rl_judge.py` from base + mask (`base_masked`), then
from the 180-example SFT adapter + mask (`sft180_masked`). Each run directory has
`config.json`, `metrics.jsonl` (one line per step), `answers.jsonl` (every
training answer with its judgment), `answers_step_XXXX.txt` (10 answers to read
every 25 steps), `eval/step_XXXX/` (short eval every 25 steps) and `evals.json`.
Results and discussion: `docs/experiment_history.md`, section 13.

Usable at step 300: base 26.0% → 43.0%; SFT 180 35.0% → 60.5%.

```sh
python -m scripts.judge_eval compare --run-dirs runs/sft_sizes_20261016/eval_180_masked \
  runs/rl_20261016/sft180_masked/eval/step_0300
```

`base_masked_aborted_memory` ran past GPU memory on WSL (steps slowed to about
170 s) and `sft180_masked_aborted_oom` ran out of memory at step 1; both were
stopped and rerun after the fixes described in section 13. Checkpoints
(`*/checkpoints/`, 2.4 GB per run) are not committed.
