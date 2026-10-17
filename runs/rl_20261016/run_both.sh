#!/usr/bin/env bash
# Section 13 of docs/experiment_history.md: GRPO with the judge as the reward, from base + mask, then from the 180-example SFT adapter + mask.
PY="${PY:-python}"
OUT=runs/rl_20261016
$PY -m scripts.rl_judge --run_dir=$OUT/base_masked --prompts=data/prompts/rl_pool.jsonl \
  --start_eval=runs/short_eval_20261016/base_1.7b_masked > $OUT/base_masked.log 2>&1
echo "base_masked exited with $?"
$PY -m scripts.rl_judge --run_dir=$OUT/sft180_masked --prompts=data/prompts/rl_pool.jsonl \
  --start_adapter=runs/sft_sizes_20261016/train_180/checkpoints/best.pt \
  --start_eval=runs/sft_sizes_20261016/eval_180_masked > $OUT/sft180_masked.log 2>&1
echo "sft180_masked exited with $?"
