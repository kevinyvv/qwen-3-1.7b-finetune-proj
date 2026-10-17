#!/usr/bin/env bash
# Section 12 of docs/experiment_history.md: 16 answers per prompt at the RL sampling temperature, for each RL starting point.
set -euo pipefail
PY="${PY:-python}"
OUT=runs/rl_check_20261016
mkdir -p $OUT
COMMON=(--data_path=data/prompts/rl_check_50.jsonl --max_new_tokens=300 --temperature=1.0
        "--seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]" --batch_size=16 --judge=True
        --no_e=True --mask_e=True --ascii_only=True)
$PY -m scripts.evaluate "${COMMON[@]}" --output_dir=$OUT/base_masked > $OUT/base_masked.log 2>&1
$PY -m scripts.evaluate "${COMMON[@]}" --path=runs/sft_sizes_20261016/train_180/checkpoints/best.pt \
  --output_dir=$OUT/sft180_masked > $OUT/sft180_masked.log 2>&1
$PY -m scripts.rl_check --run-dirs $OUT/base_masked $OUT/sft180_masked
