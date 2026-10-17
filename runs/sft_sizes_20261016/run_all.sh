#!/usr/bin/env bash
# Section 11 of docs/experiment_history.md: one fresh adapter per nested training set,
# then the short eval (masked, and instruction only) on its best checkpoint.
# Run from the repository root: bash runs/sft_sizes_20261016/run_all.sh
set -euo pipefail

PY="${PY:-python}"
DATA=data/rewrites/splits
OUT=runs/sft_sizes_20261016
SIZES="${SIZES:-15 30 45 90 135 180}"

for n in $SIZES; do
  run=$OUT/train_$n
  if [ ! -f "$run/checkpoints/best.pt" ] || [ ! -f "$run/summary.json" ]; then
    echo "== training on $n examples"
    $PY -m scripts.sft_rewrites --train_path=$DATA/train_$n.jsonl --val_path=$DATA/val.jsonl \
      --run_dir=$run --batch_size=4 --max_length=3072 --epochs=10 --learning_rate=1e-4 \
      --seed=0 --patience=10 --eval_limit=0 > $run.train.log 2>&1
    grep -E "^Epoch|Initial" $run.train.log
  fi
  for kind in masked instruction_only; do
    eval_dir=$OUT/eval_${n}_$kind
    [ -f "$eval_dir/short_eval_report.json" ] && continue
    if [ "$kind" = masked ]; then flags="--no_e=True --mask_e=True --ascii_only=True"; else flags="--no_e=True --mask_e=False --ascii_only=False"; fi
    echo "== eval $n $kind"
    $PY -m scripts.evaluate --path=$run/checkpoints/best.pt --data_path=data/prompts/short_eval_50.jsonl \
      --output_dir=$eval_dir --max_new_tokens=300 --seeds='[0,1,2,3]' --batch_size=8 --judge=True \
      $flags > $eval_dir.log 2>&1
    $PY -m scripts.judge_eval report --run-dir $eval_dir | tail -1
  done
done
