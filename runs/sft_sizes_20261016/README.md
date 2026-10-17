# Step 2.5: amount of SFT data (2026-10-16)

`run_all.sh` trains one adapter per nested training set (`train_N`) and evaluates
its best checkpoint on the short eval (`eval_N_masked`, `eval_N_instruction_only`).
Results and discussion: `docs/experiment_history.md`, section 11.

Masked usable rate: base 26.0%; 15: 10.0%; 30: 20.5%; 45: 22.5%; 90: 31.5%;
135: 34.0%; 180: 35.0%.

```sh
python -m scripts.judge_eval compare --run-dirs runs/short_eval_20261016/base_1.7b_masked \
  runs/sft_sizes_20261016/eval_90_masked runs/sft_sizes_20261016/eval_180_masked
```

Checkpoints (`train_N/checkpoints/`, about 200 MB each) are not committed.
