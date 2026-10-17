# Short-answer eval, base models (2026-10-16)

Sections 9–10 of `docs/experiment_history.md`. 50 prompts (`data/prompts/short_eval_50.jsonl`)
× seeds 0–3, 300-token cap, temperature 0.7, thinking off, judge v3 (GPT-6 Luna).

| Folder | Model | Kind | Usable | Readable (3+) |
|---|---|---|---:|---:|
| `base_1.7b_no_rule` | 1.7B | no e rule | 80.5% | 98.0% |
| `base_1.7b_masked` | 1.7B | no-e instruction + mask | 26.0% | 67.0% |
| `base_1.7b_instruction_only` | 1.7B | no-e instruction alone | 0% | 0% |
| `base_4b_no_rule` | 4B | no e rule | 87.5% | 99.5% |
| `base_4b_masked` | 4B | no-e instruction + mask | 12.5% | 45.0% |
| `base_4b_instruction_only` | 4B | no-e instruction alone | 0% | 0% |

Each folder has `short_eval_report.json` and `samples_to_read.txt`. Compare runs with:

```sh
python -m scripts.judge_eval compare --run-dirs runs/short_eval_20261016/base_1.7b_masked runs/short_eval_20261016/base_4b_masked
```

`base_1.7b_no_rule/judgments.jsonl` also holds the earlier judge-v2 scores
(72%); reports only use judgments made with the current judge prompt.
`qwen3_4b_hf_parity.json` is the 4B check against Hugging Face.
