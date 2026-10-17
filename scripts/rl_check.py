"""Do good answers ever appear? (docs/experiment_history.md, section 12)

    python -m scripts.rl_check --run-dirs runs/rl_check_x/base_masked runs/rl_check_x/sft180_masked

Each run is a judged short eval with many seeds per prompt (16 for step 3a).
Reports, per run: the share of prompts with at least one usable answer (the
gate is 20%), the expected share with at least one in a group of 8 (the RL
group size), and how often a prompt's answers get different rewards (RL learns
nothing from a group whose rewards are all equal).
"""
import argparse
from collections import Counter
from math import comb
import json
from pathlib import Path
import statistics

from core.judge import reward
from scripts.judge_eval import joined_rows

GATE = 0.20
GROUP = 8


def at_least_one(n, c, k):
    """Chance that k answers drawn without replacement from n, c of them usable, include one."""
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def check(run_dir):
    by_prompt = {}
    for row in joined_rows(run_dir):
        by_prompt.setdefault(row['id'], []).append(row)
    usable_counts, any_usable, in_group, varied = [], [], [], []
    for rows in by_prompt.values():
        scored = [r for r in rows if r['usable'] is not None]
        count = sum(r['usable'] for r in scored)
        usable_counts.append(count)
        any_usable.append(count > 0)
        if len(scored) >= GROUP:
            in_group.append(at_least_one(len(scored), count, GROUP))
        rewards = [reward(r['judgment']) for r in scored if r['judgment']]
        varied.append(len(set(rewards)) > 1)
    answers = sum(len(rows) for rows in by_prompt.values())
    result = dict(
        run_dir=str(run_dir), prompts=len(by_prompt), answers=answers,
        samples_per_prompt=sorted(Counter(len(rows) for rows in by_prompt.values())),
        usable_rate=sum(usable_counts) / answers,
        prompts_with_any_usable=sum(any_usable) / len(any_usable),
        expected_prompts_with_usable_in_group_of_8=statistics.mean(in_group) if in_group else None,
        prompts_with_varied_rewards=sum(varied) / len(varied),
        usable_per_prompt=dict(sorted(Counter(usable_counts).items())),
        gate=GATE, passes_gate=sum(any_usable) / len(any_usable) >= GATE,
    )
    (Path(run_dir) / 'rl_check.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dirs', nargs='+', required=True)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        r = check(run_dir)
        group = r['expected_prompts_with_usable_in_group_of_8']
        group_text = 'n/a' if group is None else f'{group:.0%}'
        per_prompt = '/'.join(map(str, r['samples_per_prompt']))
        print(f'{run_dir}: usable {r["usable_rate"]:.1%}; prompts with a usable answer in {per_prompt}: '
              f'{r["prompts_with_any_usable"]:.0%} (gate {GATE:.0%}: {"pass" if r["passes_gate"] else "FAIL"}); '
              f'in a group of 8: {group_text}; '
              f'prompts with varied rewards: {r["prompts_with_varied_rewards"]:.0%}')
        print(f'   usable answers per prompt -> number of prompts: {r["usable_per_prompt"]}')


if __name__ == '__main__':
    main()
