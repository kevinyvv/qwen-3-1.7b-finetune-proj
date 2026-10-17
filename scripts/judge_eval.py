"""Judge and report short-answer eval runs (docs/experiment_history.md, section 9).

    python -m scripts.judge_eval judge --run-dir runs/x      # judge saved answers, then report
    python -m scripts.judge_eval report --run-dir runs/x     # rebuild the report only
    python -m scripts.judge_eval compare --run-dirs runs/base runs/x   # first run is the baseline

Answers that fail a counting e check are not sent to the judge. Judgments are
appended to judgments.jsonl as they finish, so a stopped run resumes where it left off.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading

from core.judge import (DIMS, JUDGE_EFFORT, JUDGE_MODEL, JUDGE_SYSTEM, judge_many,
                        make_judge_client, usable)

JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest()


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answer_key(row):
    return f'{row["id"]}:{row.get("seed", 0)}'


def needs_judge(row):
    return not row['e_check_counts'] or row['e_pass']


def load_judgments(run_dir):
    """Latest judgment per answer made with the current judge prompt."""
    judgments = {}
    for row in read_rows(Path(run_dir) / 'judgments.jsonl'):
        if row['judge_prompt_sha256'] == JUDGE_PROMPT_SHA256:
            judgments[row['key']] = row
    return judgments


def judge_run(run_dir, workers=8, client=None, retry_errors=True):
    run_dir = Path(run_dir)
    responses = read_rows(run_dir / 'responses.jsonl')
    if not responses:
        raise FileNotFoundError(run_dir / 'responses.jsonl')
    done = load_judgments(run_dir)
    todo = [row for row in responses if needs_judge(row) and (
        answer_key(row) not in done or (retry_errors and done[answer_key(row)]['error']))]
    print(f'{len(responses)} answers, {sum(map(needs_judge, responses))} need the judge, '
          f'{len(todo)} still to judge', flush=True)
    if todo:
        client = client or make_judge_client()
        lock = threading.Lock()
        by_key = {answer_key(row): row for row in todo}
        with (run_dir / 'judgments.jsonl').open('a') as file:
            def save(key, judgment):
                row = by_key[key]
                record = dict(key=key, id=row['id'], seed=row.get('seed', 0), **judgment,
                              judge_model=JUDGE_MODEL, judge_effort=JUDGE_EFFORT,
                              judge_prompt_sha256=JUDGE_PROMPT_SHA256)
                with lock:
                    file.write(json.dumps(record) + '\n')
                    file.flush()
                    save.count += 1
                    if save.count % 25 == 0 or save.count == len(todo):
                        print(f'judged {save.count}/{len(todo)}', flush=True)
            save.count = 0
            judge_many(client, [(answer_key(row), row['prompt'], row['answer'], bool(row.get('truncated')))
                                for row in todo],
                       workers=workers, on_result=save)
    return report(run_dir)


def resample_rates(per_prompt, ids, draws=1000, seed=0):
    """Pooled usable rate over prompts drawn with replacement. per_prompt: {id: [bool]}."""
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        values = [v for i in (rng.choice(ids) for _ in ids) for v in per_prompt[i]]
        if values:
            rates.append(sum(values) / len(values))
    return rates


def middle_90(values):
    values = sorted(values)
    return [values[int(0.05 * (len(values) - 1))], values[int(0.95 * (len(values) - 1))]]


def resample_difference(per_prompt_a, per_prompt_b, draws=1000, seed=0):
    """Middle 90% of (rate a - rate b), drawing the same prompts for both runs."""
    ids = sorted(set(per_prompt_a) & set(per_prompt_b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        picked = [rng.choice(ids) for _ in ids]
        a = [v for i in picked for v in per_prompt_a[i]]
        b = [v for i in picked for v in per_prompt_b[i]]
        if a and b:
            diffs.append(sum(a) / len(a) - sum(b) / len(b))
    return middle_90(diffs)


def joined_rows(run_dir):
    run_dir = Path(run_dir)
    judgments = load_judgments(run_dir)
    rows = []
    for row in read_rows(run_dir / 'responses.jsonl'):
        judgment = judgments.get(answer_key(row)) if needs_judge(row) else None
        rows.append(dict(row, judgment=judgment,
                         usable=usable(row['e_pass'], row['e_check_counts'], judgment)))
    return rows


def readable_rate(rows):
    """Share of scored answers with both scores at least 3. Answers that failed a counting
    e check have no judgment, so they count as not readable."""
    values = [r['judgment'] is not None and all(r['judgment'][k] >= 3 for k in DIMS)
              for r in rows if r['usable'] is not None]
    return sum(values) / len(values) if values else None


def per_prompt_usable(rows):
    per_prompt = {}
    for row in rows:
        per_prompt.setdefault(row['id'], [])
        if row['usable'] is not None:
            per_prompt[row['id']].append(row['usable'])
    return per_prompt


def report(run_dir):
    run_dir = Path(run_dir)
    rows = joined_rows(run_dir)
    judged = [r for r in rows if r['judgment'] is not None and not r['judgment']['error']]
    waiting = [r for r in rows if needs_judge(r) and r['judgment'] is None]
    scored = [r['usable'] for r in rows if r['usable'] is not None]
    per_prompt = per_prompt_usable(rows)
    usage = {key: sum(r['judgment']['usage'][key] for r in rows if r['judgment'])
             for key in ('calls', 'input_tokens', 'cached_input_tokens', 'output_tokens')}
    summary = dict(
        run_dir=str(run_dir), answers=len(rows), prompts=len(per_prompt),
        e_check_counts=rows[0]['e_check_counts'] if rows else None,
        usable_rate=sum(scored) / len(scored) if scored else None,
        usable_rate_middle_90=middle_90(resample_rates(per_prompt, sorted(per_prompt))) if scored else None,
        usable=sum(scored),
        readable_rate=readable_rate(rows),
        judge_errors=sum(1 for r in rows if r['judgment'] and r['judgment']['error']),
        not_yet_judged=len(waiting),
        e_pass_rate=sum(r['e_pass'] for r in rows) / len(rows) if rows else None,
        sent_to_judge=sum(map(needs_judge, rows)),
        mean_scores={k: statistics.mean(r['judgment'][k] for r in judged) if judged else None for k in DIMS},
        score_counts={k: {s: sum(r['judgment'][k] == s for r in judged) for s in range(1, 6)} for k in DIMS},
        mean_answer_tokens=statistics.mean(lengths) if (lengths := [
            r['answer_tokens'] for r in rows if r.get('answer_tokens') is not None]) else None,
        judge=dict(model=JUDGE_MODEL, effort=JUDGE_EFFORT, prompt_sha256=JUDGE_PROMPT_SHA256, **usage),
        per_prompt={i: dict(usable=sum(v), scored=len(v)) for i, v in per_prompt.items()},
    )
    (run_dir / 'short_eval_report.json').write_text(json.dumps(summary, indent=2) + '\n')

    picks = random.Random(0).sample(rows, min(10, len(rows)))
    (run_dir / 'samples_to_read.txt').write_text('\n\n'.join(
        f'{n}. {r["id"]} seed {r.get("seed", 0)} | usable={r["usable"]} e_pass={r["e_pass"]} | '
        + (f'coherence={r["judgment"]["coherence"]} responds={r["judgment"]["responds"]}\n'
           f'Judge: {r["judgment"]["note"] or r["judgment"]["error"]}'
           if r['judgment'] else 'not judged')
        + f'\nPrompt: {r["prompt"]}\nAnswer: {r["answer"]}'
        for n, r in enumerate(picks, 1)) + '\n')

    line = f'usable {summary["usable"]}/{len(scored)}'
    if scored:
        low, high = summary['usable_rate_middle_90']
        line += f' = {summary["usable_rate"]:.1%} (middle 90%: {low:.1%} to {high:.1%})'
    if scored:
        line += f'; readable (3+) {summary["readable_rate"]:.1%}'
    print(f'{run_dir}: {line}; e pass {summary["e_pass_rate"]:.1%}; judge errors '
          f'{summary["judge_errors"]}; not yet judged {len(waiting)}', flush=True)
    return summary


def compare(run_dirs):
    """Print each run's usable rate, and its difference from the first run on the same prompts."""
    runs = [(Path(d), joined_rows(d)) for d in run_dirs]
    base = per_prompt_usable(runs[0][1])
    print(f'{"run":<50} {"usable":>8} {"middle 90%":>17} {"vs first":>9} {"middle 90%":>17}')
    for run_dir, rows in runs:
        per_prompt = per_prompt_usable(rows)
        scored = [v for values in per_prompt.values() for v in values]
        rate = sum(scored) / len(scored) if scored else float('nan')
        low, high = middle_90(resample_rates(per_prompt, sorted(per_prompt))) if scored else (0, 0)
        text = f'{str(run_dir)[-50:]:<50} {rate:>8.1%} {low:>8.1%}-{high:<8.1%}'
        if run_dir != runs[0][0]:
            base_scored = [v for values in base.values() for v in values]
            diff = rate - sum(base_scored) / len(base_scored)
            d_low, d_high = resample_difference(per_prompt, base)
            text += f' {diff:>+9.1%} {d_low:>+8.1%}-{d_high:<+8.1%}'
        print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['judge', 'report', 'compare'])
    parser.add_argument('--run-dir')
    parser.add_argument('--run-dirs', nargs='+')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'judge':
        judge_run(args.run_dir, workers=args.workers)
    elif args.mode == 'report':
        report(args.run_dir)
    else:
        compare(args.run_dirs)


if __name__ == '__main__':
    main()
