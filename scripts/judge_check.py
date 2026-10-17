"""Check the short-answer judge before trusting it (docs/experiment_history.md, section 9).

    python -m scripts.judge_check --run-dir runs/judge_check_20261016_v3 --report-only

The `earlier` set is built from an Oct 6 run that is not in this repo, so judging a fresh
run dir needs that folder. runs/judge_check_20261016_v3 already holds all three judged
sets; --report-only rebuilds its report from them. The full command was:

    python -m scripts.judge_check --run-dir runs/judge_check_20261016 \
        --eval-prompts data/prompts/short_eval_50.jsonl

Three sets, all judged with the e check ignored:
- known_good: the verified short no-e answers. Target: at least 90% usable.
- earlier: short answers reviewed in the Oct 6 masked pilot. Target: the judge's
  coherence pass/fail agrees with the earlier reviews on at least 80%.
- known_bad: failed outputs from the September RL runs, each paired with eval
  prompts. Target: none usable.
"""
import argparse
import json
from pathlib import Path

from core.checks import e_pass
from core.judge import PASS_SCORE
from scripts.judge_eval import joined_rows, judge_run, read_rows

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'runs/masked_adapter_pilot_20261006'
SHORT_STAGES = ('short', 'sampling') # the pilot's other stages are 2k-token answers

KNOWN_BAD = {
    'non_answer': 'i want to know what you want to know.',
    'non_answer_short': 'i want to know what you want',
    'syllable_loop': ' '.join(['Tghn'] + ['tghn'] * 30),
    'letter_column': 'I  \nm  \nn  \nt  \nl  \ns  \nd  \nf  \ng  \nh  \nk  \np  \nr  \nt  \nw  \nx  \nz',
    'bold_loop': 'N**o**t h**i**n**g**.  \nN**o**t h**i**n**g**.  \nN**o**t h**i**n**g**.',
    'consonant_chant': 'Tst, tkt, nkt, tkt, nkt, tkt.',
}


def answer_row(id_, prompt, answer, **extra):
    return dict(id=id_, seed=0, prompt=prompt, answer=answer, e_pass=e_pass(answer),
                e_check_counts=False, answer_tokens=extra.pop('answer_tokens', None), **extra)


def build_sets(eval_prompts, bad_prompt_count=8):
    verified = read_rows(ROOT / 'data/rewrites/verified_200_20261015.jsonl')
    known_good = [answer_row(r['id'], r['prompt'], r['response'])
                  for r in verified if r['band'] == 'short']

    responses = {(r['stage'], r['condition'], r['id'], r['sample']): r
                 for r in read_rows(PILOT / 'responses_compiled.jsonl')}
    reviews = {r['sample_id']: r for r in json.loads((PILOT / 'reviews_compiled.json').read_text())}
    earlier = []
    for m in json.loads((PILOT / 'mapping_compiled.json').read_text()):
        if m['stage'] not in SHORT_STAGES:
            continue
        r, review = responses[(m['stage'], m['condition'], m['id'], m['sample'])], reviews[m['sample_id']]
        earlier.append(answer_row(m['sample_id'], r['prompt'], r['answer'],
                                  answer_tokens=r['answer_tokens'], truncated=r['truncated'],
                                  condition=m['condition'],
                                  earlier_review={k: review[k] for k in (
                                      'fluency_spelling', 'coherence', 'substance',
                                      'prompt_fulfillment', 'notes')}))

    prompts = read_rows(eval_prompts)[:bad_prompt_count]
    known_bad = [answer_row(f'{name}:{p["id"]}', p['prompt'], text, bad_kind=name)
                 for name, text in KNOWN_BAD.items() for p in prompts]
    return dict(known_good=known_good, earlier=earlier, known_bad=known_bad)


def write_set(run_dir, name, rows):
    """Write responses.jsonl once; later runs reuse it so judging can resume."""
    path = run_dir / name / 'responses.jsonl'
    if path.exists():
        saved = read_rows(path)
        if [r['id'] for r in saved] != [r['id'] for r in rows]:
            raise ValueError(f'{path} holds a different set; use a new --run-dir')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def check(run_dir):
    sets = {name: joined_rows(run_dir / name) for name in ('known_good', 'earlier', 'known_bad')}
    if any(r['usable'] is None for rows in sets.values() for r in rows):
        print('Warning: some answers have judge errors or are not judged yet; they are left out.')

    def rate(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    good = rate(r['usable'] for r in sets['known_good'])
    agree_coherence, agree_usable, table = [], [], {}
    for r in sets['earlier']:
        if r['usable'] is None:
            continue
        earlier, ours = r['earlier_review'], r['judgment']
        earlier_coherent = min(earlier['fluency_spelling'], earlier['coherence']) >= PASS_SCORE
        earlier_usable = earlier_coherent and earlier['prompt_fulfillment'] >= PASS_SCORE
        agree_coherence.append(earlier_coherent == (ours['coherence'] >= PASS_SCORE))
        agree_usable.append(earlier_usable == r['usable'])
        key = f'earlier {"pass" if earlier_coherent else "fail"}, judge {"pass" if ours["coherence"] >= PASS_SCORE else "fail"}'
        table[key] = table.get(key, 0) + 1
    bad_passed = [dict(id=r['id'], coherence=r['judgment']['coherence'], responds=r['judgment']['responds'],
                       note=r['judgment']['note']) for r in sets['known_bad'] if r['usable']]

    judged = [r['judgment'] for rows in sets.values() for r in rows if r['judgment']]
    calls = sum(j['usage']['calls'] for j in judged)
    result = dict(
        known_good=dict(answers=len(sets['known_good']), usable_rate=good,
                        target='>= 0.90', passed=good is not None and good >= 0.9),
        earlier=dict(answers=len(sets['earlier']), coherence_agreement=rate(agree_coherence),
                     usable_agreement=rate(agree_usable), coherence_table=table,
                     target='coherence agreement >= 0.80',
                     passed=bool(agree_coherence) and rate(agree_coherence) >= 0.8),
        known_bad=dict(answers=len(sets['known_bad']), usable=len(bad_passed), usable_examples=bad_passed,
                       target='0 usable', passed=not bad_passed),
        judge_usage=dict(calls=calls, **{k: sum(j['usage'][k] for j in judged) for k in (
            'input_tokens', 'cached_input_tokens', 'output_tokens')}),
    )
    result['all_passed'] = all(result[k]['passed'] for k in ('known_good', 'earlier', 'known_bad'))
    (run_dir / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('usable_examples',)}
                      if isinstance(v, dict) else v for k, v in result.items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--eval-prompts', default='data/prompts/short_eval_50.jsonl')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not args.report_only:
        for name, rows in build_sets(args.eval_prompts).items():
            write_set(run_dir, name, rows)
            judge_run(run_dir / name, workers=args.workers)
    check(run_dir)


if __name__ == '__main__':
    main()
