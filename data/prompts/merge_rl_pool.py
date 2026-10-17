"""Merge Haiku pool decisions into data/prompts/rl_pool.jsonl (run from the repo root)."""
import json
import random
from collections import Counter
from pathlib import Path

S = Path(__file__).resolve().parent
rows = lambda p: [json.loads(l) for l in open(p) if l.strip()]
train = {r['id']: r for r in rows('data/prompts/train.jsonl')}

all_decisions, problems = [], []
for k in range(10):
    chunk = rows(S / f'chunk_{k:02d}.jsonl')
    path = S / f'decisions_{k:02d}.jsonl'
    if not path.exists():
        problems.append(f'chunk {k}: no decisions file')
        continue
    dec = rows(path)
    if [d['id'] for d in dec] != [r['id'] for r in chunk]:
        problems.append(f'chunk {k}: ids do not match the chunk ({len(dec)} lines)')
        continue
    keeps = sum(d['decision'] == 'keep' for d in dec)
    unique = len({d['reason'] for d in dec})
    print(f'chunk {k}: keep {keeps}/{len(dec)} ({keeps/len(dec):.0%}), distinct reasons {unique}')
    all_decisions += [dict(d, chunk=k) for d in dec]

print('problems:', problems or 'none')
kept = [train[d['id']] for d in all_decisions if d['decision'] == 'keep']
print(f'total keep {len(kept)}/{len(all_decisions)}', Counter(d['reason_code'] for d in all_decisions))
if not problems:
    Path('data/prompts/rl_pool.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in kept))
    Path('data/prompts/rl_pool_decisions.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in all_decisions))
    print('wrote data/prompts/rl_pool.jsonl and rl_pool_decisions.jsonl')
    sample = random.Random(0).sample([d for d in all_decisions if d['decision'] == 'keep'], 40)
    (S / 'spot_check.txt').write_text('\n'.join(
        f'{i}. [{train[d["id"]]["category"]}] {train[d["id"]]["prompt"][:200]!r} | e_words {d["e_words"]}'
        for i, d in enumerate(sample, 1)) + '\n')
