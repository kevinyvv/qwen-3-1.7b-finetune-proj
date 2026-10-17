"""LLM judge for short answers: is it coherent, and does it respond to the prompt.

Factual correctness is not scored. The judge prompt never mentions the letter e,
so the same prompt works for every condition. See section 9 of docs/experiment_history.md.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re

from core.llm_api import FatalAPIError

ROOT = Path(__file__).resolve().parents[1]
JUDGE_MODEL = 'gpt-6-luna'
JUDGE_EFFORT = 'none' # no reasoning; the judge check decides whether that is enough
DIMS = ('coherence', 'responds')
PASS_SCORE = 4

JUDGE_SYSTEM = """You grade short answers written by a small chat model. You will see a user's request and the model's answer. Score two things from 1 to 5.

Coherence: is the answer readable, sensible English?
5 = clear and natural. Every word is a real, correctly spelled English word, sentences are grammatical, and the ideas connect from start to finish.
4 = clear overall, with at most one minor awkward phrase or slip.
3 = understandable but with noticeable problems: broken phrasing, a misspelled or made-up word, or some repetition.
2 = hard to follow: several misspelled or made-up words, broken grammar, or looping text.
1 = gibberish, mostly repetition, or not English.
Repetition counts against coherence. If the same phrase or line appears three or more times, or the answer keeps circling back to the same words instead of moving forward, coherence is 3 at most. If most of the answer is repetition, it is 1 or 2.
Unusual or old-fashioned word choices are fine as long as the words are real and the meaning is clear.

Responds: does the answer address the request?
5 = directly attempts what was asked.
4 = attempts what was asked, with small gaps or slight drift.
3 = partly on topic, or attempts only part of the request.
2 = mostly off topic, or generic filler that could follow almost any request.
1 = ignores the request.
Do not judge factual accuracy. Wrong facts, wrong definitions or made-up details don't lower either score, as long as the answer attempts the request.

Some answers are marked as cut off: they stopped because they hit a length limit. Judge only the text that is there. Do not mark a cut-off answer down for ending mid-sentence or for not reaching later parts of the request (for example, giving 2 of 5 requested items before the cut). Ask whether what is there is coherent and on its way to doing what was asked. Do not reward or penalize length otherwise.

Reply with only this JSON object and no other text:
{"coherence": <1-5>, "responds": <1-5>, "note": "<one sentence explaining the scores>"}"""


def judge_messages(prompt, answer, cut_off=False):
    label = 'Answer (cut off at the length limit)' if cut_off else 'Answer'
    return [{'role': 'user', 'content': f'Request:\n<<<\n{prompt}\n>>>\n\n{label}:\n<<<\n{answer}\n>>>'}]


def parse_reply(text):
    """Return {'coherence', 'responds', 'note'}, or raise ValueError if the reply can't be read."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match is None:
        raise ValueError('no JSON object in reply')
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f'bad JSON: {error}') from error
    for key in DIMS:
        if type(data.get(key)) is not int or not 1 <= data[key] <= 5:
            raise ValueError(f'{key} must be an integer from 1 to 5')
    if not isinstance(data.get('note'), str):
        raise ValueError('note must be a string')
    return {'coherence': data['coherence'], 'responds': data['responds'], 'note': data['note']}


def make_judge_client(model=JUDGE_MODEL, effort=JUDGE_EFFORT):
    """OpenAI client using OPENAI_API_KEY from .env. Never the Azure key."""
    from dotenv import load_dotenv
    from core.llm_api import OpenAIClient

    load_dotenv(ROOT / '.env')
    client = OpenAIClient(model, effort=effort)
    if type(client.client).__name__ != 'OpenAI':
        raise RuntimeError('The judge must use OPENAI_API_KEY; unset AZURE_OPENAI_ENDPOINT')
    return client


def judge_one(client, prompt, answer, cut_off=False, attempts=2, max_tokens=1000):
    """Judge one answer. An unreadable reply or a failed call is retried once, then
    recorded as a judge error. Errors that would fail every call (bad key, bad
    model name) raise FatalAPIError instead."""
    usage = dict(calls=0, input_tokens=0, cached_input_tokens=0, output_tokens=0)
    error = None
    for _ in range(attempts):
        try:
            result = client.complete(system=JUDGE_SYSTEM, messages=judge_messages(prompt, answer, cut_off),
                                     max_tokens=max_tokens)
        except FatalAPIError:
            raise
        except Exception as exc: # rate limits or network errors left after the client's own retries
            error = f'{type(exc).__name__}: {exc}'
            continue
        usage['calls'] += 1
        for key in ('input_tokens', 'cached_input_tokens', 'output_tokens'):
            usage[key] += getattr(result, key)
        try:
            return dict(parse_reply(result.text), error=None, model=result.model, usage=usage)
        except ValueError as exc:
            error = f'unreadable reply: {exc}; reply was {result.text[:200]!r}'
    return dict(coherence=None, responds=None, note=None, error=error, model=None, usage=usage)


def judge_many(client, items, workers=8, on_result=None):
    """items: [(key, prompt, answer, cut_off)]. Returns {key: judgment}; calls on_result(key, judgment)
    as each finishes so callers can save progress."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_one, client, prompt, answer, cut_off): key
                   for key, prompt, answer, cut_off in items}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            if on_result is not None:
                on_result(key, results[key])
    return results


def reward(judgment, combine='sum'):
    """Both judge scores combined into one number from 0 (both 1) to 1 (both 5).
    combine='sum' adds them; combine='lowest' uses the lower score only, so a clean
    answer that skips part of the request (5 and 3) earns less than a usable 4 and 4."""
    if combine == 'sum':
        return sum(judgment[key] - 1 for key in DIMS) / (4 * len(DIMS))
    if combine == 'lowest':
        return (min(judgment[key] for key in DIMS) - 1) / 4
    raise ValueError(f'unknown combine: {combine}')


def usable(e_ok, e_check_counts, judgment):
    """True/False, or None when the judge errored (never counted as a pass or a fail)."""
    if e_check_counts and not e_ok:
        return False
    if judgment is None or judgment.get('error'):
        return None
    return all(judgment[key] >= PASS_SCORE for key in DIMS)
