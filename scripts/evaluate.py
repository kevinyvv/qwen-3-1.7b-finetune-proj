"""Save generations for constraint scoring and quality review."""

import hashlib
import json
from pathlib import Path

import torch

from core.checks import e_pass
from core.generation import generate_rollout
from core.model import Qwen3ForCausalLM
from core.sampling import build_forbidden_token_mask
from core.utils import get_encoder, load_qwen3_weights
from training.checkpointing import load_checkpoint


NO_E_INSTRUCTION = (
    "Answer in natural English without the letters 'e' or 'E'. Preserve correct "
    "spelling and answer the user's request. This restriction applies only to "
    "your final answer, not your thinking."
)


def evaluate(model, tokenizer, rows, output_dir, max_new_tokens=768, thinking_budget=None,
             temperature=0.7, seed=0, batch_size=1, no_e=True, mask_e=False,
             judge_fn=None, metadata=None, ascii_only=False, forbidden_chars="",
             preserve_added_tokens=False, seeds=None):
    """judge_fn(prompt, answer, reference) may return {'passed': bool, ...}.
    seeds runs every prompt once per seed; it defaults to [seed]."""
    seeds = [seed] if seeds is None else list(seeds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    # the "no e rule" kind has neither the instruction nor the mask, and its e check is ignored
    e_check_counts = bool(no_e or mask_e)
    config = dict(max_new_tokens=max_new_tokens, thinking_budget=thinking_budget,
                  temperature=temperature, seed=seed, seeds=seeds, batch_size=batch_size,
                  no_e=no_e, mask_e=mask_e, ascii_only=ascii_only, forbidden_chars=forbidden_chars,
                  preserve_added_tokens=preserve_added_tokens, e_check_counts=e_check_counts,
                  instruction=NO_E_INSTRUCTION if no_e else None,
                  metadata=metadata)
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    forbidden = None
    if mask_e or ascii_only or forbidden_chars:
        forbidden = build_forbidden_token_mask(
            tokenizer, model.config.vocab_size,
            ('eE' if mask_e else '') + forbidden_chars, ascii_only,
            device=next(model.parameters()).device, preserve_added_tokens=preserve_added_tokens,
        )

    records = []
    total = len(rows) * len(seeds)
    with (output_dir / 'responses.jsonl').open('w') as file:
        for seed_value in seeds:
            torch.manual_seed(seed_value)
            for start in range(0, len(rows), batch_size):
                examples = rows[start:start + batch_size]
                texts = []
                for row in examples:
                    messages = ([{'role': 'system', 'content': NO_E_INSTRUCTION}] if no_e else [])
                    messages.append({'role': 'user', 'content': row['prompt']})
                    texts.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=thinking_budget is not None,
                    ))
                batch = tokenizer(texts, padding=True, return_tensors='pt')
                rollout = generate_rollout(
                    model, tokenizer, batch, max_new_tokens, temperature, thinking_budget,
                    forbidden_token_mask=forbidden,
                )
                for index, row in enumerate(examples):
                    ids = rollout.response_ids[index]
                    valid = rollout.completion_mask[index]
                    checked = rollout.reward_mask[index]
                    answer = tokenizer.decode(ids[checked].tolist(), skip_special_tokens=True).strip()
                    raw_response = tokenizer.decode(ids[valid].tolist(), skip_special_tokens=False)
                    e_count = answer.lower().count('e')
                    quality = judge_fn(row['prompt'], answer, row.get('response')) if judge_fn else None
                    record = {
                        'id': row['id'], 'source': row['source'], 'category': row['category'],
                        'seed': seed_value,
                        'prompt': row['prompt'], 'reference': row.get('response'),
                        'answer': answer, 'raw_response': raw_response,
                        'nonempty': bool(answer), 'e_count': e_count,
                        'no_e_pass': bool(answer) and e_count == 0,
                        'e_pass': e_pass(answer), 'e_check_counts': e_check_counts,
                        'ascii_only': answer.isascii(),
                        'response_tokens': int(valid.sum()), 'answer_tokens': int(checked.sum()),
                        'forced_thinking_end': bool((valid & ~rollout.loss_mask[index]).any()),
                        'truncated': ids[valid][-1].item() != tokenizer.eos_token_id,
                        'quality': quality,
                    }
                    record['joint_pass'] = (record['no_e_pass'] and quality['passed']
                                            if quality is not None else None)
                    records.append(record)
                    file.write(json.dumps(record) + '\n')
                    file.flush()
                    print(f'{len(records)}/{total}: no_e={record["no_e_pass"]}, '
                          f'tokens={record["answer_tokens"]}, truncated={record["truncated"]}', flush=True)

    def average(key):
        values = [row[key] for row in records if row[key] is not None]
        return sum(values) / len(values) if values else None

    summary = {
        'count': len(records), 'no_e_pass_rate': average('no_e_pass'),
        'e_pass_rate': average('e_pass'),
        'nonempty_rate': average('nonempty'), 'truncation_rate': average('truncated'),
        'ascii_only_rate': average('ascii_only'),
        'forced_thinking_end_rate': average('forced_thinking_end'),
        'mean_response_tokens': average('response_tokens'),
        'mean_answer_tokens': average('answer_tokens'), 'joint_pass_rate': average('joint_pass'),
    }
    (output_dir / 'responses.txt').write_text('\n\n'.join(
        f'{i}. {row["id"]} ({row["source"]}, {row["category"]}, seed {row["seed"]})\n'
        f'Prompt: {row["prompt"]}\n\nAnswer: {row["answer"]}\n\n'
        f'e/E count: {row["e_count"]}; truncated: {row["truncated"]}'
        for i, row in enumerate(records, 1)
    ) + '\n')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main(data_path='data/prompts/val.jsonl', output_dir='runs/baseline/eval/val',
         path=None, max_new_tokens=768, thinking_budget=None, temperature=0.7,
         seed=0, batch_size=1, no_e=True, mask_e=False, use_sdpa=True, limit=None,
         weights_dir=None, ascii_only=False, forbidden_chars="", preserve_added_tokens=False,
         seeds=None, judge=False, judge_workers=8):
    """judge=True scores the saved answers with the short-answer judge after generating."""
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    data = Path(data_path).read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    torch.manual_seed(seed)
    cfg, state_dict = load_qwen3_weights(weights_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen3ForCausalLM(cfg, use_sdpa=use_sdpa)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    if path is not None:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        load_checkpoint(path, model, None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32)
    summary = evaluate(
        model, get_encoder(), rows, output_dir, max_new_tokens, thinking_budget,
        temperature, seed, batch_size, no_e, mask_e,
        metadata={'data_path': str(data_path), 'data_sha256': hashlib.sha256(data).hexdigest(),
                  'checkpoint': str(path) if path is not None else None,
                  'use_sdpa': use_sdpa, 'limit': limit,
                  'weights_dir': str(Path(weights_dir).resolve()) if weights_dir is not None else None},
        ascii_only=ascii_only, forbidden_chars=forbidden_chars,
        preserve_added_tokens=preserve_added_tokens, seeds=seeds,
    )
    if judge:
        from scripts.judge_eval import judge_run

        summary = judge_run(output_dir, workers=judge_workers)
    return summary


if __name__ == '__main__':
    import fire

    fire.Fire(main)
