"""Assistant-only SFT on no-e rewrites, with held-out loss and saved chat samples."""

import hashlib
import json
import math
from pathlib import Path
import random

import torch

from core.model import Qwen3ForCausalLM
from core.utils import get_encoder, load_qwen3_weights
from scripts.evaluate import NO_E_INSTRUCTION, evaluate
from training.sft.sft import SFTEngine
from training.sft.sft_utils import get_loader, load_sft_dataset


ROOT = Path(__file__).resolve().parent.parent


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompt_key(row):
    return ' '.join(row['prompt'].split()).casefold()


def reserved_prompts():
    reserved = read_rows(ROOT / 'data/prompts/val.jsonl') + read_rows(ROOT / 'data/prompts/test.jsonl')
    return {row['id'] for row in reserved}, {prompt_key(row) for row in reserved}


def prepare_data(data_path, output_dir, tokenizer, max_length=2048, val_size=10, seed=0):
    rows = read_rows(data_path)
    reserved_ids, reserved_prompts_ = reserved_prompts()
    seen_ids, seen_prompts = set(), set()
    accepted, excluded = [], []
    for row in rows:
        key = prompt_key(row)
        if row['id'] in reserved_ids or key in reserved_prompts_:
            raise ValueError(f'Validation/test prompt in rewrite data: {row["id"]}')
        if row['id'] in seen_ids or key in seen_prompts:
            raise ValueError(f'Duplicate rewrite prompt: {row["id"]}')
        seen_ids.add(row['id'])
        seen_prompts.add(key)
        review = row.get('review') or {}
        if row.get('status') != 'pass' or not row.get('rewrite', '').strip() or 'e' in row['rewrite'].lower():
            excluded.append({'id': row['id'], 'reason': 'failed or empty no-e target'})
        elif review.get('faithfulness', 3) < 3 or not review.get('spelling_ok', True):
            excluded.append({'id': row['id'], 'reason': 'low faithfulness or spelling review'})
        else:
            accepted.append(row)
    if not 0 < val_size < len(accepted):
        raise ValueError('val_size must leave at least one training and validation example')
    random.Random(seed).shuffle(accepted)
    datasets, counts = write_splits(output_dir, {'train': accepted[val_size:], 'val': accepted[:val_size]},
                                    tokenizer, max_length)
    audit = dict(source=str(data_path), source_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                 source_rows=len(rows), excluded=excluded, splits=counts,
                 filter='status pass, nonempty no-e target, faithfulness >= 3 and spelling_ok when reviewed',
                 factual_errors='Retained from source; review flags are not ground truth.')
    (Path(output_dir) / 'manifest.json').write_text(json.dumps(audit, indent=2))
    return datasets, audit


def prepare_fixed_split(train_path, val_path, output_dir, tokenizer, max_length):
    """Use given train/val files (id, prompt, response) as they are: no filtering, no re-split."""
    reserved_ids, reserved_prompts_ = reserved_prompts()
    splits, seen_ids, seen_prompts = {}, set(), set()
    for split, path in (('train', train_path), ('val', val_path)):
        rows = []
        for row in read_rows(path):
            key = prompt_key(row)
            if row['id'] in reserved_ids or key in reserved_prompts_:
                raise ValueError(f'Validation/test prompt in {split} data: {row["id"]}')
            if row['id'] in seen_ids or key in seen_prompts:
                raise ValueError(f'Prompt appears twice across train/val: {row["id"]}')
            if not row['response'].strip() or 'e' in row['response'].lower():
                raise ValueError(f'Target is empty or contains e: {row["id"]}')
            seen_ids.add(row['id'])
            seen_prompts.add(key)
            rows.append(dict(id=row['id'], prompt=row['prompt'], rewrite=row['response']))
        splits[split] = rows
    datasets, counts = write_splits(output_dir, splits, tokenizer, max_length)
    audit = dict(sources={split: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                          for split, path in (('train', train_path), ('val', val_path))},
                 splits=counts, filter='none: fixed split used as given')
    (Path(output_dir) / 'manifest.json').write_text(json.dumps(audit, indent=2))
    return datasets, audit


def write_splits(output_dir, splits, tokenizer, max_length):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    datasets, counts = {}, {}
    for split, pairs in splits.items():
        path = output_dir / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(dict(id=row['id'], prompt=row['prompt'], response=row['rewrite'])) + '\n'
                                for row in pairs))
        examples = load_sft_dataset(path, tokenizer, system_prompt=NO_E_INSTRUCTION)
        for row, example in zip(pairs, examples):
            if len(example['input_ids']) > max_length:
                raise ValueError(f'{row["id"]} exceeds max_length; refusing to truncate its target')
            target = [token for token in example['labels'] if token != -100]
            if target[-1] != tokenizer.eos_token_id or tokenizer.decode(target[:-1]) != row['rewrite']:
                raise ValueError(f'Target did not round-trip: {row["id"]}')
            if set(target[:-1]) & set(tokenizer.all_special_ids):
                raise ValueError(f'Control token inside target: {row["id"]}')
        datasets[split] = examples
        counts[split] = dict(examples=len(pairs), ids=[row['id'] for row in pairs],
                             supervised_tokens=sum(sum(t != -100 for t in ex['labels']) for ex in examples),
                             max_length=max(len(ex['input_ids']) for ex in examples),
                             factual_flags=sum(bool((row.get('review') or {}).get('factual_errors')) for row in pairs))
    return datasets, counts


def main(data_path=str(ROOT / 'data/rewrites/rewrites.jsonl'), run_dir='runs/rewrites_sft_r16_seed0',
         batch_size=4, max_length=2048, epochs=5, learning_rate=5e-5, val_size=10,
         seed=0, patience=2, min_delta=0.01, eval_limit=10, max_new_tokens=256, resume_from=None,
         train_path=None, val_path=None):
    """train_path and val_path (both or neither) train on a fixed split as given; data_path and
    val_size are then ignored."""
    if (train_path is None) != (val_path is None):
        raise ValueError('Pass both train_path and val_path, or neither')
    if min(batch_size, epochs, val_size, patience, max_new_tokens) < 1 or max_length < 2:
        raise ValueError('Batch, epoch, split, patience and token limits must be positive')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or min_delta < 0 or eval_limit < 0:
        raise ValueError('Invalid learning rate, min_delta or eval_limit')
    if not torch.cuda.is_available():
        raise RuntimeError('This training runner requires CUDA; aborting instead of training on CPU')
    resume_path = Path(resume_from).resolve() if resume_from is not None else None
    source_config = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        source_config = json.loads((resume_path.parent.parent / 'config.json').read_text())
        settings = dict(batch_size=batch_size, max_length=max_length, learning_rate=learning_rate,
                        seed=seed, val_size=val_size, system_prompt=NO_E_INSTRUCTION)
        if any(source_config[key] != value for key, value in settings.items()):
            raise ValueError('Resume must preserve the original training settings')
    run_dir = Path(run_dir)
    if any((run_dir / name).exists() for name in ('config.json', 'data', 'checkpoints')):
        raise FileExistsError(f'Choose a new run_dir: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = get_encoder()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if train_path is not None:
        datasets, audit = prepare_fixed_split(train_path, val_path, run_dir / 'data', tokenizer, max_length)
    else:
        datasets, audit = prepare_data(data_path, run_dir / 'data', tokenizer, max_length, val_size, seed)
    if source_config is not None:
        if audit != source_config['data']:
            raise ValueError('Resume must use the original rewrite data and split')
        for split in datasets:
            original = resume_path.parent.parent / 'data' / f'{split}.jsonl'
            if original.read_bytes() != (run_dir / 'data' / f'{split}.jsonl').read_bytes():
                raise ValueError(f'Resume {split} data differs from the saved split')
    loaders = {split: get_loader(data, tokenizer, batch_size, shuffle=split == 'train')
               for split, data in datasets.items()}
    chat_rows = read_rows(ROOT / 'data/prompts/val.jsonl')[:eval_limit]
    config = dict(data_path=None if train_path else str(data_path),
                  train_path=str(train_path) if train_path else None,
                  val_path=str(val_path) if val_path else None, batch_size=batch_size, max_length=max_length, epochs=epochs,
                  learning_rate=learning_rate, seed=seed, val_size=val_size, patience=patience,
                  min_delta=min_delta, eval_limit=eval_limit, max_new_tokens=max_new_tokens,
                  eval_ids=[row['id'] for row in chat_rows], temperature=0.7,
                  resume_from=str(resume_path) if resume_path else None,
                  resume_sha256=hashlib.sha256(resume_path.read_bytes()).hexdigest() if resume_path else None,
                  objective='assistant_only_chat', system_prompt=NO_E_INSTRUCTION,
                  thinking=False, mask_e=False, base_weights=str(ROOT / 'weights/qwen3-1.7b'),
                  lora_rank=16, lora_alpha=16, precision='bf16', adapter_precision='float32',
                  use_sdpa=True, gradient_checkpointing=True, loss_backend='liger', data=audit)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    free, total = torch.cuda.mem_get_info()
    cap = min(10 * 1024**3, free - 2 * 1024**3)
    if cap < 6 * 1024**3:
        raise RuntimeError('Insufficient free VRAM for a guarded SFT run')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    print(f'Train/val: {len(datasets["train"])}/{len(datasets["val"])}; VRAM cap: {cap / 1024**3:.2f} GiB', flush=True)
    cfg, state = load_qwen3_weights()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3ForCausalLM(cfg, use_sdpa=True)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_state_dict(state, strict=False)
    del state
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith(('.A', '.B'))
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    engine = SFTEngine(model, tokenizer, optimizer, gradient_checkpointing=True, loss_backend='liger',
                       device='cuda', path=run_dir / 'checkpoints/latest.pt', metrics_path=run_dir / 'metrics.jsonl')

    if resume_path is not None:
        engine.load_checkpoint(resume_path)
    start_epoch, start_step = engine.epoch, engine.global_step
    if start_epoch >= epochs:
        raise ValueError('epochs is the total target and must exceed the checkpoint epoch')
    if any(group['lr'] != learning_rate for group in optimizer.param_groups):
        raise ValueError('Checkpoint learning rate differs from this run')
    print(f'Starting at epoch {start_epoch}, step {start_step}; target epoch {epochs}', flush=True)

    def chat_eval(name):
        if not chat_rows:
            return None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return evaluate(model, tokenizer, chat_rows, run_dir / 'eval' / name,
                            max_new_tokens=max_new_tokens, seed=seed,
                            metadata={'run_dir': str(run_dir), 'stage': name})

    before = engine.evaluate(loaders['val'])
    if not math.isfinite(before):
        raise FloatingPointError('Non-finite initial validation loss')
    print(f'Initial held-out loss: {before:.4f}', flush=True)
    baseline = chat_eval(f'before_val{len(chat_rows)}')
    torch.manual_seed(seed)
    best_loss = plateau_loss = before
    best_epoch, stale_epochs = start_epoch, 0
    best_path = run_dir / 'checkpoints/best.pt'
    engine.save_checkpoint(best_path)
    history = []
    for epoch in range(start_epoch + 1, epochs + 1):
        engine.epoch = epoch - 1
        engine.fit(loaders['train'], loaders['val'], epochs=1,
                   eval_every=len(loaders['train']), save_every=len(loaders['train']), log_every=1)
        engine.epoch = epoch
        after = engine.evaluate(loaders['val'])
        if not math.isfinite(after):
            raise FloatingPointError('Non-finite validation loss')
        engine.save_checkpoint(engine.path)
        if after < best_loss:
            best_loss, best_epoch = after, epoch
            engine.save_checkpoint(best_path)
        if after < plateau_loss - min_delta:
            plateau_loss, stale_epochs = after, 0
        else:
            stale_epochs += 1
        history.append(dict(epoch=epoch, step=engine.global_step, val_loss=after))
        result = dict(start_epoch=start_epoch, start_step=start_step, epochs_completed=epoch,
                      global_step=engine.global_step, val_loss_before=before,
                      best_val_loss=best_loss, best_epoch=best_epoch,
                      history=history, baseline=baseline,
                      stop_reason='validation_plateau' if stale_epochs >= patience else 'epoch_limit')
        (run_dir / 'summary.json').write_text(json.dumps(result, indent=2))
        print(f'Epoch {epoch}: val={after:.4f}, best={best_loss:.4f}', flush=True)
        if stale_epochs >= patience:
            break
    optimizer.zero_grad(set_to_none=True)
    if best_epoch != engine.epoch:
        result['latest_checkpoint_eval'] = chat_eval(f'latest_val{len(chat_rows)}')
    engine.load_checkpoint(best_path)
    result['best_checkpoint_eval'] = chat_eval(f'best_val{len(chat_rows)}')
    result['peak_cuda_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    result['peak_cuda_reserved_gib'] = torch.cuda.max_memory_reserved() / 1024**3
    (run_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    import fire

    fire.Fire(main)
