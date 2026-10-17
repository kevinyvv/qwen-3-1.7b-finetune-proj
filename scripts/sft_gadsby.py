import hashlib
import json
from pathlib import Path

import torch

from core.model import Qwen3ForCausalLM
from core.utils import get_encoder, load_qwen3_weights
from training.sft.sft import SFTEngine
from training.sft.sft_utils import get_loader, load_text_dataset


DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'sft'


def limit_vram(headroom_gb=1.0):
    free, total = torch.cuda.mem_get_info()
    fraction = (free - headroom_gb * 1024**3) / total
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"VRAM cap: {fraction * total / 1024**3:.2f} GiB")


def main(run_dir='runs/gadsby_sft_r16_seed0', data_dir=str(DATA_DIR), batch_size=2,
         max_length=2048, epochs=1, learning_rate=1e-4, seed=0, use_sdpa=True,
         eval_every=10, save_every=25, log_every=1, limit=None, patience=3, min_delta=0.01):
    
    if min(batch_size, epochs) < 1 or learning_rate <= 0 or (limit is not None and limit < 1):
        raise ValueError('batch size, epochs, learning rate and limit must be positive')
    if patience < 1 or min_delta < 0:
        raise ValueError('patience must be positive and min_delta nonnegative')
    
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    if run_dir.exists():
        raise FileExistsError(f'Choose a new run_dir: {run_dir}')
    
    torch.manual_seed(seed) # is this needed
    tokenizer = get_encoder()
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_path, val_path = data_dir / 'gadsby_train.jsonl', data_dir / 'gadsby_val.jsonl'
    train_data = load_text_dataset(train_path, tokenizer, max_length)
    val_data = load_text_dataset(val_path, tokenizer, max_length)
    
    if limit is not None:
        train_data = train_data[:limit]
    if not train_data or not val_data:
        raise ValueError('Both training and validation need at least one chunk')
    
    train_loader = get_loader(train_data, tokenizer, batch_size)
    val_loader = get_loader(val_data, tokenizer, batch_size, shuffle=False)

    config = dict(batch_size=batch_size, max_length=max_length, epochs=epochs,
                  learning_rate=learning_rate, seed=seed, use_sdpa=use_sdpa, limit=limit,
                  eval_every=eval_every, save_every=save_every, log_every=log_every,
                  patience=patience, min_delta=min_delta,
                  objective='plain_text_next_token', train_chunks=len(train_data), val_chunks=len(val_data),
                  data_manifest=json.loads((data_dir / 'gadsby_manifest.json').read_text()),
                  train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                  val_sha256=hashlib.sha256(val_path.read_bytes()).hexdigest()) # are all of these necessary?
    
    run_dir.mkdir(parents=True)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Training: {len(train_data)} chunks / {len(train_loader)} steps per epoch; '
          f'validation: {len(val_data)} chunks', flush=True)
    
    if torch.cuda.is_available():
        limit_vram(headroom_gb=2.0)
        _, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(10 * 1024**3 / total,
                                                      torch.cuda.get_per_process_memory_fraction()))
        
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    cfg, state_dict = load_qwen3_weights()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    
    try:
        model = Qwen3ForCausalLM(cfg, use_sdpa=use_sdpa)
    finally:
        torch.set_default_dtype(previous_dtype)
        
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith(('.A', '.B'))
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    
    engine = SFTEngine(
        model, tokenizer, optimizer, gradient_checkpointing=True,
        loss_backend='liger' if torch.cuda.is_available() else 'torch', precision=dtype,
        path=run_dir / 'checkpoints' / 'latest.pt', metrics_path=run_dir / 'metrics.jsonl',
    )
    
    before = engine.evaluate(val_loader)
    best_loss = plateau_loss = before
    best_epoch = stale_epochs = 0
    history = []
    engine.save_checkpoint(run_dir / 'checkpoints' / 'best.pt')
    stop_reason = 'epoch_limit'
    
    try:
        for epoch in range(epochs):
            engine.epoch = epoch
            engine.fit(train_loader, val_loader, 1, eval_every, save_every, log_every)
            engine.epoch = epoch + 1
            after = engine.evaluate(val_loader)
            if not torch.isfinite(torch.tensor(after)):
                raise FloatingPointError('Non-finite validation loss')
            engine.save_checkpoint(engine.path)
            if epoch == 0:
                engine.save_checkpoint(run_dir / 'checkpoints' / 'epoch1.pt')
            if after < best_loss:
                best_loss, best_epoch = after, epoch + 1
                engine.save_checkpoint(run_dir / 'checkpoints' / 'best.pt')
            if after < plateau_loss - min_delta:
                plateau_loss, stale_epochs = after, 0
            else:
                stale_epochs += 1
            history.append({'epoch': epoch + 1, 'step': engine.global_step, 'val_loss': after})
            if stale_epochs >= patience:
                stop_reason = 'validation_plateau'
            result = {'val_loss_before': before, 'val_loss_after': after,
                      'best_val_loss': best_loss, 'best_epoch': best_epoch,
                      'epochs_completed': epoch + 1, 'steps': engine.global_step,
                      'stop_reason': stop_reason, 'history': history}
            (run_dir / 'summary.json').write_text(json.dumps(result, indent=2))
            print(f'Epoch {epoch + 1}: validation={after:.4f}, best={best_loss:.4f}, '
                  f'stale_epochs={stale_epochs}', flush=True)
            if stale_epochs >= patience:
                break
        return result
    finally:
        if torch.cuda.is_available():
            print(f'Peak CUDA allocation: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB')


if __name__ == '__main__':
    import fire

    fire.Fire(main)
