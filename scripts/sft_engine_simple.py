import torch

from pathlib import Path

from core.utils import load_qwen3_weights, get_encoder
from core.model import Qwen3ForCausalLM
from training.engine import SFTEngine
from training.sft_utils import get_loader, load_sft_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"


def limit_vram(headroom_gb=1.0):
    free, total = torch.cuda.mem_get_info()
    fraction = (free - headroom_gb * 1024**3) / total
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"VRAM cap: {fraction * total / 1024**3:.2f} GiB")


def main(dataset='tiny', batch_size=4, max_length=None, epochs=3,
         eval_every=5, save_every=50, log_every=1):
    
    if torch.cuda.is_available():
        limit_vram() 

    cfg, sd = load_qwen3_weights()
    tokenizer = get_encoder()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    qwen3 = Qwen3ForCausalLM(cfg)
    qwen3.load_state_dict(sd, strict=False)
    
    for name, p in qwen3.named_parameters():
            p.requires_grad = name.endswith((".A", ".B"))
    optimizer = torch.optim.AdamW(
        [p for p in qwen3.parameters() if p.requires_grad], lr=1e-4
    )
    
    path = Path(__file__).resolve().parent / "checkpoints" / "sft_test.pt"
    sft_engine = SFTEngine(qwen3, tokenizer, optimizer, gradient_checkpointing=True, loss_backend='liger', path=path)
    
    train_dataset = load_sft_dataset(DATA_DIR / f"{dataset}_train.jsonl", tokenizer, max_length)
    val_dataset = load_sft_dataset(DATA_DIR / f"{dataset}_val.jsonl", tokenizer, max_length)
    train_loader = get_loader(train_dataset, tokenizer, batch_size=batch_size)
    val_loader = get_loader(val_dataset, tokenizer, batch_size=batch_size, shuffle=False)

    try:
        sft_engine.fit(train_loader, val_loader, epochs=epochs, eval_every=eval_every, save_every=save_every, log_every=log_every)
    finally:
        if torch.cuda.is_available():
            print(f"peak VRAM used by pytorch: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

if __name__ == "__main__":
    import fire
    fire.Fire(main)


