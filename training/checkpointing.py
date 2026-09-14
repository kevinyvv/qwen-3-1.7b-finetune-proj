import torch
import os
from pathlib import Path

def load_checkpoint(ckpt_path, qwen3, optimizer):
    # load the lora adapters and optimizer states from ckpt path
    # return the next epoch to start training at
    if not Path(ckpt_path).exists():
        return 0    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    qwen3.load_state_dict(checkpoint['qwen3_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return epoch + 1

def save_checkpoint(ckpt_path, qwen3, optimizer, epoch):
    # save state and epoch at path
    # only need to store A and B (lora adapter states)
    checkpoint = {
        'qwen3_state_dict': {k: v for k, v in qwen3.state_dict().items() if k.endswith((".A", ".B"))},
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(ckpt_path).with_name(Path(ckpt_path).name + '.tmp')
    torch.save(checkpoint, tmp)
    os.replace(tmp, ckpt_path)
