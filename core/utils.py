import json
from pathlib import Path
from safetensors.torch import load_file
from transformers import Qwen3Config

def load_qwen3_weights():
    WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "qwen3-1.7b"

    cfg = Qwen3Config(**json.loads((WEIGHTS / "config.json").read_text()))
    index = json.loads((WEIGHTS / "model.safetensors.index.json").read_text())

    sd = {}

    for shard in sorted(set(index["weight_map"].values())):
        sd.update(load_file(WEIGHTS / shard))

    return cfg, sd

def get_encoder():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    return tokenizer