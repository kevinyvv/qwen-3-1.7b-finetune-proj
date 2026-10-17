import hashlib
import json
import random
import re

from pathlib import Path

from datasets import concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent


# prompts that ask the model to write or fix actual code
# (not "write a movie script" or "what is SQL?", those can be answered in plain words)
CODE_REQUEST = re.compile(
    r"```"
    r"|\b(write|fix|debug|implement|optimi[sz]e|refactor)\b[^.?!\n]{0,40}\b(function|code|query|regex)\b"
    r"|\bscript in (python|bash|javascript|powershell)\b",
    re.IGNORECASE,
)

# chat has a persona system prompt ("answers in rhymes"), which stacks a second rule on top of no 'e'
NO_ROBOTS_SKIP = {"Coding", "Chat"}

def word_count(text):
    return len(text.split())


def load_no_robots():
    # both splits look alike, we pool them and make our own split later
    ds = concatenate_datasets([load_dataset("HuggingFaceH4/no_robots", split=s) for s in ("train", "test")])

    rows = []
    for ex in ds:
        if ex["category"] in NO_ROBOTS_SKIP:
            continue
        # everything left is one user message and one assistant reply
        rows.append({
            "source": "no_robots",
            "category": ex["category"],
            "prompt": ex["messages"][0]["content"],
            "response": ex["messages"][1]["content"],
        })
    return rows


def load_dolly():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    rows = []
    for ex in ds:
        prompt = ex["instruction"].strip()
        if ex["context"].strip():
            prompt += "\n\n" + ex["context"].strip()
        rows.append({
            "source": "dolly",
            "category": ex["category"],
            "prompt": prompt,
            "response": ex["response"],
        })
    return rows


def main(max_prompt_words: int=150, max_response_words: int=300, val_size: int=200, test_size: int=500, seed: int=0):
    rows = load_no_robots() + load_dolly()
    print(f"loaded {len(rows)} rows")

    kept, seen = [], set()
    dropped = {"empty": 0, "code": 0, "long prompt": 0, "long response": 0, "duplicate": 0}
    for row in rows:
        row["prompt"], row["response"] = row["prompt"].strip(), row["response"].strip()
        key = " ".join(row["prompt"].lower().split())

        if not row["prompt"] or not row["response"]:
            dropped["empty"] += 1
        elif CODE_REQUEST.search(row["prompt"]):
            dropped["code"] += 1
        elif word_count(row["prompt"]) > max_prompt_words:
            dropped["long prompt"] += 1
        elif word_count(row["response"]) > max_response_words:
            dropped["long response"] += 1 # usually a prompt asking for a long essay or story
        elif key in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(key)
            row["id"] = hashlib.sha256(key.encode()).hexdigest()[:16]
            kept.append(row)

    print("dropped:", dropped)

    random.Random(seed).shuffle(kept)
    splits = {
        "val": kept[:val_size],
        "test": kept[val_size:val_size + test_size],
        "train": kept[val_size + test_size:],
    }

    for name, split in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as file:
            for row in split:
                file.write(json.dumps({k: row[k] for k in ("id", "source", "category", "prompt", "response")}) + "\n")

        by_source = {s: sum(r["source"] == s for r in split) for s in ("no_robots", "dolly")}
        print(f"{path}: {len(split)} rows {by_source}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
