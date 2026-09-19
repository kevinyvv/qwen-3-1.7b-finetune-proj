import json
import random

from pathlib import Path

from core.utils import get_encoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"


# every task has a short prompt and a long response
# so cutting an example down to max_length always leaves response tokens to train on

def count_up(rng, n):
    start, step = rng.randint(1, 500), rng.randint(1, 9)
    prompt = f"Count up from {start} in steps of {step}. Write {n} numbers separated by commas."
    response = ", ".join(str(start + i * step) for i in range(n))
    return prompt, response


def count_down(rng, n):
    step = rng.randint(1, 9)
    start = rng.randint(1, 500) + n * step # high enough that we never go below zero
    prompt = f"Count down from {start} in steps of {step}. Write {n} numbers separated by commas."
    response = ", ".join(str(start - i * step) for i in range(n))
    return prompt, response


def times_table(rng, n):
    k = rng.randint(2, 99)
    prompt = f"Write the {k} times table from {k} x 1 to {k} x {n}, one line each."
    response = "\n".join(f"{k} x {i} = {k * i}" for i in range(1, n + 1))
    return prompt, response


def squares(rng, n):
    start = rng.randint(1, 200)
    prompt = f"Write the squares of the {n} numbers starting at {start}, one line each."
    response = "\n".join(f"{i} squared = {i * i}" for i in range(start, start + n))
    return prompt, response


TASKS = [count_up, count_down, times_table, squares]


def make_example(task, seed, tokenizer, min_tokens):
    n = 100
    while True:
        # same seed every try, so only n changes and the example just gets longer
        prompt, response = task(random.Random(seed), n)
        if len(tokenizer(response)["input_ids"]) >= min_tokens:
            return {"prompt": prompt, "response": response}
        n += 50


def write_split(name, seeds, tokenizer, min_tokens):
    path = DATA_DIR / f"long_{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    lengths = []
    with path.open("w") as file:
        for seed in seeds:
            example = make_example(TASKS[seed % len(TASKS)], seed, tokenizer, min_tokens)
            lengths.append(len(tokenizer(example["response"])["input_ids"]))
            file.write(json.dumps(example) + "\n")

    print(f"{path}: {len(lengths)} examples, response tokens min={min(lengths)} max={max(lengths)}")


def main(min_tokens: int=2048, train_size: int=32, val_size: int=8):
    tokenizer = get_encoder()

    write_split("train", range(train_size), tokenizer, min_tokens)
    write_split("val", range(1000, 1000 + val_size), tokenizer, min_tokens) # different seeds than train


if __name__ == "__main__":
    import fire

    fire.Fire(main)
