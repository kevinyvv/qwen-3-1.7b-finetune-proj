import json

import torch
from functools import partial
from torch.utils.data import DataLoader

def load_sft_dataset(path, tokenizer, max_length=None):
    examples = []

    with open(path) as file:
        for line in file:
            pair = json.loads(line)

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": pair["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompt_ids = tokenizer(prompt_text)["input_ids"]
            response_ids = tokenizer(pair["response"] + tokenizer.eos_token)["input_ids"]

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids 

            examples.append({
                "input_ids": input_ids[:max_length],
                "labels": labels[:max_length],
            })

    return examples


def collate_sft(examples, pad_token_id):
    batch_size = len(examples)
    max_length = max(len(ex["input_ids"]) for ex in examples)

    input_ids = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long
    ) # fill with padding first
    attention_mask = torch.zeros_like(input_ids) 
    labels = torch.full_like(input_ids, -100) # fill with mask out val 

    for i, example in enumerate(examples):
        length = len(example["input_ids"])

        input_ids[i, :length] = torch.tensor(
            example["input_ids"], dtype=torch.long
        )
        attention_mask[i, :length] = 1
        labels[i, :length] = torch.tensor(
            example["labels"], dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    
def get_loader(dataset, tokenizer, batch_size=4, shuffle=True):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_sft,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    
    return train_loader

