import torch
from functools import partial
from torch.utils.data import DataLoader


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
    
    
def get_train_loader(train_dataset, tokenizer, batch_size=4, shuffle=True):
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_sft,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    
    return train_loader

