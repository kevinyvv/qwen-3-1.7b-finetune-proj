import json
from pathlib import Path

import torch

from tqdm import tqdm

from core.utils import load_qwen3_weights, get_encoder
from core.model import Qwen3ForCausalLM
from train_loop.checkpointing import load_checkpoint, save_checkpoint

OVERFIT_SAMPLE = "respond without using the letter 'e'"
SAMPLES = [OVERFIT_SAMPLE for _ in range(8)]

def get_feedback(tok_ids, e_table):
    reward_tensor = e_table[tok_ids]
    
    return reward_tensor
   
def discount(rewards, gamma=0.9):
    returns = torch.zeros_like(rewards)

    running = torch.zeros(
        rewards.size(0),
        device=rewards.device,
        dtype=rewards.dtype,
    )

    for t in reversed(range(rewards.size(1))):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running

    return returns

def get_e_table(tok_to_id, eos_token, vocab_size):
    e_table = torch.zeros(vocab_size, dtype=torch.float)

    for tok, idx in tok_to_id.items():
        if 'e' in tok.lower() and tok != eos_token:
            e_table[idx] = -1.
        else:
            e_table[idx] = 0.5

    return e_table

def main(max_new_tokens: int=64, epochs=10, save_every: int=5, resume: bool=True, temperature=1.2):
    cfg, sd = load_qwen3_weights()
    tokenizer = get_encoder()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    e_table = get_e_table(tokenizer.get_vocab(), tokenizer.eos_token, cfg.vocab_size)
    e_table = e_table.to(device)
    
    qwen3 = Qwen3ForCausalLM(cfg)
    qwen3.load_state_dict(sd, strict=False)
    qwen3.to(device)
    
    for name, p in qwen3.named_parameters():
        p.requires_grad = name.endswith((".A", ".B"))

    optimizer = torch.optim.AdamW(
        [p for p in qwen3.parameters() if p.requires_grad], lr=1e-4
    )

    ckpt_path = Path(__file__).resolve().parent / "checkpoints" / "latest.pt"
    start_epoch = load_checkpoint(ckpt_path, qwen3, optimizer) if resume else 0
    end_epoch = start_epoch + epochs

    if start_epoch:
        print(f"resuming from {ckpt_path} at epoch {start_epoch}")
    
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) for s in SAMPLES
    ]
    
    log_path = Path(__file__).resolve().parent / "logs" / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    batch = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    
    input_ids, attn_mask = batch.input_ids, batch.attention_mask
    positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
    
    batch_size, prompt_len = input_ids.size(0), input_ids.size(1)
    kv_caches = [None for _ in range(cfg.num_hidden_layers)]
        
    # run prefill on kv cache
    with torch.no_grad():
        prefill_logits, prefill_kv_caches = qwen3(input_ids[:, :-1], positions[:, :-1], attn_mask[:, :-1], kv_caches) 

    for epoch in tqdm(range(start_epoch, end_epoch)):
        # reset values
        input_ids, attn_mask = batch.input_ids.clone().detach(), batch.attention_mask.clone().detach()
        positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
        
        next_tok = input_ids[:, -1:] # last token of the sequence passed in   
        logits = prefill_logits.clone().detach()
        kv_caches = [[k.detach(), v.detach()] for k, v in prefill_kv_caches]
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        log_probs = torch.tensor([[] for _ in range(batch_size)]).to(device)
        masks = [] # masks for training so we dont train on pad tokens
        
        for _ in range(max_new_tokens):
            # use the last position only once we have kv cache         
            logits, kv_caches = qwen3(next_tok, positions[:, -1:], attn_mask, kv_caches) 
            
            log_logits = (logits[:, -1, :] / temperature).log_softmax(dim=-1)
            prob = log_logits.exp()
            next_tok = torch.multinomial(prob, num_samples=1)
            log_prob = log_logits.gather(dim=-1, index=next_tok)
            
            masks.append(~finished.unsqueeze(1))
            next_tok = torch.where(finished.unsqueeze(-1), tokenizer.pad_token_id, next_tok)
            finished |= next_tok.squeeze(-1) == tokenizer.eos_token_id

            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_tok)], dim=-1)
            positions = torch.cat([positions, positions[:, -1:] + 1], dim=-1)

            log_probs = torch.cat([log_probs, log_prob], dim=-1)
            
            if finished.all():
                break
        
        mask = torch.cat(masks, dim=-1).to(device) # (b, T) bool, False for padding
        gen_ids = input_ids[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        rewards = get_feedback(gen_ids, e_table)
        per_token_reward = discount(rewards)

        advantage = (per_token_reward - per_token_reward[mask].mean()) * mask
        loss = -(log_probs * advantage).sum()

        n_tokens = mask.sum(dim=-1)
        token_reward = (rewards * mask).sum(dim=-1) 
        e_rate = (-token_reward.sum() / n_tokens.sum()).item()

        record = {
            "epoch": epoch,
            "loss": loss.item(),
            "mean_reward": token_reward.mean().item(),
            "e_rate": e_rate,
            "samples": [
                {"text": t, "reward": r, "n_tokens": n}
                for t, r, n in zip(texts, token_reward.tolist(), n_tokens.tolist())
            ],
        }

        with open(log_path, 'a') as file:
            file.write(json.dumps(record) + "\n")

        print(f"epoch {epoch}, mean reward {token_reward.mean():.3f}, e-rate {e_rate:.3f}, "
              f"{texts[0][:200]!r}") # !r to see newline as \n

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % save_every == 0 or epoch + 1 == end_epoch:
            save_checkpoint(ckpt_path, qwen3, optimizer, epoch)

if __name__ == "__main__":
    import fire
    fire.Fire(main)


