import torch

from core.utils import load_qwen3_weights, get_encoder
from core.model import Qwen3ForCausalLM

from training.checkpointing import load_checkpoint

# stuff that needed to consider
# batching -- adds another dim so compute all at the same time and also need to handle next token at the same time
# qwen3 model has built in completions so only return the new tokens? (not sure if this is true -- verify this)

def main(prompts: list[str], max_new_tokens: int=32000, path=None):
    cfg, sd = load_qwen3_weights()
    tokenizer = get_encoder()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    qwen3 = Qwen3ForCausalLM(cfg)
    qwen3.load_state_dict(sd, strict=False)
    
    if path is not None:
        res = load_checkpoint(path, qwen3, None)
        assert res != 0
    
    qwen3.to(device)
    qwen3.eval()
    
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for p in prompts
    ]
    
    batch = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    input_ids, attn_mask = batch.input_ids, batch.attention_mask
    positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
    
    batch_size, prompt_len = input_ids.size(0), input_ids.size(1)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    
    kv_caches = [None for _ in range(cfg.num_hidden_layers)]
    
    with torch.no_grad():
        
        # run prefill on kv cache
        logits, kv_caches = qwen3(input_ids[:, :-1], positions[:, :-1], attn_mask[:, :-1], kv_caches)
        next_tok = input_ids[:, -1:] # last token of the sequence passed in

        # generate
        for _ in range(max_new_tokens):

            # use the last position only once we have kv cache         
            logits, kv_caches = qwen3(next_tok, positions[:, -1:], attn_mask, kv_caches) 
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [b, 1] 

            # a sequence that already hit EOS keeps emitting pad
            next_tok = torch.where(finished.unsqueeze(-1), tokenizer.pad_token_id, next_tok)
            finished |= next_tok.squeeze(-1) == tokenizer.eos_token_id

            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_tok)], dim=-1)
            positions = torch.cat([positions, positions[:, -1:] + 1], dim=-1)

            if finished.all():
                break
        
    # decode the ids back into a string
    texts = tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)

    return texts # probably needs some reworking with Fire


if __name__ == "__main__":
    import fire

    fire.Fire(main)


