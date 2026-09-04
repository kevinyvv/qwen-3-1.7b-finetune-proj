import torch
from torch import nn
import math
from transformers import Qwen3Config

from engine.layers import Linear, RMSNorm

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_position = max_position
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        
        self.q_proj = Linear(hidden_size, num_heads * head_dim, qkv_bias)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, qkv_bias)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, qkv_bias)
    
        self.o_proj = Linear(num_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        position_embs,
        hidden_states: torch.Tensor,
        attn_mask, # attn mask is of size [b, seq_len] or None
        kv_cache=None
    ) -> torch.Tensor:
        # hidden states is of size [b, seq_len, embedding_dim]
        batch_size, seq_len = hidden_states.size(dim=0), hidden_states.size(dim=1)
        
        # get q and k using proj and norm q, k
        q = self.q_proj(hidden_states) 
        # we want to norm across the head_dim, as q_proj produces [batch, seq_len, num_heads x head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # after norming it should be of [batch, n_heads, seq_len, head_dim]
        q = self.q_norm(q) 
        
        k_new = self.k_proj(hidden_states) # [batch, seq_len, num_kv_heads * head_dim]
        k_new = k_new.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_norm(k_new) # same logic for k
        
        # RoPE on Q and K
        sin, cos = position_embs
        sin, cos = sin.unsqueeze(1), cos.unsqueeze(1)
        # formula for x,y is cos x - sin x, sin y + cos y
        # y is x + head_dim//2
        # cosine we can just keep as is
        # for sine we lowkey just need to stack like -sin on top of sin
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.concat((-x2, x1), dim=-1)
                        
        q = q * cos + rotate_half(q) * sin
        k_new = k_new * cos + rotate_half(k_new) * sin
        
        
        v_new = self.v_proj(hidden_states) # [b, seq_len, n_kv_heads x head_dim] 
        # need to reshape v to be of [b, n_head, seq_len, head_dim]
        v_new = torch.transpose(torch.reshape(v_new, (batch_size, seq_len, self.num_kv_heads, self.head_dim)), 1, 2)
        
        # if kv_cache exists: 
        if kv_cache is None:
            k = k_new
            v = v_new
        else:
            k = torch.concat((kv_cache[0], k_new), dim=2)
            v = torch.concat((kv_cache[1], v_new), dim=2)
        
        new_kv_cache = [k, v]
        
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) # to match q dims
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) # to match dim of n_head 
        
        # create causal mask + attn_mask
        neg = torch.finfo(q.dtype).min
    
        k_len = k.size(2)
        causal_mask = torch.full((seq_len, k_len), neg, device=q.device, dtype=q.dtype).triu(1 + k_len - seq_len) # [seq_len, seq_len]
        mask = torch.zeros((batch_size, 1, 1, k_len), device=q.device, dtype=q.dtype)
        
        if attn_mask is not None:
            mask = mask.masked_fill(attn_mask[:, None, None, :] == 0, neg)
        
        # q is of shape [b, n_head, q_seq_len=1, head_dim] (with kv cache)
        # k is of shape [b, n_head, seq_len, head_dim]    
        # q @ k^t -> o [b, n_head, q_seq_len=1, seq_len] (with kv cache)
        o = nn.functional.softmax((q @ k.transpose(2,3)) * self.scaling + causal_mask + mask, dim=-1, dtype=torch.float32).to(q.dtype)  
        # o @ v
        output = o @ v # [b, n_heads, q_seq_len=1, head_dim] w/ kv_cache of course
        
        # resize to be [b, s, n_head*head_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)   # [b, s, nh*hd]
        
        # then return (norm is handled downstream (post attention layernorm))
        return self.o_proj(output), new_kv_cache


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int
    ) -> None:
        
        super().__init__()
        
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.silu = nn.SiLU()

    def forward(self, x):
        
        y = self.silu(self.gate_proj(x))
        x = self.up_proj(x)
        
        x = x * y
        x = self.down_proj(x)
        
        return x


class Qwen3Layer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        
    def forward(
        self,
        position_embs,
        hidden_states: torch.Tensor,
        attn_mask,
        kv_cache=None
    ):
        
        layer_normed = self.input_layernorm(hidden_states)
        attn_res, new_kv_cache = self.self_attn(position_embs, layer_normed, attn_mask, kv_cache)
        hidden_states = hidden_states + attn_res
        
        layer_normed = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(layer_normed)
        
        return hidden_states, new_kv_cache


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        
        self.config = config # so we can use it for RoPE below
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3Layer(config)
                                        for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_mask,
        kv_caches # shape [b, blocks, [2, [seq_len, num_kv_head * head_dim]]]
    ) -> torch.Tensor:
        
        x = self.embed_tokens(input_ids) # -> [batch, sequence_length, embedding_dim]
        
        rope_theta = self.config.rope_parameters['rope_theta']
        head_dim = self.config.head_dim
        
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)) # [head_dim/2]
        freqs = positions.unsqueeze(-1).float() * inv_freq        # [b, seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                   # [b, seq, head_dim] qwen3 does this rather than pairing i, i+1
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype) 
        
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            x, kv_cache = layer((sin, cos), x, attn_mask, kv_cache)
            kv_caches[i] = kv_cache 
            
        x = self.norm(x)
        # need to go from [batch, sequence_length, embedding_dim] -> [batch, sequence_length, vocab_size]
        # multiply by [embedding_dim, vocab_size]
        # can do this in the lm_head step.
        return x, kv_caches
        

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config) 
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, input_ids, positions, attn_mask, kv_caches):
        x, new_kv_caches = self.model(input_ids, positions, attn_mask, kv_caches) # x dim = [b, seq, embed_dim]       
        logits = self.lm_head(x) # [b, seq, vocab_size]

        return logits, new_kv_caches
