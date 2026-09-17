import json
from pathlib import Path
import os

import torch

from tqdm import tqdm
import string

from core.utils import load_qwen3_weights, get_encoder
from core.model import Qwen3ForCausalLM
from core.layers import lora_disabled

class TrainingEngine:
     def __init__(self):
          return # do nothing for now

class SFTEngine(TrainingEngine):
     def __init__(self, model, tokenizer, optimizer, gradient_checkpointing, loss_backend, precision='bf16', finetune_mode='lora', device=None, path=None):
          
          self.model = model
          self.tokenizer = tokenizer
          self.optimizer = optimizer
          self.gradient_checkpointing = gradient_checkpointing
          self.loss_backend = loss_backend
          self.precision = precision
          self.finetune_mode = finetune_mode
          
          if device is None:
               device = 'cuda' if torch.cuda.is_available() else 'cpu'
          self.device = device
          self.model.to(self.device)

          self.path = path
          self.epoch = 0
          self.global_step = 0
          
          if self.path is not None:
               self.load_checkpoint(self.path)
               
     def compute_loss(self, batch, invalid_val=-100):
          # assume batch is of dict of
          # input_ids, attention_mask, labels
          # need to change this assumption later maybe?
          
          batch = {k: v.to(self.device) for k,v in batch.items()}
                              
          input_ids = batch['input_ids']
          attn_mask = batch['attention_mask'] # rmr that attn mask is for extra padding tokens and 
          labels = batch['labels']
          kv_caches = [None for _ in range(self.model.config.num_hidden_layers)]
          positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
          
          logits, kv_caches = self.model(input_ids, positions, attn_mask, kv_caches)
          
          # basically for logits we want to find the probability of the labels and then push them upwards?
          logits = logits[:, :-1, :].contiguous() # shift logits left so we only compare ones we have labels for
          labels = labels[:, 1:].contiguous() # shift labels forward one to align 
          # logits are the predictions of step i
          # labels thus need to be 1 step before to match (we check the probabilities of the lables) being predicted

          log_probs = logits.log_softmax(dim=-1)
          
          valid = labels != invalid_val
          
          if not valid.any():
               raise ValueError("batch has no supervised target token")
          
          safe_labels = labels.masked_fill(~valid, 0)
          
          target_log_probs = log_probs.gather(
               dim=-1,
               index=safe_labels.unsqueeze(-1)
          ).squeeze(-1)
          
          loss = -target_log_probs[valid].mean()
          
          return loss
          
     def fit(self, train_loader, val_loader, epochs=3, eval_every=100, save_every=100):
          
          start_epoch = self.epoch
          
          self.model.train() # set to train mode
          
          for epoch in range(start_epoch, epochs+start_epoch):               
               self.epoch = epoch
               for batch in train_loader:
                    self.optimizer.zero_grad()
                    
                    loss = self.compute_loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    self.global_step += 1
                    
                    if self.global_step % eval_every == 0:
                         val_loss = self.evaluate(val_loader)
                         # do something with it after? like save it or emit it somewhere
                         self.model.train()
                    
                    if self.global_step % save_every == 0:
                         self.save_checkpoint(self.path) if self.path is not None else self.save_checkpoint()
                    # additional logging stuff 
               
          self.save_checkpoint(self.path) if self.path is not None else self.save_checkpoint()
               
          return
     
     def evaluate(self, data_loader):
          # define some metrics at the start
          
          self.model.eval()
          
          total_loss = total_tokens = 0
          
          with torch.no_grad():
               
               for batch in data_loader:
                    loss = self.compute_loss(batch)
                    num_valid_tokens = (batch["labels"][:, 1:] != -100).sum().item()
                    total_loss += loss.item() * num_valid_tokens
                    total_tokens += num_valid_tokens
                    # do something with it i guess

          # do something with all metrics collected
          
          if total_tokens == 0:
               raise AssertionError("expected valid data to evaluate on, got sample with zero valid tokens")      
          return total_loss / total_tokens
          
     def load_checkpoint(self, path):
          if not Path(path).exists():
               return 0    
          checkpoint = torch.load(path, map_location='cpu')
          
          if self.finetune_mode == "full":          
               self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
          elif self.finetune_mode == "lora":
               missing, _ = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
               # check that adapters are all there
               for k in missing:
                    assert not (k.endswith((".A", ".B")))
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          self.epoch = checkpoint['epoch']
          self.global_step = checkpoint['global_step']

     def save_checkpoint(self, path='latest'):
          state_dict = {k: v for k, v in self.model.state_dict().items() if k.endswith((".A", ".B"))} if self.finetune_mode=='lora' else self.model.state_dict()
          checkpoint = {
               'model_state_dict': state_dict,
               'optimizer_state_dict': self.optimizer.state_dict(),
               'epoch': self.epoch,
               'global_step': self.global_step
          }
          Path(path).parent.mkdir(parents=True, exist_ok=True)
          tmp = Path(path).with_name(Path(path).name + '.tmp')
          torch.save(checkpoint, tmp)
          os.replace(tmp, path)
 
 
class RLEngine(TrainingEngine):
     def __init__(self, model, tokenizer, optimizer, objective, reward_fn, config):
          pass

     def collect_rollout(self, prompts):
          """Generate without gradients; record tokens, masks, and sampling log-probs."""
          pass

     def update(self, rollout):
          """Compute training log-probs, loss, gradients, and optimizer updates."""
          pass

     def fit(self, prompt_loader):
          for prompts in prompt_loader:
               rollout = self.collect_rollout(prompts)
               rollout.rewards = self.reward_fn(rollout)
               metrics = self.update(rollout)
               pass

     def save_checkpoint(self, path):
          pass

     def load_checkpoint(self, path):
          pass
     