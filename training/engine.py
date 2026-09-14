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
     def __init__(self, model, tokenizer, optimizer, gradient_checkpointing, loss_backend, precision='bf16', finetune_mode='lora', path=None):
          
          self.model = model
          self.tokenizer = tokenizer
          self.optimizer = optimizer
          self.gradient_checkpointing = gradient_checkpointing
          self.loss_backend = loss_backend
          self.precision = precision
          self.finetune_mode = finetune_mode
          
          self.epoch = 0
          self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
          
          if self.path is not None:
               self.load_checkpoint(path)
               
     def compute_loss(self, batch):
          # assume batch is of dict of
          # input_ids, attention_mask, labels
          input_ids = batch['input_ids']
          attn_mask = batch['attention_mask']
          labels = batch['labels']
          kv_caches = [None for _ in range(self.model.config)]
          positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
          

          logits, kv_caches = self.model(input_ids, attn_mask, positions, kv_caches)
          
     def fit(self, train_loader, val_loader, epochs=3, eval_every=100, save_every=100):
          for epoch in range(epochs):
               self.model.train() # set to train mode
               
               for batch in train_loader:
                    self.optimizer.zero_grad()
                    
                    loss = self.compute_loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    global_step += 1
                    
                    if global_step % eval_every == 0:
                         val_loss = self.evaluate(val_loader)
                         # do something with it after? like save it maybe
                         self.model.train()
                    
                    if global_step % save_every == 0:
                         self.save_checkpoint(self.path) if self.path is not None else self.save_checkpoint()
                         
                    # additional logging stuff 
          return
     
     def evaluate(self, data_loader):
          # define some metrics at the start
          
          self.model.eval()
          
          with torch.no_grad():
               for batch in data_loader:
                    loss = self.compute_loss(batch)

                    # do something with it i guess
          
          # do something with all metrics collected
          
          return
          
     def load_checkpoint(self, path):
          if not Path(path).exists():
               return 0    
          checkpoint = torch.load(path, map_location='cpu')
          
          self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          self.epoch = checkpoint['epoch'] + 1

     def save_checkpoint(self, path='latest'):
          state_dict = {k: v for k, v in self.model.state_dict().items() if k.endswith((".A", ".B"))} if self.finetune_mode=='lora' else self.model.state_dict()
          checkpoint = {
               'model_state_dict': state_dict,
               'optimizer_state_dict': self.optimizer.state_dict(),
               'epoch': self.epoch
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
     