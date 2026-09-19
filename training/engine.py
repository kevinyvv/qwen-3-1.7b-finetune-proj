import json
import logging
import os
from pathlib import Path
import time

import torch

logger = logging.getLogger(__name__)

class TrainingEngine:
     def __init__(self):
          return # does nothing for now

class SFTEngine(TrainingEngine):
     def __init__(self, model, tokenizer, optimizer, gradient_checkpointing, loss_backend, precision=torch.bfloat16, finetune_mode='lora', device=None, path=None, metrics_path=None):
          
          self.model = model
          self.tokenizer = tokenizer
          self.optimizer = optimizer
          self.gradient_checkpointing = gradient_checkpointing
          self.loss_backend = loss_backend
          
          if precision not in (torch.float32, torch.bfloat16):
               raise ValueError("only fl32 and bf16 supported")
          self.precision = precision
          self.finetune_mode = finetune_mode
          
          if device is None:
               device = 'cuda' if torch.cuda.is_available() else 'cpu'
          self.device = device
          for p in self.model.parameters():
               if not p.requires_grad:
                    p.data = p.data.to(self.precision)
          self.model.to(self.device)

          self.path = path
          self.metrics_path = Path(metrics_path) if metrics_path is not None else None
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
          
          # for precision
          with torch.autocast(device_type=torch.device(self.device).type, dtype=self.precision, enabled=self.precision != torch.float32):
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
          
     def _log_metrics(self, event, **metrics):
          record = {
               "event": event,
               "global_step": self.global_step,
               "epoch": self.epoch,
               **metrics,
          }
          summary = " ".join(
               f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
               for key, value in record.items()
          )
          logger.info("%s", summary)
          if self.metrics_path is not None:
               self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
               with self.metrics_path.open("a") as file:
                    file.write(json.dumps(record) + "\n")

     def _training_time(self):
          device = torch.device(self.device)
          if device.type == "cuda":
               torch.cuda.synchronize(device) # since cuda runs async we need to sync periodically (whenever we want to measure)
          return time.perf_counter()

     def _log_training(self, loss_sum, supervised_tokens, input_tokens, started):
          seconds = self._training_time() - started
          norms = [
               p.grad.detach().float().norm(2)
               for group in self.optimizer.param_groups
               for p in group["params"]
               if p.grad is not None
          ]
          grad_norm = torch.stack(norms).norm(2).item() if norms else 0.0
          learning_rates = [group["lr"] for group in self.optimizer.param_groups]
          self._log_metrics(
               "train",
               loss=loss_sum / supervised_tokens,
               learning_rate=learning_rates[0] if len(learning_rates) == 1 else learning_rates,
               grad_norm=grad_norm,
               input_tokens=input_tokens,
               supervised_tokens=supervised_tokens,
               seconds=seconds,
               input_tokens_per_sec=input_tokens / seconds,
          )

     def fit(self, train_loader, val_loader, epochs=3, eval_every=100, save_every=100, log_every=20):
          """Log training intervals and flush them before eval, saves, and epoch ends."""
          if min(log_every, eval_every, save_every) <= 0:
               raise ValueError("logging, evaluation, and save intervals must be positive")
          # Supply a console default when the caller has not configured logging.
          logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
          start_epoch = self.epoch
          
          self.model.train() # set to train mode
          
          for epoch in range(start_epoch, epochs+start_epoch):
               logger.info("Started epoch %d at step %d", epoch, self.global_step)

               self.epoch = epoch
               loss_sum = 0.0
               input_tokens = supervised_tokens = 0
               started = self._training_time()
               
               for batch in train_loader:
                    self.optimizer.zero_grad()
                    
                    loss = self.compute_loss(batch)
                    loss.backward()

                    self.optimizer.step()
                    
                    self.global_step += 1

                    num_valid_tokens = (batch["labels"][:, 1:] != -100).sum().item()
                    loss_sum += loss.item() * num_valid_tokens
                    supervised_tokens += num_valid_tokens
                    input_tokens += batch["attention_mask"].sum().item()

                    log_due = self.global_step % log_every == 0
                    eval_due = self.global_step % eval_every == 0
                    save_due = self.global_step % save_every == 0
                    
                    if log_due or eval_due or save_due:
                         self._log_training(loss_sum, supervised_tokens, input_tokens, started)
                         loss_sum = 0.0
                         input_tokens = supervised_tokens = 0

                    if eval_due:
                         self.evaluate(val_loader)
                         self.model.train()
                    
                    if save_due:
                         self.save_checkpoint(self.path if self.path is not None else "latest")

                    if log_due or eval_due or save_due:
                         started = self._training_time()

               if supervised_tokens:
                    self._log_training(loss_sum, supervised_tokens, input_tokens, started)

          self.save_checkpoint(self.path if self.path is not None else "latest")
               
          return
     
     def evaluate(self, data_loader):
          self.model.eval()
          
          total_loss = total_tokens = 0
          
          with torch.no_grad():
               
               for batch in data_loader:
                    loss = self.compute_loss(batch)
                    num_valid_tokens = (batch["labels"][:, 1:] != -100).sum().item()
                    total_loss += loss.item() * num_valid_tokens
                    total_tokens += num_valid_tokens
          if total_tokens == 0:
               raise AssertionError("expected valid data to evaluate on, got sample with zero valid tokens")      
          val_loss = total_loss / total_tokens
          self._log_metrics("eval", loss=val_loss, supervised_tokens=total_tokens)
          return val_loss
          
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
          logger.info("Loaded checkpoint %s at step %d", path, self.global_step)

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
          logger.info("Saved checkpoint %s at step %d", path, self.global_step)
 
 
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
