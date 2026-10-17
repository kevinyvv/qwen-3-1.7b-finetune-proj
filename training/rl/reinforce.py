import logging
from pathlib import Path

import torch

from core.generation import generate_rollout
from training.engine import TrainingEngine

logger = logging.getLogger(__name__)

# okay REINFORCE for now
# figure out what actually needs to be logged later and how to make it look not cancer
class ReinforceEngine(TrainingEngine):
     def __init__(self, model, tokenizer, optimizer, reward_fn, device=None,
                  finetune_mode="full", path=None, metrics_path=None):
          super().__init__()
          
          self.model = model
          self.tokenizer = tokenizer
          self.optimizer = optimizer
          self.reward_fn = reward_fn
          self.finetune_mode = finetune_mode
          
          if device is None:
               device = 'cuda' if torch.cuda.is_available() else 'cpu'
          self.device = device
          self.model.to(self.device)
          self.path = path
          self.metrics_path = Path(metrics_path) if metrics_path is not None else None
          self.epoch = 0
          self.global_step = 0

          if self.path is not None:
               self.load_checkpoint(self.path)

     def compute_log_probs(self, batch, tokens, response_mask, temperature=1.2):
          # tokens is prompt + response, input_ids is just prompt
          batch = {k: v.to(self.device) for k,v in batch.items()}
          
          input_ids = batch['input_ids']                                        
          attn_mask = batch['attention_mask'] # rmr that attn mask is for extra padding tokens
          # Extend the prompt mask to cover the response before computing positions.
          attn_mask = torch.cat(
               [attn_mask, response_mask.to(device=attn_mask.device, dtype=attn_mask.dtype)],
                                dim=-1)
          positions = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
          kv_caches = [None for _ in range(self.model.config.num_hidden_layers)]
          
          logits, _ = self.model(tokens, positions, attn_mask, kv_caches)
          
          response_len = tokens.shape[1] - input_ids.shape[1]
          
          response_logits = logits[:, -response_len-1:-1, :]
          response_log_probs = (response_logits.float() / temperature).log_softmax(dim=-1)
          response_tokens = tokens[:, -response_len:]
          
          log_probs = response_log_probs.gather(dim=-1, index=response_tokens.unsqueeze(-1)).squeeze(-1)
     
          return log_probs   

     def compute_loss(self, batch, max_new_tokens=64, temperature=1.2, return_metrics=False,
                      thinking_budget=None):
          with torch.no_grad():
               rollout = generate_rollout(self.model, self.tokenizer, batch, max_new_tokens, temperature, thinking_budget)
               rewards = self.reward_fn(rollout.response_ids, rollout.reward_mask)

          tokens = torch.cat([batch['input_ids'].to(self.device), rollout.response_ids], dim=-1)
          token_log_probs = self.compute_log_probs(batch, tokens, rollout.completion_mask, temperature)
          sequence_log_probs = token_log_probs.masked_fill(~rollout.loss_mask, 0.0).sum(dim=-1)
          loss = -(rewards.detach() * sequence_log_probs).mean()

          if return_metrics:
               return loss, {
                    "reward_sum": rewards.detach().float().sum().item(),
                    "response_tokens": rollout.completion_mask.sum().item(),
               }
          return loss

     def _log_training(self, loss_sum, reward_sum, sequences, input_tokens, response_tokens, started):
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
               loss=loss_sum / sequences,
               reward=reward_sum / sequences,
               response_length=response_tokens / sequences,
               learning_rate=learning_rates[0] if len(learning_rates) == 1 else learning_rates,
               grad_norm=grad_norm,
               sequences=sequences,
               input_tokens=input_tokens,
               response_tokens=response_tokens,
               seconds=seconds,
               response_tokens_per_sec=response_tokens / seconds,
          )

     def fit(self, data_loader, max_new_tokens=64, temperature=1.2, epochs=1,
             val_loader=None, eval_every=100, save_every=100, log_every=20, thinking_budget=None):
          if min(epochs, eval_every, save_every, log_every) <= 0:
               raise ValueError("epochs and logging/evaluation/checkpoint intervals must be positive")
          logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
          start_epoch = self.epoch
          self.model.train()

          for epoch in range(start_epoch, start_epoch + epochs):
               self.epoch = epoch
               logger.info("Started epoch %d at step %d", epoch, self.global_step)
               loss_sum = reward_sum = 0.0
               sequences = input_tokens = response_tokens = 0
               started = self._training_time()

               for batch in data_loader:
                    self.optimizer.zero_grad()
                    loss, metrics = self.compute_loss(
                         batch, max_new_tokens, temperature, return_metrics=True,
                         thinking_budget=thinking_budget,
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1

                    batch_size = batch["input_ids"].shape[0]
                    loss_sum += loss.item() * batch_size
                    reward_sum += metrics["reward_sum"]
                    sequences += batch_size
                    input_tokens += batch["attention_mask"].sum().item()
                    response_tokens += metrics["response_tokens"]

                    log_due = self.global_step % log_every == 0
                    eval_due = val_loader is not None and self.global_step % eval_every == 0
                    save_due = self.path is not None and self.global_step % save_every == 0

                    if log_due or eval_due or save_due:
                         self._log_training(
                              loss_sum, reward_sum, sequences, input_tokens, response_tokens, started,
                         )
                         loss_sum = reward_sum = 0.0
                         sequences = input_tokens = response_tokens = 0

                    if eval_due:
                         self.evaluate(val_loader, max_new_tokens, temperature, thinking_budget)
                         self.model.train()
                    if save_due:
                         self.save_checkpoint(self.path)

                    if log_due or eval_due or save_due:
                         started = self._training_time()

               if sequences:
                    self._log_training(
                         loss_sum, reward_sum, sequences, input_tokens, response_tokens, started,
                    )
               self.epoch = epoch + 1

          if self.path is not None:
               self.save_checkpoint(self.path)

     def evaluate(self, data_loader, max_new_tokens=64, temperature=1.2, thinking_budget=None):
          self.model.eval()
          loss_sum = reward_sum = 0.0
          sequences = response_tokens = 0
          with torch.no_grad():
               for batch in data_loader:
                    loss, metrics = self.compute_loss(
                         batch, max_new_tokens, temperature, return_metrics=True,
                         thinking_budget=thinking_budget,
                    )
                    batch_size = batch["input_ids"].shape[0]
                    loss_sum += loss.item() * batch_size
                    reward_sum += metrics["reward_sum"]
                    sequences += batch_size
                    response_tokens += metrics["response_tokens"]

          if sequences == 0:
               raise ValueError("expected at least one sequence to evaluate")
          reward = reward_sum / sequences
          self._log_metrics(
               "eval",
               loss=loss_sum / sequences,
               reward=reward,
               response_length=response_tokens / sequences,
               sequences=sequences,
               response_tokens=response_tokens,
          )
          return reward
