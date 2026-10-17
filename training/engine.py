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
