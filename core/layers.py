import contextlib

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math


# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
            
    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(self.eps + torch.mean(x.pow(2), dim=-1, keepdim=True))
        return (x.to(dtype) * self.weight)
        
# Linear
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

class LoRALinear(nn.Module):
    def __init__(self, base_linear: Linear, alpha=16, r=16):
        super().__init__()
        
        object.__setattr__(self, "base", base_linear)
        n, m = base_linear.weight.shape
        self.base.weight.requires_grad = False
        
        self.A = nn.Parameter(torch.empty((r, m)))
        self.B = nn.Parameter(torch.zeros((n, r)))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) 
        # like don't really need the a=sqrt(5) here since we dont have activation functions
        # but shouldnt negatively affect anything
        
        self.scale = alpha / r
        self.enabled = True
    
    def forward(self, x):
        base_out = self.base(x) # input @ (n x m) 
        if not self.enabled:
            return base_out # base weights are frozen, so this is the reference policy
        out = x @ self.A.T @ self.B.T * self.scale # m x (m x r) x (r x n) * scale
        return base_out + out


@contextlib.contextmanager
def lora_disabled(model):
    """Run the model as the untouched base model (no adapters)."""
    adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in adapters:
        m.enabled = False
    try:
        yield model
    finally:
        for m in adapters:
            m.enabled = True
