import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


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
