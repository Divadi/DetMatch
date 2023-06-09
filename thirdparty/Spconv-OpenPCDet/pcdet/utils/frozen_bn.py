import torch
from torch import nn as nn
from torch.nn import functional as F


class FrozenBatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-5, *args, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)
        self.register_buffer('num_batches_tracked', torch.zeros(tuple()))

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias too.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            if len(x.shape) == 2:
                scale = scale.reshape(1, -1)
                bias = bias.reshape(1, -1)
            else:
                assert len(x.shape) == 3
                scale = scale.reshape(1, -1, 1)
                bias = bias.reshape(1, -1, 1)

            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )


class FrozenBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, *args, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)
        self.register_buffer('num_batches_tracked', torch.zeros(tuple()))

    def forward(self, x):
        if x.requires_grad:
            assert len(x.shape) == 4
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias too.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)

            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )