"""Quantization-aware training with fake INT4 weights.

Weights are quantised group-wise (symmetric, per output row, group size 32)
in the forward pass. A straight-through estimator passes gradients to the
fp32 master weights, so the model learns to tolerate 4-bit rounding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fake_quant_int4(w: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    g = w.reshape(out_f, in_f // group_size, group_size)
    # Symmetric signed INT4: levels in [-8, 7]; scale from max magnitude.
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(g / scale), -8, 7) * scale
    q = q.reshape(out_f, in_f)
    # Straight-through estimator: forward uses q, backward passes gradient to w.
    return w + (q - w).detach()


class FakeQuantLinear(nn.Linear):
    """nn.Linear with optional INT4 fake quantisation on the weight."""

    qat_bits_group: int = 32
    qat_enabled: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quant_int4(self.weight, self.qat_bits_group) if self.qat_enabled else self.weight
        return nn.functional.linear(x, w, self.bias)


def enable_qat(model: nn.Module, group_size: int = 32) -> int:
    """Convert every nn.Linear inside transformer blocks to FakeQuantLinear.

    The embedding and LM head stay in higher precision (they are kept at 8-bit or
    fp16 in the export). Returns the number of converted layers. The state_dict
    keys do not change, so checkpoints load either way.
    """
    count = 0
    for name, module in model.named_modules():
        if name.startswith("blocks.") and type(module) is nn.Linear:
            if module.in_features % group_size != 0:
                continue
            module.__class__ = FakeQuantLinear
            module.qat_bits_group = group_size
            module.qat_enabled = True
            count += 1
    return count
