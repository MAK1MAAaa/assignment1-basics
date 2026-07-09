from __future__ import annotations

import math

import torch
from torch import nn

from cs336_basics.Part3.linear import Linear


class SwiGLU(nn.Module):
    """Position-wise SwiGLU feed-forward network."""

    d_model: int
    d_ff: int
    w1: Linear
    w2: Linear
    w3: Linear

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._default_d_ff(d_model)

        # w1 和 w3 负责上投影到 hidden 维度，w2 负责投影回 d_model。
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    @staticmethod
    def _default_d_ff(d_model: int) -> int:
        # 作业要求约为 8 / 3 * d_model，并取为 64 的倍数；向上取整避免 hidden 维度偏小。
        return 64 * math.ceil((8 * d_model / 3) / 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(a) = a * sigmoid(a)。这里显式使用 sigmoid，符合题目允许的稳定实现方式。
        gate = self.w1(x)
        silu_gate = gate * torch.sigmoid(gate)
        return self.w2(silu_gate * self.w3(x))
