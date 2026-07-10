from __future__ import annotations

import torch
from torch import nn

from cs336_basics.Part3.multihead_self_attention_with_rope import MultiHeadSelfAttentionWithRoPE
from cs336_basics.Part3.positionwise_feedforward import SwiGLU
from cs336_basics.Part3.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """使用 RoPE、RMSNorm 与 SwiGLU 的 pre-norm Transformer block。"""

    d_model: int
    num_heads: int
    d_ff: int
    attn: MultiHeadSelfAttentionWithRoPE
    ffn: SwiGLU
    ln1: RMSNorm
    ln2: RMSNorm

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if d_ff <= 0:
            raise ValueError("d_ff must be greater than zero.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # attention 内部负责 Q/K 的 RoPE、因果 mask 和多头的合并投影。
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """对 ``(..., sequence_length, d_model)`` 的输入执行一个 pre-norm block。"""

        if x.ndim < 2:
            raise ValueError("x must include sequence and feature dimensions.")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.shape[-1]}.")

        # 每个子层均先归一化再计算，残差路径始终保留未经归一化的 block 表示。
        attention_output = self.attn(self.ln1(x), token_positions)
        x = x + attention_output

        feed_forward_output = self.ffn(self.ln2(x))
        return x + feed_forward_output
