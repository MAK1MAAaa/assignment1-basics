from __future__ import annotations

import torch
from torch import nn

from cs336_basics.Part3.embedding import Embedding
from cs336_basics.Part3.linear import Linear
from cs336_basics.Part3.rmsnorm import RMSNorm
from cs336_basics.Part3.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """由 token embedding、多个 Transformer block 与语言模型头组成的因果语言模型。"""

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    token_embeddings: Embedding
    layers: nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero.")
        if num_layers < 0:
            raise ValueError("num_layers must not be negative.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # token embedding 不叠加绝对位置编码；位置信息由每个 block 中的 RoPE 提供。
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # 输出头生成未归一化 logits；训练时由交叉熵损失函数负责 softmax。
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回每个输入 token 位置上的未归一化词表 logits。"""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length).")
        if token_ids.shape[-1] > self.context_length:
            raise ValueError(f"sequence length must not exceed context_length ({self.context_length}).")

        hidden_states = self.token_embeddings(token_ids)
        # 未传入 token_positions 时，block 使用 [0, ..., sequence_length - 1] 的连续位置。
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.lm_head(self.ln_final(hidden_states))
