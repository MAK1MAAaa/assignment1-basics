from __future__ import annotations

import math
from collections.abc import Iterable

import torch


@torch.no_grad()
def clip_gradient_norm_(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """原地裁剪所有参数梯度组成的全局 L2 norm。"""

    if max_l2_norm < 0.0:
        raise ValueError("max_l2_norm must not be negative.")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    # 使用 float32 累积平方和，减少低精度梯度在 L2 norm 计算中的数值误差。
    total_squared_norm = sum(
        torch.sum(gradient.detach().to(dtype=torch.float32).square()).item() for gradient in gradients
    )
    total_l2_norm = math.sqrt(total_squared_norm)

    # 与 PyTorch 的默认实现一致：epsilon 防止零范数时除零，且不会放大原本较小的梯度。
    clip_coefficient = min(1.0, max_l2_norm / (total_l2_norm + 1e-6))
    if clip_coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)
