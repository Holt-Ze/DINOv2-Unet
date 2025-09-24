from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params_m(model: torch.nn.Module) -> float:
    """参数量（百万级）"""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def estimate_gflops(model: torch.nn.Module, img_size: int, device: str = "cuda") -> float | None:
    """用 thop 估算FLOPs（MACs×2），失败则返回None"""
    try:
        from thop import profile

        model_ = model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        macs, _ = profile(model_, inputs=(dummy,), verbose=False)
        flops = macs * 2
        return flops / 1e9
    except Exception:
        return None
