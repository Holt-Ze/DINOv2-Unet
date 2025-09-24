from __future__ import annotations

import math

import torch


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 0):
    total_iters = max(1, epochs * steps_per_epoch)
    warmup_iters = max(0, warmup_epochs * steps_per_epoch)

    def lr_lambda(current_iter: int):
        if warmup_iters > 0 and current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = (current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: lr_lambda(it))
