from __future__ import annotations
from __future__ import annotations

from typing import Dict

import torch

from ..config import TrainConfig
from ..models import DinoV2UNet


def make_optimizers(model: DinoV2UNet, cfg: TrainConfig):
    enc_params, dec_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            enc_params.append(param)
        else:
            dec_params.append(param)
    optim = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_backbone, "name": "enc"},
            {"params": dec_params, "lr": cfg.lr, "name": "dec"},
        ],
        weight_decay=cfg.weight_decay,
    )
    return optim


def current_lrs(optimizer) -> Dict[str, float]:
    return {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)}
