from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torchvision.utils import save_image


def denorm(x: torch.Tensor, mean: Iterable[float], std: Iterable[float]):
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return x * std_t + mean_t


@torch.no_grad()
def save_visuals(
    model,
    loader,
    device: str,
    save_dir: str,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    max_batches: int = 2,
    threshold: float = 0.5,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    cnt = 0
    for imgs, _, names in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        imgs_denorm = denorm(imgs, mean, std)
        for b in range(imgs.size(0)):
            save_image(imgs_denorm[b], os.path.join(save_dir, f"{names[b]}_img.png"))
            save_image(preds[b], os.path.join(save_dir, f"{names[b]}_pred.png"))
        cnt += 1
        if cnt >= max_batches:
            break
