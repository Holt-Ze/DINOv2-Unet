from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Tuple

import torch
from torchvision.utils import save_image

from ..utils.visualization import denorm


@torch.no_grad()
def predict(model, loader, device: str, threshold: float = 0.5) -> Iterator[Tuple[Tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor]]:
    """逐批预测，返回 (文件名, 输入图像, 概率, 二值预测)。"""

    model.eval()
    for imgs, _, names in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        yield tuple(names), imgs, probs, preds


@torch.no_grad()
def save_predictions_to_dir(
    model,
    loader,
    device: str,
    save_dir: str,
    mean: Iterable[float],
    std: Iterable[float],
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (names, imgs, _, preds) in enumerate(predict(model, loader, device, threshold=threshold)):
        imgs_denorm = denorm(imgs, mean, std)
        for b, name in enumerate(names):
            save_image(imgs_denorm[b], out_dir / f"{name}_img.png")
            save_image(preds[b], out_dir / f"{name}_pred.png")
        if max_batches is not None and idx + 1 >= max_batches:
            break
