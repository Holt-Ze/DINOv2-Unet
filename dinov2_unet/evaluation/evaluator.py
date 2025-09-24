from __future__ import annotations

import torch

from ..training.losses import ComboLoss
from .metrics import dice_iou_from_logits, pixel_acc_from_logits


@torch.no_grad()
def evaluate(model, loader, device: str, loss_fn: ComboLoss | None = None):
    model.eval()
    loss_fn = loss_fn or ComboLoss(0.5, 0.5)
    total_loss, total_dice, total_iou, total_acc = 0.0, 0.0, 0.0, 0.0
    iters = 0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, msks)
        d, i = dice_iou_from_logits(logits, msks)
        a = pixel_acc_from_logits(logits, msks)
        total_loss += loss.item()
        total_dice += d
        total_iou += i
        total_acc += a
        iters += 1
    return total_loss / iters, total_dice / iters, total_iou / iters, total_acc / iters
