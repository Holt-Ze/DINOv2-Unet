from __future__ import annotations

import torch

from ..evaluation.metrics import dice_iou_from_logits, pixel_acc_from_logits
from .optim import current_lrs


def train_one_epoch(model, loader, optim, scheduler, scaler, loss_fn, device: str, epoch: int):
    model.train()
    running_loss = 0.0
    running_dice, running_iou, running_acc = 0.0, 0.0, 0.0
    iters = 0
    for it, (imgs, msks, _) in enumerate(loader):
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            with torch.amp.autocast(autocast_device):
                logits = model(imgs)
                loss = loss_fn(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(imgs)
            loss = loss_fn(logits, msks)
            loss.backward()
            optim.step()
        scheduler.step()
        with torch.no_grad():
            d, i = dice_iou_from_logits(logits, msks)
            a = pixel_acc_from_logits(logits, msks)
        running_loss += loss.item()
        running_dice += d
        running_iou += i
        running_acc += a
        iters += 1
        if it == 0 and epoch == 0:
            print("LRs@start:", current_lrs(optim))
    return running_loss / iters, running_dice / iters, running_iou / iters, running_acc / iters
