from __future__ import annotations

import torch


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    dice = (2 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def pixel_acc_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    """逐像素准确率：二值阈值后与GT一致的像素占比"""
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return (correct.float() / total).item()
