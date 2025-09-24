from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.eps
        dice = (num / den).mean()
        return 1 - dice


class BCEWithLogitsLoss2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class ComboLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = BCEWithLogitsLoss2D()
        self.dice = DiceLoss()
        self.wb, self.wd = bce_weight, dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)
