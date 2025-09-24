from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as exc:  # pragma: no cover - informative message
    raise RuntimeError("需要安装 timm: pip install timm") from exc


class VitDinoV2Encoder(nn.Module):
    """将 DINOv2 ViT 封装为语义分割编码器。"""

    def __init__(
        self,
        backbone: str = "vit_base_patch14_dinov2",
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_blocks_until: int = 9,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, dynamic_img_size=True)
        assert hasattr(self.model, "blocks"), "timm ViT 模型需要包含 blocks 属性"
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "dynamic_img_size"):
            self.model.patch_embed.dynamic_img_size = True
        self.out_indices = out_indices
        self.patch_size = 14
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_size"):
            ps = self.model.patch_embed.patch_size
            self.patch_size = ps[0] if isinstance(ps, tuple) else int(ps)
        self.embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features")
        for i, blk in enumerate(self.model.blocks):
            requires = i >= freeze_blocks_until
            for p in blk.parameters():
                p.requires_grad = requires
        self.projs = nn.ModuleList([nn.Conv2d(self.embed_dim, 256, kernel_size=1) for _ in out_indices])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int], int]:
        b, _, h, w = x.shape
        x_pe = self.model.patch_embed(x)
        if hasattr(self.model, "_pos_embed"):
            pe_out = self.model._pos_embed(x_pe)
            if isinstance(pe_out, (list, tuple)):
                x_tokens, (gh, gw) = pe_out[0], pe_out[1]
            else:
                x_tokens = pe_out
                gh, gw = h // self.patch_size, w // self.patch_size
        else:
            if x_pe.dim() == 4:
                _, _, gh, gw = x_pe.shape
                x_tokens = x_pe.flatten(2).transpose(1, 2)
            else:
                gh, gw = h // self.patch_size, w // self.patch_size
                x_tokens = x_pe
            cls_token = getattr(self.model, "cls_token", None)
            if cls_token is not None:
                cls_tokens = cls_token.expand(b, -1, -1)
                x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
            if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
                pos_embed = self.model.pos_embed
                if pos_embed.shape[1] != x_tokens.shape[1]:
                    has_cls = pos_embed.shape[1] == gh * gw + 1
                    if has_cls:
                        cls_pe, patch_pe = pos_embed[:, :1], pos_embed[:, 1:]
                    else:
                        cls_pe, patch_pe = None, pos_embed
                    s = int(round(math.sqrt(patch_pe.shape[1])))
                    patch_pe = patch_pe.transpose(1, 2).reshape(1, patch_pe.shape[2], s, s)
                    patch_pe = F.interpolate(patch_pe, size=(gh, gw), mode="bicubic", align_corners=False)
                    patch_pe = patch_pe.flatten(2).transpose(1, 2)
                    pos_embed = torch.cat([cls_pe, patch_pe], dim=1) if cls_pe is not None else patch_pe
                x_tokens = x_tokens + pos_embed

        x_tokens = self.model.pos_drop(x_tokens)

        feats_tokens = []
        for i, blk in enumerate(self.model.blocks):
            x_tokens = blk(x_tokens)
            if i in self.out_indices:
                feats_tokens.append(x_tokens)
        x_tokens = self.model.norm(x_tokens)

        feats = []
        for tok, proj in zip(feats_tokens, self.projs):
            if tok.shape[1] == (gh * gw + 1):
                tok = tok[:, 1:, :]
            fm = tok.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
            fm = proj(fm)
            feats.append(fm)
        return feats, (gh, gw), self.patch_size


class FPNUNetDecoder(nn.Module):
    def __init__(self, num_in: int, out_ch: int = 1) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([self._conv_bn_relu(256, 256) for _ in range(num_in)])
        self.smooth = nn.ModuleList([self._conv_bn_relu(256, 256) for _ in range(num_in - 1)])
        self.head = nn.Sequential(
            self._conv_bn_relu(256, 128),
            nn.Conv2d(128, out_ch, kernel_size=1),
        )

    @staticmethod
    def _conv_bn_relu(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        feat_list: List[torch.Tensor],
        grid_hw: Tuple[int, int],
        patch_size: int,
        out_size_hw: Tuple[int, int],
    ):
        del grid_hw, patch_size  # 未使用，但保留接口兼容性
        lat = [l(f) for l, f in zip(self.lateral, feat_list)]
        p = lat[-1]
        for i in range(len(lat) - 2, -1, -1):
            p = F.interpolate(p, size=lat[i].shape[-2:], mode="bilinear", align_corners=False) + lat[i]
            p = self.smooth[i](p) if i < len(self.smooth) else p
        logits = self.head(p)
        logits = F.interpolate(logits, size=out_size_hw, mode="bilinear", align_corners=False)
        return logits


class DinoV2UNet(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_base_patch14_dinov2",
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_blocks_until: int = 9,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = VitDinoV2Encoder(backbone, out_indices, pretrained, freeze_blocks_until)
        self.decoder = FPNUNetDecoder(num_in=len(out_indices), out_ch=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, (gh, gw), patch = self.encoder(x)
        logits = self.decoder(feats, (gh, gw), patch, out_size_hw=(x.shape[2], x.shape[3]))
        return logits
