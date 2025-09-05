#!/usr/bin/env python
# coding: utf-8
"""
DINOv2-UNet for Kvasir-SEG (Polyp Segmentation)
------------------------------------------------
- timm 加载 DINOv2 ViT 作为编码器
- FPN/UNet 风格解码器（跨层融合 + 逐步上采样）
- 训练、验证、Dice/IoU/PixelAcc 指标，可视化，最优权重保存
- 学习率：全组共享 warmup+cosine 调度因子（保留 enc/dec 不同基准 LR）
- 训练前统计 Params(M) 与 FLOPs(G)（需 thop；未安装则优雅降级）

用法示例（PowerShell）:
python train.py --data_dir D:\DINOv2\kvasir-seg --img_size 448 --batch_size 8 --epochs 80 --backbone vit_base_patch14_dinov2 --lr 1e-3 --lr_backbone 1e-5
"""

# ====== 环境变量：抑制无关警告 ======
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

try:
    import timm
except Exception as e:
    raise RuntimeError("需要安装 timm: pip install timm")

# =============================
# Utils
# =============================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def denorm(x: torch.Tensor, mean, std):
    mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return x * std + mean

# =============================
# Dataset
# =============================
class KvasirSEG(Dataset):
    """Kvasir-SEG 数据集
    data_dir/
      ├─ images/  *.jpg|*.png
      └─ masks/   *.png (0/255 或 0/1)
    """
    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 448,
                 mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        super().__init__()
        self.img_dir = os.path.join(data_dir, 'images')
        self.msk_dir = os.path.join(data_dir, 'masks')
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        names = [f for f in os.listdir(self.img_dir) if os.path.splitext(f)[1].lower() in exts]
        names.sort()
        # 简单划分：8/1/1
        n = len(names)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if split == 'train':
            self.names = names[:n_train]
        elif split == 'val':
            self.names = names[n_train:n_train+n_val]
        else:
            self.names = names[n_train+n_val:]
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.names)

    def _load_pair(self, name):
        img_path = os.path.join(self.img_dir, name)
        stem = os.path.splitext(name)[0]
        msk_path_png = os.path.join(self.msk_dir, stem + '.png')
        msk_path = msk_path_png if os.path.exists(msk_path_png) else os.path.join(self.msk_dir, stem + '.jpg')
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        return img, msk

    def _train_transform(self, img: Image.Image, msk: Image.Image):
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        angle = float(np.random.uniform(-10, 10))
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.rotate(msk, angle, interpolation=TF.InterpolationMode.NEAREST)
        if np.random.rand() < 0.8:
            img = TF.adjust_brightness(img, 0.9 + 0.2*np.random.rand())
            img = TF.adjust_contrast(img, 0.9 + 0.2*np.random.rand())
            img = TF.adjust_saturation(img, 0.9 + 0.2*np.random.rand())
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        msk = TF.to_tensor(msk)
        msk = (msk > 0.5).float()
        return img, msk

    def _val_transform(self, img: Image.Image, msk: Image.Image):
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        msk = TF.to_tensor(msk)
        msk = (msk > 0.5).float()
        return img, msk

    def __getitem__(self, idx):
        name = self.names[idx]
        img, msk = self._load_pair(name)
        if self.split == 'train':
            img, msk = self._train_transform(img, msk)
        else:
            img, msk = self._val_transform(img, msk)
        return img, msk, name

# =============================
# Losses & Metrics
# =============================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2,3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2,3)) + self.eps
        dice = (num / den).mean()
        return 1 - dice

class BCEWithLogitsLoss2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        return self.loss(logits, targets)

class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = BCEWithLogitsLoss2D()
        self.dice = DiceLoss()
        self.wb, self.wd = bce_weight, dice_weight
    def forward(self, logits, targets):
        return self.wb*self.bce(logits, targets) + self.wd*self.dice(logits, targets)

@torch.no_grad()
def dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = (preds + targets - preds*targets).sum(dim=(2,3))
    dice = (2*inter + eps) / (preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

@torch.no_grad()
def pixel_acc_from_logits(logits, targets, thr=0.5):
    """逐像素准确率：二值阈值后与GT一致的像素占比"""
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return (correct.float() / total).item()

def count_params_m(model: nn.Module) -> float:
    """参数量（百万级）"""
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.no_grad()
def estimate_gflops(model: nn.Module, img_size: int, device: str = "cuda") -> float | None:
    """用 thop 估算FLOPs（MACs×2），失败则返回None"""
    try:
        from thop import profile
        model_ = model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        macs, _ = profile(model_, inputs=(dummy,), verbose=False)
        flops = macs * 2  # 约定：FLOPs ≈ 2*MACs
        return flops / 1e9  # 转为 GFLOPs
    except Exception:
        return None

# =============================
# DINOv2 ViT Encoder (timm)
# =============================
class VitDinoV2Encoder(nn.Module):
    """将 DINOv2 ViT 封装为语义分割编码器。"""
    def __init__(self, backbone='vit_base_patch14_dinov2', out_indices=(2,5,8,11),
                 pretrained=True, freeze_blocks_until=9):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, dynamic_img_size=True)
        assert hasattr(self.model, 'blocks'), 'timm ViT 模型需要包含 blocks 属性'
        if hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed, 'dynamic_img_size'):
            self.model.patch_embed.dynamic_img_size = True
        self.out_indices = out_indices
        # patch size & embed dim
        self.patch_size = 14
        if hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed, 'patch_size'):
            ps = self.model.patch_embed.patch_size
            self.patch_size = ps[0] if isinstance(ps, tuple) else int(ps)
        self.embed_dim = getattr(self.model, 'embed_dim', None) or getattr(self.model, 'num_features')
        # 冻结前半部分 blocks（小数据集上更稳）
        for i, blk in enumerate(self.model.blocks):
            requires = (i >= freeze_blocks_until)
            for p in blk.parameters():
                p.requires_grad = requires
        # 选取层的 1x1 卷积投影到 256 通道
        self.projs = nn.ModuleList([nn.Conv2d(self.embed_dim, 256, kernel_size=1) for _ in out_indices])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int], int]:
        B, C, H, W = x.shape

        # 使用 timm 的内部 _pos_embed 来统一处理 patch_embed / cls / 位置编码（自动插值）
        x_pe = self.model.patch_embed(x)
        if hasattr(self.model, '_pos_embed'):
            pe_out = self.model._pos_embed(x_pe)
            if isinstance(pe_out, (list, tuple)):
                x_tokens, (Gh, Gw) = pe_out[0], pe_out[1]  # (B, 1+N, C), (Gh, Gw)
            else:
                x_tokens = pe_out
                Gh, Gw = H // self.patch_size, W // self.patch_size
        else:
            # 极少模型没有 _pos_embed：回退到手动适配 3D/4D + 插值 pos_embed
            if x_pe.dim() == 4:
                B_, C_, Gh, Gw = x_pe.shape
                x_tokens = x_pe.flatten(2).transpose(1, 2)  # (B, N, C_)
            else:
                Gh, Gw = H // self.patch_size, W // self.patch_size
                x_tokens = x_pe  # (B, N, C)
            cls_token = getattr(self.model, 'cls_token', None)
            if cls_token is not None:
                cls_tokens = cls_token.expand(B, -1, -1)
                x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
            if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
                pos_embed = self.model.pos_embed
                if pos_embed.shape[1] != x_tokens.shape[1]:
                    has_cls = (pos_embed.shape[1] == Gh*Gw + 1)
                    if has_cls:
                        cls_pe, patch_pe = pos_embed[:, :1], pos_embed[:, 1:]
                    else:
                        cls_pe, patch_pe = None, pos_embed
                    s = int(round(math.sqrt(patch_pe.shape[1])))
                    patch_pe = patch_pe.transpose(1,2).reshape(1, patch_pe.shape[2], s, s)
                    patch_pe = F.interpolate(patch_pe, size=(Gh, Gw), mode='bicubic', align_corners=False)
                    patch_pe = patch_pe.flatten(2).transpose(1,2)
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
            if tok.shape[1] == (Gh * Gw + 1):
                tok = tok[:, 1:, :]
            fm = tok.transpose(1, 2).reshape(B, self.embed_dim, Gh, Gw)
            fm = proj(fm)
            feats.append(fm)
        return feats, (Gh, Gw), self.patch_size

# =============================
# UNet/FPN 风格 Decoder
# =============================
class FPNUNetDecoder(nn.Module):
    def __init__(self, num_in: int, out_ch: int = 1):
        super().__init__()
        self.lateral = nn.ModuleList([self._conv_bn_relu(256, 256) for _ in range(num_in)])
        self.smooth = nn.ModuleList([self._conv_bn_relu(256, 256) for _ in range(num_in-1)])
        self.head = nn.Sequential(
            self._conv_bn_relu(256, 128),
            nn.Conv2d(128, out_ch, kernel_size=1)
        )

    @staticmethod
    def _conv_bn_relu(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_list: List[torch.Tensor], grid_hw: Tuple[int,int], patch_size: int, out_size_hw: Tuple[int,int]):
        lat = [l(f) for l, f in zip(self.lateral, feat_list)]
        p = lat[-1]
        for i in range(len(lat)-2, -1, -1):
            p = F.interpolate(p, size=lat[i].shape[-2:], mode='bilinear', align_corners=False) + lat[i]
            p = self.smooth[i](p) if i < len(self.smooth) else p
        logits = self.head(p)
        logits = F.interpolate(logits, size=out_size_hw, mode='bilinear', align_corners=False)
        return logits

# =============================
# 全模型封装
# =============================
class DinoV2UNet(nn.Module):
    def __init__(self, backbone='vit_base_patch14_dinov2', out_indices=(2,5,8,11),
                 pretrained=True, freeze_blocks_until=9, num_classes=1):
        super().__init__()
        self.encoder = VitDinoV2Encoder(backbone, out_indices, pretrained, freeze_blocks_until)
        self.decoder = FPNUNetDecoder(num_in=len(out_indices), out_ch=num_classes)

    def forward(self, x):
        feats, (Gh, Gw), p = self.encoder(x)
        logits = self.decoder(feats, (Gh, Gw), p, out_size_hw=(x.shape[2], x.shape[3]))
        return logits

# =============================
# 训练工具
# =============================
@dataclass
class TrainConfig:
    data_dir: str
    img_size: int = 448
    batch_size: int = 8
    epochs: int = 80
    backbone: str = 'vit_base_patch14_dinov2'
    out_indices: Tuple[int,...] = (2,5,8,11)
    lr: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    num_workers: int = 4
    no_amp: bool = False
    freeze_blocks_until: int = 9
    save_dir: str = './runs_dinov2_unet'
    seed: int = 42

def make_optimizers(model: DinoV2UNet, cfg: TrainConfig):
    enc_params, dec_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('encoder'):
            enc_params.append(p)
        else:
            dec_params.append(p)
    optim = torch.optim.AdamW([
        {'params': enc_params, 'lr': cfg.lr_backbone, 'name': 'enc'},
        {'params': dec_params, 'lr': cfg.lr,          'name': 'dec'},
    ], weight_decay=cfg.weight_decay)
    return optim

def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0):
    total_iters = max(1, epochs * steps_per_epoch)
    warmup_iters = max(0, warmup_epochs * steps_per_epoch)

    def lr_lambda(current_iter):
        if warmup_iters > 0 and current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = (current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: lr_lambda(it))

def _current_lrs(optim):
    return { (g.get('name', str(i))): g['lr'] for i, g in enumerate(optim.param_groups) }

def train_one_epoch(model, loader, optim, scheduler, scaler, loss_fn, device, epoch):
    model.train()
    running_loss = 0.0
    running_dice, running_iou, running_acc = 0.0, 0.0, 0.0
    iters = 0
    for it, (imgs, msks, _) in enumerate(loader):
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if not scaler is None:
            with torch.amp.autocast('cuda' if device.startswith('cuda') else 'cpu'):
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
        scheduler.step()  # 每步调度
        with torch.no_grad():
            d, i = dice_iou_from_logits(logits, msks)
            a = pixel_acc_from_logits(logits, msks)
        running_loss += loss.item()
        running_dice += d
        running_iou += i
        running_acc += a
        iters += 1
        if it == 0 and epoch == 0:
            print("LRs@start:", _current_lrs(optim))
    return running_loss/iters, running_dice/iters, running_iou/iters, running_acc/iters

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = ComboLoss(0.5, 0.5)
    total_loss, total_dice, total_iou, total_acc = 0.0, 0.0, 0.0, 0.0
    iters = 0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, msks)
        d, i = dice_iou_from_logits(logits, msks)
        a = pixel_acc_from_logits(logits, msks)
        total_loss += loss.item(); total_dice += d; total_iou += i; total_acc += a
        iters += 1
    return total_loss/iters, total_dice/iters, total_iou/iters, total_acc/iters

@torch.no_grad()
def save_visuals(model, loader, device, save_dir, mean, std, max_batches=2):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    cnt = 0
    for imgs, msks, names in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        imgs_denorm = denorm(imgs, mean, std)
        for b in range(imgs.size(0)):
            save_image(imgs_denorm[b], os.path.join(save_dir, f"{names[b]}_img.png"))
            save_image(preds[b], os.path.join(save_dir, f"{names[b]}_pred.png"))
        cnt += 1
        if cnt >= max_batches:
            break

# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2')
    parser.add_argument('--freeze_blocks_until', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./runs_dinov2_unet')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Datasets
    train_ds = KvasirSEG(args.data_dir, 'train', args.img_size, mean, std)
    val_ds   = KvasirSEG(args.data_dir, 'val', args.img_size, mean, std)
    test_ds  = KvasirSEG(args.data_dir, 'test', args.img_size, mean, std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model + Optim + Sched
    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__})
    model = DinoV2UNet(backbone=cfg.backbone,
                       out_indices=(2,5,8,11),
                       pretrained=True,
                       freeze_blocks_until=cfg.freeze_blocks_until,
                       num_classes=1).to(device)

    optim = make_optimizers(model, cfg)
    # AMP：新式 API
    scaler = torch.amp.GradScaler('cuda', enabled=(not cfg.no_amp))
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(optim, epochs=cfg.epochs, steps_per_epoch=iters_per_epoch, warmup_epochs=cfg.warmup_epochs)

    # 复杂度（一次性）
    params_m = count_params_m(model)
    gflops = estimate_gflops(model, cfg.img_size, device=device)
    if gflops is None:
        print(f"[Info] GFLOPs 估算未启用（未安装 thop 或计算失败）。")
    else:
        print(f"[Info] 模型复杂度：Params={params_m:.2f}M, FLOPs≈{gflops:.2f}G @ {cfg.img_size}x{cfg.img_size}")

    loss_fn = ComboLoss(0.5, 0.5)
    best_val = -1.0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou, tr_acc = train_one_epoch(model, train_loader, optim, scheduler, scaler, loss_fn, device, epoch)
        va_loss, va_dice, va_iou, va_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        extra = f" | params {params_m:.2f}M"
        if gflops is not None:
            extra += f" flops {gflops:.2f}G"

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | time {dt:.1f}s | "
            f"train: loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} acc {tr_acc:.4f} | "
            f"val: loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} acc {va_acc:.4f}"
            f"{extra}"
        )

        # 保存 best（基于 (Dice+IoU)/2）
        score = (va_dice + va_iou) / 2
        if score > best_val:
            best_val = score
            torch.save({'model': model.state_dict(), 'epoch': epoch+1}, os.path.join(cfg.save_dir, 'best.pt'))

        if (epoch+1) % 10 == 0:
            save_visuals(model, val_loader, device, os.path.join(cfg.save_dir, f'vis_ep{epoch+1:03d}'), mean, std)

    # Test（载入 best）
    ckpt = torch.load(os.path.join(cfg.save_dir, 'best.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    te_loss, te_dice, te_iou, te_acc = evaluate(model, test_loader, device)
    print(f"Test | loss {te_loss:.4f} dice {te_dice:.4f} iou {te_iou:.4f} acc {te_acc:.4f}")
    save_visuals(model, test_loader, device, os.path.join(cfg.save_dir, 'vis_test'), mean, std, max_batches=4)

if __name__ == '__main__':
    main()
