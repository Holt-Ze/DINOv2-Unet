#!/usr/bin/env python
# coding: utf-8
"""DINOv2-UNet training entry point."""

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from dinov2_unet import (
    ComboLoss,
    DinoV2UNet,
    KvasirSEG,
    TrainConfig,
    build_scheduler,
    count_params_m,
    estimate_gflops,
    evaluate,
    make_optimizers,
    save_predictions_to_dir,
    set_seed,
    train_one_epoch,
)


# ====== 环境变量：抑制无关警告 ======
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--freeze_blocks_until", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./runs_dinov2_unet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_dir", type=str, default=None, help="可选的 train/val/test 列表目录")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_ds = KvasirSEG(args.data_dir, "train", args.img_size, mean, std, split_dir=args.split_dir)
    val_ds = KvasirSEG(args.data_dir, "val", args.img_size, mean, std, split_dir=args.split_dir)
    test_ds = KvasirSEG(args.data_dir, "test", args.img_size, mean, std, split_dir=args.split_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cfg = TrainConfig(
        **{k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__}
    )
    model = DinoV2UNet(
        backbone=cfg.backbone,
        out_indices=cfg.out_indices,
        pretrained=True,
        freeze_blocks_until=cfg.freeze_blocks_until,
        num_classes=1,
    ).to(device)

    optim = make_optimizers(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(not cfg.no_amp))
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optim,
        epochs=cfg.epochs,
        steps_per_epoch=iters_per_epoch,
        warmup_epochs=cfg.warmup_epochs,
    )

    params_m = count_params_m(model)
    gflops = estimate_gflops(model, cfg.img_size, device=device)
    if gflops is None:
        print("[Info] GFLOPs 估算未启用（未安装 thop 或计算失败）。")
    else:
        print(
            f"[Info] 模型复杂度：Params={params_m:.2f}M, FLOPs≈{gflops:.2f}G @ {cfg.img_size}x{cfg.img_size}"
        )

    loss_fn = ComboLoss(0.5, 0.5)
    best_val = -1.0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou, tr_acc = train_one_epoch(
            model, train_loader, optim, scheduler, scaler, loss_fn, device, epoch
        )
        va_loss, va_dice, va_iou, va_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        extra = f" | params {params_m:.2f}M"
        if gflops is not None:
            extra += f" flops {gflops:.2f}G"

        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | time {dt:.1f}s | "
            f"train: loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} acc {tr_acc:.4f} | "
            f"val: loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} acc {va_acc:.4f}"
            f"{extra}"
        )

        score = (va_dice + va_iou) / 2
        if score > best_val:
            best_val = score
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1},
                os.path.join(cfg.save_dir, "best.pt"),
            )

        if (epoch + 1) % 10 == 0:
            save_predictions_to_dir(
                model,
                val_loader,
                device,
                os.path.join(cfg.save_dir, f"vis_ep{epoch + 1:03d}"),
                mean,
                std,
                max_batches=2,
            )

    ckpt = torch.load(os.path.join(cfg.save_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    te_loss, te_dice, te_iou, te_acc = evaluate(model, test_loader, device)
    print(
        f"Test | loss {te_loss:.4f} dice {te_dice:.4f} iou {te_iou:.4f} acc {te_acc:.4f}"
    )
    save_predictions_to_dir(
        model,
        test_loader,
        device,
        os.path.join(cfg.save_dir, "vis_test"),
        mean,
        std,
        max_batches=4,
    )


if __name__ == "__main__":
    main()
