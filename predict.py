#!/usr/bin/env python
# coding: utf-8
"""模型推理脚本。"""

import argparse

import torch
from torch.utils.data import DataLoader

from dinov2_unet import DinoV2UNet, KvasirSEG, TrainConfig, save_predictions_to_dir, set_seed


def _parse_out_indices(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("out_indices 不能为空")
    return tuple(int(p) for p in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 DINOv2-UNet 进行推理")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="保存预测掩码的目录")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--out_indices", type=_parse_out_indices, default=(2, 5, 8, 11))
    parser.add_argument("--freeze_blocks_until", type=int, default=9)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="可选，只保存前若干个批次")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        backbone=args.backbone,
        out_indices=args.out_indices,
        freeze_blocks_until=args.freeze_blocks_until,
        num_workers=args.num_workers,
        seed=args.seed,
        split_dir=args.split_dir,
    )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    dataset = KvasirSEG(cfg.data_dir, args.split, cfg.img_size, mean, std, split_dir=cfg.split_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = DinoV2UNet(
        backbone=cfg.backbone,
        out_indices=cfg.out_indices,
        pretrained=False,
        freeze_blocks_until=cfg.freeze_blocks_until,
        num_classes=1,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)

    save_predictions_to_dir(
        model,
        loader,
        device,
        args.output,
        mean,
        std,
        max_batches=args.max_batches,
    )
    print(f"预测结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
