#!/usr/bin/env python
# coding: utf-8
"""Kvasir-SEG 数据集划分脚本。"""

import argparse

from dinov2_unet import create_dataset_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按比例划分 Kvasir-SEG 数据集")
    parser.add_argument("data_dir", type=str, help="包含 images/masks 的数据集根目录")
    parser.add_argument("output", type=str, help="保存 train/val/test txt 的目录")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dataset_splits(
        data_dir=args.data_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"已在 {args.output} 生成 train/val/test 列表。")


if __name__ == "__main__":
    main()
