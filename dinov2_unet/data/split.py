from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from .dataset import _list_image_files


def create_dataset_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> None:
    """将数据集拆分为 train/val/test 三个 txt 文件，记录图像文件名。"""

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 必须小于 1.0")

    rng = random.Random(seed)
    img_dir = Path(data_dir) / "images"
    names = _list_image_files(img_dir, extensions)
    if not names:
        raise RuntimeError(f"未在 {img_dir} 下找到图像文件")

    rng.shuffle(names)
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_names = names[:n_train]
    val_names = names[n_train : n_train + n_val]
    test_names = names[n_train + n_val :]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_names in zip(("train", "val", "test"), (train_names, val_names, test_names)):
        with (out_dir / f"{split}.txt").open("w", encoding="utf-8") as f:
            for name in split_names:
                f.write(f"{name}\n")
