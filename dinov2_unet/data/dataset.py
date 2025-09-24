from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _list_image_files(img_dir: Path, extensions: Sequence[str]) -> List[str]:
    names = [f for f in os.listdir(img_dir) if img_dir.joinpath(f).suffix.lower() in extensions]
    names.sort()
    return names


def _load_split_file(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


class KvasirSEG(Dataset):
    """Kvasir-SEG 数据集加载与增强逻辑。"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 448,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        split_dir: Optional[str] = None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        super().__init__()
        self.img_dir = Path(data_dir) / "images"
        self.msk_dir = Path(data_dir) / "masks"
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val' or 'test'")

        if split_dir is not None:
            split_path = Path(split_dir)
            names = _load_split_file(split_path / f"{split}.txt")
        else:
            names = _list_image_files(self.img_dir, extensions)
            n = len(names)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            if split == "train":
                names = names[:n_train]
            elif split == "val":
                names = names[n_train : n_train + n_val]
            else:
                names = names[n_train + n_val :]

        self.names = names
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.names)

    def _load_pair(self, name: str) -> Tuple[Image.Image, Image.Image]:
        img_path = self.img_dir / name
        stem = Path(name).stem
        msk_path_png = self.msk_dir / f"{stem}.png"
        msk_path_jpg = self.msk_dir / f"{stem}.jpg"
        msk_path = msk_path_png if msk_path_png.exists() else msk_path_jpg
        if not img_path.exists() or not msk_path.exists():
            raise FileNotFoundError(f"Missing image or mask for {name}")
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")
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
            img = TF.adjust_brightness(img, 0.9 + 0.2 * np.random.rand())
            img = TF.adjust_contrast(img, 0.9 + 0.2 * np.random.rand())
            img = TF.adjust_saturation(img, 0.9 + 0.2 * np.random.rand())
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

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img, msk = self._load_pair(name)
        if self.split == "train":
            img, msk = self._train_transform(img, msk)
        else:
            img, msk = self._val_transform(img, msk)
        return img, msk, name
