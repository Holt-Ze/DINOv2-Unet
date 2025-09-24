from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    data_dir: str
    img_size: int = 448
    batch_size: int = 8
    epochs: int = 80
    backbone: str = "vit_base_patch14_dinov2"
    out_indices: Tuple[int, ...] = (2, 5, 8, 11)
    lr: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    num_workers: int = 4
    no_amp: bool = False
    freeze_blocks_until: int = 9
    save_dir: str = "./runs_dinov2_unet"
    seed: int = 42
    split_dir: Optional[str] = None
