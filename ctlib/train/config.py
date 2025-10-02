from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainCfg:
    lesions_csv: Path
    out_dir: Path
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    device: str = "cuda"  # "cuda" или "cpu"
    num_workers: int = 2
    heat_loss_w: float = 1.0
    cls_loss_w: float = 1.0
