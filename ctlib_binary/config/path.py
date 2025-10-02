from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_OUT_DIR = PROJECT_ROOT / "prepared_data" / "prepared"
MERGED_REGISTRY_CSV = DATA_OUT_DIR / "studies.csv"

SPLITS_DIR = DATA_OUT_DIR / "splits"
TRAIN_LIST = SPLITS_DIR / "train_studies.txt"
VAL_LIST   = SPLITS_DIR / "val_studies.txt"

RUNS_DIR = PROJECT_ROOT / "runs_binary"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42

TARGET_SPACING = (1.5, 1.5, 1.5)   # (z,y,x) мм
TARGET_SHAPE   = (128, 224, 224)   # (D,H,W)
HU_WINDOW      = (-1000.0, 400.0)  # клиппинг по HU
