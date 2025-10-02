from pathlib import Path

# Корень проекта (при желании скорректируй)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
# === Входные датасеты ===
# 1) COVID19_1110 (NIfTI, CT-0..CT-4 + masks)
COVID19_1110_DIR = DATA_ROOT / "COVID19_1110"

# 2) MosMedData-CT-COVID19-type VII (DICOM)
MOSMED_VII_DIR = DATA_ROOT / "MosMedData-CT-COVID19-type VII-v 1"

# 3) MosMedData-LDCT-LUNGCR-type I (DICOM)
LDCT_LUNGCR_DIR = DATA_ROOT / "MosMedData-LDCT-LUNGCR-type I-v 1"

# 4) Кастомный набор (папки с *_anon)
CUSTOM_DIR = DATA_ROOT / "dataset"

# === Выходные файлы ===
DATA_OUT_DIR = PROJECT_ROOT / "prepared_data" / "prepared"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Единый реестр всех исследований
MERGED_REGISTRY_CSV = DATA_OUT_DIR / "studies.csv"

# Тренировочно-валидационные списки (по study_key)
SPLITS_DIR = DATA_OUT_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LIST = SPLITS_DIR / "train_studies.txt"
VAL_LIST = SPLITS_DIR / "val_studies.txt"

# Параметры сплита
VAL_SPLIT = 0.2
SEED = 42
