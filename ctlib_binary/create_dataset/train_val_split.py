from pathlib import Path
import csv, random
from typing import Tuple, List
from ctlib_binary.create_dataset import path as P

def run_split(val_split: float = None, seed: int = None) -> Tuple[List[str], List[str]]:
    if val_split is None:
        val_split = P.VAL_SPLIT
    if seed is None:
        seed = P.SEED

    # читаем единый реестр
    keys: List[str] = []
    with P.MERGED_REGISTRY_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            keys.append(row["study_key"])

    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n_val = max(1, int(len(keys) * val_split))
    val_keys = set(keys[:n_val])
    train_keys = [k for k in keys if k not in val_keys]

    with P.TRAIN_LIST.open("w", encoding="utf-8") as f:
        for k in train_keys:
            f.write(k + "\n")
    with P.VAL_LIST.open("w", encoding="utf-8") as f:
        for k in sorted(val_keys):
            f.write(k + "\n")

    return train_keys, list(val_keys)

if __name__ == "__main__":
    tr, va = run_split()
    print(f"train: {len(tr)}, val: {len(va)}")
