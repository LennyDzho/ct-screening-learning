from pathlib import Path
import csv
from typing import Dict, List, Tuple
import random

def group_split_indices(
    csv_path: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int], Dict[str, int], Dict[str, int]]:
    """
    Делит данные по исследованиям (study_key), чтобы одно исследование не попало в два сплита.
    Возвращает:
      - train_indices: индексы строк CSV для трейна
      - val_indices: индексы строк CSV для валидации
      - study_counts_train: счетчик строк на исследование в трейне
      - study_counts_val:   счетчик строк на исследование в вале
    """
    rows: List[Dict[str, str]] = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    # сгруппируем построчные индексы по study_key
    by_study: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        sk = r.get("study_key", "")
        by_study.setdefault(sk, []).append(i)

    studies = list(by_study.keys())
    rng = random.Random(seed)
    rng.shuffle(studies)

    n_val_studies = max(1, int(round(len(studies) * val_ratio)))
    val_studies = set(studies[:n_val_studies])
    train_studies = set(studies[n_val_studies:])

    tr_idx: List[int] = []
    va_idx: List[int] = []
    study_counts_train: Dict[str, int] = {}
    study_counts_val: Dict[str, int] = {}

    for sk, idxs in by_study.items():
        if sk in val_studies:
            va_idx.extend(idxs)
            study_counts_val[sk] = len(idxs)
        else:
            tr_idx.extend(idxs)
            study_counts_train[sk] = len(idxs)

    return tr_idx, va_idx, study_counts_train, study_counts_val
