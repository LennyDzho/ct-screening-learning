from pathlib import Path
from typing import Dict, List, Tuple
import csv
import shutil
import numpy as np
import cv2
import yaml

from ctlib.config.paths import LESIONS_CSV, DATASET_OUT_DIR
from ctlib_yolo.dataset.yolo_helpers import (
    read_dicom_hu, resize_keep_square, bbox_from_point_diam, normalize_nodule_type
)

SPLIT = {"val_ratio": 0.2, "test_ratio": 0.1}

def _group_split_by_study(rows: List[Dict[str,str]], val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    import random
    by_study: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        sk = r.get("study_key", "")
        by_study.setdefault(sk, []).append(i)
    studies = list(by_study.keys())
    rng = random.Random(seed); rng.shuffle(studies)
    n = len(studies)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_s = set(studies[:n_test])
    val_s  = set(studies[n_test:n_test+n_val])
    train_s = set(studies[n_test+n_val:])

    tr_idx: List[int] = []; va_idx: List[int] = []; te_idx: List[int] = []
    for sk, idxs in by_study.items():
        if sk in test_s: te_idx.extend(idxs)
        elif sk in val_s: va_idx.extend(idxs)
        else: tr_idx.extend(idxs)
    return tr_idx, va_idx, te_idx

def _ensure_clean_dir(d: Path):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# --- НОВОЕ: класс из типа узла ---
def _class_from_row(row: Dict[str, str]) -> str | None:
    """
    Пытаемся взять тип из столбцов CSV:
    - 'nodule_type' (как в nodules.csv)
    - иначе 'expert_type'
    Нормализуем к one-of: 'solid' | 'part_solid' | 'ground_glass'.
    """
    t = row.get("nodule_type") or row.get("expert_type")
    t_norm = normalize_nodule_type(t)
    return t_norm

def build_yolo_dataset(
    lesions_csv: Path = LESIONS_CSV,
    out_root: Path = DATASET_OUT_DIR / "dataset_yolo",
    img_size: int = 1024,
    bbox_enlarge: float = 1.25,     # немного расширим бокс
) -> Dict[str, int]:
    # читаем CSV
    rows: List[Dict[str,str]] = []
    with Path(lesions_csv).open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if not r.get("dicom_path"):
                continue
            # нужен тип узла
            cname = _class_from_row(r)
            if cname is None:
                continue
            # координаты должны быть
            if r.get("x_px") in (None, "", "None") or r.get("y_px") in (None, "", "None"):
                continue
            rows.append(r)

    assert rows, "В lesions.csv нет строк с корректным типом узла и координатами."

    # финальный список классов — фиксированный порядок
    classes = ["solid", "part_solid", "ground_glass"]
    class2id = {c:i for i,c in enumerate(classes)}

    # сплит по исследованиям
    tr_idx, va_idx, te_idx = _group_split_by_study(rows, SPLIT["val_ratio"], SPLIT["test_ratio"], seed=42)

    # каталоги
    img_dir = {k: out_root / "images" / k for k in ("train","val","test")}
    lbl_dir = {k: out_root / "labels" / k for k in ("train","val","test")}
    for d in list(img_dir.values()) + list(lbl_dir.values()):
        _ensure_clean_dir(d)

    # цикл
    stats = {c:0 for c in classes}
    skipped = 0

    for split_name, idx_list in (("train", tr_idx), ("val", va_idx), ("test", te_idx)):
        for i in idx_list:
            r = rows[i]
            dcm_path = Path(r["dicom_path"])
            try:
                img, spacing = read_dicom_hu(dcm_path)
            except Exception:
                skipped += 1
                continue

            try:
                x = float(r["x_px"]); y = float(r["y_px"])
            except Exception:
                skipped += 1; continue

            diam_mm = r.get("diameter_mm")
            diam_mm = float(diam_mm) if diam_mm not in ("", None, "None") else None

            # ресайз
            img_resized, scale, left, top = resize_keep_square(img, img_size)
            x_r = x * scale + left
            y_r = y * scale + top

            # bbox
            x_c, y_c, w, h = bbox_from_point_diam(
                x_r, y_r, diam_mm, spacing, scale, size=img_size,
                min_px=16.0, enlarge=bbox_enlarge
            )

            # класс
            cname = _class_from_row(r)
            if cname not in class2id:
                skipped += 1; continue
            cid = class2id[cname]
            stats[cname] += 1

            base = f"{r['study_key']}_series{r['series_number']}_idx{i}"
            img_out = img_dir[split_name] / f"{base}.png"
            lbl_out = lbl_dir[split_name] / f"{base}.txt"

            cv2.imwrite(str(img_out), (img_resized*255).astype(np.uint8))
            with lbl_out.open("w", encoding="utf-8") as f:
                f.write(f"{cid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # файлы набора
    (out_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i:c for c,i in class2id.items()},
        "nc": len(classes),
    }
    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    total = sum(stats.values())
    summary = {"total": total, "train": len(tr_idx), "val": len(va_idx), "test": len(te_idx), "classes": len(classes), "skipped": skipped}
    print("[YOLO] class stats:")
    for c in classes:
        print(f"  {c}: {stats[c]}")
    print(f"[YOLO] summary: {summary}")
    return summary

if __name__ == "__main__":
    build_yolo_dataset()
