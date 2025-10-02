from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import random
import shutil
import yaml
import cv2
import numpy as np

from ctlib.config.paths import BASE_DIR

try:
    import albumentations as A
except ImportError as e:
    raise SystemExit("Albumentations не установлен. Установите: pip install albumentations") from e


def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_labels_yolo(txt: Path) -> List[Tuple[int, float, float, float, float]]:
    if not txt.exists():
        return []
    out = []
    with txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            out.append((cls, xc, yc, w, h))
    return out

def write_labels_yolo(txt: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    with txt.open("w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def build_transform(img_size: int, allow_hflip: bool = True) -> A.BasicTransform:
    # Мягкие, мед-адекватные аугментации, совместимые с твоей версией Albumentations
    t_list = [
        A.Affine(
            scale=(0.92, 1.08),
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            rotate=(-8, 8),
            shear={"x": (-4, 4), "y": 0},
            p=1.0
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=1.0),
            A.CLAHE(clip_limit=(1.0, 2.0), p=1.0),
        ], p=0.8),
        # На твоей сборке используется GaussNoise (без var_limit/mean/per_channel, чтобы не ловить несовместимость)
        A.GaussNoise(p=0.30),
        # Лёгкая "маскировка" без несовместимых аргументов
        A.GridDropout(
            ratio=0.02,
            random_offset=True,
            p=0.10
        ),
        A.ToFloat(max_value=255.0),
    ]
    if allow_hflip:
        t_list.insert(0, A.HorizontalFlip(p=0.5))

    return A.Compose(
        t_list,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.30,
            clip=True  # подрезать боксы в [0,1]
        )
    )



def collect_train_paths(data_yaml: Path) -> Tuple[Path, Path, List[Path]]:
    data = load_yaml(data_yaml)
    root = Path(data.get("path", Path(data_yaml).parent))
    img_train = root / "images" / "train"
    lbl_train = root / "labels" / "train"
    assert img_train.exists() and lbl_train.exists(), f"Не найдены {img_train} или {lbl_train}"
    imgs = sorted(list(img_train.glob("*.png")) + list(img_train.glob("*.jpg")) + list(img_train.glob("*.jpeg")))
    return img_train, lbl_train, imgs

def compute_class_counts(lbl_train: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for p in lbl_train.glob("*.txt"):
        for cls, *_ in read_labels_yolo(p):
            counts[cls] = counts.get(cls, 0) + 1
    return counts

def upsample_factor_per_image(lbl_path: Path, target_per_class: int, class_counts: Dict[int, int]) -> int:
    """Сколько аугментаций сделать для данного изображения с учётом редких классов в нём."""
    rows = read_labels_yolo(lbl_path)
    if not rows:
        return 0
    # если в изображении несколько классов, берём максимум недобора
    need_factors = []
    for cls, *_ in rows:
        cur = class_counts.get(cls, 0)
        if cur >= target_per_class:
            need_factors.append(0)
        else:
            # чем сильнее недобор, тем больше аугментаций
            missing = max(0, target_per_class - cur)
            # грубо: 1 аугментация на каждые 20 недостающих объектов
            f = max(1, int(np.ceil(missing / 20.0)))
            need_factors.append(f)
    return int(max(need_factors)) if need_factors else 0

def main():
    ap = argparse.ArgumentParser("Offline augmentation for YOLO dataset (train split)")
    ap.add_argument("--data", type=str, default=BASE_DIR / "extracted_data" / "dataset_yolo" / "data.yaml")
    ap.add_argument("--per-image", type=int, default=1, help="Базовое число аугментаций на 1 исходное изображение")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--allow-hflip", action="store_true", default=True)
    ap.add_argument("--target-per-class", type=int, default=0, help="Если >0, дублировать редкие классы до этого числа объектов")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Не сохранять, только посчитать объёмы")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    data_yaml = Path(args.data)
    img_train, lbl_train, imgs = collect_train_paths(data_yaml)

    # классы из yaml (для подписи)
    data = load_yaml(data_yaml)
    names = data.get("names")
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)
    n_classes = len(class_names)

    # базовая трансформация
    aug = build_transform(args.imgsz, allow_hflip=args.allow_hflip)

    # оценим баланс
    class_counts = compute_class_counts(lbl_train)
    print("[AUG] class counts before:", {class_names[k]: v for k, v in class_counts.items()})

    total_new = 0
    for pimg in imgs:
        plbl = lbl_train / (pimg.stem + ".txt")
        rows = read_labels_yolo(plbl)
        if not rows:
            continue

        # подготовка в формат Albumentations
        bboxes = [(xc, yc, w, h) for (_, xc, yc, w, h) in rows]
        labels = [int(cls) for (cls, *_rest) in rows]

        # сколько раз аугментировать
        n_aug = args.per_image
        if args.target_per_class > 0:
            # усилить изображения с редкими классами
            n_aug = max(n_aug, upsample_factor_per_image(plbl, args.target_per_class, class_counts))

        if n_aug <= 0:
            continue

        # читаем изображение (в оттенках серого)
        img = cv2.imread(str(pimg), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # приводим к 8-бит [0..255] и 1-канал
        if img.dtype != np.uint8:
            img8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        else:
            img8 = img.copy()

        for k in range(n_aug):
            # запуск аугментации
            transformed = aug(image=img8, bboxes=bboxes, class_labels=labels)
            im2 = transformed["image"]
            b2 = transformed["bboxes"]
            l2 = transformed["class_labels"]

            # после сильных аугментаций боксы могут исчезнуть — пропускаем пустые
            if len(b2) == 0:
                print(f"[AUG] dropped all boxes for {pimg.name} aug#{k} (visibility/min_area)")
                continue


            # имена файлов
            new_stem = f"aug_{pimg.stem}_{k:02d}"
            out_img = img_train / f"{new_stem}.png"
            out_lbl = lbl_train / f"{new_stem}.txt"

            if not args.dry_run:
                # кламп YOLO-боксов в [0,1] на всякий случай
                rows_new = []
                for ((xc, yc, w, h), c) in zip(b2, l2):
                    xc = float(np.clip(xc, 0.0, 1.0))
                    yc = float(np.clip(yc, 0.0, 1.0))
                    w = float(np.clip(w, 1e-6, 1.0))
                    h = float(np.clip(h, 1e-6, 1.0))
                    # если после клампа бокс «вышел», подправим так, чтобы он полностью лежал внутри
                    x1 = np.clip(xc - w / 2.0, 0.0, 1.0)
                    y1 = np.clip(yc - h / 2.0, 0.0, 1.0)
                    x2 = np.clip(xc + w / 2.0, 0.0, 1.0)
                    y2 = np.clip(yc + h / 2.0, 0.0, 1.0)
                    # пересобрать валидные w,h
                    w = max(1e-6, float(x2 - x1))
                    h = max(1e-6, float(y2 - y1))
                    xc = float((x1 + x2) / 2.0)
                    yc = float((y1 + y2) / 2.0)
                    rows_new.append((int(c), xc, yc, w, h))

                cv2.imwrite(str(out_img), im2 if im2.ndim == 2 else cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
                write_labels_yolo(out_lbl, rows_new)

            # обновляем счётчики классов (важно для динамического апсемплинга)
            for c in l2:
                class_counts[c] = class_counts.get(c, 0) + 1
            total_new += 1

    print(f"[AUG] new images created: {total_new}")
    print("[AUG] class counts after:", {class_names.get(k, str(k)) if isinstance(class_names, dict) else class_names[k]: v for k, v in class_counts.items()})

if __name__ == "__main__":
    main()
