import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import cv2
import pydicom
from matplotlib import pyplot as plt
from ultralytics import YOLO

from ctlib.config.paths import BASE_DIR

# ---- конфиг и утилиты ----

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

@dataclass
class EvalConfig:
    model_path: Path
    src_neg: Path
    src_pos: Path
    out_dir: Path
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 1024
    device: str = "0"  # "cpu" или "0"
    save_conf: bool = True

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _is_dicom_file(p: Path) -> bool:
    """Пробуем распознать DICOM по метаданным (работает даже без .dcm)."""
    try:
        pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
        return True
    except Exception:
        return False

def _dcm_to_png_uint8(dcm_path: Path, out_png: Path) -> Path:
    """Грубая конвертация DICOM → 8-бит PNG с простым windowing."""
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = float(ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter)
        ww = float(ds.WindowWidth[0]  if isinstance(ds.WindowWidth,  pydicom.multival.MultiValue) else ds.WindowWidth)
        lo, hi = wc - ww / 2.0, wc + ww / 2.0
    else:
        lo, hi = np.percentile(arr, 5.0), np.percentile(arr, 95.0)

    arr = np.clip((arr - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    img8 = (arr * 255.0).astype(np.uint8)
    _ensure_dir(out_png.parent)
    cv2.imwrite(str(out_png), img8)
    return out_png

def _collect_images_maybe_convert(src: Path, tmp_png_dir: Path) -> List[Path]:
    """Собирает PNG/JPG; любые иные файлы пытается разобрать как DICOM и сконвертить в PNG."""
    files, converted, total = [], 0, 0
    if not src.exists():
        print(f"[WARN] Source folder not found: {src}")
        return files
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        total += 1
        if _is_image(p):
            files.append(p)
            continue
        if _is_dicom_file(p):  # ловит и .IMA, и файлы без расширения
            rel = p.relative_to(src)
            out_png = tmp_png_dir / rel.with_suffix(".png")
            try:
                files.append(_dcm_to_png_uint8(p, out_png))
                converted += 1
            except Exception:
                # битые/нестандартные — пропускаем
                pass
    print(f"[COLLECT] {src} -> total_files={total}, as_is_images={len(files)-converted}, dicom_converted={converted}, final={len(files)}")
    return files

def _predict_folder(yolo: YOLO, paths: List[Path], save_dir: Path,
                    conf: float, iou: float, imgsz: int, device: str) -> List[Dict[str, Any]]:
    _ensure_dir(save_dir)
    if not paths:
        print(f"[WARN] No images to run in '{save_dir.name}' — skipping inference.")
        return []
    results = yolo.predict(
        source=[str(p) for p in paths],
        conf=conf, iou=iou, imgsz=imgsz, device=device,
        save=True, save_txt=True, save_conf=True,
        project=str(save_dir.parent), name=save_dir.name, exist_ok=True, verbose=False
    )
    rows: List[Dict[str, Any]] = []
    for res in results:
        img_path = Path(res.path)
        n = 0 if res.boxes is None else len(res.boxes)
        cls_names, confs = [], []
        if n > 0:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                cls_names.append(res.names.get(cls_id, str(cls_id)))
                confs.append(float(b.conf.item()))
        rows.append({
            "image_path": str(img_path),
            "n_boxes": n,
            "classes": ";".join(cls_names),
            "confs": ";".join([f"{c:.3f}" for c in confs]),
        })
    return rows

def _compute_metrics(summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = np.array([int(r["true_label"]) for r in summary], dtype=int)
    y_pred = np.array([1 if int(r["n_boxes"]) > 0 else 0 for r in summary], dtype=int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = float(TP / max(1, TP + FP))
    rec  = float(TP / max(1, TP + FN))
    acc  = float((TP + TN) / max(1, len(y_true)))
    f1   = float(2 * prec * rec / max(1e-6, (prec + rec))) if (prec + rec) > 0 else 0.0
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": prec, "recall": rec, "accuracy": acc, "f1": f1,
            "n_images": int(len(y_true))}

def _plot_confusion_matrix(metrics: Dict[str, Any], out_png: Path) -> None:
    """
    Рисует 2x2 матрицу ошибок (True vs Pred) и сохраняет в out_png.
    Формат матрицы:
        [[TN, FP],
         [FN, TP]]
    """
    TN, FP, FN, TP = metrics["TN"], metrics["FP"], metrics["FN"], metrics["TP"]
    cm = np.array([[TN, FP],
                   [FN, TP]], dtype=float)
    total = cm.sum()
    # подписи процентов (от всех изображений)
    perc = (cm / max(1.0, total)) * 100.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix (image-level)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No pathology", "Pathology"])
    ax.set_yticklabels(["No pathology", "Pathology"])

    # подписи ячеек: абсолютное значение + %
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}\n{perc[i,j]:.1f}%",
                    ha="center", va="center",
                    color="black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("count")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def _plot_counts_bar(metrics: Dict[str, Any], out_png: Path) -> None:
    """
    Бар-чарт TP/FP/TN/FN + подписи основных метрик (Precision/Recall/Acc/F1).
    """
    TN, FP, FN, TP = metrics["TN"], metrics["FP"], metrics["FN"], metrics["TP"]
    labels = ["TP", "FP", "TN", "FN"]
    values = [TP, FP, TN, FN]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title("Counts: TP / FP / TN / FN")
    ax.set_ylabel("count")

    txt = (f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
           f"Acc={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}  "
           f"N={metrics['n_images']}")
    ax.text(0.5, 0.95, txt, ha="center", va="top", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# ---- основной пайп ----

def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
    _ensure_dir(cfg.out_dir)
    tmp_png_dir = _ensure_dir(cfg.out_dir / "tmp_png")

    neg_imgs = _collect_images_maybe_convert(cfg.src_neg, tmp_png_dir / "neg")
    pos_imgs = _collect_images_maybe_convert(cfg.src_pos, tmp_png_dir / "pos")
    print(f"[INFO] collected: neg={len(neg_imgs)} images, pos={len(pos_imgs)} images")

    yolo = YOLO(str(cfg.model_path))

    save_dir_neg = cfg.out_dir / "preds" / "norma_anon"
    save_dir_pos = cfg.out_dir / "preds" / "pneumonia_anon"
    rows_neg = _predict_folder(yolo, neg_imgs, save_dir_neg, cfg.conf, cfg.iou, cfg.imgsz, cfg.device)
    rows_pos = _predict_folder(yolo, pos_imgs, save_dir_pos, cfg.conf, cfg.iou, cfg.imgsz, cfg.device)

    summary: List[Dict[str, Any]] = []
    for r in rows_neg:
        summary.append({
            "folder": "norma_anon",
            "image_path": r["image_path"],
            "n_boxes": int(r["n_boxes"]),
            "classes": r["classes"],
            "confs": r["confs"],
            "found_pathology": int(r["n_boxes"]) > 0,
            "true_label": 0,
            "correct": int((int(r["n_boxes"]) == 0))
        })
    for r in rows_pos:
        summary.append({
            "folder": "pneumonia_anon",
            "image_path": r["image_path"],
            "n_boxes": int(r["n_boxes"]),
            "classes": r["classes"],
            "confs": r["confs"],
            "found_pathology": int(r["n_boxes"]) > 0,
            "true_label": 1,
            "correct": int((int(r["n_boxes"]) > 0))
        })

    summary_csv = cfg.out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "folder", "image_path", "n_boxes", "classes", "confs",
            "found_pathology", "true_label", "correct"
        ])
        wr.writeheader()
        wr.writerows(summary)

    metrics = _compute_metrics(summary)

    # [NEW] графики метрик
    _plot_confusion_matrix(metrics, cfg.out_dir / "confusion_matrix.png")
    _plot_counts_bar(metrics, cfg.out_dir / "counts_tp_fp_tn_fn.png")

    # как и раньше: json + txt
    with (cfg.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (cfg.out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Images: {metrics['n_images']}\n"
            f"TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}\n"
            f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  "
            f"Accuracy={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}\n"
        )

    return {"out_dir": str(cfg.out_dir), "summary_csv": str(summary_csv), "metrics": metrics}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Evaluate YOLO model on two folders (norma vs pneumonia)")
    ap.add_argument("--model", type=str, default=BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt")
    ap.add_argument("--neg",   type=str, default=BASE_DIR / "data" / "norma_anon")
    ap.add_argument("--pos",   type=str, default=BASE_DIR / "data" / "pneumonia_anon")
    ap.add_argument("--out",   type=str, default=BASE_DIR / "runs" / "vis" / "yolo")
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--iou",   type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    cfg = EvalConfig(
        model_path=Path(args.model),
        src_neg=Path(args.neg),
        src_pos=Path(args.pos),
        out_dir=Path(args.out),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    info = run_eval(cfg)
    print("[DONE] results saved in:", info["out_dir"])
    print("[METRICS]", info["metrics"])
