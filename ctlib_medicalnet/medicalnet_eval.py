import json, csv, math, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom, cv2
import matplotlib.pyplot as plt

from ctlib.config.paths import BASE_DIR


# ---------- конфиг ----------
@dataclass
class EvalCfg:
    neg_root: Path
    pos_root: Path
    out_dir: Path
    weights: Path | None = None          # путь к весам MedicalNet (resnet_18_23dataset.pth)
    device: str = "cuda"
    depth: int = 64                      # целевое число срезов
    size: int = 224                      # H=W для входа
    batch: int = 1                       # инференс по 1 исследованию
    conf_thresh: float = 0.5             # порог для класса "патология"

# ---------- утилиты чтения DICOM ----------
def _read_series_to_volume(series_dir: Path) -> Tuple[np.ndarray, float]:
    files = [p for p in series_dir.rglob("*") if p.is_file()]
    items = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
            inst = int(getattr(ds, "InstanceNumber", 1e9))  # если нет — очень большое число
            items.append((inst, p))
        except Exception:
            pass

    if not items:
        raise RuntimeError(f"No readable DICOM in {series_dir}")

    items.sort(key=lambda t: (t[0], str(t[1])))
    slices, spacings = [], []
    for _, p in items:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            slices.append(arr)
            spacings.append(float(getattr(ds, "SliceThickness", 1.0)))
        except Exception:
            continue

    if not slices:
        raise RuntimeError(f"No pixel data in {series_dir}")

    vol = np.stack(slices, axis=0)
    mean_st = float(np.mean(spacings)) if spacings else 1.0
    return vol, mean_st


def _window_and_norm(vol: np.ndarray, lo: float | None=None, hi: float | None=None) -> np.ndarray:
    """Apply simple windowing per volume to [0,1]. Fallback — robust percentiles."""
    v = vol.astype(np.float32)
    if lo is None or hi is None:
        lo = np.percentile(v, 5.0)
        hi = np.percentile(v, 95.0)
    v = (v - lo) / max(1e-6, (hi - lo))
    v = np.clip(v, 0.0, 1.0)
    return v

def _resize_slice(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def _resample_depth(v: np.ndarray, target_z: int) -> np.ndarray:
    """Линейная выборка/интерполяция по Z до target_z (без сложного ресемплинга по миллиметрам)."""
    Z, H, W = v.shape
    if Z == target_z:
        return v
    # берем равномерные индексы с линейной интерполяцией
    grid = np.linspace(0, Z - 1, target_z)
    z0 = np.floor(grid).astype(int)
    z1 = np.clip(z0 + 1, 0, Z - 1)
    w1 = grid - z0
    w0 = 1.0 - w1
    out = (v[z0] * w0[:, None, None] + v[z1] * w1[:, None, None]).astype(np.float32)
    return out

def load_study_as_tensor(series_dir: Path, depth: int, size: int) -> torch.Tensor:
    vol, _ = _read_series_to_volume(series_dir)        # [Z,H,W] в HU
    vol = _window_and_norm(vol)                        # [0..1]
    vol = np.stack([_resize_slice(s, size) for s in vol], axis=0)  # [Z,S,S]
    vol = _resample_depth(vol, depth)                  # [D,S,S]
    vol = vol[None, None, ...]                         # [1,1,D,S,S] для 3D conv
    return torch.from_numpy(vol.astype(np.float32))

def _looks_like_dicom(p: Path) -> bool:
    try:
        pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
        return True
    except Exception:
        return False

def list_studies(root: Path) -> List[Path]:
    """
    Возвращает список папок-исследований:
      - если в корне есть DICOM-файлы → сам root как одно исследование;
      - иначе все подпапки, внутри которых (глубоко) есть хотя бы один DICOM.
    """
    root = Path(root)
    # кейс: DICOM-файлы прямо в корне
    direct_dicoms = [p for p in root.iterdir() if p.is_file() and _looks_like_dicom(p)]
    if direct_dicoms:
        return [root]

    studies = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        has_dicom_inside = any(_looks_like_dicom(f) for f in d.rglob("*") if f.is_file())
        if has_dicom_inside:
            studies.append(d)
    return studies


# ---------- модель ----------
class ResNet3D18(nn.Module):
    """Лёгкая обёртка над MedicalNet backbone (ResNet-18 3D) под бинарную классификацию."""
    def __init__(self, num_classes: int = 1):
        super().__init__()
        # своя простая 3D-ResNet18 (совместимые имена весов MedicalNet)
        from torch.hub import load_state_dict_from_url
        # Скелет из torchvision.video не нужен; определим вручную упрощённо через r3d_18.
        from torchvision.models.video import r3d_18
        self.backbone = r3d_18(weights=None, progress=False)
        # вход 3D: [B,3,D,H,W]; у нас 1-канал → сконвертим 1→3 повторением
        self.backbone.stem[0] = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.classifier = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,D,S,S] -> повторим канал до 3
        x = x.repeat(1, 3, 1, 1, 1)
        feat = self.backbone(x)             # [B, C]
        logit = self.classifier(feat).view(-1)
        return logit

def load_weights_medicalnet(model: nn.Module, weights_path: Path) -> None:
    """Грубая загрузка весов MedicalNet (если слои совпадают — подгрузятся)."""
    sd = torch.load(str(weights_path), map_location="cpu")
    # некоторые веса имеют ключи с префиксами — попробуем аккуратно сопоставить
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[WEIGHTS] loaded with strict=False; missing={len(missing)} unexpected={len(unexpected)}")

# ---------- инференс и метрики ----------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"TP":0,"FP":0,"TN":0,"FN":0,
                "precision":0.0,"recall":0.0,"accuracy":0.0,"f1":0.0,
                "auc":float("nan"),"ap":float("nan"),"n":0}

    y_true = np.array([r["true"] for r in rows], int)
    y_pred = np.array([r["pred"] for r in rows], int)
    y_score = np.array([r["score"] for r in rows], float)

    TP = int(((y_true==1) & (y_pred==1)).sum())
    TN = int(((y_true==0) & (y_pred==0)).sum())
    FP = int(((y_true==0) & (y_pred==1)).sum())
    FN = int(((y_true==1) & (y_pred==0)).sum())

    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    acc  = (TP + TN) / max(1, len(y_true))
    f1   = (2 * prec * rec / max(1e-6, prec + rec)) if (prec + rec) > 0 else 0.0

    # AUC/AP только если в y_true есть оба класса
    if len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = float(roc_auc_score(y_true, y_score))
            ap  = float(average_precision_score(y_true, y_score))
        except Exception:
            auc, ap = float("nan"), float("nan")
    else:
        auc, ap = float("nan"), float("nan")

    return {"TP":TP,"FP":FP,"TN":TN,"FN":FN,"precision":prec,"recall":rec,
            "accuracy":acc,"f1":f1,"auc":auc,"ap":ap,"n":int(len(y_true))}


def plot_confusion_and_curves(rows: List[Dict[str,Any]], metrics: Dict[str,Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # confusion
    cm = np.array([[metrics["TN"], metrics["FP"]],
                   [metrics["FN"], metrics["TP"]]], dtype=int)
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No pathology","Pathology"])
    ax.set_yticklabels(["No pathology","Pathology"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
    ax.set_title("Confusion matrix (study-level)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_dir/"confusion_matrix.png", dpi=150); plt.close(fig)

    # ROC/PR
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        y_true = np.array([r["true"] for r in rows], int)
        y_score = np.array([r["score"] for r in rows], float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)

        fig = plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC AUC={metrics['auc']:.3f}")
        plt.subplot(1,2,2)
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR AP={metrics['ap']:.3f}")
        plt.tight_layout(); plt.savefig(out_dir/"roc_pr_curves.png", dpi=150); plt.close(fig)
    except Exception:
        pass

    # score histogram
    y0 = [r["score"] for r in rows if r["true"]==0]
    y1 = [r["score"] for r in rows if r["true"]==1]
    fig, ax = plt.subplots(1,1,figsize=(7,4))
    ax.hist(y0, bins=20, alpha=0.6, label="No pathology")
    ax.hist(y1, bins=20, alpha=0.6, label="Pathology")
    ax.set_xlabel("probability (sigmoid)"); ax.set_ylabel("count")
    ax.legend(); ax.set_title("Score distribution")
    plt.tight_layout(); plt.savefig(out_dir/"score_hist.png", dpi=150); plt.close(fig)

# ---------- основной ран ----------
def run(cfg: EvalCfg) -> Dict[str,Any]:
    device = torch.device(cfg.device if (cfg.device!="cpu" and torch.cuda.is_available()) else "cpu")
    out = cfg.out_dir; out.mkdir(parents=True, exist_ok=True)

    # список исследований
    neg_studies = list_studies(cfg.neg_root)
    pos_studies = list_studies(cfg.pos_root)
    print(f"[INFO] studies: neg={len(neg_studies)}  pos={len(pos_studies)}")

    # модель
    model = ResNet3D18(num_classes=1).to(device)
    if cfg.weights and Path(cfg.weights).exists():
        load_weights_medicalnet(model, Path(cfg.weights))
    else:
        print("[WARN] weights not provided or not found — using random init (для честного теста укажи --weights).")
    model.eval()

    rows: List[Dict[str,Any]] = []

    @torch.no_grad()
    def infer_one(study_dir: Path, true_label: int) -> Dict[str,Any]:
        vol = load_study_as_tensor(study_dir, cfg.depth, cfg.size)  # [1,1,D,S,S]
        vol = vol.to(device)
        logit = model(vol)               # [B]
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= cfg.conf_thresh)
        return {"study": str(study_dir), "true": true_label, "score": float(prob), "pred": pred}

    # прогон
    for s in neg_studies:
        try:
            rows.append(infer_one(s, 0))
        except Exception as e:
            print(f"[SKIP] {s.name}: {e}")
    for s in pos_studies:
        try:
            rows.append(infer_one(s, 1))
        except Exception as e:
            print(f"[SKIP] {s.name}: {e}")

    # сохранить summary
    with (out/"summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["study","true","score","pred"])
        wr.writeheader(); wr.writerows(rows)

    # метрики и графики
    metrics = compute_metrics(rows)
    with (out/"metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (out/"metrics.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"N={metrics['n']}  TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}\n"
            f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  "
            f"Accuracy={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}  "
            f"AUC={metrics['auc']:.3f}  AP={metrics['ap']:.3f}\n"
        )
    plot_confusion_and_curves(rows, metrics, out)
    print("[DONE] saved to:", out)
    print("[METRICS]", metrics)
    return {"out_dir": str(out), "metrics": metrics}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate 3D-ResNet (MedicalNet-style) on two folders of CT studies")
    parser.add_argument("--model", type=str, default=BASE_DIR / "ctlib_medicalnet" / "models" / "resnet_50_23dataset.pth")
    parser.add_argument("--neg",   type=str, default=BASE_DIR / "data" / "norma_anon")
    parser.add_argument("--pos",   type=str, default=BASE_DIR / "data" / "pneumonia_anon")
    parser.add_argument("--out",   type=str, default=BASE_DIR / "runs" / "vis" / "mednet")
    parser.add_argument("--weights", type=str, default="", help="path to MedicalNet weights (resnet_18_23dataset.pth)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args()

    cfg = EvalCfg(
        neg_root=Path(args.neg),
        pos_root=Path(args.pos),
        out_dir=Path(args.out),
        weights=Path(args.model) if args.model else None,
        device=args.device,
        depth=args.depth,
        size=args.size,
        conf_thresh=args.thresh,
    )
    run(cfg)
