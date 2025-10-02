from pathlib import Path
from typing import List, Dict
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from ctlib.config.paths import LESIONS_CSV, DATASET_OUT_DIR
from ctlib.slice_heatmap import LesionSliceHeatmapDS
from ctlib.models.unet_multitask import UNetMultiTask
from ctlib.train.split import group_split_indices


def _peak_xy(hm: torch.Tensor) -> torch.Tensor:
    # hm: B,1,H,W → B,2 (x,y) пиксели
    B, _, H, W = hm.shape
    flat = hm.view(B, -1)
    idx = flat.argmax(dim=1)
    y = (idx // W).float()
    x = (idx % W).float()
    return torch.stack([x, y], dim=1)


def evaluate(
    ckpt: Path = None,
    lesions_csv: Path = LESIONS_CSV,
    out_dir: Path = DATASET_OUT_DIR / "mtl_unet" / "eval",
    batch_size: int = 8,
    threshold: float = 0.5,
) -> Path:
    """
    Запускает инференс на валид. сплите и сохраняет CSV + картинки метрик.
    Возвращает путь к eval_predictions.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "eval_predictions.csv"

    # === dataset & split (валидация) ===
    ds_all = LesionSliceHeatmapDS(lesions_csv, size=512, train=False, allow_unlabeled=True)
    _, va_idx, _, _ = group_split_indices(lesions_csv, val_ratio=0.2, seed=42)
    va_ds = Subset(ds_all, va_idx)
    ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # === model ===
    if ckpt is None:
        ckpt = DATASET_OUT_DIR / "mtl_unet" / "unet_multitask_best.pt"
    model = UNetMultiTask(in_ch=1, base=32)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    # === loop ===
    all_probs: List[float] = []
    all_true:  List[int] = []
    rows_out: List[Dict[str, str]] = []

    with torch.no_grad():
        for batch in ld:
            img = batch["image"]  # B,1,H,W
            heat_gt = batch["heatmap"]
            y = batch["label"]        # B (float, может быть NaN)
            has_lbl = torch.tensor(batch["has_label"], dtype=torch.bool)

            out = model(img)
            heat_pred = out["heat"]   # logits
            cls_logit = out["cls_logit"]  # B,1

            # классификация (только там, где есть метка)
            prob = torch.sigmoid(cls_logit).squeeze(1)  # B
            pred = (prob >= threshold).to(torch.int32)

            # локализация: пиковые координаты и L2
            peak_pred = _peak_xy(torch.sigmoid(heat_pred))  # B,2
            peak_gt   = _peak_xy(heat_gt)                   # B,2
            l2 = torch.norm(peak_pred - peak_gt, dim=1)     # B

            B = img.size(0)
            for i in range(B):
                meta = batch["meta"]
                study_key = meta["study_key"][i]
                dcm_path  = meta["dicom_path"][i]
                y_true = None
                if has_lbl[i]:
                    y_true = int(y[i].item())
                    all_true.append(y_true)
                    all_probs.append(float(prob[i].item()))

                rows_out.append({
                    "study_key": study_key,
                    "dicom_path": dcm_path,
                    "series_number": str(meta["series_number"][i]),
                    "y_true": ("" if y_true is None else str(y_true)),
                    "y_prob": f"{float(prob[i].item()):.6f}",
                    "y_pred": str(int(pred[i].item())),
                    "l2px": f"{float(l2[i].item()):.3f}",
                    "x_peak_pred": f"{float(peak_pred[i,0].item()):.2f}",
                    "y_peak_pred": f"{float(peak_pred[i,1].item()):.2f}",
                    "x_peak_gt":   f"{float(peak_gt[i,0].item()):.2f}",
                    "y_peak_gt":   f"{float(peak_gt[i,1].item()):.2f}",
                })

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wr.writeheader()
        wr.writerows(rows_out)

    if len(set(all_true)) >= 2:
        auc = roc_auc_score(all_true, all_probs)
        fpr, tpr, _ = roc_curve(all_true, all_probs)
        # ROC
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1], linestyle="--")
        plt.title(f"ROC-AUC = {auc:.3f}")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
        plt.savefig(out_dir / "roc.png", dpi=150); plt.close()

        y_pred = [1 if p >= threshold else 0 for p in all_probs]
        cm = confusion_matrix(all_true, y_pred, labels=[0,1])
        plt.figure(figsize=(4.5,4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion matrix")
        plt.xticks([0,1], ["benign(0)","malig(1)"])
        plt.yticks([0,1], ["benign(0)","malig(1)"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i,j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close()

        acc = (cm[0,0] + cm[1,1]) / np.maximum(1, cm.sum())
        print(f"[EVAL] Labeled samples: {len(all_true)} | AUC={auc:.3f} | ACC@{threshold:.2f}={acc:.3f}")
        print(f"[EVAL] Confusion matrix:\n{cm}")
    else:
        print("[EVAL] Недостаточно размеченных примеров в валидации для AUC/CM.")

    print(f"[EVAL] saved: {pred_csv}")
    return pred_csv


if __name__ == "__main__":
    evaluate()
