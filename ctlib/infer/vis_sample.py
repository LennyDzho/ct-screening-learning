from pathlib import Path
import torch
import matplotlib.pyplot as plt

from ctlib.config.paths import LESIONS_CSV, DATASET_OUT_DIR
from ctlib.slice_heatmap import LesionSliceHeatmapDS
from ctlib.models.unet_multitask import UNetMultiTask
from ctlib.train.split import group_split_indices

def visualize_val_sample(idx_in_val: int = 0, ckpt: Path = None, threshold: float = 0.5):
    # датасет (без аугментаций)
    ds_all = LesionSliceHeatmapDS(LESIONS_CSV, size=512, train=False, allow_unlabeled=True)
    _, va_idx, _, _ = group_split_indices(LESIONS_CSV, val_ratio=0.2, seed=42)
    assert 0 <= idx_in_val < len(va_idx), f"idx_in_val={idx_in_val} вне диапазона 0..{len(va_idx)-1}"
    sample = ds_all[va_idx[idx_in_val]]

    # модель
    if ckpt is None:
        ckpt = DATASET_OUT_DIR / "mtl_unet" / "unet_multitask_best.pt"
    model = UNetMultiTask(in_ch=1, base=32)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    img = sample["image"].unsqueeze(0)
    gt  = sample["heatmap"]
    y_true = sample["label"].item() if sample["label"].numel() == 1 else float("nan")
    meta = sample["meta"]

    with torch.no_grad():
        out = model(img)
        heat = out["heat"].squeeze(0).squeeze(0)  # H,W (logits)
        p = torch.sigmoid(out["cls_logit"]).item()
        y_pred = 1 if p >= threshold else 0

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img.squeeze(0).squeeze(0).numpy(), cmap="gray")
    axs[0].set_title("Image"); axs[0].axis("off")

    axs[1].imshow(gt.squeeze(0).numpy(), vmin=0, vmax=1); axs[1].set_title("GT heat"); axs[1].axis("off")

    axs[2].imshow(torch.sigmoid(heat).numpy(), vmin=0, vmax=1)
    axs[2].set_title(f"Pred heat | p={p:.2f} | pred={y_pred} | true={int(y_true) if y_true==y_true else 'NA'}")
    axs[2].axis("off")

    fig.suptitle(f'{meta["study_key"]} | series={meta["series_number"]}', y=1.02)
    plt.tight_layout()
    plt.show()
