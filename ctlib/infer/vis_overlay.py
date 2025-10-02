from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from ctlib.config.paths import LESIONS_CSV, DATASET_OUT_DIR
from ctlib.slice_heatmap import LesionSliceHeatmapDS
from ctlib.models.unet_multitask import UNetMultiTask
from ctlib.train.split import group_split_indices

def _peak_xy(hm: torch.Tensor) -> tuple[float, float]:
    """Возвращает (x,y) пика теплокарты hm (H,W) как float."""
    H, W = hm.shape
    flat = hm.view(-1)
    idx = int(flat.argmax().item())
    y = idx // W
    x = idx % W
    return float(x), float(y)

def visualize_val_overlay(
    idx_in_val: int = 0,
    ckpt: Path | None = None,
    alpha: float = 0.35,
    save_path: Path | None = None,
):
    """
    Показывает 3 картинки: GT, Pred, и их совместное наложение на КТ.
    idx_in_val — индекс валидационного примера (после группового сплита).
    Возвращает dict с метаданными/метриками (p, y_true, y_pred, study_key, series_number, save_path).
    """
    ds_all = LesionSliceHeatmapDS(LESIONS_CSV, size=512, train=False, allow_unlabeled=True)
    _, va_idx, _, _ = group_split_indices(LESIONS_CSV, val_ratio=0.2, seed=42)
    assert 0 <= idx_in_val < len(va_idx), f"idx_in_val={idx_in_val} вне 0..{len(va_idx)-1}"
    sample = ds_all[va_idx[idx_in_val]]

    img  = sample["image"].squeeze(0)      # [H,W]
    gt_h = sample["heatmap"].squeeze(0)    # [H,W]
    y_t  = sample["label"].item() if sample["label"].numel()==1 else float("nan")
    meta = sample["meta"]

    if ckpt is None:
        ckpt = DATASET_OUT_DIR / "mtl_unet" / "unet_multitask_best.pt"
    model = UNetMultiTask(in_ch=1, base=32)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(img.unsqueeze(0).unsqueeze(0))  # 1,1,H,W
        pred_h_logits = out["heat"].squeeze(0).squeeze(0)
        pred_h = torch.sigmoid(pred_h_logits)
        p = torch.sigmoid(out["cls_logit"]).item()
        y_pred = int(p >= 0.5)

    # центры
    x_gt, y_gt = _peak_xy(gt_h)
    x_pr, y_pr = _peak_xy(pred_h)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{meta["study_key"]} | series={meta["series_number"]} | p_mal={p:.2f} | '
                 f'pred={y_pred} | true={int(y_t) if y_t==y_t else "NA"}',
                 y=1.03)

    axs[0].imshow(img.numpy(), cmap="gray")
    axs[0].imshow(gt_h.numpy(), alpha=alpha)
    axs[0].contour(gt_h.numpy(), levels=[0.5], colors="lime", linewidths=1.0)
    axs[0].scatter([x_gt], [y_gt], c="lime", marker="x", s=60, linewidths=2)
    axs[0].set_title("GT (контур + центр)")
    axs[0].axis("off")

    axs[1].imshow(img.numpy(), cmap="gray")
    axs[1].imshow(pred_h.numpy(), alpha=alpha)
    axs[1].contour(pred_h.numpy(), levels=[0.5], colors="red", linewidths=1.0)
    axs[1].scatter([x_pr], [y_pr], c="red", marker="x", s=60, linewidths=2)
    axs[1].set_title(f"Pred (контур + центр) | p={p:.2f}")
    axs[1].axis("off")

    axs[2].imshow(img.numpy(), cmap="gray")
    axs[2].contour(gt_h.numpy(), levels=[0.5], colors="lime", linewidths=1.0)
    axs[2].contour(pred_h.numpy(), levels=[0.5], colors="red", linewidths=1.0)
    axs[2].scatter([x_gt], [y_gt], c="lime", marker="x", s=50)
    axs[2].scatter([x_pr], [y_pr], c="red", marker="x", s=50)
    axs[2].set_title("Overlay: GT (зел) vs Pred (красн)")
    axs[2].axis("off")

    from matplotlib.lines import Line2D
    proxy = [Line2D([0], [0], color="lime", lw=2, label="GT"),
             Line2D([0], [0], color="red", lw=2, label="Pred")]
    axs[2].legend(handles=proxy, loc="lower right")


    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[VIS] saved {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return {
        "p": float(p),
        "y_true": (None if not (y_t==y_t) else int(y_t)),
        "y_pred": int(y_pred),
        "study_key": meta["study_key"],
        "series_number": meta["series_number"],
        "save_path": str(save_path) if save_path else None,
    }

def save_val_overlays(n: int = 12, start: int = 0, out_dir: Path = DATASET_OUT_DIR / "mtl_unet" / "eval" / "overlays"):
    """Сохраняет n подряд картинок (по индексам валидации). Может включать несколько срезов одного исследования."""
    _, va_idx, _, _ = group_split_indices(LESIONS_CSV, val_ratio=0.2, seed=42)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(start, min(start+n, len(va_idx))):
        save_path = out_dir / f"val_{i:04d}.png"
        visualize_val_overlay(idx_in_val=i, save_path=save_path)

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

def save_val_overlays_by_study(
    n: int = 100,
    out_dir: Path = DATASET_OUT_DIR / "mtl_unet" / "eval" / "overlays_by_study",
    start_study: int = 0,
):
    """
    Сохраняет оверлеи для n РАЗНЫХ исследований (не более одного среза на study_key).
    Файлы именуются как: idx_{k}_study_{study}_p{p:.2f}_t{true}_p{pred}.png
    """
    ds_all = LesionSliceHeatmapDS(LESIONS_CSV, size=512, train=False, allow_unlabeled=True)
    _, va_idx, _, _ = group_split_indices(LESIONS_CSV, val_ratio=0.2, seed=42)

    # выберем по одному индексу на каждое исследование
    seen = set()
    unique_indices = []
    for j in va_idx:
        sk = ds_all[j]["meta"]["study_key"]
        if sk in seen:
            continue
        seen.add(sk)
        unique_indices.append(j)
    # сдвиг по списку исследований
    unique_indices = unique_indices[start_study:start_study + n]

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for k, j in enumerate(unique_indices):
        info = visualize_val_overlay(idx_in_val=va_idx.index(j), save_path=None)
        sk = _sanitize(info["study_key"])
        p  = info["p"]; yt = info["y_true"]; yp = info["y_pred"]
        fname = f"idx{k:03d}_study_{sk}_p{p:.2f}_t{('NA' if yt is None else yt)}_p{yp}.png"
        save_path = out_dir / fname
        # перерисуем прямо в файл, чтобы зафиксировать изображение
        visualize_val_overlay(idx_in_val=va_idx.index(j), save_path=save_path)

    print(f"[VIS] saved {len(unique_indices)} overlays to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Визуализация предсказаний модели на КТ")
    parser.add_argument("--idx", type=int, default=0, help="Индекс примера из валидационного сплита")
    parser.add_argument("--save", type=str, default=None, help="Путь для сохранения картинки (по умолчанию show)")
    parser.add_argument("--batch", type=int, default=0, help="Сохранить подряд N примеров начиная с --idx (без группировки по исследованиям)")
    parser.add_argument("--by-study", action="store_true", default=True, help="Сохранять по одному срезу на исследование")
    parser.add_argument("--n", type=int, default=100, help="Сколько исследований/картинок сохранить при --by-study")
    parser.add_argument("--start-study", type=int, default=0, help="Сдвиг по списку уникальных исследований (для пагинации)")
    args = parser.parse_args()

    if args.by_study:
        save_val_overlays_by_study(n=args.n, start_study=args.start_study)
    elif args.batch > 0:
        save_val_overlays(n=args.batch, start=args.idx,
                          out_dir=DATASET_OUT_DIR / "mtl_unet" / "eval" / "overlays")
    else:
        visualize_val_overlay(idx_in_val=args.idx,
                              save_path=Path(args.save) if args.save else None)
