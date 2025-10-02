import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from ctlib.config.paths import LESIONS_CSV, DATASET_OUT_DIR   # не DATASET_OUT_DIR
from ctlib.models.unet_multitask import UNetMultiTask
from ctlib.slice_heatmap import LesionSliceHeatmapDS
from ctlib.train.config import TrainCfg
from ctlib.train.split import group_split_indices

def _peak_xy(hm: torch.Tensor) -> torch.Tensor:
    B, _, H, W = hm.shape
    flat = hm.view(B, -1)
    idx = flat.argmax(dim=1)
    y = (idx // W).float()
    x = (idx % W).float()
    return torch.stack([x, y], dim=1)

def _append_history_row(path: Path, row: Dict[str, Any]) -> None:
    """
    Пишем строку истории. Если файл уже существует, но его заголовок не совпадает
    с актуальным набором полей — делаем бэкап и пересоздаём файл с новым хедером.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_loss","val_loss",
        "train_heat","train_cls",
        "val_heat","val_cls",
        "auc","l2px",
        "n_train","n_val","batch_size","lr",
        "labeled_tr","labeled_val",
    ]

    need_header = True
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                existing_fields = rd.fieldnames or []
            if set(existing_fields) == set(fields):
                need_header = False
            else:
                backup = path.with_suffix(path.suffix + ".bak")
                path.replace(backup)
                print(f"[history] Header changed → old file moved to: {backup.name}")
        except Exception:
            backup = path.with_suffix(path.suffix + ".bak")
            path.replace(backup)
            print(f"[history] Corrupt history → moved to: {backup.name}")

    with path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if need_header:
            wr.writeheader()
        wr.writerow({k: row.get(k) for k in fields})


def _plot_curves(history_csv: Path, out_dir: Path) -> None:
    """Читаем историю и строим графики, устойчиво к неполным колонкам."""
    if not history_csv.exists():
        return

    rows: List[Dict[str, str]] = []
    with history_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        return

    def _f(r: Dict[str, str], key: str, default: float = float("nan")) -> float:
        v = r.get(key, "")
        try:
            return float(v) if v not in ("", "nan", "None") else default
        except Exception:
            return default

    ep  = [int(r.get("epoch", i+1)) for i, r in enumerate(rows)]
    tr  = [_f(r, "train_loss", float("nan")) for r in rows]
    va  = [_f(r, "val_loss",   float("nan")) for r in rows]
    th  = [_f(r, "train_heat", float("nan")) for r in rows]
    tc  = [_f(r, "train_cls",  float("nan")) for r in rows]
    vh  = [_f(r, "val_heat",   float("nan")) for r in rows]
    vc  = [_f(r, "val_cls",    float("nan")) for r in rows]
    auc = [_f(r, "auc",        float("nan")) for r in rows]
    l2  = [_f(r, "l2px",       float("nan")) for r in rows]

    plt.figure(figsize=(6,4))
    plt.plot(ep, tr, label="train (total)")
    plt.plot(ep, va, label="val (total)")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Total Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()

    if any(not np.isnan(x) for x in th+tc+vh+vc):
        plt.figure(figsize=(7,4))
        if any(not np.isnan(x) for x in th): plt.plot(ep, th, label="train heat")
        if any(not np.isnan(x) for x in tc): plt.plot(ep, tc, label="train cls")
        if any(not np.isnan(x) for x in vh): plt.plot(ep, vh, "--", label="val heat")
        if any(not np.isnan(x) for x in vc): plt.plot(ep, vc, "--", label="val cls")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss components"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "loss_components.png", dpi=150)
        plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(ep, auc, marker="o")
    plt.xlabel("epoch"); plt.ylabel("AUC"); plt.title("ROC-AUC"); plt.tight_layout()
    plt.savefig(out_dir / "auc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(ep, l2, marker="o")
    plt.xlabel("epoch"); plt.ylabel("L2 (px)"); plt.title("Localization error"); plt.tight_layout()
    plt.savefig(out_dir / "l2px.png", dpi=150)
    plt.close()

def _compute_pos_weight(csv_path: Path) -> float:
    pos = neg = 0
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            t = r.get("target_malignant")
            if t in ("", "None", None): continue
            if int(t) == 1: pos += 1
            else: neg += 1
    return float(neg) / max(1.0, float(pos)) if pos > 0 else 1.0

def run_training(cfg: TrainCfg) -> Dict[str, Any]:
    # === DATA ===
    ds_all = LesionSliceHeatmapDS(cfg.lesions_csv, size=512, train=True, allow_unlabeled=True)
    tr_idx, va_idx, cnt_tr, cnt_va = group_split_indices(cfg.lesions_csv, val_ratio=0.2, seed=42)
    tr_ds = Subset(ds_all, tr_idx)
    va_ds = Subset(ds_all, va_idx)

    ds_all.train = True
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    ds_all.train = False
    val_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    n_tr, n_val = len(tr_ds), len(va_ds)

    print(f"Датасет загружен: N={len(ds_all)} из {cfg.lesions_csv}")
    print(f"Исследований: train={len(cnt_tr)}, val={len(cnt_va)} | Строк: train={n_tr}, val={n_val}")
    print(f"tr_ld (batches) = {len(tr_ld)}  |  val_ld (batches) = {len(val_ld)}  при batch_size={cfg.batch_size}")

    # === MODEL ===
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = UNetMultiTask(in_ch=1, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    pos_w = torch.tensor([_compute_pos_weight(cfg.lesions_csv)], device=device)

    # === IO ===
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); best_path = out_dir / "unet_multitask_best.pt"
    history_csv = out_dir / "history.csv"

    for epoch in range(1, cfg.epochs + 1):
        # === TRAIN ===
        ds_all.train = True
        model.train()
        tr_loss_sum = tr_heat_sum = tr_cls_sum = 0.0
        labeled_tr = 0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False)
        for batch in pbar:
            img = batch["image"].to(device)
            heat_gt = batch["heatmap"].to(device)
            y = batch["label"].to(device).unsqueeze(1)
            has_lbl = batch["has_label"].to(device).bool() if torch.is_tensor(batch["has_label"]) \
                      else torch.as_tensor(batch["has_label"], device=device, dtype=torch.bool)

            out = model(img)
            heat_pred = out["heat"]
            cls_logit = out["cls_logit"]

            loss_heat = F.binary_cross_entropy_with_logits(heat_pred, heat_gt)
            if has_lbl.any():
                loss_cls = F.binary_cross_entropy_with_logits(cls_logit[has_lbl], y[has_lbl], pos_weight=pos_w)
                labeled_tr += int(has_lbl.sum().item())
            else:
                loss_cls = torch.zeros((), device=device)

            loss = cfg.heat_loss_w * loss_heat + cfg.cls_loss_w * loss_cls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            # аккумулируем суммы для усреднения
            bs = img.size(0)
            tr_loss_sum += loss.item() * bs
            tr_heat_sum += loss_heat.item() * bs
            tr_cls_sum  += (loss_cls.item() * bs if loss_cls.numel() else 0.0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        tr_loss = tr_loss_sum / max(1, n_tr)
        tr_heat = tr_heat_sum / max(1, n_tr)
        tr_cls  = tr_cls_sum  / max(1, n_tr)

        # === VAL ===
        ds_all.train = False
        model.eval()
        val_loss_sum = val_heat_sum = val_cls_sum = 0.0
        preds: List[float] = []; targets: List[float] = []; l2_px: List[float] = []
        labeled_val = 0
        pbar_v = tqdm(val_ld, desc=f"Epoch {epoch}/{cfg.epochs} [val]", leave=False)
        with torch.no_grad():
            for batch in pbar_v:
                img = batch["image"].to(device)
                heat_gt = batch["heatmap"].to(device)
                y = batch["label"].to(device).unsqueeze(1)
                has_lbl = batch["has_label"].to(device).bool() if torch.is_tensor(batch["has_label"]) \
                          else torch.as_tensor(batch["has_label"], device=device, dtype=torch.bool)

                out = model(img)
                heat_pred = out["heat"]; cls_logit = out["cls_logit"]

                loss_heat = F.binary_cross_entropy_with_logits(heat_pred, heat_gt)
                if has_lbl.any():
                    loss_cls = F.binary_cross_entropy_with_logits(cls_logit[has_lbl], y[has_lbl], pos_weight=pos_w)
                    preds.extend(torch.sigmoid(cls_logit[has_lbl]).cpu().view(-1).tolist())
                    targets.extend(y[has_lbl].cpu().view(-1).tolist())
                    labeled_val += int(has_lbl.sum().item())
                else:
                    loss_cls = torch.zeros((), device=device)

                loss = cfg.heat_loss_w * loss_heat + cfg.cls_loss_w * loss_cls

                bs = img.size(0)
                val_loss_sum += loss.item() * bs
                val_heat_sum += loss_heat.item() * bs
                val_cls_sum  += (loss_cls.item() * bs if loss_cls.numel() else 0.0)

                # L2 локализации
                p_pred = _peak_xy(torch.sigmoid(heat_pred))
                p_gt   = _peak_xy(heat_gt)
                l2 = torch.norm(p_pred - p_gt, dim=1)
                l2_px.extend(l2.cpu().tolist())

                pbar_v.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss = val_loss_sum / max(1, n_val)
        val_heat = val_heat_sum / max(1, n_val)
        val_cls  = val_cls_sum  / max(1, n_val)

        try:
            auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        mean_l2 = float(sum(l2_px) / max(1, len(l2_px)))

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train={tr_loss:.4f} (heat={tr_heat:.4f}, cls={tr_cls:.4f})  "
            f"val={val_loss:.4f} (heat={val_heat:.4f}, cls={val_cls:.4f})  "
            f"AUC={auc:.3f}  L2px={mean_l2:.2f}  "
            f"studies: train={len(cnt_tr)}, val={len(cnt_va)}  labeled_tr={labeled_tr}  labeled_val={labeled_val}"
        )

        # best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best to {best_path.name}")

        _append_history_row(history_csv, {
            "epoch": epoch,
            "train_loss": tr_loss, "val_loss": val_loss,
            "train_heat": tr_heat, "train_cls": tr_cls,
            "val_heat": val_heat, "val_cls": val_cls,
            "auc": auc, "l2px": mean_l2,
            "n_train": n_tr, "n_val": n_val,
            "batch_size": cfg.batch_size, "lr": cfg.lr,
            "labeled_tr": labeled_tr, "labeled_val": labeled_val,
        })
        _plot_curves(history_csv, out_dir)

    return {"best_ckpt": str(best_path), "best_val": best_val}

if __name__ == "__main__":
    cfg = TrainCfg(
        lesions_csv= LESIONS_CSV,
        out_dir    = DATASET_OUT_DIR / "mtl_unet",
        epochs=100,
        batch_size=8,
        lr=3e-4,
        device="cuda",
        num_workers=2,
        heat_loss_w=2.0,
        cls_loss_w=1.0,
    )
    run_training(cfg)
