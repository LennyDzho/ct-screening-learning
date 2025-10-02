import argparse, csv, json, warnings, random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

import config.path as P
from ctlib_binary.datasets.volume_ds import VolumeBinaryDataset
from ctlib_binary.models.medicalnet_r3d18 import R3D18Binary
from ctlib_binary.utils.csv_registry import load_registry

warnings.filterwarnings("ignore", message="Precision is ill-defined.*")

# ---------- utils ----------
def _seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _sigmoid(x): return 1.0 / (1.0 + torch.exp(-x))

def compute_metrics_from_np(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, probs))
        else:
            out["roc_auc"] = float("nan")
        out["pr_auc"]  = float(average_precision_score(y, probs))
        pred = (probs >= 0.5).astype("int32")
        out["acc"] = float(accuracy_score(y, pred))
        out["f1"]  = float(f1_score(y, pred, zero_division=0))
    except Exception:
        pred = (probs >= 0.5).astype("int32")
        out["acc"] = float((pred == y).mean())
    return out

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        probs = _sigmoid(logits).cpu().numpy()
        y     = labels.cpu().numpy()
    return compute_metrics_from_np(probs, y)

def save_predictions_csv(out_csv: Path, keys: List[str], logits: torch.Tensor, labels: torch.Tensor):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    probs = _sigmoid(logits).cpu().numpy()
    lg    = logits.detach().cpu().numpy()
    y     = labels.cpu().numpy()
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["study_key","label","logit","prob","pred"])
        w.writeheader()
        for k, yy, lo, pr in zip(keys, y, lg, probs):
            w.writerow({
                "study_key": k,
                "label": int(yy),
                "logit": float(lo),
                "prob": float(pr),
                "pred": int(pr >= 0.5),
            })

def save_confusion_and_report(out_png: Path, out_txt: Path, out_json: Path, logits: torch.Tensor, labels: torch.Tensor):
    try:
        from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
    except Exception:
        return
    probs = _sigmoid(logits).cpu().numpy()
    y_true = labels.cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4,4), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    report_txt = classification_report(
        y_true, y_pred, target_names=["norma(0)", "pathology(1)"], zero_division=0
    )
    out_txt.write_text(report_txt, encoding="utf-8")

    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
    summary = {
        "labels": [0,1],
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "support": sup.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def plot_learning_curves(history: List[Dict[str, float]], out_png: Path):
    import pandas as pd
    df = pd.DataFrame(history)
    fig = plt.figure(figsize=(9,6), dpi=150)
    ax1 = fig.add_subplot(221)
    for split in ["train","val"]:
        dd = df[df["split"]==split]
        ax1.plot(dd["epoch"], dd["loss"], label=split)
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()
    ax2 = fig.add_subplot(222)
    for split in ["train","val"]:
        dd = df[df["split"]==split]
        ax2.plot(dd["epoch"], dd["roc_auc"], label=split)
    ax2.set_title("ROC AUC"); ax2.set_xlabel("epoch"); ax2.set_ylabel("auc"); ax2.legend()
    ax3 = fig.add_subplot(223)
    for split in ["train","val"]:
        dd = df[df["split"]==split]
        ax3.plot(dd["epoch"], dd["acc"], label=split)
    ax3.set_title("Accuracy"); ax3.set_xlabel("epoch"); ax3.set_ylabel("acc"); ax3.legend()
    ax4 = fig.add_subplot(224)
    for split in ["train","val"]:
        dd = df[df["split"]==split]
        ax4.plot(dd["epoch"], dd["f1"], label=split)
    ax4.set_title("F1"); ax4.set_xlabel("epoch"); ax4.set_ylabel("f1"); ax4.legend()
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# ---------- train ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies_csv", type=Path, default=P.MERGED_REGISTRY_CSV)
    ap.add_argument("--train_list", type=Path, default=P.TRAIN_LIST)
    ap.add_argument("--val_list",   type=Path, default=P.VAL_LIST)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=Path, default=P.RUNS_DIR / "binary_r3d18")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--no_tqdm", action="store_true", help="disable progress bars")
    ap.add_argument("--log_interval", type=int, default=10, help="каждые N батчей логировать промежуточные метрики")
    ap.add_argument("--tensorboard", action="store_true", help="логировать в TensorBoard")
    ap.add_argument("--balance", choices=["none","loss","sampler","both"], default="sampler",
                    help="борьба с дисбалансом: 'loss' -> pos_weight, 'sampler' -> WeightedRandomSampler")
    ap.add_argument("--seed", type=int, default=42)
    # ускорение
    ap.add_argument("--cache_dir", type=Path, default=P.RUNS_DIR / "cache_volumes")
    ap.add_argument("--target_shape", type=str, default="96,192,192", help="D,H,W")
    ap.add_argument("--compile", action="store_true", help="torch.compile для модели")
    args = ap.parse_args()

    _seed_everything(args.seed)

    # быстрые матмулы на Ada/Ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir = args.out_dir / "epochs"; epochs_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = args.out_dir / "metrics.jsonl"
    history_csv   = args.out_dir / "history.csv"
    steps_csv     = args.out_dir / "steps.csv"

    tb = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=args.out_dir / "tb")
        except Exception:
            tb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- splits & class stats ---
    from ctlib_binary.utils.csv_registry import load_split_lists
    train_keys, val_keys = load_split_lists(args.train_list, args.val_list)
    registry = load_registry(args.studies_csv)
    train_labels = np.array([int(registry[k]["label"]) for k in train_keys], dtype=np.int64)
    n_pos = int(train_labels.sum()); n_neg = int((1 - train_labels).sum())
    pos_weight_val = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    print(f"[INFO] class balance (train): pos={n_pos}, neg={n_neg}, pos_weight={pos_weight_val:.4f}")

    # --- parse target shape ---
    td, th, tw = map(int, args.target_shape.split(","))
    target_shape = (td, th, tw)

    # --- datasets with cache ---
    train_ds = VolumeBinaryDataset(args.studies_csv, train_keys, augment=True,
                                   cache_dir=args.cache_dir, target_shape=target_shape)
    val_ds   = VolumeBinaryDataset(args.studies_csv, val_keys,   augment=False,
                                   cache_dir=args.cache_dir, target_shape=target_shape)

    # --- sampler (по желанию) ---
    sampler = None
    if args.balance in {"sampler", "both"}:
        class_count = np.array([n_neg, n_pos], dtype=np.float64)
        class_weight = 1.0 / np.clip(class_count, 1, None)
        sample_weights = class_weight[train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

    # --- DataLoaders: быстрые настройки ---
    common_dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers,
    )
    if args.num_workers > 0:
        common_dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler, **common_dl_kwargs
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **common_dl_kwargs
    )

    # --- модель ---
    model = R3D18Binary(pretrained=args.pretrained).to(device)
    model = model.to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.balance in {"loss", "both"}:
        pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    if args.compile:
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # headers
    if not history_csv.exists():
        with history_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","split","loss","roc_auc","pr_auc","acc","f1"])
            writer.writeheader()
    if not steps_csv.exists():
        with steps_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","split","step","seen","loss","roc_auc","acc","f1"])
            writer.writeheader()

    best_auc = -1.0
    epoch_history: List[Dict[str, float]] = []
    global_step_tr = 0
    global_step_va = 0

    for epoch in range(1, args.epochs + 1):
        epoch_dir = epochs_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # ===== TRAIN =====
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        tr_logits, tr_labels, tr_keys = [], [], []

        it = train_ld if args.no_tqdm else tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        running_loss = 0.0; running_n = 0

        for bidx, (x, y, bkeys) in enumerate(it, start=1):
            x = x.to(device, non_blocking=True)
            # В ЭТОМ месте x уже [B,1,D,H,W] -> можно безопасно установить формат памяти:
            x = x.contiguous(memory_format=torch.channels_last_3d)

            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logit = model(x)
                loss = loss_fn(logit, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            tr_loss_sum += float(loss.item()) * bs
            tr_n += bs
            running_loss += float(loss.item()) * bs
            running_n += bs
            tr_logits.append(logit.detach().cpu())
            tr_labels.append(y.detach().cpu())
            tr_keys.extend(list(bkeys))
            global_step_tr += 1

            if (bidx % args.log_interval) == 0:
                tmp_logits = torch.cat(tr_logits, 0)
                tmp_labels = torch.cat(tr_labels, 0)
                m = compute_metrics(tmp_logits, tmp_labels)
                avg_loss = running_loss / max(1, running_n)
                if not args.no_tqdm:
                    it.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{m.get('acc',0):.3f}",
                                   f1=f"{m.get('f1',0):.3f}", auc=f"{m.get('roc_auc',np.nan):.3f}")
                with steps_csv.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch","split","step","seen","loss","roc_auc","acc","f1"])
                    writer.writerow({"epoch": epoch, "split": "train", "step": global_step_tr, "seen": tr_n,
                                     "loss": avg_loss, "roc_auc": m.get("roc_auc", np.nan),
                                     "acc": m.get("acc", np.nan), "f1": m.get("f1", np.nan)})
                running_loss = 0.0; running_n = 0

        tr_logits = torch.cat(tr_logits, 0)
        tr_labels = torch.cat(tr_labels, 0)
        tr_loss = tr_loss_sum / max(1, tr_n)
        tr_metrics = compute_metrics(tr_logits, tr_labels)

        save_predictions_csv(epoch_dir / "train_preds.csv", tr_keys, tr_logits, tr_labels)
        save_confusion_and_report(epoch_dir / "train_cm.png", epoch_dir / "train_report.txt",
                                  epoch_dir / "train_report.json", tr_logits, tr_labels)

        # ===== VAL =====
        model.eval()
        va_loss_sum, va_n = 0.0, 0
        va_logits, va_labels, va_keys = [], [], []
        itv = val_ld if args.no_tqdm else tqdm(val_ld, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False)

        running_loss_v = 0.0; running_n_v = 0
        with torch.no_grad():
            for bidx, (x, y, bkeys) in enumerate(itv, start=1):
                x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logit = model(x)
                    loss = loss_fn(logit, y)
                bs = x.size(0)
                va_loss_sum += float(loss.item()) * bs
                va_n += bs
                running_loss_v += float(loss.item()) * bs
                running_n_v += bs
                va_logits.append(logit.cpu())
                va_labels.append(y.cpu())
                va_keys.extend(list(bkeys))
                global_step_va += 1

                if (bidx % args.log_interval) == 0:
                    tmp_logits = torch.cat(va_logits, 0)
                    tmp_labels = torch.cat(va_labels, 0)
                    m = compute_metrics(tmp_logits, tmp_labels)
                    avg_loss = running_loss_v / max(1, running_n_v)
                    if not args.no_tqdm:
                        itv.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{m.get('acc',0):.3f}",
                                        f1=f"{m.get('f1',0):.3f}", auc=f"{m.get('roc_auc',np.nan):.3f}")
                    with steps_csv.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["epoch","split","step","seen","loss","roc_auc","acc","f1"])
                        writer.writerow({"epoch": epoch, "split": "val", "step": global_step_va, "seen": va_n,
                                         "loss": avg_loss, "roc_auc": m.get("roc_auc", np.nan),
                                         "acc": m.get("acc", np.nan), "f1": m.get("f1", np.nan)})
                    running_loss_v = 0.0; running_n_v = 0

        va_logits = torch.cat(va_logits, 0)
        va_labels = torch.cat(va_labels, 0)
        va_loss = va_loss_sum / max(1, va_n)
        va_metrics = compute_metrics(va_logits, va_labels)

        save_predictions_csv(epoch_dir / "val_preds.csv", va_keys, va_logits, va_labels)
        save_confusion_and_report(epoch_dir / "val_cm.png", epoch_dir / "val_report.txt",
                                  epoch_dir / "val_report.json", va_logits, va_labels)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"AUC {va_metrics.get('roc_auc','-'):.4f} | "
              f"ACC {va_metrics.get('acc','-'):.4f} | F1 {va_metrics.get('f1','-'):.4f}")

        # history CSV
        if not history_csv.exists():
            with history_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch","split","loss","roc_auc","pr_auc","acc","f1"])
                writer.writeheader()
        with history_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","split","loss","roc_auc","pr_auc","acc","f1"])
            writer.writerow({"epoch": epoch, "split": "train", "loss": tr_loss, **tr_metrics})
            writer.writerow({"epoch": epoch, "split": "val",   "loss": va_loss, **va_metrics})

        # JSONL per-epoch
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            json.dump({"epoch": epoch, "split": "train", **tr_metrics, "loss": tr_loss}, f); f.write("\n")
            json.dump({"epoch": epoch, "split": "val",   **va_metrics, "loss": va_loss}, f); f.write("\n")

        # checkpoints
        ckpt_path = epoch_dir / "weights.pth"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "metrics": {"train": tr_metrics, "val": va_metrics}}, ckpt_path)
        cur_auc = va_metrics.get("roc_auc", -1.0)
        if cur_auc is not None and cur_auc > best_auc:
            best_auc = cur_auc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "metrics": va_metrics}, args.out_dir / "best.pth")

        epoch_history.append({"epoch": epoch, "split": "train", "loss": tr_loss, **tr_metrics})
        epoch_history.append({"epoch": epoch, "split": "val",   "loss": va_loss, **va_metrics})

        if tb is not None:
            tb.add_scalar("train/loss", tr_loss, epoch)
            if tr_metrics.get("roc_auc")==tr_metrics.get("roc_auc"):
                tb.add_scalar("train/auc", tr_metrics["roc_auc"], epoch)
            tb.add_scalar("train/acc", tr_metrics.get("acc",0), epoch)
            tb.add_scalar("train/f1",  tr_metrics.get("f1",0), epoch)
            tb.add_scalar("val/loss", va_loss, epoch)
            if va_metrics.get("roc_auc")==va_metrics.get("roc_auc"):
                tb.add_scalar("val/auc", va_metrics["roc_auc"], epoch)
            tb.add_scalar("val/acc", va_metrics.get("acc",0), epoch)
            tb.add_scalar("val/f1",  va_metrics.get("f1",0), epoch)

    try:
        plot_learning_curves(epoch_history, args.out_dir / "learning_curves.png")
    except Exception:
        pass

    if tb is not None:
        tb.close()

    print(f"Done. Best ROC AUC = {best_auc:.4f}")

if __name__ == "__main__":
    main()
