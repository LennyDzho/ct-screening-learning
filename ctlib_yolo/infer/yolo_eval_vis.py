from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv
import math

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from ctlib.config.paths import BASE_DIR


# ---------- utils ----------

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_labels_yolo(txt: Path) -> List[Tuple[int, float, float, float, float]]:
    """Возвращает список (cls, xc, yc, w, h) в нормированных координатах [0..1]. Если файла нет — пусто."""
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

def yolo_to_xyxy(box: Tuple[float, float, float, float], W: int, H: int) -> Tuple[int, int, int, int]:
    """(xc, yc, w, h) norm -> (x1,y1,x2,y2) px."""
    xc, yc, w, h = box
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def clip_xyxy(x1,y1,x2,y2,W,H):
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

def draw_box(img, x1,y1,x2,y2, color, label: Optional[str]=None, thickness: int=2):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(0, y1-6)
        cv2.rectangle(img, (x1, y_text-th-4), (x1+tw+4, y_text), color, -1)
        cv2.putText(img, label, (x1+2, y_text-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

# ---------- main vis ----------

def visualize_split_predictions(
    data_yaml: Path,
    model_path: Path,
    split: str = "val",
    out_dir: Path = Path("runs/vis/yolo"),
    limit: int = 200,
    conf: float = 0.25,
    iou_match: float = 0.30,
    overlay_mode: str = "side",  # "side" (GT|Pred) или "overlay"
    save_csv: bool = True,
) -> Dict[str, int]:
    """
    Рисует GT (зелёный) и предсказания (красный). Совпадения (IoU>=iou_match и тот же класс) учитываются как TP.
    Сохраняет изображения и CSV со сводкой по кадрам.
    """
    data = load_yaml(Path(data_yaml))
    root = Path(data.get("path", Path(data_yaml).parent))
    names = data.get("names")
    if isinstance(names, dict):
        # {id: name} -> список
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)

    img_dir = root / f"images/{split}"
    lbl_dir = root / f"labels/{split}"
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    # сбор файлов
    imgs = sorted([p for p in img_dir.glob("*.png")] + [p for p in img_dir.glob("*.jpg")] + [p for p in img_dir.glob("*.jpeg")])
    if limit and limit > 0:
        imgs = imgs[:min(limit, len(imgs))]

    # CSV
    csv_path = out_dir / "per_image_metrics.csv"
    if save_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["image", "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall"])

    totals = {"gt":0, "pred":0, "tp":0, "fp":0, "fn":0}

    for pimg in imgs:
        img = cv2.imread(str(pimg), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape[:2]
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # GT
        ptxt = lbl_dir / (pimg.stem + ".txt")
        gts_norm = read_labels_yolo(ptxt)
        gts: List[Tuple[int,Tuple[int,int,int,int]]] = []
        for (cid, xc, yc, w, h) in gts_norm:
            x1,y1,x2,y2 = yolo_to_xyxy((xc,yc,w,h), W, H)
            x1,y1,x2,y2 = clip_xyxy(x1,y1,x2,y2,W,H)
            gts.append((cid, (x1,y1,x2,y2)))

        # предсказания
        results = model.predict(source=str(pimg), conf=conf, verbose=False, imgsz=max(H,W))
        boxes = results[0].boxes
        preds: List[Tuple[int, float, Tuple[int,int,int,int]]] = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = map(int, xyxy)
                cid = int(b.cls.item())
                score = float(b.conf.item())
                preds.append((cid, score, clip_xyxy(x1,y1,x2,y2,W,H)))

        # matching: гриди по IoU и совпадению класса
        used_gt = set()
        used_pr = set()
        matches: List[Tuple[int,int,float]] = []  # (gt_idx, pred_idx, IoU)
        for j, (pcid, pscore, pb) in enumerate(preds):
            best_iou = 0.0; best_i = -1
            for i, (gcid, gb) in enumerate(gts):
                if i in used_gt:
                    continue
                if gcid != pcid:
                    continue
                iou = iou_xyxy(pb, gb)
                if iou >= iou_match and iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0:
                used_gt.add(best_i); used_pr.add(j)
                matches.append((best_i, j, best_iou))

        tp = len(matches)
        fp = len(preds) - tp
        fn = len(gts) - tp
        precision = (tp / (tp + fp)) if (tp+fp) > 0 else 0.0
        recall    = (tp / (tp + fn)) if (tp+fn) > 0 else 0.0

        totals["gt"] += len(gts); totals["pred"] += len(preds)
        totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn

        # визуализация
        vis_gt = color_img.copy()
        for (cid, (x1,y1,x2,y2)) in gts:
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            draw_box(vis_gt, x1,y1,x2,y2, color=(0,255,0), label=f"GT:{cname}")

        vis_pr = color_img.copy()
        for j, (cid, score, (x1,y1,x2,y2)) in enumerate(preds):
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            is_tp = j in used_pr
            color = (0,255,255) if is_tp else (0,0,255)  # TP=желтый, FP=красный
            lab = f"PR:{cname} {score:.2f}"
            draw_box(vis_pr, x1,y1,x2,y2, color=color, label=lab)

        if overlay_mode == "overlay":
            vis = color_img.copy()
            # GT зелёные
            for (cid, (x1,y1,x2,y2)) in gts:
                cname = class_names[cid] if cid < len(class_names) else str(cid)
                draw_box(vis, x1,y1,x2,y2, color=(0,255,0), label=f"GT:{cname}")
            # Pred красные/желтые
            for j, (cid, score, (x1,y1,x2,y2)) in enumerate(preds):
                cname = class_names[cid] if cid < len(class_names) else str(cid)
                is_tp = j in used_pr
                color = (0,255,255) if is_tp else (0,0,255)
                lab = f"PR:{cname} {score:.2f}"
                draw_box(vis, x1,y1,x2,y2, color=color, label=lab)
            out_img = vis
        else:
            # side-by-side
            pad = 10
            h = max(vis_gt.shape[0], vis_pr.shape[0])
            w = vis_gt.shape[1] + vis_pr.shape[1] + pad
            out_img = np.zeros((h, w, 3), dtype=np.uint8)
            out_img[:vis_gt.shape[0], :vis_gt.shape[1]] = vis_gt
            out_img[:vis_pr.shape[0], vis_gt.shape[1]+pad:vis_gt.shape[1]+pad+vis_pr.shape[1]] = vis_pr
            # заголовок
            title = f"{pimg.name} | GT={len(gts)} PR={len(preds)} TP={tp} FP={fp} FN={fn} P={precision:.2f} R={recall:.2f}"
            cv2.putText(out_img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        out_path = out_dir / f"{pimg.stem}_vis.jpg"
        cv2.imwrite(str(out_path), out_img)

        if save_csv:
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow([pimg.name, len(gts), len(preds), tp, fp, fn, precision, recall])

    # финальная сводка
    P = (totals["tp"] / (totals["tp"] + totals["fp"])) if (totals["tp"]+totals["fp"])>0 else 0.0
    R = (totals["tp"] / (totals["tp"] + totals["fn"])) if (totals["tp"]+totals["fn"])>0 else 0.0
    print(f"[DONE] {split}: images={len(imgs)}  GT={totals['gt']}  PR={totals['pred']}  TP={totals['tp']}  FP={totals['fp']}  FN={totals['fn']}  P={P:.3f}  R={R:.3f}")
    if save_csv:
        print(f"[CSV] {csv_path}")
    print(f"[OUT] {out_dir}")
    return totals

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("YOLO eval & visualization (GT vs Pred)")
    parser.add_argument("--data", type=str, default=BASE_DIR / "extracted_data" / "dataset_yolo" / "data.yaml")
    parser.add_argument("--model", type=str, default=BASE_DIR / "runs" / "detect" / "train2" / "weights" / "best.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    parser.add_argument("--out", type=str, default=BASE_DIR / "runs" / "vis" / "yolo")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou",  type=float, default=0.30, help="IoU threshold for TP matching (same class)")
    parser.add_argument("--mode", type=str, default="side", choices=["side","overlay"])
    args = parser.parse_args()

    visualize_split_predictions(
        data_yaml=Path(args.data),
        model_path=Path(args.model),
        split=args.split,
        out_dir=Path(args.out),
        limit=args.limit,
        conf=args.conf,
        iou_match=args.iou,
        overlay_mode=args.mode,
    )
