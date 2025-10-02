from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import csv
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
import cv2
import random

DEFAULT_SIZE = 512

def _to_hu_and_window(ds, arr: np.ndarray, w_center: float = -600.0, w_width: float = 1500.0) -> np.ndarray:
    # HU = pixel_value * RescaleSlope + RescaleIntercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr.astype(np.float32) * slope + intercept
    # применим окно: центр/ширина -> [L, H]
    L = w_center - w_width / 2.0
    H = w_center + w_width / 2.0
    hu = np.clip(hu, L, H)
    # нормировка в [0,1]
    hu = (hu - L) / max(1e-6, (H - L))
    return hu.astype(np.float32)

def _read_dicom_array(path: Path) -> Tuple[np.ndarray, Optional[Tuple[float,float]]]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    img = _to_hu_and_window(ds, arr, w_center=-600.0, w_width=1500.0)  # лёгочное окно

    spacing = None
    try:
        ps = ds.PixelSpacing  # [row_mm, col_mm]
        spacing = (float(ps[0]), float(ps[1]))
    except Exception:
        spacing = None
    return img, spacing

def _resize_keep_aspect(img: np.ndarray, target: int = DEFAULT_SIZE) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape
    s = target / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=img2.dtype)
    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas[top:top+nh, left:left+nw] = img2
    return canvas, s, left, top

def _make_heatmap(h: int, w: int, cx: float, cy: float, sigma_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    hm = np.exp(-d2 / (2.0 * (sigma_px ** 2))).astype(np.float32)
    return hm

def _augment_pair(img: np.ndarray, hm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # H-flip
    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1]); hm = np.ascontiguousarray(hm[:, ::-1])
    # V-flip
    if random.random() < 0.2:
        img = np.ascontiguousarray(img[::-1, :]); hm = np.ascontiguousarray(hm[::-1, :])
    # 90*k
    if random.random() < 0.2:
        k = random.choice([1, 2, 3])
        img = np.ascontiguousarray(np.rot90(img, k)); hm  = np.ascontiguousarray(np.rot90(hm, k))
    # brightness/contrast
    if random.random() < 0.5:
        alpha = 0.8 + 0.4 * random.random()
        beta  = -0.1 + 0.2 * random.random()
        img = np.clip(img * alpha + beta, 0.0, 1.0).astype(np.float32)
    return img, hm

class LesionSliceHeatmapDS(Dataset):
    def __init__(self, csv_path: Path, size: int = DEFAULT_SIZE, train: bool = True, allow_unlabeled: bool = True):
        self.csv_path = Path(csv_path)
        self.size = size
        self.train = train
        self.allow_unlabeled = allow_unlabeled
        self.rows = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                if not r.get("dicom_path"):
                    continue
                if not self.allow_unlabeled and r.get("target_malignant") in ("", "None", None):
                    continue
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        dcm_path = Path(r["dicom_path"])
        img, spacing = _read_dicom_array(dcm_path)

        x = float(r["x_px"]); y = float(r["y_px"])
        diam_mm = r.get("diameter_mm")
        diam_mm = float(diam_mm) if diam_mm not in ("", None, "None") else None

        img_resized, scale, left, top = _resize_keep_aspect(img, self.size)
        x_r = x * scale + left
        y_r = y * scale + top

        if diam_mm and spacing:
            px_per_mm = (1.0 / spacing[0] + 1.0 / spacing[1]) / 2.0
            sigma_px = max(2.0, (diam_mm * px_per_mm * scale) / 2.5)
        else:
            sigma_px = 6.0

        heat = _make_heatmap(self.size, self.size, x_r, y_r, sigma_px)

        if self.train:
            img_resized, heat = _augment_pair(img_resized, heat)

        image_t = torch.from_numpy(img_resized[None, ...]).float()
        heat_t  = torch.from_numpy(heat[None, ...]).float()

        has_label = r.get("target_malignant") not in ("", "None", None)
        label_t = (torch.tensor(float(r["target_malignant"])) if has_label
                   else torch.tensor(float("nan")))

        return {
            "image": image_t,
            "heatmap": heat_t,
            "label": label_t,
            "has_label": has_label,
            "meta": {
                "study_key": r["study_key"],
                "dicom_path": r["dicom_path"],
                "series_number": r["series_number"],
                "x_resized": x_r,
                "y_resized": y_r,
                "sigma_px": sigma_px,
            }
        }
