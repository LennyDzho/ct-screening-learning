from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pydicom
import cv2

DEFAULT_SIZE = 1024

def read_dicom_hu(path: Path, w_center: float = -600.0, w_width: float = 1500.0) -> Tuple[np.ndarray, Optional[Tuple[float,float]]]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    L = w_center - w_width/2.0
    H = w_center + w_width/2.0
    hu = np.clip(hu, L, H)
    img = (hu - L) / max(1e-6, (H - L))
    spacing = None
    try:
        ps = ds.PixelSpacing  # [row_mm, col_mm]
        spacing = (float(ps[0]), float(ps[1]))
    except Exception:
        spacing = None
    return img.astype(np.float32), spacing

def resize_keep_square(img: np.ndarray, target: int = DEFAULT_SIZE):
    h, w = img.shape
    s = target / max(h, w)
    nh, nw = int(round(h*s)), int(round(w*s))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=img2.dtype)
    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas[top:top+nh, left:left+nw] = img2
    return canvas, s, left, top

def bbox_from_point_diam(x_px: float, y_px: float, diam_mm: Optional[float], spacing: Optional[Tuple[float,float]],
                         scale: float, size: int = DEFAULT_SIZE, min_px: float = 16.0, enlarge: float = 1.0) -> Tuple[float,float,float,float]:
    """
    YOLO bbox (x_c, y_c, w, h) в [0..1]. Квадрат по диаметру; enlarge>1.0 можно чуть расширить бокс.
    """
    if diam_mm and spacing:
        px_per_mm = (1.0/spacing[0] + 1.0/spacing[1]) / 2.0
        w_px = max(min_px, diam_mm * px_per_mm * scale) * float(enlarge)
    else:
        w_px = min_px
    h_px = w_px

    x_c = np.clip(x_px, 0, size-1)
    y_c = np.clip(y_px, 0, size-1)
    return x_c/size, y_c/size, w_px/size, h_px/size

# --- НОВОЕ: нормализация типа узла ---

_NODULE_TYPE_MAP = {
    "#0S": "solid", "#0_S": "solid", "#0-S": "solid", "0S": "solid", "0_S": "solid",
    "#1PS": "part_solid", "#1_PS": "part_solid", "#1-PS": "part_solid", "1PS": "part_solid", "1_PS": "part_solid",
    "#2GG": "ground_glass", "#2_GG": "ground_glass", "#2-GG": "ground_glass", "2GG": "ground_glass", "2_GG": "ground_glass",
}

def normalize_nodule_type(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = str(s).strip().upper().replace(" ", "")
    return _NODULE_TYPE_MAP.get(key)
