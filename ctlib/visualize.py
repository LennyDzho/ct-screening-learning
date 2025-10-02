from pathlib import Path
from typing import Optional, Tuple
import pydicom
import numpy as np
import matplotlib.pyplot as plt

import csv
import math

from ctlib.config.paths import LESIONS_CSV


def _load_dicom_array(path: Path) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    # нормализация для отображения
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)

    spacing = None
    try:
        ps = ds.PixelSpacing  # [row_spacing_mm, col_spacing_mm]
        spacing = (float(ps[0]), float(ps[1]))
    except Exception:
        spacing = None
    return arr, spacing

def visualize_from_csv(lesions_csv: Path = LESIONS_CSV, idx: int = 0) -> None:
    """
    Визуализирует запись с индексом idx из lesions.csv.
    """
    rows = []
    with lesions_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    if not rows:
        print("CSV пуст.")
        return
    # if idx < 0 or idx >= len(rows):
    #     print(f"Некорректный idx={idx}, доступно 0..{len(rows)-1}")
    #     return

    r = rows[idx]
    dcm_path = Path(r["dicom_path"])
    x = float(r["x_px"])
    y = float(r["y_px"])
    diameter_mm = r.get("diameter_mm")
    diameter_mm = float(diameter_mm) if diameter_mm not in (None, "", "None") else None

    arr, spacing = _load_dicom_array(dcm_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="gray")
    plt.scatter([x], [y], s=40, marker="x")
    title = f'{r["study_key"]} | series={r["series_number"]} | zΔ={float(r["z_diff_mm"]):.2f} mm | target={r["target_malignant"]}'
    plt.title(title)

    # Нарисуем круг диаметра при наличии PixelSpacing
    if spacing and diameter_mm:
        # диаметр в пикселях приблизительно по колоночному шагу
        # можно усреднить по двум осям
        px_per_mm = (1.0 / spacing[0] + 1.0 / spacing[1]) / 2.0
        radius_px = (diameter_mm * px_per_mm) / 2.0
        theta = np.linspace(0, 2 * math.pi, 256)
        cx = x + radius_px * np.cos(theta)
        cy = y + radius_px * np.sin(theta)
        plt.plot(cx, cy, linewidth=1)

    plt.show()


if __name__ == "__main__":
    visualize_from_csv(idx=70)