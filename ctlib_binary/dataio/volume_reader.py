from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import os
import numpy as np
# optional deps
try:
    import nibabel as nib  # NIfTI
except Exception:
    nib = None

try:
    import SimpleITK as sitk  # универсальный fallback и для DICOM, и для NIfTI
except Exception:
    sitk = None

from scipy.ndimage import zoom  # pip install scipy

try:
    import pydicom  # DICOM
    # ↓↓↓ добавьте этот блок сразу после импорта pydicom ↓↓↓
    import logging, warnings
    from pydicom import config as pdc

    # 1) не ругаться на нестрогие значения (и так по умолчанию False, но явно укажем)
    pdc.enforce_valid_values = False

    # 2) приглушить именно спам про VR UI (UserWarning из pydicom.valuerep)
    warnings.filterwarnings(
        "ignore",
        message=r"Invalid value for VR UI: .*",
        category=UserWarning,
        module="pydicom.valuerep",
    )

    # 3) дополнительно снизим уровень логгера pydicom (на всякий случай)
    pdc.logger.setLevel(logging.ERROR)
except Exception:
    pydicom = None


# ---------------- NIfTI ----------------

def _read_nifti(nifti_path: Path) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    if nib is not None:
        img = nib.load(str(nifti_path))
        data = np.asarray(img.get_fdata()).astype(np.float32)
        aff = img.affine
        spacing = (float(abs(aff[2,2])), float(abs(aff[1,1])), float(abs(aff[0,0])))
        if data.ndim == 4:
            data = data[..., 0]
        # иногда данные в (H,W,D), перекинем в (D,H,W)
        if data.shape[0] < 16 and data.shape[-1] >= 16:
            data = np.transpose(data, (2,0,1))
        return data, spacing
    elif sitk is not None:
        img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)
        sp = img.GetSpacing()  # (x,y,z)
        spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
        return arr, spacing
    else:
        raise RuntimeError("Install nibabel or SimpleITK to read NIfTI")


# ---------------- DICOM (robust) ----------------

def _float_tuple(x) -> tuple:
    try:
        return tuple(float(v) for v in x)
    except Exception:
        return tuple()

def _round_tuple(t: tuple, ndigits: int = 5) -> tuple:
    return tuple(round(v, ndigits) for v in t)

def _safe_get(ds, name: str, default=None):
    return getattr(ds, name, default) if ds is not None else default

def _series_key(ds) -> tuple:
    rows = int(_safe_get(ds, "Rows", 0))
    cols = int(_safe_get(ds, "Columns", 0))
    pxsp = _round_tuple(_float_tuple(_safe_get(ds, "PixelSpacing", (1.0, 1.0))))
    iop  = _round_tuple(_float_tuple(_safe_get(ds, "ImageOrientationPatient", (1,0,0,0,1,0))))
    suid = str(_safe_get(ds, "SeriesInstanceUID", ""))
    return (rows, cols, pxsp, iop, suid)

def _read_dicom_series_pydicom(series_dir) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    """Устойчивый pydicom-парсер: series_dir может быть str или Path."""
    assert pydicom is not None, "pydicom is required (or install SimpleITK fallback)"
    series_dir = Path(series_dir)  # <<< ВАЖНО: приводим к Path

    files: List[Path] = [p for p in series_dir.rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError(f"No files in {series_dir}")

    parsed: List[pydicom.dataset.FileDataset] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=False, force=True)
            # pixel_array может бросить при отсутствии PixelData — проверяем наличие тега
            if not hasattr(ds, "PixelData"):
                continue
            _ = ds.pixel_array  # прогреваем
            parsed.append(ds)
        except Exception:
            continue

    if not parsed:
        raise RuntimeError(f"No DICOM slices readable in {series_dir} (maybe compressed? try SimpleITK)")

    # сгруппируем по совместимому ключу
    groups: Dict[tuple, List[pydicom.dataset.FileDataset]] = {}
    for ds in parsed:
        key = _series_key(ds)
        groups.setdefault(key, []).append(ds)

    # выберем самую большую группу по количеству срезов
    key_best = max(groups.keys(), key=lambda k: len(groups[k]))
    slices = groups[key_best]
    if len(slices) < 2:
        raise RuntimeError(f"Found only {len(slices)} slice(s) in best group for {series_dir}")

    # отсортируем
    def sort_key(ds):
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        ipp = _safe_get(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) >= 3:
            return float(ipp[2])
        return 0
    slices.sort(key=sort_key)

    # проверим, что форма совпадает
    shapes = {(int(_safe_get(s, "Rows", 0)), int(_safe_get(s, "Columns", 0))) for s in slices}
    if len(shapes) != 1:
        # выбросим любые срезы, чья форма отличается от моды
        from collections import Counter
        counts = Counter([(int(_safe_get(s, "Rows", 0)), int(_safe_get(s, "Columns", 0))) for s in slices])
        target_shape, _ = counts.most_common(1)[0]
        slices = [s for s in slices if (int(_safe_get(s, "Rows", 0)), int(_safe_get(s, "Columns", 0))) == target_shape]
        if len(slices) < 2:
            raise RuntimeError("All input arrays must have the same shape (after filtering <2 slices left)")

    # соберём объём
    vol = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    # в HU
    slope = float(_safe_get(slices[0], "RescaleSlope", 1.0))
    intercept = float(_safe_get(slices[0], "RescaleIntercept", 0.0))
    vol = vol * slope + intercept

    # spacing
    try:
        py, px = _float_tuple(_safe_get(slices[0], "PixelSpacing", (1.0, 1.0)))  # (row, col) -> (y, x)
    except Exception:
        py = px = 1.0

    # dz из позиций либо SliceThickness
    dz = None
    try:
        z0 = float(_safe_get(slices[0], "ImagePositionPatient", (0,0,0))[2])
        z1 = float(_safe_get(slices[1], "ImagePositionPatient", (0,0,0))[2])
        dz = abs(z1 - z0)
    except Exception:
        pass
    if not dz or dz <= 0:
        dz = float(_safe_get(slices[0], "SliceThickness", 1.0))

    spacing = (float(dz), float(py), float(px))  # (z,y,x)
    return vol.astype(np.float32), spacing


def _read_dicom_series_sitk(series_dir) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    """SITK fallback: series_dir может быть str или Path."""
    assert sitk is not None, "Install SimpleITK for DICOM fallback"
    series_dir = Path(series_dir)  # <<< ВАЖНО

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not series_ids:
        files = [str(p) for p in series_dir.rglob("*.dcm")]
        if not files:
            files = [str(p) for p in series_dir.rglob("*") if p.is_file()]
        if not files:
            raise RuntimeError(f"No DICOM files for SimpleITK in {series_dir}")
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        best_id, best_len = None, -1
        for sid in series_ids:
            file_names = reader.GetGDCMSeriesFileNames(str(series_dir), sid)
            if len(file_names) > best_len:
                best_len = len(file_names); best_id = sid
        files = reader.GetGDCMSeriesFileNames(str(series_dir), best_id)
        reader.SetFileNames(files)
        img = reader.Execute()

    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)
    sp = img.GetSpacing()  # (x,y,z)
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
    return arr, spacing


def _read_dicom_series(series_dir: Path) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    # сначала пробуем pydicom (быстрее), при проблемах — SimpleITK
    force_sitk = os.environ.get("FORCE_SITK", "0").lower() in {"1", "true", "yes"}
    if force_sitk and sitk is not None:
        return _read_dicom_series_sitk(series_dir)

    if pydicom is not None:
        try:
            return _read_dicom_series_pydicom(series_dir)
        except Exception as e:
            if sitk is None:
                raise
            # fallback
            return _read_dicom_series_sitk(series_dir)
    elif sitk is not None:
        return _read_dicom_series_sitk(series_dir)
    else:
        raise RuntimeError("Install pydicom or SimpleITK to read DICOM")


# --------------- Common preprocessing ---------------

def window_hu(vol: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-6)
    return vol

def resample_iso(vol: np.ndarray, src_spacing: Tuple[float,float,float], tgt_spacing: Tuple[float,float,float]) -> np.ndarray:
    # если целевой шаг почти равен исходному, пропускаем ресэмпл
    if tgt_spacing is None:
        return vol
    eps = 1e-6
    zf = [src_spacing[i] / max(tgt_spacing[i], eps) for i in range(3)]
    if all(abs(f - 1.0) < 1e-3 for f in zf):
        return vol
    return zoom(vol, zoom=zf, order=1)


def pad_or_crop_center(vol: np.ndarray, target_shape: Tuple[int,int,int]) -> np.ndarray:
    D,H,W = vol.shape
    tD,tH,tW = target_shape
    # pad
    pad_z = max(0, tD - D)
    pad_y = max(0, tH - H)
    pad_x = max(0, tW - W)
    if pad_z or pad_y or pad_x:
        vol = np.pad(vol,
                     ((pad_z//2, pad_z - pad_z//2),
                      (pad_y//2, pad_y - pad_y//2),
                      (pad_x//2, pad_x - pad_x//2)),
                     mode="constant", constant_values=0.0)
        D,H,W = vol.shape
    # crop center
    sz = max(0, (D - tD) // 2)
    sy = max(0, (H - tH) // 2)
    sx = max(0, (W - tW) // 2)
    return vol[sz:sz+tD, sy:sy+tH, sx:sx+tW]

def load_volume(path: Union[str, Path],
                path_type: str,
                hu_window: Tuple[float,float],
                target_spacing: Optional[Tuple[float,float,float]],
                target_shape: Tuple[int,int,int]) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    path = Path(path)
    if path_type == "nifti":
        vol, sp = _read_nifti(path)
    elif path_type == "dicom":
        vol, sp = _read_dicom_series(path)
    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    vol = window_hu(vol, hu_window[0], hu_window[1])
    if target_spacing is not None:
        vol = resample_iso(vol, sp, target_spacing)
        sp = target_spacing
    vol = pad_or_crop_center(vol, target_shape)
    return vol.astype(np.float32), sp
