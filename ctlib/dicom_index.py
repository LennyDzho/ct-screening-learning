from pathlib import Path
from typing import Dict, List, Any, Optional
import pydicom

SliceEntry = Dict[str, Any]  # {"path": Path, "z_mm": float, "instance_number": Optional[int]}
SeriesIndex = Dict[int, List[SliceEntry]]  # series_number -> list of slices

def _read_z_mm(ds: pydicom.dataset.FileDataset) -> Optional[float]:
    # Предпочтительно ImagePositionPatient (0020,0032) [x,y,z]
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp and len(ipp) == 3:
        try:
            return float(ipp[2])
        except Exception:
            pass
    # Фолбэк: SliceLocation (0020,1041)
    sl = getattr(ds, "SliceLocation", None)
    if sl is not None:
        try:
            return float(sl)
        except Exception:
            pass
    return None

def build_series_index(study_dir: Path) -> SeriesIndex:
    """
    Строит индекс серий исследования: {SeriesNumber: [ {path, z_mm, instance_number}, ... ]}
    """
    series: SeriesIndex = {}
    for dcm_path in sorted(study_dir.glob("*")):
        if not dcm_path.is_file():
            continue
        # Файлы могут быть без расширения или с .dcm/CT_*. Берём всё и пытаемся прочитать DICOM.
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        except Exception:
            continue

        series_number = getattr(ds, "SeriesNumber", None)
        if series_number is None:
            # без SeriesNumber поместим в виртуальную серию 0
            series_number = 0

        z_mm = _read_z_mm(ds)
        instance_number = getattr(ds, "InstanceNumber", None)

        entry: SliceEntry = {
            "path": dcm_path,
            "z_mm": z_mm,
            "instance_number": instance_number,
        }
        series.setdefault(int(series_number), []).append(entry)

    # Отсортируем срезы в каждой серии по z (если есть), иначе по InstanceNumber
    for sn, slices in series.items():
        slices.sort(key=lambda e: (
            float("inf") if e["z_mm"] is None else e["z_mm"],
            float("inf") if e["instance_number"] is None else int(e["instance_number"])
        ))
    return series
