import argparse
import time
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch

from ctlib_binary.dataio.volume_reader import (
    load_volume,
    window_hu,
    pad_or_crop_center,
)
from ctlib_binary.models.medicalnet_r3d18 import R3D18Binary

# --- optional libs for UID/series/slice enumeration ---
try:
    import pydicom
except Exception:
    pydicom = None
try:
    import SimpleITK as sitk
except Exception:
    sitk = None

PROJECT_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = PROJECT_DIR / "tmp_extract"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------ utils ------------------------

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))

def _extract_if_zip(p: Path, workdir: Path) -> Path:
    if p.is_dir():
        root = p
    elif p.suffix.lower() == ".zip":
        root = workdir / p.stem
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(p), "r") as zf:
                zf.extractall(root)
        # автопровал при единственной вложенной папке
        while True:
            subs = [d for d in root.iterdir() if d.is_dir()]
            files = [f for f in root.iterdir() if f.is_file()]
            if len(subs) == 1 and not files:
                root = subs[0]
                continue
            break
    else:
        root = p
    return root

def _looks_like_study_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    files = [p for p in d.iterdir() if p.is_file()]
    if len(files) < 3:
        return False
    if pydicom is not None:
        for f in files[:10]:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                if getattr(ds, "SOPClassUID", None) is not None or getattr(ds, "StudyInstanceUID", None) is not None:
                    return True
            except Exception:
                continue
    return True

def _iter_study_dirs_from_container(container: Path) -> List[Path]:
    subdirs = [d for d in container.iterdir() if d.is_dir()]
    studies = [d for d in subdirs if _looks_like_study_dir(d)]
    if studies:
        return studies
    return [container] if _looks_like_study_dir(container) else []

def _rnd_tuple(t): return tuple(round(float(x), 5) for x in t)

def _group_key_for_series(ds) -> tuple:
    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))
    pxsp = _rnd_tuple(getattr(ds, "PixelSpacing", [1.0, 1.0])) if hasattr(ds, "PixelSpacing") else (1.0, 1.0)
    iop  = _rnd_tuple(getattr(ds, "ImageOrientationPatient", [1,0,0,0,1,0])) if hasattr(ds, "ImageOrientationPatient") else (1,0,0,0,1,0)
    suid = str(getattr(ds, "SeriesInstanceUID", ""))
    return (rows, cols, pxsp, iop, suid)

def _choose_best_series_uids(study_dir: Path) -> Tuple[str, str]:
    # pydicom группировка
    if pydicom is not None:
        parsed = []
        for fp in study_dir.rglob("*"):
            if not fp.is_file(): continue
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                if hasattr(ds, "SOPClassUID") or hasattr(ds, "StudyInstanceUID"):
                    parsed.append(ds)
            except Exception:
                pass
        if parsed:
            groups: Dict[tuple, List] = {}
            for ds in parsed:
                groups.setdefault(_group_key_for_series(ds), []).append(ds)
            best = max(groups.keys(), key=lambda k: len(groups[k]))
            one = groups[best][0]
            return str(getattr(one, "StudyInstanceUID", "") or ""), str(getattr(one, "SeriesInstanceUID", "") or "")
    # SITK fallback
    if sitk is not None:
        reader = sitk.ImageSeriesReader()
        sids = reader.GetGDCMSeriesIDs(str(study_dir)) or []
        best_id, best_len = "", -1
        for sid in sids:
            files = reader.GetGDCMSeriesFileNames(str(study_dir), sid)
            if len(files) > best_len:
                best_len, best_id = len(files), sid
        return "", best_id
    return "", ""

def _list_files_for_series(study_dir: Path, series_uid: str) -> List[Path]:
    """Файлы серии в корректном порядке (SITK)."""
    assert sitk is not None, "SimpleITK is required for per-slice mode"
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(study_dir), series_uid)
    return [Path(f) for f in files]

def _read_series_array_and_spacing(study_dir: Path, series_uid: str) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    assert sitk is not None, "SimpleITK is required for per-slice mode"
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(study_dir), series_uid)
    reader.SetFileNames(files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W), по порядку files
    sp = img.GetSpacing()  # (x,y,z)
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))  # (z,y,x)
    return arr, spacing

def _sop_uids_for_files(files: List[Path]) -> List[str]:
    uids = []
    if pydicom is not None:
        for f in files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                uids.append(str(getattr(ds, "SOPInstanceUID", "")) or "")
            except Exception:
                uids.append("")
    else:
        uids = [""] * len(files)
    return uids

def _make_subvolume_centered_on_slice(vol: np.ndarray, z: int, target_dhw: Tuple[int,int,int]) -> np.ndarray:
    """Берём окно глубины D, центрированное на срезе z, с нулевой подпадкой по краям; H,W приводим pad_or_crop_center."""
    D, H, W = vol.shape
    tD, tH, tW = target_dhw
    half = tD // 2
    # подпадка по Z
    pad_before = max(0, half - z)
    pad_after  = max(0, (z + half + 1) - D) if (tD % 2 == 1) else max(0, (z + half) - D + 1)
    if pad_before or pad_after:
        vol_padded = np.pad(vol, ((pad_before, pad_after), (0,0), (0,0)), mode="constant", constant_values=0.0)
        z_shift = z + pad_before
    else:
        vol_padded = vol
        z_shift = z
    start = z_shift - half
    if tD % 2 == 1:
        stop = start + tD
    else:
        # чётная глубина — окно [z-half, z+half)
        stop = start + tD
    sub = vol_padded[start:stop, :, :]
    # H,W к целевому размеру
    sub = pad_or_crop_center(sub, (tD, tH, tW))
    return sub.astype(np.float32)

# ------------------------ inference ------------------------

def run_infer(
    inputs: List[str],
    weights: Path,
    out_xlsx: Path,
    hu_min: float = -600.0,
    hu_max: float = 1500.0,
    target_shape: Tuple[int, int, int] = (96, 192, 192),
    device_str: Optional[str] = None,
    threshold: float = 0.49,
    split_by_series: bool = False,
    per_slice: bool = True,
) -> None:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # модель
    model = R3D18Binary(pretrained=False).to(device)
    ckpt = torch.load(str(weights), map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.to(memory_format=torch.channels_last_3d)

    rows = []

    # входы → контейнеры
    containers: List[Path] = []
    for name in inputs:
        p = (PROJECT_DIR / name).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Вход не найден: {p}")
        containers.append(_extract_if_zip(p, EXTRACT_DIR))

    # исследования (подпапки первого уровня или сам контейнер)
    study_dirs: List[Path] = []
    for c in containers:
        study_dirs.extend(_iter_study_dirs_from_container(c))

    for study_path in study_dirs:
        # выясняем «лучшую» серию (и/или все серии)
        study_uid_best, series_uid_best = _choose_best_series_uids(study_path)

        if per_slice:
            # обязательно через SITK: прочитать массив и список файлов в правильном порядке
            if not series_uid_best:
                # если не нашли, но SITK доступен — берём первую серию
                if sitk is None:
                    raise RuntimeError("per-slice режим требует SimpleITK")
                reader = sitk.ImageSeriesReader()
                sids = reader.GetGDCMSeriesIDs(str(study_path)) or []
                if not sids:
                    continue
                series_uid_best = sids[0]

            files = _list_files_for_series(study_path, series_uid_best)
            sop_uids = _sop_uids_for_files(files)
            raw_vol, spacing = _read_series_array_and_spacing(study_path, series_uid_best)  # (D,H,W)
            # HU-окно
            raw_vol = window_hu(raw_vol, hu_min, hu_max)  # 0..1

            D = raw_vol.shape[0]
            for z in range(D):
                t0 = time.perf_counter()
                status = "Success"
                prob = np.nan
                pred = ""

                try:
                    sub = _make_subvolume_centered_on_slice(raw_vol, z, target_shape)  # (tD,tH,tW)
                    x = torch.from_numpy(sub).unsqueeze(0).unsqueeze(0)                 # [1,1,D,H,W]
                    x = x.contiguous(memory_format=torch.channels_last_3d).to(device, non_blocking=True)
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                        logit = model(x)
                    prob = float(_sigmoid(logit)[0].item())
                    pred = int(prob >= threshold)
                except Exception:
                    status = "Failure"

                dt = float(time.perf_counter() - t0)
                rows.append({
                    "path_to_study": str(study_path),
                    "study_uid": study_uid_best,
                    "series_uid": series_uid_best,
                    "slice_index": z,                         # доп. колонка
                    "sop_instance_uid": sop_uids[z] if z < len(sop_uids) else "",  # доп. колонка
                    "probability_of_pathology": float(prob) if prob == prob else np.nan,
                    "pathology": int(pred) if pred != "" else "",
                    "processing_status": status,
                    "time_of_processing": dt,
                })
        else:
            # старые режимы: 1 строка на исследование или по сериям
            if split_by_series:
                # перечислить серии
                if sitk is None:
                    raise RuntimeError("--split_by_series требует SimpleITK")
                reader = sitk.ImageSeriesReader()
                sids = reader.GetGDCMSeriesIDs(str(study_path)) or []
                series_list = sids if sids else ([series_uid_best] if series_uid_best else [])
            else:
                series_list = [series_uid_best]

            for series_uid in series_list:
                t0 = time.perf_counter()
                status = "Success"
                prob = np.nan
                pred = ""
                try:
                    vol, _ = load_volume(
                        path=study_path,
                        path_type="dicom",
                        hu_window=(hu_min, hu_max),
                        target_spacing=None,
                        target_shape=target_shape,
                    )
                    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
                    x = x.contiguous(memory_format=torch.channels_last_3d).to(device, non_blocking=True)
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                        logit = model(x)
                    prob = float(_sigmoid(logit)[0].item())
                    pred = int(prob >= threshold)
                except Exception:
                    status = "Failure"
                dt = float(time.perf_counter() - t0)
                rows.append({
                    "path_to_study": str(study_path),
                    "study_uid": study_uid_best,
                    "series_uid": series_uid or series_uid_best,
                    "probability_of_pathology": float(prob) if prob == prob else np.nan,
                    "pathology": int(pred) if pred != "" else "",
                    "processing_status": status,
                    "time_of_processing": dt,
                })

    # колонки:
    base_cols = [
        "path_to_study",
        "study_uid",
        "series_uid",
        "probability_of_pathology",
        "pathology",
        "processing_status",
        "time_of_processing",
    ]
    if per_slice:
        cols = ["path_to_study", "study_uid", "series_uid", "slice_index", "sop_instance_uid",
                "probability_of_pathology", "pathology", "processing_status", "time_of_processing"]
    else:
        cols = base_cols

    out_xlsx = (out_xlsx if Path(out_xlsx).is_absolute() else (PROJECT_DIR / out_xlsx)).resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=cols)
    df.to_excel(out_xlsx, index=False)
    print(f"[INFO] saved: {out_xlsx} (rows: {len(df)})")

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser("Hackathon binary classifier inference")
    ap.add_argument("inputs", nargs="+", help="имена zip/папок в корне проекта; внутри — ПАПКИ-исследования")
    ap.add_argument("--weights", type=Path, default=PROJECT_DIR / "runs_binary" / "binary_r3d18" / "best.pth")
    ap.add_argument("--out_xlsx", type=Path, default=PROJECT_DIR / "result" / "hackathon_results.xlsx")
    ap.add_argument("--hu_min", type=float, default=-600.0)
    ap.add_argument("--hu_max", type=float, default=1500.0)
    ap.add_argument("--shape", type=str, default="96,192,192", help="D,H,W")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (auto)")
    ap.add_argument("--threshold", type=float, default=0.5, help="порог для бинаризации")
    ap.add_argument("--split_by_series", action="store_true", help="отдельная строка на каждую серию в исследовании")
    # ap.add_argument("--per_slice", action="store_true", help="ОТДЕЛЬНАЯ СТРОКА НА КАЖДЫЙ DICOM-СРЕЗ (добавит slice_index и sop_instance_uid)")
    args = ap.parse_args()

    td, th, tw = map(int, args.shape.split(","))
    run_infer(
        inputs=args.inputs,
        weights=args.weights,
        out_xlsx=args.out_xlsx,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_shape=(td, th, tw),
        device_str=args.device,
        threshold=args.threshold,
        split_by_series=args.split_by_series,
        # per_slice=args.per_slice,
    )

if __name__ == "__main__":
    main()
