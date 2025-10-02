from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from .dicom_index import build_series_index

MatchResult = Dict[str, Any]
# {"series_number": int, "dicom_path": Path, "z_slice_mm": float, "z_diff_mm": float}

def find_best_slice_for_lesion(
    study_dir: Path,
    z_mm: float,
    series_candidates: Optional[List[int]] = None,
) -> Optional[MatchResult]:
    """
    Ищет лучший срез для очага с глубиной z_mm (мм).
    Если series_candidates непуст, приоритетно ищем в них (в порядке указания).
    """
    series_index = build_series_index(study_dir)
    if not series_index:
        return None

    def _best_in_series(sn: int) -> Optional[MatchResult]:
        slices = series_index.get(sn)
        if not slices:
            return None
        best = None
        best_diff = float("inf")
        best_z = None
        best_path = None
        for s in slices:
            z_s = s.get("z_mm")
            if z_s is None:
                continue
            diff = abs(z_s - z_mm)
            if diff < best_diff:
                best_diff = diff
                best = s
                best_z = z_s
                best_path = s["path"]
        if best is None:
            return None
        return {
            "series_number": sn,
            "dicom_path": best_path,
            "z_slice_mm": best_z,
            "z_diff_mm": best_diff,
        }

    # 1) Пробуем среди кандидатов
    if series_candidates:
        for sn in series_candidates:
            res = _best_in_series(int(sn))
            if res:
                return res

    # 2) Иначе среди всех серий
    across_best = None
    across_diff = float("inf")
    for sn in series_index.keys():
        res = _best_in_series(sn)
        if res and res["z_diff_mm"] < across_diff:
            across_diff = res["z_diff_mm"]
            across_best = res

    return across_best
