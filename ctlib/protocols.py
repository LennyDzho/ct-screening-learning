import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Нормализация типов очагов (варианты с подчёркиваниями и т.п.)
_TYPE_ALIASES = {
    "#0S": "#0S",
    "#0_S": "#0S",
    "#1PS": "#1PS",
    "#1_PS": "#1PS",
    "#2GG": "#2GG",
    "#2_GG": "#2GG",
}

def _norm_type(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().upper()
    return _TYPE_ALIASES.get(s, s)

def _parse_series_list(series_no_raw: Optional[Any]) -> List[int]:
    if series_no_raw is None or series_no_raw == "":
        return []
    if isinstance(series_no_raw, (int, float)):
        try:
            return [int(series_no_raw)]
        except Exception:
            return []
    parts = [p.strip() for p in str(series_no_raw).split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out

def load_protocol(path: Path) -> Dict[str, Any]:
    """
    Читает JSON, устойчив к UTF-8 BOM.
    """
    # UTF-8 with BOM → используем utf-8-sig
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def iter_lesions_from_protocol(protocol_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = protocol_json.get("ids", {})
    accession = ids.get("accession number")
    study_id = ids.get("study id")
    study_key = f"{accession}_{study_id}" if accession and study_id else (accession or study_id or "")

    nodules = protocol_json.get("nodules")
    if not isinstance(nodules, list):
        return []

    out: List[Dict[str, Any]] = []
    for cluster_idx, cluster in enumerate(nodules):
        if not isinstance(cluster, list):
            continue
        for mark_idx, nodule in enumerate(cluster):
            if not isinstance(nodule, dict):
                continue
            for doctor_id, payload in nodule.items():
                if not isinstance(doctor_id, str):
                    continue
                if not isinstance(payload, dict):
                    continue

                x = payload.get("x")
                y = payload.get("y")
                z = payload.get("z")
                z_type = payload.get("z type")
                diameter = payload.get("diameter (mm)")
                series_candidates = _parse_series_list(payload.get("series no"))
                ntype = _norm_type(payload.get("type"))

                expert_decision_raw = payload.get("expert decision")
                if isinstance(expert_decision_raw, list) and expert_decision_raw:
                    expert = expert_decision_raw[0]
                elif isinstance(expert_decision_raw, dict):
                    expert = expert_decision_raw
                else:
                    expert = {}

                expert_type = _norm_type(expert.get("type"))
                expert_decision = expert.get("decision")
                expert_malignant = expert.get("malignant")

                out.append({
                    "study_key": study_key,
                    "cluster_idx": cluster_idx,
                    "mark_idx": mark_idx,
                    "doctor_id": doctor_id,
                    "x_px": x,
                    "y_px": y,
                    "z_mm": z,
                    "z_type": z_type,
                    "diameter_mm": diameter,
                    "series_candidates": series_candidates,
                    "nodule_type": ntype,
                    "expert_type": expert_type,
                    "expert_decision": expert_decision,
                    "expert_malignant": expert_malignant,
                })
    return out

def read_all_protocols(protocols_dir: Path) -> List[Dict[str, Any]]:
    print(protocols_dir)
    out: List[Dict[str, Any]] = []
    for p in sorted(protocols_dir.glob("*.json")):
        try:
            proto = load_protocol(p)
        except Exception as e:
            print(f"[protocols] load failed: {p} → {e}")
            continue
        try:
            out.extend(iter_lesions_from_protocol(proto))
        except Exception as e:
            print(f"[protocols] parse failed: {p} → {e}")
            continue
    return out
