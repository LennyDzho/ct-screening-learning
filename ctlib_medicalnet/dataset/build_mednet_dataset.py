from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse, json, csv, random, re

from ctlib.config.paths import BASE_DIR

# ---------- нормализация типа узла ----------
_NODULE_TYPE_MAP = {
    "#0S":"solid","#0_S":"solid","#0-S":"solid","0S":"solid","0_S":"solid",
    "#1PS":"part_solid","#1_PS":"part_solid","#1-PS":"part_solid","1PS":"part_solid","1_PS":"part_solid",
    "#2GG":"ground_glass","#2_GG":"ground_glass","#2-GG":"ground_glass","2GG":"ground_glass","2_GG":"ground_glass",
}
def norm_type(s: Optional[str]) -> Optional[str]:
    if not s: return None
    key = str(s).strip().upper().replace(" ", "")
    return _NODULE_TYPE_MAP.get(key)

# ---------- правила «есть патология» ----------
def decision_is_positive(decision: str) -> bool:
    # можно сузить, если нужно: только "confirmed"
    return str(decision).strip().lower() in {"confirmed", "confirmed_partially", "confirmed partially"}

def nodule_is_positive(nod: Dict) -> Tuple[bool, Optional[str]]:
    """
    Возвращает (is_pathology, type_class):
      - True, если есть экспертное решение и оно положительное;
      - тип нормализуем в one-of: solid | part_solid | ground_glass (если указан).
    """
    t = norm_type(nod.get("type"))
    decs = nod.get("expert decision") or nod.get("expert_decision") or []
    ok = any(decision_is_positive(d.get("decision","")) for d in decs if isinstance(d, dict))
    # при желании можно проверять размер:
    # proper = any(bool(d.get("proper size", False)) for d in decs if isinstance(d, dict))
    # ok = ok and proper
    return bool(ok), t

# ---------- парсинг одного JSON ----------
@dataclass
class StudyRow:
    study_id: str
    accession: str
    gender: str
    age: str
    has_pathology: int
    n_nodules: int
    n_solid: int
    n_part_solid: int
    n_ground_glass: int
    types: str          # например: "solid:2;part_solid:1"
    json_path: str
    study_dir: str      # найденная папка DICOM (или "")
    series_hint: str    # что было в json ('series no'), может помочь в поиске

def parse_json_file(p: Path) -> StudyRow | None:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    ids = obj.get("ids", {})
    study_id = str(ids.get("study id") or ids.get("study_id") or "")
    accession = str(ids.get("accession number") or ids.get("accession_number") or "")
    age = str(ids.get("age") or "")
    gender = str(ids.get("gender") or "")

    n_total = 0
    n_pos = 0
    c_s = c_ps = c_gg = 0
    series_hint = ""

    nodules = obj.get("nodules") or []
    for nlist in nodules:
        # элемент nlist — словарь по врачам; берём все непустые
        for annot in (nlist[0] if isinstance(nlist, list) and nlist else nlist):
            pass
        # json из примера — список с одним словарём; заберём его
        if isinstance(nlist, list) and nlist and isinstance(nlist[0], dict):
            dct = nlist[0]
        elif isinstance(nlist, dict):
            dct = nlist
        else:
            continue

        # у словаря dct несколько ключей-врачей; берём первую непустую запись
        rec: Optional[Dict] = None
        for key, item in dct.items():
            if isinstance(item, dict):
                rec = item; break
        if not isinstance(rec, dict):
            continue

        n_total += 1
        pos, t = nodule_is_positive(rec)
        if pos: n_pos += 1
        if t == "solid": c_s += 1
        elif t == "part_solid": c_ps += 1
        elif t == "ground_glass": c_gg += 1

        if not series_hint and isinstance(rec.get("series no"), (str,int)):
            series_hint = str(rec.get("series no"))

    has_path = 1 if n_pos > 0 else 0
    types_str = ";".join([f"solid:{c_s}", f"part_solid:{c_ps}", f"ground_glass:{c_gg}"])
    return StudyRow(
        study_id=study_id, accession=accession, gender=gender, age=age,
        has_pathology=has_path, n_nodules=n_total,
        n_solid=c_s, n_part_solid=c_ps, n_ground_glass=c_gg,
        types=types_str, json_path=str(p), study_dir="", series_hint=series_hint
    )

# ---------- поиск папки с DICOM ----------
def find_study_dir(dicom_root: Path, study_id: str, accession: str) -> str:
    if not dicom_root.exists(): return ""
    candidates: List[Path] = []
    tokens = [t for t in [study_id, accession] if t]
    if not tokens: return ""
    # ищем неглубоким проходом: имя папки или подстрока в пути
    for d in dicom_root.rglob("*"):
        if d.is_dir():
            name = d.name
            if any(t and t in name for t in tokens):
                candidates.append(d)
    # эвристика: предпочесть более длинное совпадение
    def score(d: Path) -> int:
        s = 0
        path_str = str(d)
        for t in tokens:
            if t and t in path_str: s += len(t)
        return s
    if candidates:
        best = max(candidates, key=score)
        return str(best)
    return ""

# ---------- сплит по исследованиям ----------
def group_split(rows: List[StudyRow], val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    studies = list(range(len(rows)))
    rnd = random.Random(seed); rnd.shuffle(studies)
    n = len(studies)
    n_test = max(1, int(round(n * test_ratio))) if n > 5 else max(0, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n > 5 else max(0, int(round(n * val_ratio)))
    te = set(studies[:n_test])
    va = set(studies[n_test:n_test+n_val])
    tr = set(studies[n_test+n_val:])
    tr_i = [i for i in studies if i in tr]
    va_i = [i for i in studies if i in va]
    te_i = [i for i in studies if i in te]
    return tr_i, va_i, te_i

# ---------- сохранение ----------
def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)

def write_txt(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Build MedNet dataset (study-level) from JSON annotations")
    ap.add_argument("--json-root", type=str, default=BASE_DIR / "extracted_data")
    ap.add_argument("--dicom-root", type=str, default=BASE_DIR / "extracted_data" / ".extracted")
    ap.add_argument("--out", type=str, default=BASE_DIR / "extracted_data" / "dataset_mednet")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-dicom", action="store_true", help="отбрасывать исследования, для которых не найдена папка с DICOM")
    args = ap.parse_args()

    json_root = Path(args.json_root)
    dicom_root = Path(args.dicom_root)
    out_root  = Path(args.out)

    json_files = sorted(json_root.rglob("*.json"))
    rows: List[StudyRow] = []
    for p in json_files:
        r = parse_json_file(p)
        if not r: continue
        r.study_dir = find_study_dir(dicom_root, r.study_id, r.accession)
        if args.require_dicom and not r.study_dir:
            continue
        rows.append(r)

    # фильтр: должны быть оба класса для обучения
    n_pos = sum(r.has_pathology for r in rows)
    n_neg = len(rows) - n_pos
    print(f"[INFO] parsed studies: total={len(rows)}  pos={n_pos}  neg={n_neg}")

    # сплит
    tr, va, te = group_split(rows, args.val_ratio, args.test_ratio, seed=args.seed)

    # manifest_all.csv
    write_csv(out_root/"manifest_all.csv", [r.__dict__ for r in rows])

    # по сплитам
    def sel(idxs: List[int]) -> List[StudyRow]: return [rows[i] for i in idxs]
    splits = {
        "train": sel(tr),
        "val":   sel(va),
        "test":  sel(te),
    }

    for split_name, lst in splits.items():
        # csv с метаданными
        write_csv(out_root/f"{split_name}.csv", [
            {
                "study_id": r.study_id,
                "study_dir": r.study_dir,
                "label": r.has_pathology,
                "n_nodules": r.n_nodules,
                "n_solid": r.n_solid, "n_part_solid": r.n_part_solid, "n_ground_glass": r.n_ground_glass,
                "types": r.types,
                "json_path": r.json_path,
            } for r in lst
        ])
        # простые списки путей (удобно под даталоадер)
        write_txt(out_root/f"{split_name}.txt", [r.study_dir for r in lst if r.study_dir])

    # сводка
    print("[DONE] saved dataset manifest to:", out_root)
    for k, lst in splits.items():
        pos = sum(r.has_pathology for r in lst); neg = len(lst) - pos
        print(f"  {k:5s}: n={len(lst)}  pos={pos}  neg={neg}")

if __name__ == "__main__":
    main()
