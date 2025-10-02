from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import pandas as pd

from ctlib_binary.create_dataset.study_types import StudyRow
from ctlib_binary.create_dataset import path as P
from ctlib_binary.create_dataset.utils_io import write_registry_csv

# ---------- нормализация и словарь синонимов

def _norm_name(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9а-яё]+", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

COL_CANDIDATES: Dict[str, set[str]] = {
    "patient_uid": {
        "patient_uid","patient","patient_id","patientidx","patient_uid_anon",
        "пациент","идентификатор_пациента","код_пациента",
    },
    "study_sequential_id": {
        "study_sequential_id","study_seq_id","study_id","seq_id","studyindex",
        "номер_исследования","исследование","номер_стадии",
    },
    "study_date": {
        "study_date","date","exam_date","study_dt","дата","дата_исследования",
    },
    "folder": {
        "folder","path","dir","directory","study_folder","folder_rel",
        "папка","каталог","путь",
    },
    "verified_covid": {
        "verified_covid_19","verified_covid19","covid","label","pathology",
        "covid_positive","is_covid","gt","метка","патология",
    },
}

POSITIVES = {"+","1","yes","true","covid","positive","патология","есть","да"}
NEGATIVES = {"-","0","no","false","norma","норма","отрицательно","нет",""}

def _to_bool_label(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    if s in POSITIVES: return 1
    if s in NEGATIVES: return 0
    return 0

# ---------- детект заголовков на листе

def _combine_header_rows(rows: List[List[str]]) -> List[str]:
    """Комбинируем до двух верхних строк в заголовок: если в первой пусто/Unnamed, дополняем второй."""
    if not rows:
        return []
    h1 = [str(x) if pd.notna(x) else "" for x in rows[0]]
    if len(rows) == 1:
        return h1
    h2 = [str(x) if pd.notna(x) else "" for x in rows[1]]
    out = []
    for a, b in zip(h1, h2 + [""] * (len(h1) - len(h2))):
        a_norm = a.strip()
        b_norm = b.strip()
        if a_norm and not a_norm.lower().startswith("unnamed"):
            out.append(a_norm)
        elif b_norm and not b_norm.lower().startswith("unnamed"):
            out.append(b_norm)
        else:
            out.append("")
    return out

def _find_header_row(df_raw: pd.DataFrame, max_scan: int = 20) -> Tuple[int, List[str]]:
    """
    Пытаемся найти строку(и) заголовков:
    1) смотрим пару строк (i) и (i+1), комбинируем их;
    2) нормализуем и ищем наличие хотя бы одного из колонок-пути (folder/path/dir/папка...) и метки.
    """
    folder_like = {_norm_name(x) for x in ["folder","path","dir","directory","study_folder","folder_rel","папка","каталог","путь"]}
    label_like  = {_norm_name(x) for x in ["verified_covid_19","verified_covid19","covid","label","pathology","метка","патология"]}

    rows = min(max_scan, len(df_raw))
    for i in range(rows):
        # собираем 1 или 2 строки для заголовка
        base = df_raw.iloc[i].tolist()
        header = _combine_header_rows([base] + ([df_raw.iloc[i+1].tolist()] if i+1 < len(df_raw) else []))
        norm = [_norm_name(x) for x in header]
        if not norm or all(c.startswith("unnamed") or c == "" for c in norm):
            continue
        if (set(norm) & folder_like) and (set(norm) & label_like):
            return i, header
    # если не нашли — попробуем взять первую непустую строку
    for i in range(rows):
        header = [str(x) for x in df_raw.iloc[i].tolist()]
        norm = [_norm_name(x) for x in header]
        if any(x and not x.startswith("unnamed") for x in norm):
            return i, header
    return 0, [str(x) for x in df_raw.iloc[0].tolist()]

def _read_registry_any_sheet(xlsx_path: Path) -> pd.DataFrame:
    """Читаем все листы, пытаемся на каждом детектить заголовки. Возвращаем первый, где нашли колонку пути."""
    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as e:
        raise RuntimeError(f"Не удалось открыть Excel: {xlsx_path}\n{e}")

    folder_like = {_norm_name(x) for x in ["folder","path","dir","directory","study_folder","folder_rel","папка","каталог","путь"]}
    for sheet in xls.sheet_names:
        df_try = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, dtype=str)
        header_row, header = _find_header_row(df_try)
        # применим заголовок и обрежем верх
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, dtype=str)
        df = df.iloc[header_row+1:].reset_index(drop=True)
        df.columns = header[:len(df.columns)]
        # нормализованные имена
        norm_cols = [_norm_name(c) for c in df.columns]
        # признак — нашли ли что-то похожее на колонку пути
        if set(norm_cols) & folder_like:
            # сохраним превью для отладки
            preview = df.head(30)
            P.DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
            preview.to_csv(P.DATA_OUT_DIR / "debug_mosmed_vii_preview.csv", index=False)
            return df
    # если ни на одном листе не нашли путь — вернём первый лист как есть (чтобы словить подсказку об отсутствующих колонках)
    return pd.read_excel(xlsx_path, sheet_name=0)

def _map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    norm_map = {c: _norm_name(c) for c in df.columns}
    inv: Dict[str, str] = {}
    for c, n in norm_map.items():
        inv.setdefault(n, c)

    def pick(key: str) -> Optional[str]:
        for cand in COL_CANDIDATES[key]:
            n = _norm_name(cand)
            if n in inv:
                return inv[n]
        return None

    return {
        "patient_uid": pick("patient_uid"),
        "study_sequential_id": pick("study_sequential_id"),
        "study_date": pick("study_date"),
        "folder": pick("folder"),
        "verified_covid": pick("verified_covid"),
    }

# ---------- основная сборка

def build() -> List[StudyRow]:
    xlsx = P.MOSMED_VII_DIR / "dataset_registry.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Файл не найден: {xlsx}\nПроверь MOSMED_VII_DIR в path.py")

    df = _read_registry_any_sheet(xlsx)

    colmap = _map_columns(df)
    # покажем, что именно сопоставилось (для отладки)
    print(colmap.items())

    missing = [k for k, v in colmap.items() if v is None]
    if missing:
        raise RuntimeError(
            "Missing required columns for MosMed VII registry: "
            f"{missing}\n"
            f"paths tried: {xlsx}\n"
            f"Detected columns: {list(df.columns)}\n"
            f"Подсказка: открой data/prepared/debug_mosmed_vii_preview.csv и посмотри, как pandas видит таблицу. "
            "Убедись, что там есть колонка с путём (folder/path/dir/папка/каталог) и колонка метки (verified_covid-19/pathology/метка)."
        )

    rows: List[StudyRow] = []
    for _, r in df.iterrows():
        folder_rel = str(r[colmap["folder"]]).strip() if colmap["folder"] else ""
        if not folder_rel:
            continue
        label_raw = r[colmap["verified_covid"]] if colmap["verified_covid"] else ""
        label = _to_bool_label(label_raw)

        patient_uid = str(r[colmap["patient_uid"]]).strip() if colmap["patient_uid"] else "NA"
        seq_id     = str(r[colmap["study_sequential_id"]]).strip() if colmap["study_sequential_id"] else "NA"
        study_date = str(r[colmap["study_date"]]).strip() if colmap["study_date"] else "NA"

        folder_path = Path(folder_rel)
        if not folder_path.is_absolute():
            folder_path = (P.MOSMED_VII_DIR / folder_path).resolve()

        study_key = f"MOSMED_VII_{patient_uid}_{seq_id}_{study_date}"

        rows.append(StudyRow(
            dataset="MosMed_VII",
            study_key=study_key,
            path=str(folder_path),
            path_type="dicom",
            label=label,
            source_category=None,
        ))
    return rows

if __name__ == "__main__":
    out = build()
    out_csv = P.DATA_OUT_DIR / "mosmed_vii.csv"
    write_registry_csv(out, out_csv)
    print(f"Wrote {len(out)} rows to {out_csv}")
