from pathlib import Path
from typing import List
from ctlib_binary.create_dataset.study_types import StudyRow
from ctlib_binary.create_dataset.utils_io import read_xlsx
from ctlib_binary.create_dataset import path as P

def _candidate_study_dir(root: Path, study_uid: str) -> Path:
    # по README структура: studies/studyUID_X/seriesUID_X/*.dcm
    # будем указывать путь до studies/<study_uid>
    return (root / "studies" / study_uid)

def build() -> List[StudyRow]:
    xlsx = P.LDCT_LUNGCR_DIR / "dataset_registry.xlsx"
    df = read_xlsx(xlsx)
    rows: List[StudyRow] = []
    for _, r in df.iterrows():
        uid = str(r["study_instance_anon"]).strip()
        label = int(r["pathology"])
        study_dir = _candidate_study_dir(P.LDCT_LUNGCR_DIR, uid)
        study_key = f"LDCT_LUNGCR_{uid}"
        rows.append(StudyRow(
            dataset="LDCT_LUNGCR",
            study_key=study_key,
            path=str(study_dir.resolve()),
            path_type="dicom",
            label=label,
            source_category=None,
        ))
    return rows

if __name__ == "__main__":
    out = build()
    from ctlib_binary.create_dataset.utils_io import write_registry_csv
    write_registry_csv(out, P.DATA_OUT_DIR / "ldct_lungcr.csv")
