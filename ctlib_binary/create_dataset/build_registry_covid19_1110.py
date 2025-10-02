from pathlib import Path
from typing import List, Iterator

from ctlib_binary.create_dataset.study_types import StudyRow
from ctlib_binary.create_dataset.utils_io import read_xlsx, write_registry_csv
from ctlib_binary.create_dataset import path as P

def _category_to_label(cat: str) -> int:
    # CT-0 => 0, CT-1..4 => 1
    return 0 if cat.strip().upper() == "CT-0" else 1

def build() -> List[StudyRow]:
    xlsx = P.COVID19_1110_DIR / "dataset_registry.xlsx"
    df = read_xlsx(xlsx)
    rows: List[StudyRow] = []
    for _, r in df.iterrows():
        study_id = str(r["study_id"])
        category = str(r["category"]).strip()
        study_file = str(r["study_file"]).strip()  # '/studies/CT-0/study_0001.nii.gz'
        mask_file = str(r.get("mask_file", "")).strip() if "mask_file" in df.columns else ""

        nifti_path = (P.COVID19_1110_DIR / study_file.lstrip("/")).resolve()
        mask_path = (P.COVID19_1110_DIR / mask_file.lstrip("/")).resolve() if mask_file else None

        label = _category_to_label(category)
        study_key = f"COVID19_1110_{study_id}"

        rows.append(StudyRow(
            dataset="COVID19_1110",
            study_key=study_key,
            path=str(nifti_path),
            path_type="nifti",
            label=label,
            source_category=category,
            mask_path=str(mask_path) if mask_path else None,
        ))
    return rows

if __name__ == "__main__":
    out = build()
    from ctlib_binary.create_dataset.utils_io import write_registry_csv, read_xlsx

    write_registry_csv(out, P.DATA_OUT_DIR / "covid19_1110.csv")
