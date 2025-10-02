from pathlib import Path
from typing import List
from ctlib_binary.create_dataset.study_types import StudyRow
from ctlib_binary.create_dataset import path as P

def _list_dirs(d: Path) -> List[Path]:
    return [p for p in d.iterdir() if p.is_dir()]

def build() -> List[StudyRow]:
    rows: List[StudyRow] = []
    # НОРМА
    norma = P.CUSTOM_DIR / "norma_anon"
    if norma.exists():
        for sd in _list_dirs(norma):
            rows.append(StudyRow(
                dataset="CUSTOM",
                study_key=f"CUSTOM_{sd.name}",
                path=str(sd.resolve()),
                path_type="dicom",
                label=0,
                source_category="norma",
            ))
    # ПАТОЛОГИИ
    pneumo = P.CUSTOM_DIR / "pneumonia_anon"
    if pneumo.exists():
        for sd in _list_dirs(pneumo):
            rows.append(StudyRow(
                dataset="CUSTOM",
                study_key=f"CUSTOM_{sd.name}",
                path=str(sd.resolve()),
                path_type="dicom",
                label=1,
                source_category="pneumonia",
            ))
    ptx = P.CUSTOM_DIR / "pneumotorax_anon"
    if ptx.exists():
        for sd in _list_dirs(ptx):
            rows.append(StudyRow(
                dataset="CUSTOM",
                study_key=f"CUSTOM_{sd.name}",
                path=str(sd.resolve()),
                path_type="dicom",
                label=1,
                source_category="pneumothorax",
            ))
    return rows

if __name__ == "__main__":
    out = build()
    from ctlib_binary.create_dataset.utils_io import write_registry_csv
    write_registry_csv(out, P.DATA_OUT_DIR / "custom.csv")
