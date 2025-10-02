import csv
from pathlib import Path
from typing import Iterable, List
import pandas as pd
from ctlib_binary.create_dataset.study_types import StudyRow

def read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def write_registry_csv(rows: Iterable[StudyRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "study_key", "path", "path_type",
                "label", "source_category", "mask_path"
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "dataset": r.dataset,
                "study_key": r.study_key,
                "path": r.path,
                "path_type": r.path_type,
                "label": r.label,
                "source_category": r.source_category or "",
                "mask_path": r.mask_path or "",
            })
