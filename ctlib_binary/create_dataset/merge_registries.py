from pathlib import Path
from typing import List
import pandas as pd
from ctlib_binary.create_dataset.utils_io import write_registry_csv
from ctlib_binary.create_dataset.study_types import StudyRow
from ctlib_binary.create_dataset import path as P
from ctlib_binary.create_dataset.build_registry_covid19_1110 import build as build_c19
from ctlib_binary.create_dataset.build_registry_mosmed_vii import build as build_vii
from ctlib_binary.create_dataset.build_registry_ldct_lungcr import build as build_lungcr
from ctlib_binary.create_dataset.build_registry_custom import build as build_custom

def merge() -> List[StudyRow]:
    rows: List[StudyRow] = []
    rows.extend(build_c19())
    rows.extend(build_vii())
    rows.extend(build_lungcr())
    rows.extend(build_custom())
    # Удалим дубликаты по study_key на всякий случай
    uniq = {}
    for r in rows:
        if r.study_key not in uniq:
            uniq[r.study_key] = r
    return list(uniq.values())

if __name__ == "__main__":
    merged = merge()
    write_registry_csv(merged, P.MERGED_REGISTRY_CSV)
    print(f"Wrote {len(merged)} rows to {P.MERGED_REGISTRY_CSV}")
