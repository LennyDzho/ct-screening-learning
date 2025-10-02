from pathlib import Path
from typing import Dict, Any, List, Tuple
import csv

def load_registry(studies_csv: Path) -> Dict[str, Dict[str, Any]]:
    reg: Dict[str, Dict[str, Any]] = {}
    with studies_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            reg[row["study_key"]] = row
    return reg

def load_split_lists(train_txt: Path, val_txt: Path) -> Tuple[List[str], List[str]]:
    def read_list(p: Path) -> List[str]:
        with p.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return read_list(train_txt), read_list(val_txt)
