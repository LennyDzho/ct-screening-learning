from pathlib import Path
import csv, sys
from ctlib_binary.create_dataset import path as P

def main() -> int:
    missing = 0
    with P.MERGED_REGISTRY_CSV.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = Path(row["path"])
            if not p.exists():
                missing += 1
                print(f"[MISSING] {row['study_key']} -> {p}", file=sys.stderr)
            m = row.get("mask_path", "")
            if m:
                mp = Path(m)
                if not mp.exists():
                    print(f"[WARN MASK] {row['study_key']} mask not found: {mp}", file=sys.stderr)
    if missing == 0:
        print("All study paths exist.")
    else:
        print(f"Missing {missing} study paths.", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
