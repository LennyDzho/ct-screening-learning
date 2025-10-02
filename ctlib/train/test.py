from pathlib import Path
import csv
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().resolve().parents[1]
LESIONS_CSV = PROJECT_ROOT / "data" / "dataset_out" / "lesions.csv"

def main():
    if not LESIONS_CSV.exists():
        print(f"Файл не найден: {LESIONS_CSV}")
        return

    rows = []
    with LESIONS_CSV.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    print(f"Всего строк в lesions.csv: {len(rows)}")
    by_target = Counter(int(r["target_malignant"]) for r in rows)
    print(f"Классы: benign(0)={by_target.get(0,0)}, malignant(1)={by_target.get(1,0)}")

    studies = Counter(r["study_key"] for r in rows)
    print(f"Исследований покрыто: {len(studies)} (среднее очагов/исследование: {len(rows)/max(1,len(studies)):.2f})")

    # Проверка путей DICOM
    missing_paths = sum(1 for r in rows if not Path(r["dicom_path"]).exists())
    print(f"DICOM-пути отсутствуют (должно быть 0): {missing_paths}")

    # Пример: топ-5 исследований по количеству очагов
    print("Топ-5 исследований по числу очагов:")
    for sk, cnt in studies.most_common(5):
        print(f"  {sk}: {cnt}")

if __name__ == "__main__":
    main()
