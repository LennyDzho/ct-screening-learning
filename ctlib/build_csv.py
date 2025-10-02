import csv
from pathlib import Path

from ctlib.config import PROTOCOL_DIR, EXTRACT_DIR, LESIONS_CSV
from ctlib.match_slice import find_best_slice_for_lesion
from ctlib.protocols import read_all_protocols



def _study_dir_for(study_key: str) -> Path:
    return EXTRACT_DIR / study_key

def build_lesions_csv(protocols_dir: Path = PROTOCOL_DIR, out_csv: Path = LESIONS_CSV) -> int:
    """
    Создаёт итоговый CSV с очагами и сопоставленными DICOM-срезами.
    Возвращает количество записанных строк.
    """
    records = read_all_protocols(protocols_dir)

    header = [
        # ключи исследования
        "study_key",
        # координаты в изображении и глубина
        "x_px", "y_px", "z_mm", "z_type",
        # геометрия очага
        "diameter_mm",
        # типы
        "nodule_type", "expert_type",
        # решение эксперта
        "expert_decision", "expert_malignant", "target_malignant",
        # индекс кластера/метки/врач
        "cluster_idx", "mark_idx", "doctor_id",
        # выбранный DICOM-срез
        "series_number", "dicom_path", "z_slice_mm", "z_diff_mm",
        # исходный список кандидатов серий (для отладки)
        "series_candidates",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()

        for rec in records:
            study_key = rec["study_key"]
            z_mm = rec["z_mm"]
            series_candidates = rec.get("series_candidates") or []

            # пропустим записи без z_mm
            if z_mm is None:
                continue

            study_dir = _study_dir_for(study_key)
            if not study_dir.exists():
                # исследования пока не распакованы/отсутствуют — можно логировать, но сейчас просто пропустим
                continue

            match = find_best_slice_for_lesion(
                study_dir=study_dir,
                z_mm=float(z_mm),
                series_candidates=series_candidates
            )

            if match is None:
                # Не удалось сопоставить срез — пропустим (или можно писать без dicom_path)
                continue

            target = None
            # Приоритет — expert_malignant, если он есть (True/False), иначе None.
            if rec.get("expert_malignant") is True:
                target = 1
            elif rec.get("expert_malignant") is False:
                target = 0
            # если None — можно пропустить, чтобы в обучении были только верифицированные
            if target is None:
                continue

            row = {
                "study_key": study_key,

                "x_px": rec.get("x_px"),
                "y_px": rec.get("y_px"),
                "z_mm": rec.get("z_mm"),
                "z_type": rec.get("z_type"),

                "diameter_mm": rec.get("diameter_mm"),

                "nodule_type": rec.get("nodule_type"),
                "expert_type": rec.get("expert_type"),

                "expert_decision": rec.get("expert_decision"),
                "expert_malignant": rec.get("expert_malignant"),
                "target_malignant": target,

                "cluster_idx": rec.get("cluster_idx"),
                "mark_idx": rec.get("mark_idx"),
                "doctor_id": rec.get("doctor_id"),

                "series_number": match["series_number"],
                "dicom_path": str(match["dicom_path"]),
                "z_slice_mm": match["z_slice_mm"],
                "z_diff_mm": match["z_diff_mm"],

                "series_candidates": ",".join(str(s) for s in series_candidates) if series_candidates else "",
            }
            wr.writerow(row)
            n_written += 1

    return n_written

if __name__ == "__main__":
    n = build_lesions_csv()
    print(f"written: {n} rows -> {LESIONS_CSV}")
