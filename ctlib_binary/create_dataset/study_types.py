from dataclasses import dataclass
from typing import Optional, Literal

Label = Literal[0, 1]  # 0 = норма, 1 = патология

@dataclass
class StudyRow:
    dataset: str               # имя набора
    study_key: str             # уникальный идентификатор внутри объединенного реестра
    path: str                  # путь к файлу (NIfTI) или директории (DICOM)
    path_type: Literal["nifti", "dicom"]
    label: Label               # 0/1
    source_category: Optional[str] = None   # например, CT-0..CT-4
    mask_path: Optional[str] = None         # при наличии
