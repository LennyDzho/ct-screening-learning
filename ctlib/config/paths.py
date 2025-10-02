from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
DICOM_DIR = DATA_DIR / "dicom"
PROTOCOL_DIR = DATA_DIR / "protocols"


EXTRACT_DIR = BASE_DIR / "extracted_data" / ".extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

MEDICALNET_WEIGHTS_PATH = BASE_DIR / "ctlib" / "models" / "medicalnet_pretrained.pth"

PREPARED_DIR = BASE_DIR / "extracted_data" / "prepared"
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

LESIONS_CSV = PREPARED_DIR / "lesions.csv"

DATASET_OUT_DIR = BASE_DIR / "extracted_data"

if __name__ == "__main__":
    print("BASE_DIR =", BASE_DIR)
    print("DATA_DIR =", DATA_DIR)
    print("DICOM_DIR =", DICOM_DIR)
    print("PROTOCOL_DIR =", PROTOCOL_DIR)
    print("EXTRACT_DIR =", EXTRACT_DIR)
    print("PREPARED_DIR =", PREPARED_DIR)