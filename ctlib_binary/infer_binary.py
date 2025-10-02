import argparse, csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import config.path as P
from ctlib_binary.datasets.volume_ds import VolumeBinaryDataset
from ctlib_binary.models.medicalnet_r3d18 import R3D18Binary
from ctlib_binary.utils.csv_registry import load_split_lists

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies_csv", type=Path, default=P.MERGED_REGISTRY_CSV)
    ap.add_argument("--keys_txt", type=Path, default=P.VAL_LIST)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--out_csv", type=Path, default=P.RUNS_DIR / "predictions.csv")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys_txt.read_text(encoding="utf-8").splitlines() if k.strip()]

    ds = VolumeBinaryDataset(args.studies_csv, keys, augment=False)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = R3D18Binary(pretrained=False).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def sigmoid(x): return 1.0 / (1.0 + torch.exp(-x))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["study_key","label","logit","prob"])
        writer.writeheader()
        with torch.no_grad():
            for x, y, keys_batch in ld:
                x = x.to(device, non_blocking=True)
                logit = model(x)
                prob = sigmoid(logit).cpu()
                for i in range(x.size(0)):
                    writer.writerow({
                        "study_key": keys_batch[i],
                        "label": int(y[i].item()),
                        "logit": float(logit[i].cpu().item()),
                        "prob": float(prob[i].item())
                    })
    print(f"Saved predictions to {args.out_csv}")

if __name__ == "__main__":
    main()
