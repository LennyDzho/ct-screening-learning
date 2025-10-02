from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import re, hashlib

from ctlib_binary.utils.csv_registry import load_registry
from ctlib_binary.dataio.volume_reader import load_volume  # сигнатура: (path, path_type, hu_window, target_spacing, target_shape)

def _safe_name(s: str) -> str:
    # оставляем буквы/цифры/._- , остальное -> _
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    # убираем повторяющиеся "_"
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:128]  # ограничим длину

def _hash_name(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


class VolumeBinaryDataset(Dataset):
    def __init__(self,
                 studies_csv: Path,
                 keys: List[str],
                 augment: bool = False,
                 cache_dir: Optional[Path] = None,
                 target_shape: Tuple[int,int,int] = (96, 192, 192),
                 hu_wl: int = -600,
                 hu_ww: int = 1500):
        """
        Возвращает тензор [1, D, H, W] (float32, 0..1) в формате channels_last_3d.
        Кэширует уже предобработанные объёмы (после load_volume).
        """
        super().__init__()
        self.reg = load_registry(studies_csv)
        self.keys = keys
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.target_shape = target_shape
        self.hu_wl = hu_wl
        self.hu_ww = hu_ww
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.keys)

    def _cache_path(self, key: str) -> Path:
        row = self.reg[key]
        ptype = _safe_name(row.get("path_type", "unk"))
        d, h, w = self.target_shape
        safe_key = _safe_name(key)               # ← убираем пробелы, двоеточия, и т.п.
        base = f"{safe_key}_{ptype}_D{d}x{h}x{w}_wl{self.hu_wl}_ww{self.hu_ww}.npy"
        if self.cache_dir is None:
            return Path("__nocache__") / base
        out = self.cache_dir / base
        # если вдруг путь слишком длинный/недопустимый — запасной вариант по хэшу
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            return out
        except Exception:
            h = _hash_name(f"{key}|{ptype}|{d}x{h}x{w}|{self.hu_wl}|{self.hu_ww}")
            return self.cache_dir / f"{h}.npy"

    def _load_or_build(self, key: str) -> np.ndarray:
        if self.cache_dir is not None:
            cpath = self._cache_path(key)
            if cpath.exists():
                arr = np.load(cpath, mmap_mode="r")
                return np.asarray(arr)

        row = self.reg[key]
        path = Path(row["path"])                        # ← гарантируем Path
        path_type = row.get("path_type", "nifti")

        vol, _spacing = load_volume(
            path=path,
            path_type=path_type,
            hu_window=(self.hu_wl, self.hu_ww),
            target_spacing=None,
            target_shape=self.target_shape
        )

        vol = np.asarray(vol, dtype=np.float32)
        if np.nanmin(vol) < 0.0 or np.nanmax(vol) > 1.0:
            np.clip(vol, 0.0, 1.0, out=vol)

        if self.cache_dir is not None:
            cpath = self._cache_path(key)
            try:
                np.save(cpath, vol)
            except OSError:
                # резерв: короткое имя по хэшу
                h = _hash_name(f"{key}|{path_type}|{self.target_shape}|{self.hu_wl}|{self.hu_ww}")
                np.save(self.cache_dir / f"{h}.npy", vol)

        return vol

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        y = float(self.reg[key]["label"])

        vol = self._load_or_build(key)  # [D,H,W]

        # лёгкие аугментации
        # if self.augment:
        #     # случайный флип по ширине
        #     if np.random.rand() < 0.5:
        #         vol = vol[:, :, ::-1].copy()
        #     # яркость/контраст очень мягко
        #     if np.random.rand() < 0.3:
        #         scale = 0.9 + 0.2 * np.random.rand()
        #         vol = np.clip(vol * scale, 0.0, 1.0, out=vol)

        t = torch.from_numpy(vol.copy()).unsqueeze(0)  # [1,D,H,W]
        # t = t.to(memory_format=torch.channels_last_3d)  # для ускорения 3D свёрток

        y = torch.tensor(y, dtype=torch.float32)
        return t, y, key
