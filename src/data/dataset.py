"""PlantVillageDataset — reads the persisted split CSVs.

Two modes:

* ``mode="flat"`` — returns ``(image, class_id)`` where ``class_id`` is the
  global 38-class (species, disease) label. Used by the flat baseline.
* ``mode="hierarchical"`` — returns ``(image, species_id, disease_id_in_species)``.
  ``disease_id_in_species`` is the *per-species* disease index, which is what the
  per-species disease heads predict.

The dataset never re-scans the image folders: it consumes the dataframe produced
by :mod:`src.data.splits` (loaded from ``data/splits/<split>.csv``), so every
experiment sees the identical, persisted split.

Images are decoded with OpenCV (BGR->RGB) because the albumentations pipelines
operate on numpy HWC uint8 arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.splits import load_split

VALID_MODES = ("flat", "hierarchical")


def _read_rgb(path: Path) -> np.ndarray:
    """Decode an image to an RGB uint8 HWC numpy array.

    Uses ``np.fromfile`` + ``cv2.imdecode`` rather than ``cv2.imread`` because
    OpenCV's ``imread`` cannot open paths containing non-ASCII characters on
    Windows. ``np.fromfile`` opens via Python's Unicode-aware IO, so the byte
    buffer is decoded path-agnostically.
    """
    import cv2

    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"Empty or missing image: {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class PlantVillageDataset(Dataset):
    """Torch dataset over a persisted PlantVillage split.

    Parameters
    ----------
    df:
        Labelled split dataframe (must contain at least ``filepath`` and the
        relevant label columns). Mutually exclusive with ``splits_dir``/``split``.
    data_root:
        Root the ``filepath`` column is relative to (the ``color`` folder).
    mode:
        ``"flat"`` or ``"hierarchical"``.
    transform:
        An albumentations ``Compose`` (called as ``transform(image=arr)``).
    species_filter:
        If set (only valid in hierarchical mode), keep only rows of that
        ``species_id``. Used when training a single per-species disease head.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        df: pd.DataFrame | None = None,
        splits_dir: str | Path | None = None,
        split: str | None = None,
        mode: str = "flat",
        transform: Callable | None = None,
        species_filter: int | None = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if (df is None) == (splits_dir is None):
            raise ValueError("Provide exactly one of `df` or (`splits_dir` + `split`).")

        if df is None:
            if split is None:
                raise ValueError("`split` is required when loading from `splits_dir`.")
            df = load_split(splits_dir, split)

        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform

        required = {"flat": ["filepath", "class_id"],
                    "hierarchical": ["filepath", "species_id", "disease_id_in_species"]}[mode]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Split dataframe missing columns for mode={mode!r}: {missing}")

        if species_filter is not None:
            if mode != "hierarchical":
                raise ValueError("species_filter is only valid in hierarchical mode.")
            df = df[df["species_id"] == species_filter]
            if len(df) == 0:
                raise ValueError(f"No rows for species_id={species_filter}.")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        row = self.df.iloc[idx]
        image = _read_rgb(self.data_root / row["filepath"])

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.mode == "flat":
            return image, int(row["class_id"])
        return image, int(row["species_id"]), int(row["disease_id_in_species"])

    # ---- helpers for loss weighting / sampling ------------------------------
    def label_column(self) -> str:
        """Name of the target column for the current mode (for sampler/weights)."""
        return "class_id" if self.mode == "flat" else "disease_id_in_species"

    def labels(self) -> np.ndarray:
        """Integer target labels as a numpy array (for the current mode)."""
        return self.df[self.label_column()].to_numpy()

    def class_counts(self) -> np.ndarray:
        """Per-class sample counts, indexed by label id (0..num_classes-1)."""
        labels = self.labels()
        num = int(labels.max()) + 1 if len(labels) else 0
        return np.bincount(labels, minlength=num)
