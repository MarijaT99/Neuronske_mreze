"""PlantVillageDataset — čita sačuvane split CSV fajlove.

Dva režima:

* ``mode="flat"`` — vraća ``(image, class_id)`` gde je ``class_id`` globalni
  38-class (species, disease) label. Koristi ga flat baseline.
* ``mode="hierarchical"`` — vraća ``(image, species_id, disease_id_in_species)``.
  ``disease_id_in_species`` je *po-species* indeks bolesti, što je upravo ono što
  predviđaju po-species disease head-ovi.

Dataset nikada ponovo ne skenira foldere sa slikama: koristi dataframe koji
proizvodi :mod:`src.data.splits` (učitan iz ``data/splits/<split>.csv``), tako da
svaki eksperiment vidi identičan, sačuvani split.

Slike se dekodiraju pomoću OpenCV-a (BGR->RGB) jer albumentations pipeline-i
rade nad numpy HWC uint8 nizovima.
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
    """Dekodiraj sliku u RGB uint8 HWC numpy niz.

    Koristi ``np.fromfile`` + ``cv2.imdecode`` umesto ``cv2.imread`` jer
    OpenCV-ov ``imread`` ne može da otvori putanje sa ne-ASCII karakterima na
    Windows-u. ``np.fromfile`` otvara preko Python-ovog Unicode-svesnog IO-a, pa se
    bajt bafer dekodira nezavisno od putanje.
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
    """Torch dataset nad sačuvanim PlantVillage split-om.

    Parametri
    ----------
    df:
        Labelirani split dataframe (mora sadržati barem ``filepath`` i
        odgovarajuće label kolone). Međusobno isključiv sa ``splits_dir``/``split``.
    data_root:
        Koren u odnosu na koji je relativna kolona ``filepath`` (``color`` folder).
    mode:
        ``"flat"`` ili ``"hierarchical"``.
    transform:
        Albumentations ``Compose`` (poziva se kao ``transform(image=arr)``).
    species_filter:
        Ako je postavljen (validan samo u hijerarhijskom režimu), zadržava samo
        redove tog ``species_id``-a. Koristi se pri treniranju jednog po-species
        disease head-a.
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

    # ---- pomoćne funkcije za loss weighting / sampling ----------------------
    def label_column(self) -> str:
        """Naziv ciljne kolone za trenutni režim (za sampler/težine)."""
        return "class_id" if self.mode == "flat" else "disease_id_in_species"

    def labels(self) -> np.ndarray:
        """Celobrojni ciljni label-i kao numpy niz (za trenutni režim)."""
        return self.df[self.label_column()].to_numpy()

    def class_counts(self) -> np.ndarray:
        """Broj uzoraka po klasi, indeksiran po label id-u (0..num_classes-1)."""
        labels = self.labels()
        num = int(labels.max()) + 1 if len(labels) else 0
        return np.bincount(labels, minlength=num)
