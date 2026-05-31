"""Stratified, persisted train/val/test splitting for PlantVillage.

The PlantVillage ``color`` dataset is laid out as one folder per
``<Species>___<Disease>`` class::

    <data_root>/
        Apple___Apple_scab/*.jpg
        Apple___Black_rot/*.jpg
        ...
        Tomato___healthy/*.jpg

There are 38 such folders spanning 14 species. The dataset is imbalanced at
*both* hierarchy levels, so:

* The split is **stratified on the 38-class (species, disease) label**. This
  simultaneously preserves the species distribution (a species' support is the
  sum of its disease classes) and the per-species disease distribution.
* The split is **persisted to disk** (``train.csv`` / ``val.csv`` / ``test.csv``
  + ``label_maps.json``) and must never be regenerated on the fly — downstream
  code loads these files so every experiment sees identical data.

Label conventions (all deterministic, derived from alphabetical sorting):

* ``species_id``      — global species index, 0 .. 13.
* ``class_id``        — global flat index, 0 .. 37 (the 38-class label).
* ``disease_id_in_species`` — per-species disease index, 0 .. (k_species - 1),
                       used by the hierarchical disease classifiers.

Paths stored in the CSVs are **relative to ``data_root``** so the splits are
portable between the local machine and Kaggle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# PlantVillage uses a triple-underscore separator between species and disease.
CLASS_SEP = "___"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class LabelMaps:
    """Bidirectional label encodings, derived deterministically from the data."""

    species_to_id: dict[str, int]
    class_to_id: dict[str, int]  # full "Species___Disease" -> 0..37
    # per species_id: {disease_name -> local disease id}
    disease_in_species_to_id: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def id_to_species(self) -> dict[int, str]:
        return {v: k for k, v in self.species_to_id.items()}

    @property
    def id_to_class(self) -> dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}

    @property
    def num_species(self) -> int:
        return len(self.species_to_id)

    @property
    def num_classes(self) -> int:
        return len(self.class_to_id)

    def num_diseases_for(self, species_id: int) -> int:
        return len(self.disease_in_species_to_id[species_id])

    def to_json(self, path: str | Path) -> None:
        payload = {
            "species_to_id": self.species_to_id,
            "class_to_id": self.class_to_id,
            # JSON keys must be strings
            "disease_in_species_to_id": {
                str(sid): mapping for sid, mapping in self.disease_in_species_to_id.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelMaps":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            species_to_id=payload["species_to_id"],
            class_to_id=payload["class_to_id"],
            disease_in_species_to_id={
                int(sid): mapping
                for sid, mapping in payload["disease_in_species_to_id"].items()
            },
        )


def _parse_class_name(class_name: str) -> tuple[str, str]:
    """Split a ``Species___Disease`` folder name into (species, disease)."""
    if CLASS_SEP not in class_name:
        raise ValueError(
            f"Class folder {class_name!r} does not contain the expected "
            f"separator {CLASS_SEP!r}. Are you pointing at the 'color' dataset root?"
        )
    species, disease = class_name.split(CLASS_SEP, 1)
    return species, disease


def scan_dataset(data_root: str | Path) -> pd.DataFrame:
    """Walk ``data_root`` and build a per-image dataframe.

    Returns a frame with columns:
    ``filepath`` (relative to data_root), ``class_name``, ``species``, ``disease``.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    class_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subfolders found under {data_root}. "
            "Expected one folder per 'Species___Disease' class."
        )

    rows: list[dict] = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        species, disease = _parse_class_name(class_name)
        images = sorted(
            p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS and p.is_file()
        )
        if not images:
            raise FileNotFoundError(f"No images found in class folder: {class_dir}")
        for img in images:
            rows.append(
                {
                    "filepath": img.relative_to(data_root).as_posix(),
                    "class_name": class_name,
                    "species": species,
                    "disease": disease,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("filepath").reset_index(drop=True)


def build_label_maps(df: pd.DataFrame) -> LabelMaps:
    """Build deterministic label encodings from the scanned dataframe."""
    species_sorted = sorted(df["species"].unique())
    species_to_id = {s: i for i, s in enumerate(species_sorted)}

    class_sorted = sorted(df["class_name"].unique())
    class_to_id = {c: i for i, c in enumerate(class_sorted)}

    disease_in_species_to_id: dict[int, dict[str, int]] = {}
    for species, sid in species_to_id.items():
        diseases = sorted(df.loc[df["species"] == species, "disease"].unique())
        disease_in_species_to_id[sid] = {d: i for i, d in enumerate(diseases)}

    return LabelMaps(
        species_to_id=species_to_id,
        class_to_id=class_to_id,
        disease_in_species_to_id=disease_in_species_to_id,
    )


def encode_labels(df: pd.DataFrame, maps: LabelMaps) -> pd.DataFrame:
    """Add integer label columns to the dataframe."""
    df = df.copy()
    df["species_id"] = df["species"].map(maps.species_to_id).astype(int)
    df["class_id"] = df["class_name"].map(maps.class_to_id).astype(int)
    df["disease_id_in_species"] = [
        maps.disease_in_species_to_id[sid][dis]
        for sid, dis in zip(df["species_id"], df["disease"])
    ]
    return df


def stratified_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    stratify_col: str = "class_id",
) -> pd.DataFrame:
    """Add a ``split`` column with values in {train, val, test}.

    Stratification is done on ``stratify_col`` (the 38-class label by default),
    which preserves both the species and per-species disease distributions.
    Performed as two stratified splits: first hold out (val+test), then divide
    that holdout into val and test.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1 (train needs a share too).")

    df = df.reset_index(drop=True)
    holdout_size = val_size + test_size

    train_idx, holdout_idx = train_test_split(
        df.index,
        test_size=holdout_size,
        random_state=seed,
        stratify=df[stratify_col],
    )

    # Within the holdout, test's share is test_size / (val_size + test_size).
    rel_test_size = test_size / holdout_size
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=rel_test_size,
        random_state=seed,
        stratify=df.loc[holdout_idx, stratify_col],
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def make_splits(
    data_root: str | Path,
    out_dir: str | Path,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, LabelMaps]:
    """End-to-end: scan, encode, stratified-split, and persist.

    Writes ``train.csv``, ``val.csv``, ``test.csv``, ``all.csv`` and
    ``label_maps.json`` under ``out_dir``. Returns the full labelled dataframe
    and the label maps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(data_root)
    maps = build_label_maps(df)
    df = encode_labels(df, maps)
    df = stratified_split(df, val_size=val_size, test_size=test_size, seed=seed)

    maps.to_json(out_dir / "label_maps.json")
    df.to_csv(out_dir / "all.csv", index=False)
    for split_name in ("train", "val", "test"):
        df[df["split"] == split_name].to_csv(out_dir / f"{split_name}.csv", index=False)

    return df, maps


def load_split(splits_dir: str | Path, split: str) -> pd.DataFrame:
    """Load a persisted split ('train' | 'val' | 'test' | 'all')."""
    path = Path(splits_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/prepare_splits.py first."
        )
    return pd.read_csv(path)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split class counts plus the imbalance ratio (max/min support)."""
    counts = (
        df.groupby(["split", "class_name"]).size().unstack("split", fill_value=0)
    )
    for col in ("train", "val", "test"):
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[["train", "val", "test"]].sum(axis=1)
    return counts.sort_values("total", ascending=False)
