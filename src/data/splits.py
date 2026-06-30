"""Stratified, sačuvano train/val/test deljenje za PlantVillage.

PlantVillage ``color`` dataset je organizovan kao jedan folder po
``<Species>___<Disease>`` klasi::

    <data_root>/
        Apple___Apple_scab/*.jpg
        Apple___Black_rot/*.jpg
        ...
        Tomato___healthy/*.jpg

Postoji 38 takvih foldera koji obuhvataju 14 species. Dataset je nebalansiran na
*oba* nivoa hijerarhije, pa:

* Split je **stratified po 38-class (species, disease) label-u**. To
  istovremeno čuva i raspodelu species (support jedne species je zbir njenih
  disease klasa) i po-species raspodelu bolesti.
* Split je **sačuvan na disk** (``train.csv`` / ``val.csv`` / ``test.csv``
  + ``label_maps.json``) i nikada se ne sme regenerisati u hodu - kod nizvodno
  učitava ove fajlove tako da svaki eksperiment vidi identične podatke.

Konvencije za label-e (sve determinističke, izvedene iz alfabetskog sortiranja):

* ``species_id``      - globalni indeks species, 0 .. 13.
* ``class_id``        - globalni flat indeks, 0 .. 37 (38-class label).
* ``disease_id_in_species`` - po-species indeks bolesti, 0 .. (k_species - 1),
                       koji koriste hijerarhijski disease klasifikatori.

Putanje sačuvane u CSV fajlovima su **relativne u odnosu na ``data_root``** tako
da su split-ovi prenosivi između lokalne mašine i Kaggle-a.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# PlantVillage koristi separator od tri donje crte između species i disease.
CLASS_SEP = "___"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class LabelMaps:
    """Dvosmerna kodiranja label-a, deterministički izvedena iz podataka."""

    species_to_id: dict[str, int]
    class_to_id: dict[str, int]  # pun "Species___Disease" -> 0..37
    # po species_id: {disease_name -> lokalni disease id}
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
            # JSON ključevi moraju biti string-ovi
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
    """Razdvoji naziv foldera ``Species___Disease`` na (species, disease)."""
    if CLASS_SEP not in class_name:
        raise ValueError(
            f"Class folder {class_name!r} does not contain the expected "
            f"separator {CLASS_SEP!r}. Are you pointing at the 'color' dataset root?"
        )
    species, disease = class_name.split(CLASS_SEP, 1)
    return species, disease


def scan_dataset(data_root: str | Path) -> pd.DataFrame:
    """Prođi kroz ``data_root`` i izgradi dataframe po slici.

    Vraća frame sa kolonama:
    ``filepath`` (relativno u odnosu na data_root), ``class_name``, ``species``, ``disease``.
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
    """Izgradi determinističko kodiranje label-a iz skeniranog dataframe-a."""
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
    """Dodaj celobrojne label kolone u dataframe."""
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
    """Dodaj kolonu ``split`` sa vrednostima u {train, val, test}.

    Stratifikacija se radi po ``stratify_col`` (podrazumevano 38-class label),
    što čuva i raspodelu species i po-species raspodelu bolesti.
    Izvodi se kao dva stratified split-a: prvo se izdvoji holdout (val+test),
    zatim se taj holdout deli na val i test.
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

    # Unutar holdout-a, udeo test-a je test_size / (val_size + test_size).
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
    """Od početka do kraja: skeniranje, kodiranje, stratified-split i čuvanje.

    Upisuje ``train.csv``, ``val.csv``, ``test.csv``, ``all.csv`` i
    ``label_maps.json`` u ``out_dir``. Vraća kompletan labelirani dataframe
    i label mape.
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
    """Učitaj sačuvani split ('train' | 'val' | 'test' | 'all')."""
    path = Path(splits_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/prepare_splits.py first."
        )
    return pd.read_csv(path)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Broj klasa po split-u plus odnos disbalansa (max/min support)."""
    counts = (
        df.groupby(["split", "class_name"]).size().unstack("split", fill_value=0)
    )
    for col in ("train", "val", "test"):
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[["train", "val", "test"]].sum(axis=1)
    return counts.sort_values("total", ascending=False)
