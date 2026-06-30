"""Pravi trajni, stratified train/val/test split.

Pokreni jednom. Rezultujući CSV fajlovi u ``--out-dir`` su verzionisani i ne smeju
se ponovo generisati za pojedinačne eksperimente — svaki model čita isti split.

Primer::

    python scripts/prepare_splits.py \
        --data-root "data/raw/plantvillage dataset/color" \
        --out-dir data/splits \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Omogući uvoz ``src`` kada se pokreće kao običan skript.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import make_splits, split_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create persisted stratified splits.")
    p.add_argument(
        "--data-root",
        required=True,
        help="Path to the PlantVillage 'color' root (one folder per Species___Disease).",
    )
    p.add_argument("--out-dir", default="data/splits", help="Where to write the split CSVs.")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, maps = make_splits(
        data_root=args.data_root,
        out_dir=args.out_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"Total images : {len(df):,}")
    print(f"Species (L1) : {maps.num_species}")
    print(f"Classes (L2) : {maps.num_classes}")
    print("\nPer-split totals:")
    print(df["split"].value_counts().to_string())

    summary = split_summary(df)
    imb = summary["total"].max() / summary["total"].min()
    print(f"\nClass imbalance ratio (max/min support): {imb:.1f}x")
    print(f"\nSplits written to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
