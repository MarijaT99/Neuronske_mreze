"""Create the persisted, stratified train/val/test split.

Run once. The resulting CSVs under ``--out-dir`` are versioned and must not be
regenerated for individual experiments — every model reads the same split.

Example::

    python scripts/prepare_splits.py \
        --data-root "data/raw/plantvillage dataset/color" \
        --out-dir data/splits \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
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
