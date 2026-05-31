"""DataLoader factory.

Builds train/val/test loaders from the persisted splits, wiring in:

* the right transform pipeline per split (heavy aug for train, minimal for eval),
* optional ``WeightedRandomSampler`` for imbalance (train only),
* reproducible shuffling via a seeded generator + ``seed_worker``.

``sampler`` and ``shuffle`` are mutually exclusive in PyTorch; when the weighted
sampler is requested we drop shuffle (the sampler already randomizes order).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageDataset
from src.data.transforms import build_eval_transforms, build_train_transforms
from src.training.samplers import make_weighted_sampler
from src.utils.seed import make_generator, seed_worker


def build_dataset(
    *,
    data_root: str | Path,
    splits_dir: str | Path,
    split: str,
    mode: str = "flat",
    img_size: int = 224,
    aug_strength: str = "heavy",
    species_filter: int | None = None,
) -> PlantVillageDataset:
    transform = (
        build_train_transforms(img_size, aug_strength=aug_strength)
        if split == "train"
        else build_eval_transforms(img_size)
    )
    return PlantVillageDataset(
        data_root=data_root,
        splits_dir=splits_dir,
        split=split,
        mode=mode,
        transform=transform,
        species_filter=species_filter,
    )


def build_loader(
    dataset: PlantVillageDataset,
    *,
    batch_size: int = 32,
    is_train: bool = False,
    use_weighted_sampler: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool | None = None,
) -> DataLoader:
    """Wrap a dataset in a DataLoader with reproducible behaviour."""
    sampler = None
    shuffle = False
    if is_train:
        if use_weighted_sampler:
            sampler = make_weighted_sampler(dataset.labels())
        else:
            shuffle = True
    if drop_last is None:
        drop_last = is_train

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )


def build_loaders(
    *,
    data_root: str | Path,
    splits_dir: str | Path,
    mode: str = "flat",
    img_size: int = 224,
    batch_size: int = 32,
    aug_strength: str = "heavy",
    use_weighted_sampler: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    species_filter: int | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test loaders in one call. Returns a dict keyed by split."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = build_dataset(
            data_root=data_root,
            splits_dir=splits_dir,
            split=split,
            mode=mode,
            img_size=img_size,
            aug_strength=aug_strength,
            species_filter=species_filter,
        )
        loaders[split] = build_loader(
            ds,
            batch_size=batch_size,
            is_train=(split == "train"),
            use_weighted_sampler=use_weighted_sampler and split == "train",
            num_workers=num_workers,
            seed=seed,
        )
    return loaders
