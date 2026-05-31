"""Albumentations transform pipelines.

Two pipelines:

* ``build_train_transforms`` — **heavy** augmentation. This is deliberate: the
  known PlantVillage failure mode is that models latch onto the near-uniform
  per-class background instead of the lesion. Aggressive color jitter, random
  erasing (CoarseDropout), perspective/affine warps and flips push the model to
  rely on leaf/lesion texture rather than background cues. This is our main
  mitigation for the background-bias limitation (alongside Grad-CAM analysis).
* ``build_eval_transforms`` — **minimal**: resize + normalize only. No
  augmentation on val/test, ever.

Normalization defaults to ImageNet statistics (we use ImageNet-pretrained
backbones). EDA confirmed PlantVillage's per-channel stats are close to these.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(
    img_size: int = 224,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    aug_strength: str = "heavy",
) -> A.Compose:
    """Training augmentation pipeline.

    ``aug_strength`` in {"light", "medium", "heavy"} scales how aggressive the
    augmentation is. "heavy" is the default and recommended for the
    background-bias mitigation; "light"/"medium" are provided for ablations.
    """
    if aug_strength not in {"light", "medium", "heavy"}:
        raise ValueError(f"aug_strength must be light|medium|heavy, got {aug_strength!r}")

    geometric: list = [
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0) if aug_strength == "heavy" else (0.8, 1.0),
            ratio=(0.8, 1.25),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    if aug_strength in {"medium", "heavy"}:
        geometric.append(
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.0625),
                rotate=(-25, 25),
                shear=(-8, 8),
                p=0.5,
            )
        )
    if aug_strength == "heavy":
        geometric.append(A.Perspective(scale=(0.05, 0.1), p=0.3))

    # Color/lighting — attack the uniform-background cue most directly.
    cj_strength = {"light": 0.1, "medium": 0.2, "heavy": 0.3}[aug_strength]
    photometric: list = [
        A.ColorJitter(
            brightness=cj_strength,
            contrast=cj_strength,
            saturation=cj_strength,
            hue=cj_strength / 3,
            p=0.7,
        ),
        A.RandomBrightnessContrast(p=0.3),
    ]
    if aug_strength in {"medium", "heavy"}:
        photometric.append(
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(),
                    A.MotionBlur(blur_limit=5),
                ],
                p=0.3,
            )
        )

    occlusion: list = []
    if aug_strength in {"medium", "heavy"}:
        # Random erasing — forces use of multiple leaf regions, not one patch.
        max_holes = 8 if aug_strength == "heavy" else 4
        hole_frac = 0.15 if aug_strength == "heavy" else 0.1
        occlusion.append(
            A.CoarseDropout(
                num_holes_range=(1, max_holes),
                hole_height_range=(0.02, hole_frac),
                hole_width_range=(0.02, hole_frac),
                p=0.3,
            )
        )

    return A.Compose(
        [
            *geometric,
            *photometric,
            *occlusion,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(
    img_size: int = 224,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """Validation/test pipeline: deterministic resize + normalize only."""
    # Resize slightly larger then center-crop — standard ImageNet eval recipe.
    resize = int(round(img_size * 1.14))
    return A.Compose(
        [
            A.Resize(height=resize, width=resize),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_transforms(split: str, img_size: int = 224, **kwargs) -> A.Compose:
    """Convenience dispatcher: 'train' -> heavy aug, else -> eval transforms."""
    if split == "train":
        return build_train_transforms(img_size, **kwargs)
    eval_kwargs = {k: v for k, v in kwargs.items() if k in {"mean", "std"}}
    return build_eval_transforms(img_size, **eval_kwargs)
