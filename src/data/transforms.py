"""Albumentations transform pipeline-i.

Dva pipeline-a:

* ``build_train_transforms`` - **jak** augmentation. To je namerno: poznati
  način otkazivanja kod PlantVillage-a je da se modeli zakače za skoro uniforman
  per-class background umesto za leziju. Agresivan color jitter, random
  erasing (CoarseDropout), perspective/affine deformacije i flip-ovi teraju model
  da se oslanja na teksturu lista/lezije pre nego na background tragove. Ovo je naša
  glavna mera protiv background bias ograničenja (uz Grad-CAM analizu).
* ``build_eval_transforms`` - **minimalan**: samo resize + normalize. Nikada nema
  augmentation-a na val/test-u.

Normalizacija podrazumevano koristi ImageNet statistiku (koristimo ImageNet
pretrained backbone-ove). EDA je potvrdila da su PlantVillage per-channel statistike
bliske ovima.
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
    """Pipeline za augmentation pri treniranju.

    ``aug_strength`` u {"light", "medium", "heavy"} podešava koliko je agresivan
    augmentation. "heavy" je podrazumevan i preporučen za ublažavanje background
    bias-a; "light"/"medium" su predviđeni za ablacije.
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

    # Boja/osvetljenje - najdirektnije napada trag uniformnog background-a.
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
        # Random erasing - prisiljava korišćenje više regiona lista, ne jednog dela.
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
    """Pipeline za validaciju/test: samo deterministički resize + normalize."""
    # Resize na malo veću dimenziju pa center-crop - standardna ImageNet eval procedura.
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
    """Praktičan dispečer: 'train' -> jak aug, inače -> eval transformacije."""
    if split == "train":
        return build_train_transforms(img_size, **kwargs)
    eval_kwargs = {k: v for k, v in kwargs.items() if k in {"mean", "std"}}
    return build_eval_transforms(img_size, **eval_kwargs)
