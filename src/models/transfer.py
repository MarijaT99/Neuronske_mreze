"""Wrapper-i za modele zasnovane na transfer learning-u (ResNet-50, EfficientNet-B0).

Oba backbone-a se kreiraju kroz ``timm`` sa novim klasifikacionim head-om sa
``num_classes`` izlaza. Podržana su dva režima treniranja:

* ``mode="feature_extraction"`` — zamrzni ceo backbone, treniraj samo head.
* ``mode="finetune"``           — odmrzni poslednjih ``unfreeze_blocks`` stage-ova
                                  (uz head); ostatak ostaje zamrznut. Prosledi
                                  ``unfreeze_blocks=-1`` da se odmrzne sve.

Korišćenje timm-ovog podrazumevanog ``features_only=False`` uz resetovan
``num_classes`` daje jedan tensor logits-a, što odgovara interfejsu baseline CNN-a
tako da je Trainer nezavisan od modela.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

SUPPORTED = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
}


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _local_pretrained_file(timm_name: str) -> str | None:
    """Vraća lokalni fajl sa pretrained težinama za ``timm_name`` ako postoji.

    Postavi ``PDH_PRETRAINED_DIR`` na folder koji sadrži timm safetensors fajlove
    kada runtime nema internet (npr. Kaggle): tada se ImageNet težine učitavaju sa
    diska umesto preuzimanja sa HuggingFace-a. Fajlovi se traže po podrazumevanom
    hub tag-u modela (``resnet50.a1_in1k.safetensors``), pa zatim po nekoliko
    rezervnih opcija.
    """
    d = os.environ.get("PDH_PRETRAINED_DIR")
    if not d or not os.path.isdir(d):
        return None
    import timm

    try:
        cfg = timm.get_pretrained_cfg(timm_name, allow_unregistered=False)
        tag = (getattr(cfg, "hf_hub_id", "") or "").split("/")[-1]
    except Exception:
        tag = ""
    for fname in (f"{tag}.safetensors", f"{timm_name}.safetensors", "model.safetensors"):
        if fname == ".safetensors":
            continue
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    return None


class TransferModel(nn.Module):
    """timm backbone + resetovan head, sa feature-extraction / finetune režimima.

    Parameters
    ----------
    arch:
        Jedan od ``SUPPORTED`` ("resnet50", "efficientnet_b0").
    num_classes:
        Izlazni logits-i.
    pretrained:
        Učitaj ImageNet pretrained težine (True za transfer learning).
    mode:
        "feature_extraction" ili "finetune".
    unfreeze_blocks:
        U finetune režimu, koliko poslednjih backbone stage-ova odmrznuti
        (-1 = sve). Ignoriše se u feature_extraction režimu.
    drop_rate:
        Dropout pre klasifikatora (prosleđuje se timm-u).
    """

    def __init__(
        self,
        arch: str,
        num_classes: int,
        *,
        pretrained: bool = True,
        mode: str = "finetune",
        unfreeze_blocks: int = 2,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        if arch not in SUPPORTED:
            raise ValueError(f"arch must be one of {sorted(SUPPORTED)}, got {arch!r}")
        if mode not in {"feature_extraction", "finetune"}:
            raise ValueError("mode must be 'feature_extraction' or 'finetune'")

        import timm

        self.arch = arch
        self.mode = mode
        create_kwargs = dict(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        if pretrained:
            local = _local_pretrained_file(SUPPORTED[arch])
            if local is not None:
                # Učitaj ImageNet težine sa diska (offline); timm prilagođava head.
                create_kwargs["pretrained_cfg_overlay"] = dict(file=local)
        self.backbone = timm.create_model(SUPPORTED[arch], **create_kwargs)
        self._configure_trainable(mode, unfreeze_blocks)

    # ------------------------------------------------------------------ #
    def _classifier_module(self) -> nn.Module:
        """Vraća head modul (timm ga izlaže preko get_classifier())."""
        return self.backbone.get_classifier()

    def _stage_modules(self) -> list[nn.Module]:
        """Uređena lista backbone stage-ova za selektivno odmrzavanje.

        ResNet: layer1..layer4. EfficientNet: deca blocks Sequential-a.
        """
        if self.arch == "resnet50":
            return [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
        # efficientnet_b0: timm izlaže .blocks kao Sequential stage-ova
        return list(self.backbone.blocks.children())

    def _configure_trainable(self, mode: str, unfreeze_blocks: int) -> None:
        if mode == "feature_extraction":
            _set_requires_grad(self.backbone, False)
            _set_requires_grad(self._classifier_module(), True)
            return

        # finetune
        if unfreeze_blocks < 0:
            _set_requires_grad(self.backbone, True)
            return

        # Zamrzni sve, pa odmrzni head + poslednjih N stage-ova.
        _set_requires_grad(self.backbone, False)
        _set_requires_grad(self._classifier_module(), True)
        stages = self._stage_modules()
        for stage in stages[-unfreeze_blocks:] if unfreeze_blocks else []:
            _set_requires_grad(stage, True)

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_transfer_model(arch: str, num_classes: int, **kwargs) -> TransferModel:
    """Praktičan factory koji preslikava config-driven konstrukciju."""
    return TransferModel(arch, num_classes, **kwargs)
