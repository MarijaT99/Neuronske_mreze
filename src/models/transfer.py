"""Transfer-learning model wrappers (ResNet-50, EfficientNet-B0).

Both backbones are created through ``timm`` with a fresh classification head of
``num_classes`` outputs. Two training regimes are supported:

* ``mode="feature_extraction"`` — freeze the whole backbone, train only the head.
* ``mode="finetune"``           — unfreeze the last ``unfreeze_blocks`` stages
                                  (plus the head); the rest stays frozen. Pass
                                  ``unfreeze_blocks=-1`` to unfreeze everything.

Using timm's ``features_only=False`` default with ``num_classes`` reset gives a
single tensor of logits, matching the baseline CNN's interface so the Trainer is
model-agnostic.
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
    """Return a local pretrained-weights file for ``timm_name`` if available.

    Set ``PDH_PRETRAINED_DIR`` to a folder holding the timm safetensors when the
    runtime has no internet (e.g. Kaggle): we then load the ImageNet weights from
    disk instead of downloading from HuggingFace. Files are matched by the model's
    default hub tag (``resnet50.a1_in1k.safetensors``), then a couple of fallbacks.
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
    """A timm backbone + reset head, with feature-extract / finetune regimes.

    Parameters
    ----------
    arch:
        One of ``SUPPORTED`` ("resnet50", "efficientnet_b0").
    num_classes:
        Output logits.
    pretrained:
        Load ImageNet-pretrained weights (True for transfer learning).
    mode:
        "feature_extraction" or "finetune".
    unfreeze_blocks:
        In finetune mode, how many trailing backbone stages to unfreeze
        (-1 = all). Ignored in feature_extraction mode.
    drop_rate:
        Dropout before the classifier (passed to timm).
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
                # Load ImageNet weights from disk (offline); timm adapts the head.
                create_kwargs["pretrained_cfg_overlay"] = dict(file=local)
        self.backbone = timm.create_model(SUPPORTED[arch], **create_kwargs)
        self._configure_trainable(mode, unfreeze_blocks)

    # ------------------------------------------------------------------ #
    def _classifier_module(self) -> nn.Module:
        """Return the head module (timm exposes it via get_classifier())."""
        return self.backbone.get_classifier()

    def _stage_modules(self) -> list[nn.Module]:
        """Ordered list of backbone stages for selective unfreezing.

        ResNet: layer1..layer4. EfficientNet: the blocks Sequential children.
        """
        if self.arch == "resnet50":
            return [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
        # efficientnet_b0: timm exposes .blocks as a Sequential of stages
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

        # Freeze everything, then unfreeze the head + last N stages.
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
    """Convenience factory mirroring config-driven construction."""
    return TransferModel(arch, num_classes, **kwargs)
