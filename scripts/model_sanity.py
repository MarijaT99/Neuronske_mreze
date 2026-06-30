"""Sanity check modela — forward prolazi + invarijante zamrzavanja/režima.

Pokreće se na CPU-u sa nasumičnim tenzorima (bez podataka, bez preuzimanja pretrained
težina) pa je brz i offline. Proverava:
* BaselineCNN daje izlaz (B, num_classes) za flat i veličine head-a po species;
* TransferModel (resnet50, efficientnet_b0) daje izlaz ispravnog oblika;
* feature_extraction zamrzava backbone (samo je head trainable);
* fine-tuning sa unfreeze_blocks odmrzava head + samo N završnih stadijuma;
* build_model factory dispečuje sve tipove;
* HierarchicalClassifier.predict vraća validne species/disease id-jeve i usmerava
  trivijalne species na disease id 0 (proverava se putanja error propagation-a).

Upisuje izveštaj u $TEMP; izlazi sa kodom različitim od nule pri bilo kom padu.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baseline_cnn import BaselineCNN  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.hierarchical import HierarchicalClassifier  # noqa: E402
from src.models.transfer import TransferModel  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _report_path() -> Path:
    if "--report" in sys.argv:
        return Path(sys.argv[sys.argv.index("--report") + 1])
    if os.environ.get("REPORT_PATH"):
        return Path(os.environ["REPORT_PATH"])
    return Path(tempfile.gettempdir()) / "pdh_model_sanity.txt"


REPORT = _report_path()
report: list[str] = []
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    report.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    seed_everything(42)
    x = torch.randn(2, 3, 224, 224)

    # --- BaselineCNN ------------------------------------------------------
    m = BaselineCNN(num_classes=38)
    out = m(x)
    check("baseline: flat output (2,38)", tuple(out.shape) == (2, 38), str(tuple(out.shape)))
    params = sum(p.numel() for p in m.parameters())
    check("baseline: param count modest (<6M)", params < 6_000_000, f"{params:,} params")

    m_head = BaselineCNN(num_classes=10)  # npr. disease head za Tomato
    check("baseline: per-species head (2,10)", tuple(m_head(x).shape) == (2, 10))

    # --- TransferModel: resnet50 -----------------------------------------
    rn_fe = TransferModel("resnet50", 38, pretrained=False, mode="feature_extraction")
    out = rn_fe(x)
    check("resnet50 FE: output (2,38)", tuple(out.shape) == (2, 38), str(tuple(out.shape)))
    # backbone zamrznut, head trainable
    head_params = list(rn_fe._classifier_module().parameters())
    check("resnet50 FE: head is trainable", all(p.requires_grad for p in head_params))
    body_trainable = sum(
        p.requires_grad for n, p in rn_fe.named_parameters() if not n.startswith("backbone.fc")
    )
    check("resnet50 FE: backbone frozen", rn_fe.num_trainable_params() == sum(p.numel() for p in head_params),
          f"trainable={rn_fe.num_trainable_params():,}")

    rn_ft = TransferModel("resnet50", 38, pretrained=False, mode="finetune", unfreeze_blocks=2)
    t_ft = rn_ft.num_trainable_params()
    check("resnet50 FT(2): more trainable than FE, less than all",
          rn_fe.num_trainable_params() < t_ft < rn_ft.num_total_params(),
          f"FE={rn_fe.num_trainable_params():,} FT={t_ft:,} total={rn_ft.num_total_params():,}")
    # layer4 treba da bude trainable, layer1 treba da bude zamrznut
    check("resnet50 FT(2): layer4 unfrozen",
          all(p.requires_grad for p in rn_ft.backbone.layer4.parameters()))
    check("resnet50 FT(2): layer1 frozen",
          not any(p.requires_grad for p in rn_ft.backbone.layer1.parameters()))

    # --- TransferModel: efficientnet_b0 ----------------------------------
    eff = TransferModel("efficientnet_b0", 14, pretrained=False, mode="finetune", unfreeze_blocks=2)
    check("efficientnet_b0: output (2,14)", tuple(eff(x).shape) == (2, 14), str(tuple(eff(x).shape)))
    check("efficientnet_b0 FT(2): some but not all trainable",
          0 < eff.num_trainable_params() < eff.num_total_params())

    # --- factory ----------------------------------------------------------
    fm = build_model({"type": "baseline_cnn", "dropout": 0.3}, num_classes=38)
    check("factory: baseline_cnn", isinstance(fm, BaselineCNN) and tuple(fm(x).shape) == (2, 38))
    fr = build_model({"type": "resnet50", "pretrained": False, "mode": "feature_extraction"}, 38)
    check("factory: resnet50", isinstance(fr, TransferModel) and tuple(fr(x).shape) == (2, 38))

    # --- HierarchicalClassifier ------------------------------------------
    # Mini postavka: 3 species; species 0 ima 4 bolesti, species 1 ima 2,
    # species 2 je trivijalan (nema head -> disease id 0).
    species_model = BaselineCNN(num_classes=3)
    disease_models = {0: BaselineCNN(num_classes=4), 1: BaselineCNN(num_classes=2)}
    hc = HierarchicalClassifier(species_model, disease_models, num_species=3)
    check("hier: trivial species detected", hc.trivial_species == [2], str(hc.trivial_species))

    xb = torch.randn(8, 3, 224, 224)
    pred = hc.predict(xb)
    check("hier: species_id shape (8,)", tuple(pred.species_id.shape) == (8,))
    check("hier: disease_id shape (8,)", tuple(pred.disease_id_in_species.shape) == (8,))
    check("hier: species ids in [0,3)",
          bool(((pred.species_id >= 0) & (pred.species_id < 3)).all()))
    # disease id-jevi validni za predviđeni species
    valid = True
    sizes = {0: 4, 1: 2, 2: 1}
    for s, d in zip(pred.species_id.tolist(), pred.disease_id_in_species.tolist()):
        if not (0 <= d < sizes[s]):
            valid = False
            break
    check("hier: disease id valid for predicted species", valid)
    # trivijalan species uvek -> disease 0
    trivial_mask = pred.species_id == 2
    if trivial_mask.any():
        check("hier: trivial species -> disease 0",
              bool((pred.disease_id_in_species[trivial_mask] == 0).all()))
    else:
        report.append("[INFO] hier: no trivial-species samples in this random batch")

    # --- state_dict kruži kroz kontejner i vraća se ispravno -------------
    sd = hc.state_dict()
    check("hier: state_dict non-empty and includes heads",
          any("disease_models.0" in k for k in sd) and any("species_model" in k for k in sd))

    REPORT.write_text(
        "\n".join(["=== MODEL SANITY ==="] + report
                  + ["", f"RESULT: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}"]),
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        REPORT.write_text("MODEL SANITY CRASHED:\n" + traceback.format_exc(), encoding="utf-8")
        code = 2
    sys.exit(code)
