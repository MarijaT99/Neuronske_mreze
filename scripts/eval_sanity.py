"""Sanity check za nasleđivanje config-a + hijerarhijsku evaluaciju.

Offline, bez podataka. Proverava:
* load_config razrešava `extends:` i radi dubinsko spajanje (override-i eksperimenta
  pobeđuju, bazni ključevi opstaju);
* svi isporučeni config-i se učitavaju i izlažu obavezne ključeve;
* evaluate_hierarchical: savršena predviđanja -> macro F1 1.0 na svakom nivou;
* obračun error propagation-a: greška u species iznuđuje end-to-end grešku i
  pripisuje se species-u, a ne bolesti;
* mapiranje globalnih id-jeva kruži i vraća se (species, disease-in-species) -> 38-klasni id.

Upisuje izveštaj u $TEMP; izlazi sa kodom različitim od nule pri padu.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from src.data.splits import LabelMaps
from src.evaluation.hierarchical_eval import evaluate_hierarchical
from src.utils.config import load_config

ROOT = Path(_bootstrap.ROOT)


def _report_path() -> Path:
    if "--report" in sys.argv:
        return Path(sys.argv[sys.argv.index("--report") + 1])
    if os.environ.get("REPORT_PATH"):
        return Path(os.environ["REPORT_PATH"])
    return Path(tempfile.gettempdir()) / "pdh_eval_sanity.txt"


REPORT = _report_path()
report: list[str] = []
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    report.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    # --- nasleđivanje config-a -------------------------------------------
    cfg = load_config(ROOT / "configs" / "resnet50_flat.yaml")
    check("config: extends merged base (seed present)", cfg.get("seed") == 42)
    check("config: override wins (epochs=25 not base 30)", cfg.training.epochs == 25,
          f"epochs={cfg.training.epochs}")
    check("config: base key survives (img_size=224)", cfg.data.img_size == 224)
    check("config: 'extends' key removed", "extends" not in cfg)
    check("config: task/model present", cfg.task.type == "flat" and cfg.model.type == "resnet50")

    for name in ["baseline_cnn_flat", "resnet50_flat", "resnet50_hierarchical",
                 "efficientnet_hierarchical"]:
        c = load_config(ROOT / "configs" / f"{name}.yaml")
        ok = c.experiment.name == name and "training" in c and "imbalance" in c
        check(f"config: {name} loads with required keys", ok)

    # --- label mape za evaluaciju (učitaj stvarne sačuvane mape) ---------
    maps = LabelMaps.from_json(ROOT / "data" / "splits" / "label_maps.json")
    check("maps: 14 species, 38 classes", maps.num_species == 14 and maps.num_classes == 38)

    # --- hijerarhijska evaluacija: savršena predviđanja ------------------
    # Napravi mali poravnati uzorak direktno iz stvarne strukture label-a.
    rng = np.random.default_rng(0)
    rows = []  # (species_id, disease_in_species) — bolest u okviru species-a
    for sid in range(maps.num_species):
        k = maps.num_diseases_for(sid)
        for d in range(k):
            rows.append((sid, d))
    rows = rows * 3  # ponovi da bi metrike bile stabilne
    arr = np.array(rows)
    true_s, true_d = arr[:, 0], arr[:, 1]

    perfect = evaluate_hierarchical(
        true_species=true_s, true_disease_in_species=true_d,
        pred_species=true_s.copy(), pred_disease_in_species=true_d.copy(), maps=maps,
    )
    check("hier-eval: perfect -> end-to-end macro F1 = 1.0",
          abs(perfect.end_to_end.macro_f1 - 1.0) < 1e-9)
    check("hier-eval: perfect -> species macro F1 = 1.0",
          abs(perfect.species.macro_f1 - 1.0) < 1e-9)
    check("hier-eval: perfect -> 0 errors",
          perfect.errors_from_species == 0 and perfect.errors_from_disease_only == 0)

    # --- error propagation: pokvari species na nekim uzorcima ------------
    pred_s = true_s.copy()
    pred_d = true_d.copy()
    # Promeni species za 20% uzoraka u drugi species; bolest ostaje onakva kakvu
    # je predvideo (pogrešan) head — simuliramo time što zadržavamo pred_d, ali on
    # sada pripada pogrešnom species-u. To treba da se broji kao greška uzrokovana species-om.
    n = len(true_s)
    flip_idx = rng.choice(n, size=n // 5, replace=False)
    for i in flip_idx:
        pred_s[i] = (true_s[i] + 1) % maps.num_species

    res = evaluate_hierarchical(
        true_species=true_s, true_disease_in_species=true_d,
        pred_species=pred_s, pred_disease_in_species=pred_d, maps=maps,
    )
    check("hier-eval: species errors == number flipped",
          res.n - res.species_correct == len(flip_idx),
          f"flipped={len(flip_idx)} species_wrong={res.n - res.species_correct}")
    check("hier-eval: every species error is an end-to-end error",
          res.errors_from_species == len(flip_idx),
          f"errors_from_species={res.errors_from_species}")
    check("hier-eval: no disease-only errors here (disease preds were correct)",
          res.errors_from_disease_only == 0,
          f"disease_only={res.errors_from_disease_only}")
    share = res.summary()["share_of_errors_due_to_species"]
    check("hier-eval: all errors attributed to species (share=1.0)",
          abs(share - 1.0) < 1e-9, f"share={share:.3f}")

    # --- greška samo u bolesti: tačan species, pogrešna bolest -----------
    pred_s2 = true_s.copy()
    pred_d2 = true_d.copy()
    # Pronađi Tomato uzorak (k=10) i pokvari samo njegovu bolest.
    tomato = maps.species_to_id["Tomato"]
    tmask = np.where(true_s == tomato)[0]
    corrupt = tmask[:5]
    for i in corrupt:
        pred_d2[i] = (true_d[i] + 1) % maps.num_diseases_for(tomato)
    res2 = evaluate_hierarchical(
        true_species=true_s, true_disease_in_species=true_d,
        pred_species=pred_s2, pred_disease_in_species=pred_d2, maps=maps,
    )
    check("hier-eval: disease-only errors counted, none from species",
          res2.errors_from_disease_only == len(corrupt) and res2.errors_from_species == 0,
          f"disease_only={res2.errors_from_disease_only} species={res2.errors_from_species}")

    REPORT.write_text(
        "\n".join(["=== EVAL SANITY ==="] + report
                  + ["", f"RESULT: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}"]),
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        REPORT.write_text("EVAL SANITY CRASHED:\n" + traceback.format_exc(), encoding="utf-8")
        code = 2
    sys.exit(code)
