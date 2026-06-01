"""Sanity for error_analysis non-plotting logic (no matplotlib/grad-cam needed).

Verifies: module imports despite heavy deps being absent (lazy imports);
find_misclassified returns the right indices and respects max_per_pair;
most_confused_pairs ranks off-diagonal confusions; _default_target_layer picks a
sensible layer for BaselineCNN / TransferModel.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from src.evaluation import error_analysis as ea
from src.evaluation.metrics import confusion


def _report_path() -> Path:
    if "--report" in sys.argv:
        return Path(sys.argv[sys.argv.index("--report") + 1])
    return Path(os.environ.get("REPORT_PATH",
                Path(tempfile.gettempdir()) / "pdh_ea_sanity.txt"))


REPORT = _report_path()
report: list[str] = []
failures: list[str] = []


def check(name, cond, detail=""):
    report.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    # find_misclassified
    yt = np.array([0, 0, 1, 1, 2, 2, 2])
    yp = np.array([0, 1, 1, 0, 2, 0, 0])  # wrong at idx 1,3,5,6
    wrong = ea.find_misclassified(yt, yp)
    check("find_misclassified: correct indices", list(wrong) == [1, 3, 5, 6], str(list(wrong)))
    capped = ea.find_misclassified(yt, yp, max_per_pair=1)
    # pairs: (0,1)@1, (1,0)@3, (2,0)@5, (2,0)@6 -> cap drops the 2nd (2,0)
    check("find_misclassified: max_per_pair caps duplicates",
          list(capped) == [1, 3, 5], str(list(capped)))

    # most_confused_pairs
    cm = confusion(yt, yp, num_classes=3)
    pairs = ea.most_confused_pairs(cm, ["a", "b", "c"], top_k=5)
    top = (pairs.iloc[0]["true"], pairs.iloc[0]["predicted"], int(pairs.iloc[0]["count"]))
    check("most_confused_pairs: top pair is (c,a) count 2", top == ("c", "a", 2), str(top))
    check("most_confused_pairs: no diagonal entries",
          not ((pairs["true"] == pairs["predicted"]).any()))

    # target-layer inference
    import torch.nn as nn
    from src.models.baseline_cnn import BaselineCNN
    from src.models.transfer import TransferModel

    bl = BaselineCNN(num_classes=5)
    layer = ea._default_target_layer(bl)
    check("target layer: BaselineCNN -> a module", isinstance(layer, nn.Module))

    rn = TransferModel("resnet50", 5, pretrained=False)
    rl = ea._default_target_layer(rn)
    check("target layer: resnet50 -> Bottleneck-ish module", isinstance(rl, nn.Module))

    eff = TransferModel("efficientnet_b0", 5, pretrained=False)
    el = ea._default_target_layer(eff)
    check("target layer: efficientnet -> a module", isinstance(el, nn.Module))

    REPORT.write_text(
        "\n".join(["=== ERROR-ANALYSIS SANITY ==="] + report
                  + ["", f"RESULT: {'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}"]),
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        REPORT.write_text("ERROR-ANALYSIS SANITY CRASHED:\n" + traceback.format_exc(), encoding="utf-8")
        code = 2
    sys.exit(code)
