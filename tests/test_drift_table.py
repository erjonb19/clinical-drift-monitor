"""The drift table's ship rule: highest mean AUROC on the two real sites for the centralized
models, FPR@95TPR breaking ties; method C's sensitivity row never takes part."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from drift_table import DETECTORS, SENSITIVITY, SHIFTS, build  # noqa: E402


def result(
    kind: str, seed: int, aurocs: dict[str, float], fprs: dict[str, float]
) -> dict[str, Any]:
    metrics = {
        det: {s: {"auroc": aurocs.get(det, 0.6), "fpr95": fprs.get(det, 0.5)} for s in SHIFTS}
        for det in (*DETECTORS, SENSITIVITY)
    }
    return {"kind": kind, "seed": seed, "metrics": metrics}


def test_ship_rule_uses_centralized_real_sites_and_breaks_ties_on_fpr() -> None:
    central = {"gram": 0.80, "energy": 0.80, SENSITIVITY: 0.99}
    results = [result("final", s, central, {"gram": 0.4, "energy": 0.3}) for s in range(3)]
    # Federated models favour max softmax: they must not change the decision.
    results += [result(k, s, {"max_softmax": 0.95}, {}) for k in ("fedavg", "fedprox")
                for s in range(3)]  # fmt: skip
    out = build(results)
    assert out["ship"] == "energy"  # tied AUROC with Gram, lower FPR@95TPR
    assert SENSITIVITY not in out["ranking"]
    assert out["table"]["centralized"]["gram"]["buenos_aires"]["auroc"]["n"] == 3
