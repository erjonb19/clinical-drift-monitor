"""Backlog item 1, after replay month 12: score the new public ISIC images that arrive each
month and report drift signals that need no labels
(``results/phase3/monthly/post_replay_spec.json``).

The owner's rules (2026-11-06): score whatever new images arrive; report label-free drift
signals (input or embedding drift, prediction mix, confidence); a month with no new images
records ``no_new_data`` explicitly, never a silent green run; labelled metrics come later.
The proposals the owner confirmed: new means a public ISIC ID the job has not seen; images are
scored with the ``champion`` model; each signal is compared with the validation baseline; the
status is ``scored``, ``no_new_data`` or ``failed``. The tests and thresholds are fixed in the
spec, committed before any such run. Every signal is computed the same way for the baseline and
the month, with statistics fitted on the baseline only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chisquare, ks_2samp

from cdm.monitor import drift_test, psi

Array = NDArray[np.float64]
SCORED, NO_NEW_DATA, FAILED = "scored", "no_new_data", "failed"
ALERT_P = 0.01
# Below this a month's signals are counts only and never alert. 100 is the repo's minimum for a
# reported metric (cdm.eval.MIN_GROUP); with no drift, PSI over 10 bins averages about 0.78 at
# 30 images and 0.10 at 100 (a review's synthetic check, 2026-11-09, not a committed result), so
# the 0.2 PSI floor only means something from about 100 images.
MIN_IMAGES = 100
EMBEDDING_COMPONENTS = 5
EMBEDDING_PSI = 0.2
MIX_TVD = 0.1
CONFIDENCE_SHIFT = 0.05
MIN_EXPECTED = 5.0  # chi-square: classes expected below this are pooled


def new_image_ids(listed: Iterable[str], seen: Iterable[str]) -> list[str]:
    """Public ISIC IDs the job has not seen before, sorted, without duplicates."""
    known = set(seen)
    return sorted({i for i in listed if i not in known})


def embedding_drift(
    baseline: Array, current: Array, k: int = EMBEDDING_COMPONENTS
) -> dict[str, Any]:
    """Each image's features projected onto the baseline's top-k principal components (fitted on
    the baseline only); a KS test and PSI per component. Alert when any component has
    KS p < ALERT_P / k (Bonferroni) and PSI >= EMBEDDING_PSI."""
    mean = baseline.mean(axis=0)
    _, _, vt = np.linalg.svd(baseline - mean, full_matrices=False)
    comps = vt[:k]
    b, c = (baseline - mean) @ comps.T, (current - mean) @ comps.T
    per = []
    for j in range(comps.shape[0]):
        ks = ks_2samp(c[:, j], b[:, j])
        per.append({"component": j, "ks_p": float(ks.pvalue), "psi": psi(b[:, j], c[:, j])})
    alpha = ALERT_P / len(per)
    alert = any(p["ks_p"] < alpha and p["psi"] >= EMBEDDING_PSI for p in per)
    return {"components": per, "alpha": alpha, "alert": bool(alert)}


def prediction_mix(
    baseline: Sequence[int], current: Sequence[int], n_classes: int
) -> dict[str, Any]:
    """The month's predicted classes against the baseline's predicted-class shares: a chi-square
    goodness-of-fit test and the total variation distance. Alert when p < ALERT_P and
    TVD >= MIX_TVD.

    Classes expected below MIN_EXPECTED images are pooled into one cell; if that cell is still
    below MIN_EXPECTED it joins the smallest other cell. A class the baseline never predicted
    stays in the pool, so the strongest shift (a new class) is tested, not dropped. With fewer
    than two cells no chi-square test is possible: ``chi2_p`` is None and the TVD alone decides.
    The same holds when the month predicts a class the baseline never predicted (expected 0):
    merged into another cell it could hide behind a class that vanished, so the classes are
    listed in ``unseen_classes`` and the TVD alone decides.
    """
    p = np.bincount(np.asarray(baseline), minlength=n_classes) / len(baseline)
    obs = np.bincount(np.asarray(current), minlength=n_classes).astype(np.float64)
    exp = p * len(current)
    keep = exp >= MIN_EXPECTED
    o, e = [float(x) for x in obs[keep]], [float(x) for x in exp[keep]]
    if (~keep).any():
        po, pe = float(obs[~keep].sum()), float(exp[~keep].sum())
        if pe >= MIN_EXPECTED or not e:
            o.append(po)
            e.append(pe)
        else:
            j = int(np.argmin(e))
            o[j] += po
            e[j] += pe
    tvd = float(0.5 * np.abs(obs / len(current) - p).sum())
    unseen = [int(i) for i in np.flatnonzero((p == 0) & (obs > 0))]
    chi2_p: float | None = None
    if not unseen and len(e) >= 2 and min(e) > 0:
        chi2_p = float(chisquare(o, e).pvalue)
        alert = chi2_p < ALERT_P and tvd >= MIX_TVD
    else:
        alert = tvd >= MIX_TVD
    return {
        "observed_share": (obs / len(current)).tolist(),
        "baseline_share": p.tolist(),
        "chi2_p": chi2_p,
        "tvd": tvd,
        "unseen_classes": unseen,
        "alert": bool(alert),
    }


def confidence_drift(baseline: Array, current: Array) -> dict[str, Any]:
    """Top softmax probability: KS test against the baseline's, and the change in its median.
    Alert when p < ALERT_P and |median change| >= CONFIDENCE_SHIFT."""
    ks = ks_2samp(current, baseline)
    shift = float(np.median(current) - np.median(baseline))
    return {
        "ks_p": float(ks.pvalue),
        "median_change": shift,
        "alert": bool(ks.pvalue < ALERT_P and abs(shift) >= CONFIDENCE_SHIFT),
    }


def label_free_signals(
    baseline: dict[str, Any], current: dict[str, Any], n_classes: int
) -> dict[str, Any]:
    """All four signals for one month. Each dict holds ``ood`` (scores), ``features`` (n x d),
    ``predicted`` (class indices) and ``top_prob`` (top softmax probability)."""
    n = len(current["ood"])
    b, c = baseline, current
    signals = {
        "ood": drift_test(np.asarray(b["ood"]), np.asarray(c["ood"])),
        "embedding": embedding_drift(np.asarray(b["features"]), np.asarray(c["features"])),
        "prediction_mix": prediction_mix(b["predicted"], c["predicted"], n_classes),
        "confidence": confidence_drift(np.asarray(b["top_prob"]), np.asarray(c["top_prob"])),
    }
    if n < MIN_IMAGES:
        for s in signals.values():
            s["alert"] = False
            if "warning" in s:
                s["warning"] = False
        return {"images": n, "note": f"counts only: under {MIN_IMAGES} images", **signals}
    return {"images": n, **signals}


def month_status(n_new: int, error: str | None = None) -> dict[str, Any]:
    """The month's explicit status: ``failed`` (with the error), ``no_new_data`` or ``scored``."""
    if error:
        return {"status": FAILED, "error": error}
    if n_new == 0:
        return {"status": NO_NEW_DATA, "new_images": 0}
    return {"status": SCORED, "new_images": n_new}


def month_alert(signals: dict[str, Any]) -> bool:
    """The month is in alert when any of the four signals alerts."""
    return any(
        bool(signals[k]["alert"]) for k in ("ood", "embedding", "prediction_mix", "confidence")
    )
