"""A serving bundle for CI only: an untrained EfficientNet-B0 and a random detector, so the API's
Docker image can be built and smoke-tested without the real weights (never in git). Its outputs
mean nothing. Writes dist/test_bundle by default (gitignored); it refuses to overwrite a
folder holding another bundle, so the real one cannot be clobbered.

Usage: python scripts/make_test_bundle.py [--out dist/test_bundle]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cdm.ood import MahalanobisDetector
from cdm.serving import detector_arrays
from cdm.train import build_model, weights_hash

REPO = Path(__file__).resolve().parents[1]
VERSION = "ci-test-bundle-untrained"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "dist" / "test_bundle")
    out = parser.parse_args().out
    existing = out / "manifest.json"
    if existing.exists() and json.loads(existing.read_text())["version"] != VERSION:
        raise SystemExit(f"{out} holds a real bundle: refusing to overwrite it")
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = build_model(pretrained=False, arch="b0").eval()
    torch.save(model.state_dict(), OUT / "seed0.pt")
    rng = np.random.default_rng(0)
    det = MahalanobisDetector(8).fit(rng.normal(size=(140, 1280)), np.repeat(np.arange(7), 20))
    np.savez(OUT / "seed0_detector.npz", **detector_arrays(det))  # type: ignore[arg-type]
    files = ["seed0.pt", "seed0_detector.npz"]
    manifest = {
        "version": VERSION,
        "abstain_threshold": 1e12,
        "refer_threshold": 0.0638,
        "seeds": [
            {
                "seed": 0,
                "checkpoint": files[0],
                "detector": files[1],
                "weights_sha256": weights_hash(model),
            }
        ],  # fmt: skip
        "sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
        "note": "CI test bundle: untrained weights, meaningless outputs",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
