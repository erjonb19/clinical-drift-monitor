"""Backlog item 21: the docker compose demo's client, bundle fetch, rate-limit setting and
compose file. No network: ISIC and the API are faked."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cdm import compose_demo

REPO = Path(__file__).resolve().parents[1]


def manifest(tmp: Path, name: str, blobs: dict[str, bytes]) -> Path:
    path = tmp / name
    lines = ["isic_id,lesion_id,diagnosis,license,attribution,url,sha256"]
    for isic_id, blob in blobs.items():
        url = f"{compose_demo.ISIC_FILES[1]}{isic_id}.jpg"
        lines.append(f"{isic_id},IL_{isic_id},Benign,CC-0,Anonymous,{url},"
                     f"{hashlib.sha256(blob).hexdigest()}")  # fmt: skip
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_file_whose_hash_differs_is_refused(tmp_path: Path) -> None:
    rows = compose_demo.read_manifest(manifest(tmp_path, "m.csv", {"ISIC_1": b"right"}))
    with pytest.raises(ValueError, match="SHA-256"):
        compose_demo.image_bytes(rows[0], tmp_path / "cache", lambda url: b"wrong")


def test_only_isic_urls_with_full_hashes_are_read(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    header = "isic_id,lesion_id,diagnosis,license,attribution,url,sha256\n"
    path.write_text(header + "ISIC_1,IL_1,B,CC-0,A,https://example.com/x.jpg," + "0" * 64 + "\n")
    with pytest.raises(ValueError, match="unexpected URL"):
        compose_demo.read_manifest(path)


def test_the_client_waits_on_the_rate_limit_and_runs_the_drift_test(tmp_path: Path) -> None:
    ref = {f"ISIC_R{i}": f"r{i}".encode() for i in range(30)}
    new = {f"ISIC_S{i}": f"s{i}".encode() for i in range(30)}
    blobs = {**ref, **new}
    calls: list[int] = []

    def post(data: bytes) -> tuple[int, dict[str, Any]]:
        calls.append(1)
        if len(calls) == 1:
            return 429, {}
        shifted = data.startswith(b"s")
        score = 50.0 + int(data[1:]) if shifted else 10.0 + int(data[1:]) / 10
        return 200, {"ood_score": score, "abstain": shifted, "model_version": "test"}

    def fetch(url: str) -> bytes:
        return blobs[url.rsplit("/", 1)[1].removesuffix(".jpg")]

    waits: list[float] = []
    cache = tmp_path / "cache"
    reference = compose_demo.score(
        compose_demo.read_manifest(manifest(tmp_path, "r.csv", ref)), cache, fetch, post,
        sleep=waits.append,
    )  # fmt: skip
    shifted = compose_demo.score(
        compose_demo.read_manifest(manifest(tmp_path, "s.csv", new)), cache, fetch, post
    )
    assert waits == [6.0] and len(reference) == len(shifted) == 30
    out = compose_demo.summary(reference, shifted)
    assert out["drift"]["alert"] and out["shifted"]["abstain_share"] == 1.0
    assert "not the monitoring job's comparison" in out["comparison"]
    same = compose_demo.summary(reference, reference)
    assert not same["drift"]["alert"]


def test_a_refused_image_stops_the_run(tmp_path: Path) -> None:
    rows = compose_demo.read_manifest(manifest(tmp_path, "m.csv", {"ISIC_1": b"x"}))
    with pytest.raises(RuntimeError, match="answered 400"):
        compose_demo.score(rows, tmp_path / "c", lambda u: b"x", lambda d: (400, {}))


def test_rate_limits_come_from_the_environment_or_stop_the_api() -> None:
    from app.api.main import limiter_from_env

    default = limiter_from_env({})
    assert (default.per_client, default.total) == (10, 60)
    raised = limiter_from_env({"API_RATE_PER_CLIENT": "120", "API_RATE_TOTAL": "240"})
    assert (raised.per_client, raised.total) == (120, 240)
    for bad in ("0", "-1", "ten", "", "10001", "1.5"):
        with pytest.raises(ValueError, match="whole number"):
            limiter_from_env({"API_RATE_PER_CLIENT": bad})


def fetch_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fetch_champion_bundle", REPO / "scripts" / "fetch_champion_bundle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_bundle_fetch_refuses_without_a_release_asset(tmp_path: Path) -> None:
    module = fetch_script()
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"bundle_asset": {"url": None, "sha256": None}}))
    with pytest.raises(SystemExit, match="OWNER_STEPS"):
        module.asset(config)
    config.write_text(json.dumps({"bundle_asset": {"url": "https://example.com/b.tgz",
                                                   "sha256": "a" * 64}}))  # fmt: skip
    with pytest.raises(SystemExit, match="release asset"):
        module.asset(config)
    taken = tmp_path / "bundle"
    taken.mkdir()
    with pytest.raises(SystemExit, match="never overwritten"):
        module.main(["--dest", str(taken), "--config", str(config)])


def test_the_compose_file_keeps_the_api_local_and_the_config_read_only() -> None:
    text = (REPO / "compose.yaml").read_text("utf-8")
    assert '"127.0.0.1:7860:7860"' in text  # not published on every interface
    assert "./config:/demo/config:ro" in text
    assert text.count("read_only: true") == 2 and text.count("no-new-privileges:true") == 2
    assert "privileged" not in text and "SECRET" not in text.upper()
