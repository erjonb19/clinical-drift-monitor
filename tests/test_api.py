"""app/api: the page carries the notice, predictions come back with the licence, uploads stay in
memory and are size limited, clients are rate limited (and cannot spoof their address), bad
bodies are refused, and nothing but counts is kept."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import LICENSE, MAX_BYTES, RateLimiter, create_app
from cdm.serving import NOTICE


class FakeService:
    version = "fake"

    def predict(self, image: bytes) -> dict[str, Any]:
        Image.open(io.BytesIO(image)).verify()
        return {
            "label": "nv",
            "probabilities": {"nv": 0.9, "mel": 0.1},
            "ood_score": 1.0,
            "abstain": False,
            "refer_melanoma": False,
            "model_version": self.version,
            "notice": NOTICE,
            "debug_path": "/secret/internal",  # a field the response model must drop
        }


def jpeg(size: tuple[int, int] = (256, 256)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 150, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def client(limiter: RateLimiter | None = None, hops: int = 0) -> TestClient:
    app = create_app(lambda: FakeService(), limiter, hops)  # type: ignore[arg-type, return-value]
    return TestClient(app)


def post(c: TestClient, body: bytes, **headers: str) -> int:
    headers = {"content-type": "image/jpeg", **headers}
    return int(c.post("/predict", content=body, headers=headers).status_code)


def test_page_and_health_carry_the_notice() -> None:
    c = client()
    text = c.get("/").text
    assert NOTICE in text and "non-commercial" in text and "Clínic" in text
    health = c.get("/health").json()
    assert health["notice"] == NOTICE and health["predictions"] == 0


def test_prediction_comes_back_with_the_licence_and_only_counts_are_kept() -> None:
    c = client()
    r = c.post("/predict", content=jpeg(), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200 and r.json()["license"] == LICENSE
    health = c.get("/health").json()
    assert health["predictions"] == 1 and health["latency_ms_p50"] is not None


def test_a_large_upload_is_never_written_to_disk() -> None:
    temp = Path(tempfile.gettempdir())
    before = set(temp.iterdir())
    big = jpeg((2400, 1800)) + b"\0" * (2 * 1024 * 1024)  # over Starlette's 1 MB spool size
    assert post(client(), big) in (200, 400)
    assert set(temp.iterdir()) == before


def test_bodies_over_the_cap_and_bad_bodies_are_refused() -> None:
    c = client()
    assert post(c, b"\0" * (MAX_BYTES + 1)) == 413
    assert post(c, b"not an image") == 400
    assert c.get("/health").json()["rejected"] == 2


def test_rate_limit_and_a_spoofed_forwarded_address() -> None:
    c = client(RateLimiter(per_client=1, total=100), hops=1)
    # The proxy appends the real client last; a client-chosen first entry changes nothing.
    assert post(c, jpeg(), **{"x-forwarded-for": "9.9.9.9, 1.1.1.1"}) == 200
    assert post(c, jpeg(), **{"x-forwarded-for": "8.8.8.8, 1.1.1.1"}) == 429
    assert post(c, jpeg(), **{"x-forwarded-for": "2.2.2.2"}) == 200
    assert c.get("/health").json()["rate_limited"] == 1


def test_rate_limiter_window_slides_and_memory_stays_bounded() -> None:
    now = [0.0]
    limiter = RateLimiter(per_client=1, total=1000, window=60, max_clients=5, clock=lambda: now[0])
    assert limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")
    for i in range(50):
        limiter.allow(f"c{i}")
    assert len(limiter.clients) <= 5


def test_the_input_contract_quarantines_with_a_reason_and_counts_it() -> None:
    """Backlog item 18: a readable image that breaks an enforced rule gets 422 and its reason,
    is counted as quarantined (not rejected), and is never scored."""
    c = client()
    small = c.post("/predict", content=jpeg((100, 100)), headers={"content-type": "image/jpeg"})
    assert small.status_code == 422 and "resolution" in small.json()["detail"]
    wide = c.post("/predict", content=jpeg((1200, 300)), headers={"content-type": "image/jpeg"})
    assert wide.status_code == 422 and "aspect ratio" in wide.json()["detail"]
    ok = c.post("/predict", content=jpeg(), headers={"content-type": "image/jpeg"})
    assert ok.status_code == 200 and isinstance(ok.json()["input_flags"], list)
    health = c.get("/health").json()
    assert health["quarantined"] == 2 and health["rejected"] == 0 and health["predictions"] == 1


def test_a_decompression_bomb_gets_400_not_a_server_error() -> None:
    buf = io.BytesIO()
    Image.new("L", (12000, 9000)).save(buf, format="PNG")
    c = client()
    r = c.post("/predict", content=buf.getvalue(), headers={"content-type": "image/png"})
    assert r.status_code == 400 and c.get("/health").json()["rejected"] == 1


def test_every_response_carries_the_security_headers() -> None:
    """Backlog item 44: the page, JSON answers, refusals and unknown paths all get the headers,
    and the CSP's script hash matches the page's inline script, so the page still works."""
    c = client()
    page = c.get("/")
    script = page.text.split("<script>")[1].split("</script>")[0]
    want = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    for r in (page, c.get("/health"), c.post("/predict", content=b"x"), c.get("/nope")):
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
        assert r.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
        csp = r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in csp and "default-src 'none'" in csp
        script_src = next(d for d in csp.split("; ") if d.startswith("script-src"))
        assert script_src == f"script-src 'sha256-{want}'"  # no 'unsafe-inline' for scripts


def test_the_schema_is_not_served_and_responses_keep_only_their_fields() -> None:
    c = client()
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert c.get(path).status_code == 404
    out = c.post("/predict", content=jpeg(), headers={"content-type": "image/jpeg"}).json()
    assert "debug_path" not in out
    assert set(out) == {
        "label", "probabilities", "ood_score", "abstain", "refer_melanoma", "model_version",
        "notice", "input_flags", "license",
    }  # fmt: skip
    assert set(c.get("/health").json()) == {
        "status", "model_version", "notice", "predictions", "rejected", "quarantined",
        "rate_limited", "latency_ms_p50", "latency_ms_p95",
    }  # fmt: skip


def test_a_refusal_is_logged_without_address_or_body(caplog: pytest.LogCaptureFixture) -> None:
    c = client(RateLimiter(per_client=1, total=100), hops=1)
    secret = b"SECRET-BODY-BYTES"
    with caplog.at_level(logging.WARNING, logger="cdm.api.security"):
        assert post(c, b"not an image " + secret, **{"x-forwarded-for": "203.0.113.7"}) == 400
        assert post(c, jpeg(), **{"x-forwarded-for": "203.0.113.7"}) == 429
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "cdm.api.security"]
    assert [(e["outcome"], e["reason"]) for e in events] == [
        ("rejected", "unreadable"),
        ("rate_limited", "rate_limit"),
    ]
    assert all(set(e) == {"event", "time", "request_id", "outcome", "reason"} for e in events)
    assert events[0]["request_id"] != events[1]["request_id"]
    text = caplog.text
    assert "203.0.113.7" not in text and "SECRET-BODY" not in text and "testclient" not in text
