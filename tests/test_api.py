"""app/api: the page carries the notice, predictions come back with the licence, uploads stay in
memory and are size limited, clients are rate limited (and cannot spoof their address), bad
bodies are refused, and nothing but counts is kept."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import LICENSE, MAX_BYTES, RateLimiter, create_app
from cdm.serving import NOTICE


class FakeService:
    version = "fake"

    def predict(self, image: bytes) -> dict[str, Any]:
        Image.open(io.BytesIO(image)).verify()
        return {"label": "nv", "abstain": False, "notice": NOTICE}


def jpeg(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 150, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def client(limiter: RateLimiter | None = None, hops: int = 0) -> TestClient:
    app = create_app(lambda: FakeService(), limiter, hops)  # type: ignore[arg-type, return-value]
    return TestClient(app)


def post(c: TestClient, body: bytes, **headers: str) -> int:
    headers = {"content-type": "image/jpeg", **headers}
    return c.post("/predict", content=body, headers=headers).status_code


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
