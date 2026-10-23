"""app/api: the page carries the notice, predictions come back with the licence, uploads are
rate limited and size limited, bad files are refused, and nothing but counts is kept."""

from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import LICENSE, RateLimiter, create_app
from cdm.serving import NOTICE


class FakeService:
    version = "fake"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def predict(self, image: bytes) -> dict[str, Any]:
        Image.open(io.BytesIO(image)).verify()
        self.seen.append(len(image))
        return {"label": "nv", "abstain": False, "notice": NOTICE}


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 150, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def client(limiter: RateLimiter | None = None) -> tuple[TestClient, FakeService]:
    fake = FakeService()
    return TestClient(create_app(lambda: fake, limiter)), fake  # type: ignore[arg-type, return-value]


def test_page_and_health_carry_the_notice() -> None:
    c, _ = client()
    assert NOTICE in c.get("/").text and "non-commercial" in c.get("/").text
    health = c.get("/health").json()
    assert health["notice"] == NOTICE and health["predictions"] == 0


def test_prediction_comes_back_with_the_licence_and_only_counts_are_kept() -> None:
    c, _ = client()
    r = c.post("/predict", files={"image": ("lesion.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["license"] == LICENSE
    health = c.get("/health").json()
    assert health["predictions"] == 1 and health["latency_ms_p50"] is not None
    assert "lesion.jpg" not in str(health)


def test_bad_file_and_rate_limit() -> None:
    c, _ = client(RateLimiter(per_client=2, total=100))
    assert (
        c.post("/predict", files={"image": ("x.jpg", b"not an image", "image/jpeg")}).status_code
        == 400
    )
    assert c.post("/predict", files={"image": ("x.jpg", jpeg(), "image/jpeg")}).status_code == 200
    assert c.post("/predict", files={"image": ("x.jpg", jpeg(), "image/jpeg")}).status_code == 429
    assert c.get("/health").json()["rate_limited"] == 1


def test_rate_limiter_window_slides() -> None:
    now = [0.0]
    limiter = RateLimiter(per_client=1, total=10, window=60, clock=lambda: now[0])
    assert limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")


def test_the_rate_limit_uses_the_forwarded_client_behind_a_proxy() -> None:
    c, _ = client(RateLimiter(per_client=1, total=100))
    files = {"image": ("x.jpg", jpeg(), "image/jpeg")}
    assert (
        c.post("/predict", files=files, headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    )
    assert (
        c.post("/predict", files=files, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
    )
    assert (
        c.post("/predict", files=files, headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429
    )
