"""Public research demo of the drift monitor's classifier (Phase 3): FastAPI on CPU.

POST /predict takes one image and returns the shipped ensemble's label and class
probabilities, a Mahalanobis OOD score, an abstain flag (score above the threshold chosen on
validation) and a melanoma referral flag. Research demo, not for medical use.

Privacy and limits (owner's conditions, 2026-10-22):
- An uploaded image is read into memory for one request and never written or logged. The logs
  and /health hold counts and timing only.
- Requests are rate limited per client address and in total.
- The page and every response carry the notice, the licence terms and attribution.

Run: BUNDLE_DIR=<serving bundle> uvicorn app.api.main:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from cdm.serving import NOTICE, Service

MAX_BYTES = 10 * 1024 * 1024
LICENSE = (
    "Model weights and outputs are for non-commercial research only. The model was trained on "
    "ISIC Archive images: HAM10000 (CC BY-NC 4.0; Tschandl, Rosendahl and Kittler, Sci. Data "
    "2018), BCN20000 from Hospital Clinic de Barcelona (CC BY-NC) and MSK collections (CC0). "
    "Attribution: International Skin Imaging Collaboration (ISIC) Archive."
)
log = logging.getLogger("cdm.api")


class RateLimiter:
    """Sliding one-minute windows: ``per_client`` requests per address and ``total`` overall."""

    def __init__(self, per_client: int = 10, total: int = 60, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:  # fmt: skip
        self.per_client, self.total, self.window, self.clock = per_client, total, window, clock
        self.clients: dict[str, deque[float]] = {}
        self.all: deque[float] = deque()
        self.lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = self.clock()
        with self.lock:
            for q in (self.all, self.clients.setdefault(client, deque())):
                while q and now - q[0] > self.window:
                    q.popleft()
            mine = self.clients[client]
            if len(self.all) >= self.total or len(mine) >= self.per_client:
                return False
            self.all.append(now)
            mine.append(now)
            return True


class Stats:
    """Counts and timing only: no image, filename, hash or address is kept."""

    def __init__(self) -> None:
        self.counts = {"predictions": 0, "rejected": 0, "rate_limited": 0}
        self.millis: deque[float] = deque(maxlen=1000)
        self.lock = threading.Lock()

    def add(self, key: str, millis: float | None = None) -> None:
        with self.lock:
            self.counts[key] += 1
            if millis is not None:
                self.millis.append(millis)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            times = sorted(self.millis)

        def pick(q: float) -> float | None:
            return round(times[min(len(times) - 1, int(q * len(times)))], 1) if times else None

        return {**self.counts, "latency_ms_p50": pick(0.5), "latency_ms_p95": pick(0.95)}


def page() -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Skin lesion classifier: research demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: system-ui, sans-serif; max-width: 42rem; margin: 2rem auto;
padding: 0 1rem">
<p style="border: 2px solid #b00; padding: .75rem; font-weight: 600">{NOTICE} Do not use it to
diagnose, screen or triage anyone. Uploaded images are processed in memory and never stored or
logged.</p>
<h1>Skin lesion classifier with a drift score</h1>
<p>Upload one dermoscopic image. The answer gives seven-class probabilities, an
out-of-distribution score, and an <em>abstain</em> flag when the image looks unlike the training
data.</p>
<form action="/predict" method="post" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required> <button type="submit">Classify</button>
</form>
<p><small>{LICENSE}</small></p>
<p><small>Source and evaluation: github.com/erjonb19/clinical-drift-monitor</small></p>
</body></html>"""


def create_app(load: Callable[[], Service], limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="Skin lesion classifier (research demo)", docs_url=None, redoc_url=None)
    limits = limiter or RateLimiter()
    stats = Stats()
    state: dict[str, Service] = {}

    def service() -> Service:
        if "s" not in state:
            state["s"] = load()
        return state["s"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_version": service().version, "notice": NOTICE,
                **stats.snapshot()}  # fmt: skip

    @app.post("/predict")
    async def predict(request: Request, image: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        # Behind a hosting proxy every request comes from the proxy, so the first forwarded
        # address is the client. It is used for the rate limit in memory and never logged.
        forwarded = request.headers.get("x-forwarded-for", "")
        client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
        if not limits.allow(client):
            stats.add("rate_limited")
            raise HTTPException(429, "rate limit: try again in a minute")
        data = await image.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            stats.add("rejected")
            raise HTTPException(413, "image larger than 10 MB")
        start = time.perf_counter()
        try:
            out = service().predict(data)
        except Exception:  # noqa: BLE001 - any undecodable upload is the client's error
            stats.add("rejected")
            raise HTTPException(400, "could not read that file as an image") from None
        finally:
            del data
        millis = 1000 * (time.perf_counter() - start)
        stats.add("predictions", millis)
        log.info("prediction in %.0f ms", millis)
        return {**out, "license": LICENSE}

    return app


def _load() -> Service:
    return Service.load(Path(os.environ.get("BUNDLE_DIR", "/app/bundle")))


app = create_app(_load)
