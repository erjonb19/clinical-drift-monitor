"""Public research demo of the drift monitor's classifier (Phase 3): FastAPI on CPU.

POST /predict takes one image as the raw request body (Content-Type image/*) and returns the
shipped ensemble's label and class probabilities, a Mahalanobis OOD score, an abstain flag
(score above the threshold chosen on validation) and a melanoma referral flag. Research demo,
not for medical use.

Privacy and limits (owner's conditions, 2026-10-22):
- The body is read from the request stream into memory, with a hard size cap checked as bytes
  arrive; it is never written to disk (no multipart form, which Starlette spools to a temporary
  file above 1 MB) and never logged. /health holds counts and timing only, and there is no
  access log. The hosting platform's own proxy may log addresses; this code does not.
- Requests are rate limited per client and in total, before any body is read.
- The page and every response carry the notice, the licence terms and attribution.

Run: BUNDLE_DIR=<serving bundle> uvicorn app.api.main:app --host 0.0.0.0 --port 7860
Behind a proxy (Hugging Face Spaces), set TRUSTED_PROXY_HOPS=1 so the client address for the
rate limit is the entry that proxy appended to X-Forwarded-For.

Hardening (backlog item 44): every response the app builds carries security headers (a CSP
that allows only the page's own inline script, by hash, and no framing), the OpenAPI schema is
not served, both routes have response models so no extra field can leak, and refusals are
logged as structured events with a time, request id, outcome and reason code, never an address
or a body. HSTS is set by the host at deploy, not here. Starlette's own plain-text 500 page sits
outside the middleware and has no headers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from cdm import contract
from cdm.serving import NOTICE, Service

MAX_BYTES = 10 * 1024 * 1024
LICENSE = (
    "Model weights and outputs are for non-commercial research only. The model was trained on "
    "ISIC Archive images: HAM10000 (CC BY-NC 4.0; Tschandl, Rosendahl and Kittler, Sci. Data "
    "2018), BCN20000 from Hospital Clínic de Barcelona (CC BY-NC; Combalia et al. 2019) and MSK "
    "collections (CC0). Attribution: International Skin Imaging Collaboration (ISIC) Archive; "
    "per-image attribution is in the project's data card."
)


class RateLimiter:
    """Sliding windows: ``per_client`` requests per address and ``total`` overall. The total is
    checked first, empty client windows are dropped, and at most ``max_clients`` addresses are
    tracked (the least recently seen go first), so memory stays bounded."""

    def __init__(self, per_client: int = 10, total: int = 60, window: float = 60.0,
                 max_clients: int = 10_000,
                 clock: Callable[[], float] = time.monotonic) -> None:  # fmt: skip
        self.per_client, self.total, self.window, self.clock = per_client, total, window, clock
        self.max_clients = max_clients
        self.clients: OrderedDict[str, deque[float]] = OrderedDict()
        self.all: deque[float] = deque()
        self.lock = threading.Lock()

    def _trim(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def allow(self, client: str) -> bool:
        now = self.clock()
        with self.lock:
            self._trim(self.all, now)
            if len(self.all) >= self.total:
                return False
            mine = self.clients.get(client)
            if mine is not None:
                self._trim(mine, now)
                if len(mine) >= self.per_client:
                    return False
            else:
                mine = deque()
                self.clients[client] = mine
                while len(self.clients) > self.max_clients:
                    self.clients.popitem(last=False)
            self.clients.move_to_end(client)
            self.all.append(now)
            mine.append(now)
            for key in [k for k, q in self.clients.items() if not q]:
                del self.clients[key]
            return True


CONTRACT = contract.load_config()
SECURITY_LOG = logging.getLogger("cdm.api.security")


class Prediction(BaseModel):
    """The only fields /predict returns."""

    label: str
    probabilities: dict[str, float]
    ood_score: float
    abstain: bool
    refer_melanoma: bool
    model_version: str
    notice: str
    input_flags: list[str]
    license: str


class Health(BaseModel):
    """The only fields /health returns: status, version and counts."""

    status: str
    model_version: str
    notice: str
    predictions: int
    rejected: int
    quarantined: int
    rate_limited: int
    latency_ms_p50: float | None
    latency_ms_p95: float | None


def security_event(outcome: str, reason: str) -> None:
    """One structured line per refusal: UTC time, a fresh request id, the outcome and a reason
    code. No address, header, filename or body is logged (the privacy conditions above)."""
    event = {"event": "api_refusal", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "request_id": uuid.uuid4().hex, "outcome": outcome, "reason": reason}  # fmt: skip
    SECURITY_LOG.warning(json.dumps(event, sort_keys=True))


class Stats:
    """Counts and timing only: no image, filename, hash or address is kept."""

    def __init__(self) -> None:
        self.counts = {"predictions": 0, "rejected": 0, "quarantined": 0, "rate_limited": 0}
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


def client_address(request: Request, trusted_hops: int) -> str:
    """The client for the rate limit: with ``trusted_hops`` proxies in front, the entry the
    nearest trusted proxy appended to X-Forwarded-For (counted from the right, so a client
    cannot choose it); otherwise the connecting address. Kept in memory only."""
    if trusted_hops > 0:
        hops = [
            h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()
        ]
        if len(hops) >= trusted_hops:
            return hops[-trusted_hops]
    return request.client.host if request.client else "?"


async def read_capped(request: Request, limit: int) -> bytes:
    """The request body, in memory, refused as soon as it passes ``limit`` bytes."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "image larger than 10 MB")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "image larger than 10 MB")
    return bytes(body)


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
<p>Choose one dermoscopic image. The answer gives seven-class probabilities, an
out-of-distribution score, and an <em>abstain</em> flag when the image looks unlike the training
data.</p>
<input type="file" id="f" accept="image/*"> <button id="b">Classify</button>
<pre id="out" style="white-space: pre-wrap"></pre>
<script>
document.getElementById("b").onclick = async () => {{
  const file = document.getElementById("f").files[0];
  if (!file) return;
  const r = await fetch("/predict", {{method: "POST", body: file,
    headers: {{"Content-Type": file.type || "application/octet-stream"}}}});
  document.getElementById("out").textContent = JSON.stringify(await r.json(), null, 2);
}};
</script>
<p><small>{LICENSE}</small></p>
<p><small>Source and evaluation: github.com/erjonb19/clinical-drift-monitor</small></p>
</body></html>"""


def script_hash(html: str) -> str:
    """The CSP source for the page's one inline script: sha256 of its exact text. The page is
    this module's own constant HTML; plain string splitting (no regex) requires exactly one
    bare <script>...</script> and refuses any other script tag, in any letter case."""
    lower = html.lower()
    if lower.count("<script") != 1 or lower.count("</script") != 1 or html.count("<script>") != 1:
        raise ValueError("the page must have exactly one inline script")
    _, _, rest = html.partition("<script>")
    script, found, _ = rest.partition("</script>")
    if not found:
        raise ValueError("the page must have exactly one inline script")
    digest = hashlib.sha256(script.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def security_headers(html: str) -> dict[str, str]:
    """Headers for every response. The page uses inline style attributes, so styles stay
    'unsafe-inline'; scripts run only if they match the hash, and the page cannot be framed."""
    csp = (
        "default-src 'none'; "
        f"script-src {script_hash(html)}; "
        "style-src 'unsafe-inline'; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )
    return {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }


def create_app(
    load: Callable[[], Service], limiter: RateLimiter | None = None, trusted_hops: int = 0
) -> FastAPI:
    app = FastAPI(
        title="Skin lesion classifier (research demo)",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    headers = security_headers(page())

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        response: Response = await call_next(request)
        response.headers.update(headers)
        return response

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

    @app.get("/health", response_model=Health)
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_version": service().version, "notice": NOTICE,
                **stats.snapshot()}  # fmt: skip

    @app.post("/predict", response_model=Prediction)
    async def predict(request: Request) -> dict[str, Any]:
        if not limits.allow(client_address(request, trusted_hops)):
            stats.add("rate_limited")
            security_event("rate_limited", "rate_limit")
            raise HTTPException(429, "rate limit: try again in a minute")
        try:
            data = await read_capped(request, MAX_BYTES)
        except HTTPException:
            stats.add("rejected")
            security_event("rejected", "too_large")
            raise
        # The input contract (backlog item 18): an unreadable body is the client's error (400);
        # a readable image that breaks an enforced rule is quarantined with its reason (422).
        # Duplicates are never checked here: the API keeps no hash of any upload.
        verdict = await run_in_threadpool(contract.check, data, CONTRACT, (), False)
        if verdict.unreadable:
            stats.add("rejected")
            security_event("rejected", "unreadable")
            raise HTTPException(400, "could not read that body as an image")
        if not verdict.ok:
            stats.add("quarantined")
            security_event("quarantined", "input_contract")
            raise HTTPException(422, "; ".join(verdict.quarantined))
        model = service()  # a bundle problem is a server error, not the client's file
        start = time.perf_counter()
        try:
            out = await run_in_threadpool(model.predict, data)
        except Exception:  # noqa: BLE001 - any undecodable upload is the client's error
            stats.add("rejected")
            security_event("rejected", "undecodable")
            raise HTTPException(400, "could not read that body as an image") from None
        finally:
            del data
        stats.add("predictions", 1000 * (time.perf_counter() - start))
        return {**out, "input_flags": verdict.flags, "license": LICENSE}

    return app


def _load() -> Service:
    return Service.load(Path(os.environ.get("BUNDLE_DIR", "/app/bundle")))


def limiter_from_env(env: dict[str, str]) -> RateLimiter:
    """The rate limits, 10 per client and 60 in total a minute unless set. Backlog item 21: the
    local compose demo raises them for its own client. A value that is not a whole number from 1
    to 10,000 stops the API at startup (fail closed), never falls back to no limit."""
    values = {}
    for name, default in (("API_RATE_PER_CLIENT", 10), ("API_RATE_TOTAL", 60)):
        raw = env.get(name, str(default))
        if not (raw.isascii() and raw.isdigit()) or not 1 <= int(raw) <= 10_000:
            raise ValueError(f"{name} must be a whole number from 1 to 10000")
        values[name] = int(raw)
    return RateLimiter(per_client=values["API_RATE_PER_CLIENT"], total=values["API_RATE_TOTAL"])


app = create_app(
    _load,
    limiter=limiter_from_env(dict(os.environ)),
    trusted_hops=int(os.environ.get("TRUSTED_PROXY_HOPS", "0")),
)
