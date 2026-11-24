# Overnight run, 2026-11-11

Repo-side work only, as asked. No paid model calls, no Haiku spend, no GPU run, no
Databricks deploy or workspace change, no release tag or image push, no change to
`.claude/settings.json` or any permission setting, no force push, no data deleted. The
scheduler probe (item 1) was left for you. Under the new merge rule (#85), every PR below
was merged only with CI green on its final commit and after a fresh review by a separate
agent that found nothing blocking.

## Merged (26 PRs, in merge order)

| PR | Item | What |
|---|---|---|
| #84 | 22 | The security audit (you approved it). |
| #89 | 11 | Dependabot holds fastapi, open_clip_torch and transformers to patches; #83 closed (fastapi 0.162 broke flwr 1.39's pin). |
| #90 | 40 | `.gitignore` covers the standard's secret and state files. |
| #85 | 11, 24, 43 | The standing merge rule in CLAUDE.md and the PLAN.md decisions log; items 11, 24 and 43 ticked. |
| #95 | 29 | Images over 50 megapixels refused from the header, before decoding. |
| #86 | 45 | The $5 Haiku cap fails closed when the spend log is missing or shrinks. |
| #94 | 36 | ruff's security rules (`S`) on, each exception with its reason. |
| #101 | 33 | CIFAR-10 files are MD5-checked before unpickling. |
| #96 | 31 | `torch.load(weights_only=True)` everywhere; torch floor 2.6. |
| #102 | 38 | Dependabot cooldown of 7 days. |
| #98 | 47 | Demo and loader hardening. Built, not ticked: it waits for your redeploy. |
| #91 | 41 | `.env.example` with every variable the code reads. |
| #107 | (fix) | Main's lint, broken by #94 plus #98 (see "What went wrong"). |
| #87 | 28 | Read-only workflow tokens; no checkout keeps credentials. |
| #92 | 42 | `SECURITY.md` points to private vulnerability reporting. |
| #99 | 46 | Formula-like text in the G5 CSV error sheets written as text. |
| #103 | 44 | API: security headers, no OpenAPI schema, response models, refusal events. |
| #105 | 30 | Uploads decoded only by the JPEG and PNG plugins. |
| #106 | 32 | The bundle manifest must hash every checkpoint and detector; the deploy script checks the same. |
| #88 | 25 | gitleaks over the full history on every push, with a planted-key self-test. |
| #100 | 35 | The Docker base image pinned by digest. |
| #110 | 11 | Dependabot holds mpmath to patches (sympy 1.14 needs < 1.4). |
| #93 | 39 | Trivy fails on fixable HIGH. Passing needed two image changes: setuptools removed (old vendored jaraco.context and wheel), and Debian security updates applied (apt, zlib1g). |
| #109 | 9 | One-page change-control plan; a test regenerates the model card, plan and standards doc and fails if any is stale. |
| #111 | 34 | Every GitHub Action pinned to a commit. |
| #112 | (Dependabot) | 13 Python updates (see below). |

Order: item 45 first, then the Mediums, 29 and 31, then the Lows, then item 9, as you asked.
The merge order differed slightly from the work order: each PR merged when its CI went green.

## Closed, held or open

- **#83** (Dependabot): closed. Its fastapi 0.162 conflicts with flwr 1.39's `fastapi<0.139`. #89 holds fastapi to patches.
- **#104** (Dependabot, 18 updates): opened 06:09 UTC and closed by Dependabot itself at 07:33 ("updatable in another way"). I did not review it; #108 replaced it.
- **#108** (Dependabot, 14 updates): reviewed and held. Its mpmath 1.4.1 cannot be installed with sympy 1.14.0, which is pinned in the same constraints file. CI was green only because the Dockerfile installs torch, with sympy and mpmath, before the constraints apply. After #110 merged, Dependabot closed #108 itself.
- **#112** (Dependabot, 13 updates): merged. It is #108 without mpmath, with filelock at 4.2.8 instead of 4.2.7. A dry run shows the constraints resolve, and all of CI passed. Points for you:
  - `pyproject.toml` now has onnxruntime 1.31.1, while `databricks/onnx_export_job.json` and the committed ONNX results use 1.31.0;
  - the image runs starlette 1.10.0, while CI's API tests run 1.3.1 (flwr's cap);
  - the constraints file's header still says it passed the local Docker test on 2026-10-23.
- **#97** (item 23, `.claude/settings.json`): open, for you. I will not merge it.
- **#50**: not touched; it waits for your threshold decisions.
- **This PR**: the README and PROGRESS lines for every item above, the blocked notes, a BACKLOG guard test, and this report.

## Blocked, with the reason written under the item

- **Items 26 and 27** (uv lock; hashed image installs): uv is approved, but this cloud session's proxy returns 403 for `download.pytorch.org`, so the CPU torch wheels can't be locked or hashed. Unblock: allow that host in the environment's network settings, or run `uv lock` on the laptop.
- **Item 37** (zizmor): a new tool, so it needs your OK under directive 11.
- **Item 47**: built and merged, waiting on your Streamlit redeploy.
- **Item 23**: #97, your decision.
- **Item 1**: the probe, which you run tomorrow.

Everything else in the Order of work also waits on you: GPU or Databricks runs, accounts, or #50's thresholds.

## Only you can check or do

1. **Required checks:** add `gitleaks` (the Secrets workflow) to branch protection on `main`.
2. **Default workflow permission:** Settings, Actions, General, "Workflow permissions": set it to read-only. I can't read it without admin access.
3. **Admin bypass:** branch protection is enforced for non-admins only, so you can still merge without the checks. Turn on "Do not allow bypassing" if you want it to bind you too.
4. **Anthropic Console:** put a spend limit on the Haiku key. Item 45's client-side cap is not the real backstop.
5. **Dependabot alerts:** the push banner reported 5 on `main` (4 high, 1 moderate). I can't list them without admin access. They are probably the accepted `cryptography` and `uv` findings held by flwr (`config/pip_audit_ignore.json`). Please confirm, or tell me which they are.
6. **Streamlit redeploy** for item 47, then a quick look at the demo.
7. **#97**: review the proposed `.claude/settings.json`.
8. **Network:** allow `download.pytorch.org` for items 26 and 27, or lock on the laptop.
9. **zizmor**: OK it or not (item 37).
10. **Two image decisions:**
    - The API image now runs `apt-get upgrade`. That fixes Trivy's Debian HIGHs, but two builds from the same pinned digest can differ. The alternative is to wait for a rebuilt base image and update the digest.
    - The Dockerfile's torch line installs torch's own dependencies (sympy, mpmath, filelock and others) without the constraints file, so those constraint pins aren't applied in the image.
11. **#50** thresholds, and the **scheduler probe**, as planned.

## What went wrong (and the check that now catches it)

- **Main's lint was red for about 54 minutes** (06:39 to 07:33 UTC). #94 turned on ruff's `S` rules, and #98 added test SQL that `S608` flags. Each PR passed alone. Before merging #98, I checked lint on the merged tree, but my script read ruff's last output line, which is blank, so it reported "ok". #107 fixed main. Every later merged-tree check used exit codes, and before the last batch of merges I merged all the remaining PRs together locally and ran lint, mypy and the affected tests.
- **My automatic conflict resolver duplicated BACKLOG lines** in #91 and #93: an item listed twice, once ticked. Review caught it before merge, and I resolved later conflicts by hand. `tests/test_backlog.py` (this PR) now fails on any duplicated or missing item number.
- **Two text edits silently failed to apply** in #103: a test assertion and a BACKLOG note, because my script replaced a string without asserting it was found. Review caught both, and every later edit asserted its match.
- **CI queue:** I cancelled 67 queued runs for superseded commits (21, then 46) to unblock the queue. That was runs only; no code or history was touched. (The Actions page shows more cancellations, because the workflows also cancel a branch's older run on every push.)
- **Rate limit:** a review agent hit the 5-hour usage limit, and reviews paused until the reset at 09:20 UTC.
- **Two final commits applied only a reviewer's own wording nits** and merged without another pass: #102's note and #109's manifest wording.

## How to check each item

**Item 9 (model card, change-control plan)**
1. Run `python scripts/build_docs.py`, then `git diff docs/`. Expected: no change.
2. Run `pytest tests/test_build_docs.py`.
3. Read `docs/change-control-plan.md`. It should fit on one page.

**Items 25, 28, 34, 35, 38 and 39 (supply chain)**
1. Run `pytest tests/test_supply_chain.py`.
2. Open any PR's checks: "Secrets / gitleaks" runs, and the Docker job's Trivy step lists no fixable HIGH or CRITICAL finding.
3. Run `grep -rn "uses:" .github/workflows/`. Expected: every line has a SHA and a `# v` comment.

**Items 29, 30, 32 and 44 (uploads, bundle, API)**
1. Run `pytest tests/test_contract.py tests/test_serving.py tests/test_api.py`.

**Item 31:** run `pytest tests/test_weights_only.py`.

**Item 33:** run `pytest tests/test_build_benchmarks.py`.

**Item 36:** run `ruff check .`. Expected: "All checks passed!".

**Items 41 and 42:** run `pytest tests/test_env_example.py tests/test_security_md.py`, and read `SECURITY.md`.

**Items 45 and 46:** run `pytest tests/test_extract.py`.

**Item 47** (after your redeploy): on the demo, a cohort definition that tries `read_csv` gets a generic error.
