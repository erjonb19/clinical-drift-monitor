# Security audit, 2026-11-10 (backlog item 22)

Plan only: this audit changes no code. It checks the repo against `docs/SECURITY_STANDARDS.md`
(Section 19.3 and every section that applies to this stack) at `main` 877b608. Each gap is
reported in the standard's SEVERITY format and has a backlog item (items 23 to 47) for the owner
to place. The owner reviews and merges this PR himself.

**Context that sets severity.** Public repo, public data only (ISIC, MTSamples), no PHI. The
FastAPI service is not deployed (it runs in CI and locally). The Streamlit cohort demo is
deployed on Streamlit Community Cloud. Databricks jobs run in the owner's Free Edition
workspace, deployed only from the owner's laptop. One paid key exists (Claude Haiku, capped at
$5 in total, about $0.03 spent).

**Summary.** No Critical and no High finding. No secret was found in the working tree or in any
of the 414 commits on every branch. 5 Medium findings (supply chain and repo settings) and 26
Low ones. One Low finding is also a silent failure of the repo's own making (the image
contract's decompression-bomb limit, built on 2026-11-09), recorded as silent failure #16.

**How it was done.** Four read-only passes: secrets and repo settings; dependencies, containers,
model files and CI; application, data and LLM code; and the Claude Code settings. Where a
finding rests on a live check, it says so. Things that could not be checked from a cloud
session are listed at the end.

---

## Section 19.3 (new repo setup), item by item

| Check | State | Finding |
|---|---|---|
| `docs/SECURITY_STANDARDS.md`; Prime Directives in `CLAUDE.md` | Done (#80) | - |
| `.claude/settings.json` from Section 17.2 | Missing | Proposal below; item 23 |
| `.gitignore` covers secrets and state; `.env.example` committed | Partly (`.env` only) | items 40, 41 |
| `pyproject.toml` cooldown and script settings (Section 4) | Missing | item 26 |
| `pre-commit` with gitleaks | Missing | item 25 |
| CI security job, SHA-pinned actions, read-only default permissions | Partly | items 28, 34, 37 |
| Branch protection on `main` | **Off** (checked 2026-11-10) | item 24 |
| Secret scanning, push protection, Dependabot alerts | Scanning and push protection on; security updates off; alerts not visible | item 43 |
| `SECURITY.md` (here: GitHub private vulnerability reporting, per CLAUDE.md) | Missing; private reporting off | item 42 |

---

## Medium

```
[SEVERITY: Medium] Branch protection is off on main
Where: repo settings (GET /repos/erjonb19/clinical-drift-monitor/branches/main: "protected": false; no rulesets)
Risk: Anything with push rights, an agent session included, can push or force-push to main without CI or review, which undercuts every gate this repo relies on.
Fix: A ruleset on main: require a pull request and the quality, image and CodeQL analyze checks; block force-push and deletion. (The owner planned this on 2026-11-09.)
Standard: SEC.md §19.3, OWASP A08:2025, CWE-284
```
Backlog item 24.

```
[SEVERITY: Medium] No secret scanning before commit or in CI
Where: repo root (no .pre-commit-config.yaml); .github/workflows/*.yml (no gitleaks or trufflehog)
Risk: Nothing local or in CI stops a generic secret from being committed; GitHub push protection catches provider-pattern keys (such as Anthropic's) but its non-provider patterns are off.
Fix: .pre-commit-config.yaml with the gitleaks hook pinned to a version; a CI job running gitleaks over the full history (action pinned by SHA, contents: read); optionally enable non-provider patterns.
Standard: SEC.md §3.3, §19.3; OWASP A02:2025; CWE-798
```
Backlog item 25.

```
[SEVERITY: Medium] No lockfile, no hashes and no release-age cooldown for Python; CI and the deployed demo install floating versions
Where: pyproject.toml:12-17, 21-27, 33-40, 60-68 (">=" ranges); .github/workflows/ci.yml:28-30, detector-quality.yml:27-29 (unpinned torch, floating extras); benchmark.yml:28-30, docker.yml:28-30; app/requirements.txt:4 ("-e .[cohort,demo]" for the Streamlit Cloud app)
Risk: Each CI run and each Streamlit rebuild installs whatever is newest on PyPI, so a freshly published malicious release would run in CI (with the workflow token) or in the public demo before anyone reviewed it; pip-audit only catches known CVEs.
Fix: uv with a committed uv.lock and [tool.uv] exclude-newer = "7 days"; "uv sync --locked" in every workflow; for pip consumers (Streamlit, Docker) "uv export --frozen" with hashes and "pip install --require-hashes". Adopting uv is a new tool: owner's OK first (directive 11).
Standard: OWASP A03:2025, CWE-494, CWE-1357, SEC.md §4.3, §16.5
```
Backlog item 26. Checked against uv's settings reference (2026-11-10): `exclude-newer` "Accepts
RFC 3339 timestamps (e.g., 2006-12-02T02:07:43Z), a "friendly" duration (e.g., 24 hours, 1 week,
30 days), or an ISO 8601 duration (e.g., PT24H, P7D, P30D)", and `exclude-newer-package` can
exempt one package.

```
[SEVERITY: Medium] The API image installs without hashes, and its torch step bypasses the constraints
Where: Dockerfile:9 (torch==2.14.0 and torchvision==0.29.0 from the PyTorch index: no hashes, not under -c constraints.txt, so torch's own dependencies resolve freely); Dockerfile:14 (-c constraints.txt without --require-hashes; the setuptools>=68 build backend is not constrained); docker/constraints.txt (exact pins, no hashes)
Risk: A tampered or new artifact for a pinned version, or an unlisted dependency, would land in the image unnoticed. The image is only built in CI and never pushed, which limits the impact today.
Fix: A hashed requirements file for the image (including the +cpu torch wheels), installed with "pip install --no-deps --require-hashes", then the package with --no-build-isolation (or setuptools pinned in the hashed file).
Standard: OWASP A03:2025, A08:2025, CWE-494, SEC.md §4.3, §4.4
```
Backlog item 27.

```
[SEVERITY: Medium] The CI workflow has no permissions block
Where: .github/workflows/ci.yml (no "permissions:"; the other five workflows set contents: read)
Risk: On push the job gets the repository's default token scope, which may be read and write, while it installs floating packages (item 26) and checkout keeps the token in .git/config, so a compromised dependency could push to the repo.
Fix: "permissions: contents: read" at the top of ci.yml; "persist-credentials: false" on checkouts that do not push; set the repository's default workflow permission to read-only.
Standard: OWASP A01:2025, A03:2025, CWE-250, SEC.md §16.5
```
Backlog item 28.

---

## Low

```
[SEVERITY: Low] The image contract's decompression-bomb limit is 100 megapixels, not the 50 its comments state (silent failure #16)
Where: src/cdm/contract.py:26-28, 102-106; src/cdm/serving.py:34-36, 129-130; app/api/main.py:207, 217; tests/test_contract.py (tests 108 MP only)
Risk: Pillow raises only above twice MAX_IMAGE_PIXELS and merely warns between 50 and 100 MP, so a 58 KB PNG of 60 MP is accepted and decoded twice per request; checked live on 2026-11-10: contract.check returned unreadable=False, quarantined=[] with a DecompressionBombWarning. The API is not deployed; on a 512 MB host this would be Medium.
Fix: Treat DecompressionBombWarning as an error (warnings filter) or check img.size against a hard cap right after Image.open, before load; add a 60 MP test; optionally limit concurrent decodes.
Standard: CWE-409, CWE-400, OWASP API4:2023, SEC.md §7.8, §8
```
Backlog item 29.

```
[SEVERITY: Low] Every Pillow decoder parses uploads before the JPEG/PNG allowlist
Where: src/cdm/contract.py:102-111; src/cdm/serving.py:129
Risk: Image.open and load run whichever plugin matches the bytes (TIFF, PSD, ICO and others) before the format check, exposing past and future plugin CVEs.
Fix: Image.open(..., formats=["JPEG", "PNG"]) in both places, driven by CONTRACT["enforced"]["formats"].
Standard: CWE-20, ASVS V5, SEC.md §7.9
```
Backlog item 30.

```
[SEVERITY: Low] torch.load without weights_only=True, and a declared floor where the unsafe default applies (Section 4.5)
Where: src/cdm/serving.py:117; src/cdm/gold.py:45; scripts/phase0_cifar_check.py:74; tests/test_pipeline.py:110; databricks/notebooks/ batch_drift_reference.py:113, final_test.py:155, drift_table.py:210, review_worksheet.py:125, triage.py:83, shipped_features.py:94, onnx_export.py:112, retrain.py:166, ladder_batch.py:178, serving_bundle.py:93, posthoc_ladder.py:58, analysis_look.py:102, ba_recalibration.py:75, evaluate_checkpoint.py:57; pyproject.toml:23 and five job files (torch>=2.4)
Risk: Before torch 2.6 torch.load unpickles by default, which runs code; every environment in use resolves to 2.14 (safe default) but nothing enforces it. Checkpoints come from the owner's own volumes and bundles; none are committed.
Fix: weights_only=True at every call; floor torch>=2.6 in pyproject and the job files; a test that fails on torch.load without weights_only=True; later, safetensors for the bundle.
Standard: CWE-502, OWASP A08:2025, LLM04:2026, SEC.md §4.5
```
Backlog item 31.

```
[SEVERITY: Low] The serving bundle's check trusts a manifest beside the files and does not require every checkpoint to be listed
Where: src/cdm/serving.py:110-119
Risk: The SHA-256 check runs before torch.load (good), but the manifest sits in the same folder, and a manifest with a partial sha256 map loads unhashed files; file names are joined to the folder unchecked.
Fix: Require each seed's checkpoint and detector to be keys of manifest["sha256"]; reject names with "/" or ".."; optionally take the expected manifest SHA-256 from configuration.
Standard: CWE-502, CWE-345, SEC.md §4.5
```
Backlog item 32.

```
[SEVERITY: Low] Unverified pickle.load of the CIFAR-10 batches
Where: scripts/build_benchmarks.py:66 (the noqa: S301 there is inert, since S rules are off)
Risk: A swapped file in the owner's --data folder would run code when this local script runs.
Fix: Check the published checksum of each CIFAR file before unpickling, as cdm.data does for HAM10000.
Standard: CWE-502, SEC.md §4.5
```
Backlog item 33.

```
[SEVERITY: Low] GitHub-owned actions are pinned to moving tags, and a test allows it
Where: actions/checkout@v7, actions/setup-python@v7, actions/upload-artifact@v7 in ci.yml, benchmark.yml, codeql.yml, detector-quality.yml, docker.yml, drift-watchdog.yml; tests/test_supply_chain.py (exempts the actions and github owners)
Risk: A moved or compromised tag would run new code in CI without review.
Fix: Pin each to its full commit SHA with a version comment (Dependabot keeps SHAs current); drop the owner exemption from the test.
Standard: OWASP A03:2025, CWE-829, SEC.md §16.5
```
Backlog item 34.

```
[SEVERITY: Low] The Docker base image is pinned by tag, not digest
Where: Dockerfile:4 (FROM python:3.11.17-slim)
Risk: The tag can be re-pushed, so a rebuild can pick up different base layers with no change in the repo.
Fix: FROM python:3.11.17-slim@sha256:<digest>; Dependabot's docker updates keep the digest current.
Standard: OWASP A03:2025, SEC.md §4.4
```
Backlog item 35.

```
[SEVERITY: Low] Ruff's security rules (S) are off
Where: pyproject.toml:88 (select = ["E", "F", "I", "B", "UP", "SIM", "NPY"])
Risk: New pickle, torch.load, subprocess-with-shell or hard-coded-secret patterns are not flagged at lint time (CodeQL covers part of this).
Fix: Add "S" to select, then fix each hit or give it a reasoned per-line noqa.
Standard: SEC.md §16.3, §18
```
Backlog item 36.

```
[SEVERITY: Low] No workflow linter (zizmor)
Where: .github/workflows/
Risk: Workflow misconfigurations such as the two above (items 28 and 34) are not caught automatically (CodeQL's actions analysis covers part).
Fix: A zizmor job, pinned by SHA, on changes to .github/workflows/**.
Standard: SEC.md §16.5, §18
```
Backlog item 37.

```
[SEVERITY: Low] Dependabot has no cooldown
Where: .github/dependabot.yml (pip, docker, github-actions: no "cooldown:")
Risk: Update PRs open on the day a release appears; each is reviewed and CI-gated, which limits the risk.
Fix: "cooldown: default-days: 7" on each ecosystem (longer semver-major-days if wanted).
Standard: SEC.md §4.6, §16.5
```
Backlog item 38. Checked against GitHub's Dependabot options reference (2026-11-10): `cooldown`
takes `default-days`, `semver-major-days`, `semver-minor-days`, `semver-patch-days` (integers)
and `include` / `exclude` lists.

```
[SEVERITY: Low] The image scan fails on fixable CRITICAL findings only
Where: .github/workflows/docker.yml (severity: CRITICAL, ignore-unfixed: true); tests/test_supply_chain.py
Risk: Fixable HIGH vulnerabilities in the image never fail the build; the standard asks for critical and high to be fixed within days.
Fix: severity: CRITICAL,HIGH, with accepted findings in .trivyignore with reasons as today; update the test.
Standard: SEC.md §4.4, §4.6
```
Backlog item 39.

```
[SEVERITY: Low] .gitignore misses most of the entries Section 3.1 requires
Where: .gitignore:15 (only ".env" under "# Secrets")
Risk: .env.local, *.pem, *.key, *.p12, *.tfstate, .terraform/, secrets/ and credential JSON would be committable if created (checked with git check-ignore).
Fix: Add .env.*, !.env.example, *.pem, *.key, *.p12, *.pfx, *.tfstate, *.tfstate.*, .terraform/, secrets/, *credentials*.json, service-account*.json, .databrickscfg.
Standard: SEC.md §3.1, §19.3, CWE-538
```
Backlog item 40.

```
[SEVERITY: Low] No .env.example
Where: repo root
Risk: Variable names live only in prose, so a real value is more likely to be put somewhere it gets committed.
Fix: Commit .env.example with names and placeholders only: ANTHROPIC_API_KEY, DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, CDM_DATA, CDM_NOTES_DIR, BUNDLE_DIR, TRUSTED_PROXY_HOPS.
Standard: SEC.md §3.1, §19.3
```
Backlog item 41.

```
[SEVERITY: Low] No private way to report a vulnerability
Where: repo root (no SECURITY.md); repo settings (private vulnerability reporting: disabled)
Risk: A finder has no private channel and may open a public issue.
Fix: Turn on private vulnerability reporting (owner, repo settings); add a SECURITY.md that points to it and states the scope (no deployed service, public data only), with no personal email (CLAUDE.md, note 4).
Standard: SEC.md §19.3
```
Backlog item 42.

```
[SEVERITY: Low] Dependabot security updates are off; Dependabot alerts not visible
Where: repo settings (security_and_analysis.dependabot_security_updates: disabled; alerts need authentication to read)
Risk: Fix PRs for vulnerable dependencies are not raised automatically; dependabot.yml covers version updates only.
Fix: Turn on Dependabot alerts and security updates (owner, Settings, Code security).
Standard: SEC.md §19.3, §21, OWASP A03:2025
```
Backlog item 43.

```
[SEVERITY: Low] The watchdog's token asks for the all-apis scope
Where: scripts/watchdog.py:55
Risk: If the service principal's secret leaked, only the principal's own grants (planned: Can View on one job) would limit it; the token scope narrows nothing.
Fix: Keep the principal to Can View on job 524433148303929 only and record it; use a narrower scope if Databricks offers one for reading jobs.
Standard: SEC.md §3.1, CWE-250
```
Folded into item 19 (least-privilege access), not a new item.

```
[SEVERITY: Low] The API has no authentication (already planned)
Where: app/api/main.py:185-224
Risk: Anyone who can reach the service can call /predict and /health; rate limits are the only control. Acceptable while the API is not deployed.
Fix: Backlog item 19's per-client keys: a router-level dependency, hashed keys compared with secrets.compare_digest, and a 401 test.
Standard: OWASP A07:2025, API2:2023, SEC.md §5.3, §16.3
```
Already backlog item 19; no new item.

```
[SEVERITY: Low] No security headers on the API's page or responses
Where: app/api/main.py:143-169 (page with an inline script), 172-226 (no middleware)
Risk: No CSP, nosniff, frame-ancestors or Referrer-Policy, so the page can be framed; there is no XSS sink today (constants only; output through textContent).
Fix: A small middleware: X-Content-Type-Options: nosniff, Referrer-Policy: strict-origin-when-cross-origin, a CSP with a hash for the inline script and frame-ancestors 'none'; HSTS at deploy.
Standard: OWASP A02:2025, SEC.md §9
```
Backlog item 44 (with the next two: API hardening before any deploy).

```
[SEVERITY: Low] /openapi.json is exposed and routes have no response models
Where: app/api/main.py:175 (docs and redoc off, openapi_url left on), 189-195 (dict[str, Any])
Risk: The schema is public, and nothing pins output fields, so a later change could leak a field.
Fix: openapi_url=None (or gated); Pydantic response models for /predict and /health.
Standard: OWASP API9:2023, SEC.md §6.2, §16.3
```
Backlog item 44.

```
[SEVERITY: Low] The API logs no security events
Where: app/api/main.py (no logger; rate-limit hits only counted)
Risk: Rate-limit storms, bomb attempts and repeated rejections leave no trace. Not logging addresses or bodies is a deliberate privacy choice and stays.
Fix: Structured events with no address or body: request id, outcome (rate_limited, rejected, quarantined) and a reason code.
Standard: OWASP A09:2025, SEC.md §12.1
```
Backlog item 44.

```
[SEVERITY: Low] The $5 Haiku cap fails open if its spend log is missing
Where: src/cdm/extract.py:46 (hard-coded price), 245-259 (total() is 0 when the file is absent); scripts/extraction.py:62
Risk: Moving or deleting ~/.cdm/llm_spend.jsonl resets the recorded spend to $0, and a wrong price constant undercounts; the cap is client-side only.
Fix: A hard spend limit on the Anthropic Console for this key (owner; the real backstop); refuse when the log is missing but a marker file says spending has started; a test that dates the price constant.
Standard: LLM10:2026, SEC.md §14, §12.3
```
Backlog item 45.

```
[SEVERITY: Low] Formula injection in the owner's G5 error sheet
Where: scripts/extraction.py:108-110, 169-178 (raw model reply and note text written to CSV)
Risk: A cell starting with =, +, - or @ becomes a formula when opened in a spreadsheet; the .xlsx path already guards against this. Owner-only local file, public notes.
Fix: Prefix risky leading characters with ' before writing, or write only the validated value plus a truncated raw string.
Standard: CWE-1236, LLM05:2026, SEC.md §7.8
```
Backlog item 46.

```
[SEVERITY: Low] The public demo's DuckDB connection keeps file and network access
Where: src/cdm/cohort.py:657-668 (used by the deployed demo through src/cdm/demo.py:210)
Risk: The SQL is built only from allowlisted integers and identifiers, so injection is not reachable today; a future bug would have file and network access.
Fix: After registering the tables: SET enable_external_access = false; SET lock_configuration = true.
Standard: CWE-89 (defence in depth), SEC.md §7.2, §14.2
```
Backlog item 47 (with the next three: the public demo).

```
[SEVERITY: Low] The public demo may show full tracebacks
Where: app/cohort_demo.py:28-30, 68 (fetch and load not wrapped); no .streamlit/config.toml
Risk: Streamlit's default error details show stack traces, for example when the ISIC API is down, revealing paths and code (the repo is public, so the impact is low).
Fix: .streamlit/config.toml with [client] showErrorDetails = "none"; a generic st.error around the fetch.
Standard: OWASP A10:2025, SEC.md §11
```
Backlog item 47.

```
[SEVERITY: Low] The demo's resources grow with each visitor
Where: src/cdm/demo.py:207 (tempfile.mkdtemp never removed); src/cdm/cohort.py:619-639 (re-runs every earlier definition on each Run)
Risk: Temporary folders pile up, and one visitor making many distinct definitions costs more CPU per Run; availability only.
Fix: TemporaryDirectory or cleanup at session end; a cap on distinct definitions per session.
Standard: OWASP API4:2023, SEC.md §8
```
Backlog item 47.

```
[SEVERITY: Low] The ISIC client follows server-supplied "next" URLs with no host check or size cap
Where: src/cdm/sources.py:47-68 (used by the deployed demo)
Risk: A spoofed or compromised response could send requests to another host or return a huge body; TLS to a fixed host makes this unlikely.
Fix: Require https and netloc == "api.isic-archive.com" for "next"; read with a byte cap.
Standard: CWE-918, OWASP A01:2025, SEC.md §7.6
```
Backlog item 47.

Informational, no item: a local home path in a committed docstring
(`scripts/build_benchmarks.py:13`; it shows the owner's local user name, which the GitHub handle
already reveals); no SBOM (optional in Section 4.6 for non-production projects); the demo's
small-cell guard is per browser session, which the page states and which matters only on
non-public data.

---

## Proposed `.claude/settings.json` (Section 17.2, tailored; not added)

The owner reviews this; nothing is created by this PR. Permission syntax was checked against
Claude Code's current docs on 2026-11-10 (code.claude.com/docs/en/permissions,
/permission-modes, /settings, /cloud-environments, /sandboxing). Key points from them: rules are
evaluated "deny, then ask, then allow", and "a matching ask rule prompts even when a more
specific allow rule also matches"; `Bash(ls *)` and `Bash(ls:*)` are equivalent; a Bash rule
"isn't a security boundary around the program" (`git -C . push` is not matched by
`Bash(git push *)`); MCP tools are matched as `mcp__<server>__*`; in `dontAsk` mode Claude Code
"Auto-denies every call that would otherwise prompt"; cloud sessions pick the mode from a
dropdown and read the repo's `.claude/settings.json`.

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "defaultMode": "acceptEdits",
    "disableBypassPermissionsMode": "disable",
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./**/.env)",
      "Read(./**/.env.*)",
      "Read(./secrets/**)",
      "Read(./**/*.pem)",
      "Read(./**/*.key)",
      "Read(./**/*.p12)",
      "Read(./**/*.tfstate)",
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)",
      "Read(~/.config/gcloud/**)",
      "Read(~/.kube/**)",
      "Read(~/.npmrc)",
      "Read(~/.pypirc)",
      "Read(~/.databrickscfg)",
      "Read(~/.config/gh/**)",
      "Edit(./.claude/**)",
      "Edit(./config/pip_audit_ignore.json)",
      "Edit(./.trivyignore)",
      "Bash(git push --force *)",
      "Bash(git push -f *)",
      "Bash(git push * --force*)",
      "Bash(git push * -f *)",
      "Bash(git push * --mirror*)",
      "Bash(git push * --delete *)",
      "Bash(git push * main)",
      "Bash(git push * HEAD:main*)",
      "Bash(git commit * --no-verify*)",
      "Bash(rm -rf *)",
      "Bash(rm -fr *)",
      "Bash(sudo *)",
      "Bash(wget *)",
      "Bash(printenv*)",
      "Bash(env)",
      "Bash(curl * -X *)",
      "Bash(curl * --request *)",
      "Bash(curl * -d *)",
      "Bash(curl * --data*)",
      "Bash(curl * -F *)",
      "Bash(curl * --form *)",
      "Bash(curl * -T *)",
      "Bash(curl * --upload-file *)",
      "Bash(gh api * -X *)",
      "Bash(gh api * --method *)",
      "Bash(gh api * -f *)",
      "Bash(gh api * -F *)",
      "Bash(gh api * --input *)"
    ],
    "ask": [
      "Bash(pip install *)",
      "Bash(python -m pip install *)",
      "Bash(docker run *)",
      "Bash(databricks *)",
      "Bash(hf *)",
      "Bash(gh release *)",
      "Bash(python scripts/deploy_space.py*)",
      "Bash(python scripts/deploy_databricks.py*)",
      "Bash(python scripts/stage_research_job.py*)",
      "Bash(python scripts/move_champion.py*)",
      "Bash(python scripts/log_evidence.py*)",
      "Bash(python scripts/extraction.py*)",
      "Edit(./.github/**)",
      "WebFetch"
    ],
    "allow": [
      "Bash(git status*)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git show *)",
      "Bash(git fetch *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git worktree *)",
      "Bash(git checkout *)",
      "Bash(git rebase origin/main)",
      "Bash(git push -u origin claude/*)",
      "Bash(git push origin claude/*)",
      "Bash(pytest *)",
      "Bash(python -m pytest *)",
      "Bash(mypy*)",
      "Bash(python -m mypy*)",
      "Bash(ruff check *)",
      "Bash(ruff format *)",
      "Bash(curl -s https://api.github.com/*)",
      "Bash(curl -sS https://api.github.com/*)",
      "Bash(gh api repos/*)",
      "mcp__github__get_*",
      "mcp__github__list_*",
      "mcp__github__pull_request_read",
      "mcp__github__create_pull_request",
      "mcp__github__update_pull_request",
      "mcp__github__merge_pull_request"
    ]
  }
}
```

| Change from Section 17.2 | Why |
|---|---|
| `defaultMode: acceptEdits` (was `default`) | In `default` the first edit prompts and an unattended run waits; `acceptEdits` accepts edits in the working directory. For overnight runs the owner picks `dontAsk` in the cloud dropdown, so every `ask` below becomes a denial instead of a stall. |
| Deny list kept; added `*.p12`, `~/.databrickscfg`, `~/.config/gh/**` | `*.p12` is in §3.1; the repo's deploy scripts use the Databricks CLI profile; `gh` keeps its token there. |
| New Edit denies: `.claude/**`, `config/pip_audit_ignore.json`, `.trivyignore` | The agent must not widen its own permissions or silence the audit and scan gates (directive 15). |
| Push and commit denies widened | Force, mirror, delete and pushes to `main`; `--no-verify` would skip the gitleaks hook (item 25). Not a boundary (docs): branch protection (item 24) is the real control. |
| `git push *` moved from ask to a narrow allow (`claude/*` branches) | Ask beats allow, so an ask on every push would block the allowed ones. This repo's working branches are `claude/...`. |
| npm, pnpm, npx, uv, terraform, kubectl, helm, supabase, vercel removed | Not used in this repo (checked by grep); an unmatched command already prompts (or is denied under `dontAsk`). They come back if a tool is adopted (uv, item 26). |
| `pip install` stays in **ask** | Directive 11: no installs without asking. Under `dontAsk` it is denied, so an unattended run cannot install anything; it writes the need down and moves on (CLAUDE.md note 1). If the owner wants unattended runs to install the repo's own pinned dependencies, that needs a SECURITY-EXCEPTION entry, his decision. |
| `curl`: ask removed; read-only GETs to `api.github.com` allowed; write flags denied | Used to read CI status. The deny patterns are spaced (`curl * -d *`) on purpose: an unspaced `curl *-d*` would match the "-d" in "clinical-drift" and block every call to this repo's API. Combined flags (`-sSd`) can still slip past: not a boundary. |
| `wget` denied (was ask) | Not used; nothing needs it unattended. |
| Deploy, registry, Haiku and Databricks commands in ask | They write outside the repo or spend (directive 13); under `dontAsk` they are denied, which matches the rule that the owner deploys. |
| `Edit(./.github/**)` in ask | §17.3: read every CI diff. Under `dontAsk` CI changes are denied in unattended runs; the owner may prefer to allow them, since every workflow change still goes through a reviewed PR. |
| GitHub MCP: reads, PR create/update and merge allowed | The owner's standing merge OK (2026-11-10, BACKLOG rules). Branch protection with required checks (item 24) is what makes an agent merge safe; without it, move merge to ask. |
| No `sandbox` block | Whether the sandbox runs in the cloud VM is not documented; a sandbox that fails to start would stop every session. Turn it on with `/sandbox` on the laptop instead (user settings). |

Not verified (docs silent or not reachable): whether a committed `defaultMode` sets the cloud
dropdown's default; whether workspace trust gates committed `allow` rules in cloud sessions; the
exact MCP tool names in a cloud session. Check after adding: start a cloud session, run
`/permissions`, then try `ruff check .` (allowed), `curl -X POST https://api.github.com/x`
(denied) and a push to `main` (denied).

---

## Could not be checked from this session

- The repository's default workflow token permission, Dependabot alerts, collaborators, deploy
  keys and Actions secrets (need authentication).
- Streamlit Community Cloud settings and secrets for the deployed demo.
- Whether an Anthropic Console spend limit is set, and the current Haiku price.
- What the Databricks jobs declared with `torch>=2.4` actually resolve to.
- The scope of the Databricks service principal, which does not exist yet.

## Process note

During this work a review agent installed pytest into its own environment to run tests, without
asking (directive 11). It touched nothing in the repo; reported here so the owner knows.
