# CLAUDE.md — Clinical Imaging Drift Monitor

Working context for Claude Code in this repo. Read this and PLAN.md before any work.

## What this is
A personal portfolio project: a skin lesion classifier served with a drift (out-of-distribution) score, a monthly drift job, and grounded LLM incident reports. It grows out of my Purdue ECE 570 course project (solo work). The original notebook and slides live in `reference/` and are read-only history, not code to build on.

PLAN.md is the source of truth for scope and order. Do not add features that are not in PLAN.md. If something in the plan turns out wrong or impossible, stop and tell me, and propose a change to PLAN.md before coding around it.

## How to work with me
- Work one phase at a time, in order, starting with Phase 0. Do not start a phase until the previous phase's "Done when" is met and I have confirmed it.
- Before each phase, write a short plan (files to create, commands to run) and wait for my OK.
- When you give me steps to run, use short numbered steps that say exactly what to open and what to run. Keep the number of files small.
- Tick the PLAN.md checkboxes as items are finished.

## Honesty rules (most important)
- Every number in the README, results tables, or anywhere else must come from a run that is committed in this repo. No estimates presented as results.
- Label staged (synthetic) drift as staged everywhere it appears. Real drift and staged drift are never mixed in one number.
- Keep `docs/BUILT_VS_PLANNED.md`: built, scaffolded, not started. Nothing is described as built until it runs.
- Keep `docs/silent-failures.md`: every bug that reported success while being wrong, with the check that now catches it (reckoner pattern).
- If a result looks too good or too bad (for example an AUROC far below 50%), stop and investigate before reporting it.

## Hard constraints
- Zero cost. No paid services, no paid API keys. Allowed: Databricks Free Edition, Colab Pro (already paid for), Kaggle free GPU, Render free tier, Streamlit Community Cloud, Gemini API free tier, GitHub Actions on a public repo.
- No Azure and no GCP billing for this project.
- Databricks is central (I am working toward the Databricks Data Engineer Associate and ML Associate certifications). Prefer native Databricks features (Unity Catalog, Delta, Auto Loader, Lakeflow Declarative Pipelines, Lakeflow Jobs, MLflow with Unity Catalog registry) where Free Edition supports them. Free Edition is serverless only and non-commercial.
- One ML framework: PyTorch. Do not use TensorFlow or Keras.
- Respect each ISIC image's license (CC-0, CC-BY, CC-BY-NC). Record license and attribution per image.
- Never commit data files, model weights, or secrets. Secrets go in environment variables.

## Engineering standards
- Python 3.11, typed code, ruff, pytest. CI must pass on every push.
- Splits are by lesion, never by image. A test enforces this.
- Fix random seeds; report results as mean and range over 3 seeds.
- Every OOD score is computed the same way for in-distribution and OOD data, with statistics fit on training data only. A test checks that scores point the right way.
- Small, reviewed commits. One logical change per commit.

## Patterns to reuse from my other public repos
- github.com/erjonb19/reckoner: manifests with checksums, silent-failures doc, BUILT_VS_PLANNED doc, README with results first.
- github.com/erjonb19/governed-clinical-agent: groundedness check, eval harness that separates provider outages from accuracy drops, CI eval gate, Databricks Delta publishing, write-audit-publish data quality gates.
Read these repos for the pattern, then write fresh code here. Do not copy large files wholesale.
