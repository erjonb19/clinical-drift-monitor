# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting:
open the repository's **Security** tab and choose **Report a vulnerability**
(https://github.com/erjonb19/clinical-drift-monitor/security/advisories/new). Do not open a
public issue for a security problem.

You can expect an acknowledgement within a week. This is a personal portfolio project
maintained by one person, so fixes are best effort.

## Scope

- The code in this repository: the `cdm` package, the scripts, the FastAPI research API
  (`app/api`), the Streamlit cohort demo (`app/cohort_demo.py`), the Databricks notebooks
  and job definitions, and the GitHub Actions workflows.
- The public cohort demo on Streamlit Community Cloud runs this code on public data.
- The FastAPI service is not deployed; it runs in CI and locally only.

Out of scope: the third-party services the project uses (GitHub, Databricks, Streamlit
Community Cloud, the ISIC Archive, Anthropic), which have their own reporting channels.

## Data

The repository holds code, configuration and results, no patient data. Some committed files
list public ISIC Archive identifiers (image, lesion and patient IDs) for splits and per-image
results; no images and no note text are committed. The data the project has run on are public:
ISIC images and metadata under their own licenses, and public MTSamples note text.
`cdm.extract` also has a loader for MIMIC-IV-Note (PhysioNet, credentialed access) that stays
off unless `CDM_TIER2=1` is set and never sends notes to an outside API; no committed result
uses it, and no MIMIC data is in the repository. If you believe a file in this
repository contains personal or protected health information, report it as above.

## How this repository is checked

The owner's security standard is `docs/SECURITY_STANDARDS.md`; the latest audit against it
is `docs/security-audit-2026-11-10.md`.
