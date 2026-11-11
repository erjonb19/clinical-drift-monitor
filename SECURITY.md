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

The repository holds code, configuration and aggregate results only, no patient data. The
data it works on are public: ISIC images and metadata under their own licenses, and public
MTSamples note text. If you believe a file in this
repository contains personal or protected health information, report it as above.

## How this repository is checked

The owner's security standard is `docs/SECURITY_STANDARDS.md`; the latest audit against it
is `docs/security-audit-2026-11-10.md`.
