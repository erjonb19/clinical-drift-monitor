# Governance evidence packet

Version 1, 2026-10-21.

How data access, requests and AI outputs are governed in this project, and where the evidence
for each control lives. This is a **portfolio build on public research data**. No protected
health information is held anywhere in it. The restricted tier below is designed but not yet
enforced in code, and it is empty today. It exists so that credentialed data, such as
MIMIC-IV-Note or an institution's own OMOP, can plug in without redesigning the controls.

**Status.** A control marked with a phase, such as (G2) or (G5), is built in that phase and is
only a plan until that phase's PR is merged. [BUILT_VS_PLANNED.md](BUILT_VS_PLANNED.md) says
what runs today. Requests the owner makes up to exercise the service get the `synthetic` label
by hand, so they are counted apart from real ones.

## 1. Access tiers

| Tier | What it holds | Where | Who | Leaves the tier only as |
| --- | --- | --- | --- | --- |
| **0: Public** | Code; concept map (OMOP concept IDs only); aggregate results with cells of at least 11; synthetic requests; the demo | GitHub repo, MLflow evidence runs, Streamlit demo | Anyone | (already public) |
| **1: Workspace** | Row-level public research data (images, silver, OMOP tables); concept names and codes (SNOMED content under the Affiliate License); model checkpoints | Databricks Free Edition workspace (`workspace.cdm`, volume `workspace.cdm.raw`) | Workspace owner | Aggregates that pass small-cell suppression; code |
| **2: Restricted** | Credentialed or identifiable data: none today. Planned: MIMIC-IV-Note (PhysioNet DUA), and the data-intake platform's OMOP | A separate catalog or schema with no grants to tier 0 identities | Named, credentialed users only | Aggregates only, after output review (section 3); never row-level, never note text |

**Rules that hold at every tier**
- **Small cells:** a count under 11 is suppressed in any output leaving tier 1 or 2. So is
  any count from which a suppressed cell could be recovered by subtraction. The cohort
  service enforces both within one result (G2). Across requests it is a reviewer's check
  (phenotypes/README.md, limits).
- **No SNOMED text in tier 0.** OMOP concept IDs are public; SNOMED codes and names are not
  redistributed.
- **Licenses travel with the data.** Per-image license and attribution are kept from ingest
  to `omop_image_occurrence`. CC-BY-NC images and HAM10000 (CC BY-NC 4.0) are for
  non-commercial use only.
- **Secrets** (Databricks tokens, `ANTHROPIC_API_KEY`) live in environment variables or
  `.env`, which is git-ignored. They are never committed or logged.

## 2. Data use checklist (one per request)

Each request issue (`.github/ISSUE_TEMPLATE/data_request.yml`) is checked against this list
before work starts. The answers are recorded in the issue.

- [ ] **Question** is specific enough to define a cohort and an output.
- [ ] **Data sources** named, each with its license or data use agreement.
- [ ] **Tier** of every source and of the requested output identified.
- [ ] **Minimum necessary:** only the columns and rows the question needs.
- [ ] **Approval status:** public data (none needed), IRB exempt, IRB approved, or DUA signed.
  Restricted data needs the document's reference.
- [ ] **Output form:** counts, aggregates, a cohort definition, or row-level data (tier 1+
  only).
- [ ] **Small-cell rule** acknowledged by the requester.
- [ ] **Cohort definition** saved as a versioned file in `phenotypes/`, and approved by hash
  before it runs.
- [ ] **Output reviewed** before release: suppression applied, no identifiers, no free text.
- [ ] **Retention:** where the output lives and when it is deleted.

## 3. Risk tiering

| Risk | Examples | Required before release |
| --- | --- | --- |
| **Low** | Counts and feasibility numbers on public tier 1 data, from an approved phenotype | Automatic small-cell suppression; audit log entry |
| **Medium** | A new phenotype; row-level export of public data; any model or LLM output shown to a requester | Human approval of the exact definition version; reviewer sign-off in the issue |
| **High** | Anything touching tier 2; LLM processing of clinical notes; outputs that could identify a person; anything used to make a decision about a patient | Approval by the data owner and, for real clinical data, IRB or privacy office; output review by a second person; escalation (section 4) if anything is unexpected |

AI outputs carry their own controls (G5):
- extraction results are scored against hand labels before use (`docs/EVAL_PLAN.md` Q7);
- LLM calls are logged with model, prompt hash and cost;
- CI never calls a live model;
- a concept ID an LLM returns must already exist in the concept map, or it is rejected.

## 4. Escalation path

| Trigger | First response | Escalate to | Record |
| --- | --- | --- | --- |
| A data or OMOP gate fails | The pipeline update stops by design; investigate | Data owner, if the source changed | Gate result table; `docs/silent-failures.md` if it had passed wrongly |
| A request asks for a suppressed cell, or for counts that would recover one | Suppress it in the output, or refuse the definition; explain the small-cell rule | Approver, if the requester disputes it | Cohort service audit log (refusal entry) (G2) |
| A cohort definition changes after approval | The service refuses to run it (hash mismatch) | Approver re-approves the new version | Audit log (G2) |
| Unexpected identifiers or free text in an output | Stop release; delete the output | Data owner; for tier 2, the privacy office | Issue comment and audit log |
| A license conflict (for example a commercial use of CC-BY-NC data) | Stop | Project owner | Issue |
| An LLM returns a concept not in the map, or extraction accuracy falls below the baseline | Reject the output | Project owner, before any further use | G5 results and MLflow (G5) |
| LLM spend reaches the daily budget ($5) | Stop calls | Project owner | Spend log (G5) |

In this portfolio, every role (data engineer, approver, data owner) is the project owner,
and the privacy office is named only as the step a real deployment would add. The roles are
kept separate in the audit log so the workflow is visible.

The synthetic requests (issues labelled `synthetic`) are the one exception: Claude Code
recorded their decisions as a stand-in reviewer, in
`results/research/g2_cohort/synthetic_approvals.jsonl`. Those decisions do not count for real
requests: `phenotypes/approvals.jsonl` holds only the project owner's own decisions.

## 5. License and agreement register

| Source | Terms | Constraint here |
| --- | --- | --- |
| Repository code | MIT (`LICENSE`) | Code only: not the data, and not the course material in `reference/` |
| ISIC Archive images | CC-0, CC-BY or CC-BY-NC, per image | Attribution kept per image; NC images non-commercial |
| HAM10000 | CC BY-NC 4.0 | Non-commercial; cited |
| SNOMED CT (via Athena) | SNOMED International Affiliate License | Codes and names stay in tier 1; only OMOP concept IDs are public |
| OMOP vocabularies, Athena | OHDSI terms of use | Download kept outside the repo |
| BiomedCLIP | MIT; model card says research use only | Never served (Phase B) |
| ISIC metadata in the public demo (G6) | Per image; PAD-UFES-20 (collection 406) is CC-BY, attribution to UFES | Counts only, no images, non-commercial portfolio use, attribution on the page. Checked 2026-10-22 against the ISIC FAQ (licenses are per image; CC-BY-NC is non-commercial). Not deployed until the owner approves |
| MTSamples | mtsamples.com allows printing, sharing or linking the sample reports for educational purposes, with credit; no open license for redistribution. (Kaggle's CC0 label comes from the person who scraped the site, not the publisher.) | Note text stays out of the repo: the labelling sheet, LLM cache and error sheet live in a folder outside it; only aggregates are committed |
| MIMIC-IV-Note (future) | PhysioNet credentialed DUA | Tier 2 only; code and aggregates only in the repo; never sent to an outside LLM API (the extractor refuses it) |

## 6. Evidence index

| Control | Evidence |
| --- | --- |
| Lesion-level splits; patients documented | `patients_in_one_split` and split gates; `tests/test_frozen_splits.py` |
| Data quality gates fail the update | `gate_results`, `omop_gate_results`; the broken demo job |
| Every number traceable | `results/mlflow_runs.json`; `tests/test_evidence.py` checks the hashes in CI |
| Bugs that reported success | `docs/silent-failures.md` |
| Test looks decided in advance | `spec.json` files with `committed_before_running`; look locks in the workspace volume (`databricks/notebooks/analysis_look.py`, `drift_table.py`) |
| Concept mapping verified, not guessed | `results/research/a_omop/concept_check.json` |
| Requests scoped and tracked | GitHub issues from the request form, which adds each one to the [project board](https://github.com/users/erjonb19/projects/2) |
| Cohort access governed | Phase G2: allowlist, approval by hash, audit log, suppression tests |
