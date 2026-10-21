# Phenotype library

Versioned cohort definitions for the governed cohort service (`src/cdm/cohort.py`,
`scripts/cohort.py`). Each definition lives at `<name>/v<version>.yaml`. A definition is
never edited after it is approved: a change is a new version, and needs a new approval.

These cohorts run on **public research images** (HAM10000 and ISIC Archive) in OMOP form,
not on patient records. Concept IDs come from `config/omop_concepts.csv`. Their names stay in
the workspace's concept table, because SNOMED text is not published here.

## Definitions

| Phenotype | Question it answers |
| --- | --- |
| `melanoma_any` | How many people have any melanoma? In situ is listed separately, because the general melanoma concept does not include it. |
| `melanoma_histopathology_adults` | Trial feasibility: adults with melanoma and a histopathology procedure |
| `keratinocyte_cancer_head_neck` | Basal or squamous cell carcinoma with an image of the head or neck |
| `nevus_without_histopathology` | Nevi with no biopsy: the cohort whose labels are weaker |

## The language

Only these keys are accepted. Anything else is refused before a query is built.

```yaml
name: melanoma_histopathology_adults   # must match the folder
version: 1                             # must match the file name
title: Adults with melanoma and a histopathology procedure
purpose: why the cohort is needed
author: who wrote it
index:                                 # the qualifying condition; age is measured here
  condition_concept_ids: [4162276, 141232, 40391314]
  age_min: 18                          # optional, integer years
  age_max: 90                          # optional
include:                               # each: one key, a list of concept IDs
  - procedure_concept_ids: [43531286]
exclude:
  - anatomic_site_concept_ids: [4263673]
breakdown: [gender_concept_id, age_band]   # optional; only these two
```

Criterion keys:
- `condition_concept_ids`
- `procedure_concept_ids`
- `anatomic_site_concept_ids`
- `modality_concept_ids`
- `gender_concept_ids`

Criteria are person-level, as in OMOP cohort tools. A person who has the index condition
on one lesion and a procedure on another meets both.

## Workflow

1. **Request.** An investigator opens a request issue from the form. The question, cohort,
   data needed and approval status are recorded there.
2. **Define.** The data engineer writes `phenotypes/<name>/v1.yaml`. Check it with
   `python scripts/cohort.py check`, which prints its hash and approval status.
3. **Approve.** A reviewer records a decision on that exact definition:
   `python scripts/cohort.py decide <file> approve --reviewer "<name>" --reason "<why>" --request <issue URL>`.
   - Decisions go to `phenotypes/approvals.jsonl`.
   - `escalate` hands the item to someone else and does not approve it.
4. **Run.** `python scripts/cohort.py run <file> --requester "<role>" --request <issue URL> --engine databricks`.
   - The output is an attrition table (people left after each step) and the breakdown.
   - Counts from 1 to 10 show as `<11`.
   - A count of 11 or more is shown as `suppressed` where it would let someone work out a
     hidden one by subtraction.
5. **Audit.** Every refusal, decision and run is appended to a hash-chained log.
   `python scripts/cohort.py verify` checks the chain.

## Limits

- Differencing across two separate requests (two cohorts that differ by a few people) is not
  detected. A reviewer checks for it when approving similar definitions.
- In this data, a person is a real patient only where ISIC records a patient ID. Elsewhere one
  person is one lesion (`docs/data_dictionary.md`).
- Sex is concept 0 until the Gender vocabulary is downloaded, so a `gender_concept_id`
  breakdown has one group for now.
