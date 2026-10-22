# Cohort demo: 3-minute walkthrough script

For a screen recording of `app/cohort_demo.py`. About 3 minutes at a normal pace. Words to
say are in quotes; what to do is in brackets.

**0:00 – 0:20. What this is**
[Open the app. The page title and caption are on screen.]
"This is a governed cohort service over OMOP tables. It's a portfolio build on public
research images, the PAD-UFES-20 collection from the ISIC Archive. It shows counts only. No
images, and no patient records."

**0:20 – 0:50. The data and the vocabulary**
[Scroll to the phenotype picker; leave "From the phenotype library" selected.]
"When the app starts, it pulls the collection's public metadata and builds OMOP CDM tables
with the same code the Databricks pipeline uses: person, condition occurrence, procedure
occurrence and image occurrence. Diagnoses and body sites are OMOP concept IDs, looked up in
Athena. The SNOMED names aren't redistributed, so concepts show as IDs with an Athena link."

**0:50 – 1:20. A phenotype is a versioned file**
[Pick melanoma_any. Point at the YAML and the hash, then open "Concepts in this definition".]
"An investigator's question becomes a phenotype: a versioned YAML file in an allowlisted
language. No free SQL. The service hashes the exact definition. Status: pending."

**1:20 – 1:45. Refusal, then approval**
[Click Run.]
"It refuses: nobody has approved this exact definition."
[Click Approve, then Run.]
"Here I'm the reviewer, in a scratch approvals file for this session only. Approval is tied
to the hash. Change one concept ID and it's a new version that needs a new approval."

**1:45 – 2:20. Suppressed results**
[Point at the attrition table, then the breakdown.]
"The output is an attrition table and a breakdown. Any count from 1 to 10 is suppressed,
and so is anything that would let you work it out by subtraction. That includes subtraction
against results released earlier in this session for nested definitions (each visitor has
their own log here; the real service keeps one). Every hidden cell carries the same label,
because a label like 'under 11' would itself give numbers away."

**2:20 – 2:45. Build your own**
[Switch to "Build your own". Pick two diagnoses, set an age range, tick histopathology,
choose a breakdown. Approve and run.]
"A visitor can build a cohort from the controls. It goes through the same parser, the same
approval and the same suppression."

**2:45 – 3:00. The audit log**
[Scroll to the audit log.]
"Every refusal, decision and run is in a hash-chained audit log, so an edited entry breaks
the chain. The repository's README links the protocol, the governance packet and the
service metrics."
