# Project walkthrough: 3-minute script

For a screen recording of the repository on GitHub, about 3 minutes at a normal pace. Words to
say are in quotes; what to show is in brackets. Every number is from a committed results file,
named on screen when it appears. (The cohort demo has its own script:
[demo_walkthrough.md](demo_walkthrough.md).)

**0:00 – 0:20. What this is**
[Open the README. The title and "not for clinical use" line are on screen.]
"This is a skin lesion classifier with a drift monitor, built on Databricks Free Edition at no
cost. It is a portfolio project on public ISIC images, a research prototype, and not for
clinical use."

**0:20 – 0:50. The problem it is built around**
[Scroll to the Phase 2 results table.]
"The shipped model is a three-seed ensemble. On the test split of the four hospitals it was
trained on, its balanced accuracy is 66.5 percent. At two hospitals it never saw, Buenos Aires
and a smartphone collection from Brazil, it falls to 55.9 and 39.4 percent. Nothing in the
model's output says so. That silent drop is why the project has a drift monitor."
[Point at `results/phase2/v2/test/final_test.json` beside the table.]

**0:50 – 1:20. How the numbers were protected**
[Open `docs/EVAL_PLAN.md`, then any `results/**/spec.json`.]
"Every test result follows a specification committed before its single look at the test set.
Splits are by lesion, never by image, and a test enforces it. A file of silent failures lists
every bug that reported success while being wrong, and the check that now catches it."
[Show `docs/silent-failures.md` for two seconds.]

**1:20 – 2:00. It runs as a service**
[Open `docs/ARCHITECTURE.md`; the diagram is on screen.]
"The data lands through a Databricks pipeline with quality gates that fail the update. The
model is served by a FastAPI service that refuses bad input and abstains when an image looks
unlike its training data. A monthly job replays real new-hospital images: over twelve months it
alerted on every fully shifted month and on none of the in-distribution ones. A retraining
rule, written down before the run, declined a challenger that gained six points at the new
hospital but lost more than allowed on the original ones."
[Point at the dashed lines.]
"Dashed lines are planned or hand-run steps; the page says which."

**2:00 – 2:35. Engineered like production**
[Open the README's "Production readiness" list, then a recent pull request's checks.]
"Each backlog item ships as one reviewed pull request. CI runs the tests, a type check, an
accuracy gate on committed metrics, secret scanning, a workflow audit, and a scan of the API
image that fails on any fixable high-severity finding. Every action and the base image are
pinned, and the image's system packages come from a fixed Debian snapshot, so a build is
reproducible."

**2:35 – 3:00. What it does not claim**
[Open `docs/BUILT_VS_PLANNED.md`.]
"This file is the line between built and planned. The API is not deployed; batch scoring at
scale and detector quality on staged shifts have not run yet; and staged drift, when it runs,
is labelled staged. The code, results and decisions are all in the repository."
