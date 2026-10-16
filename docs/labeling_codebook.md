# Labeling Codebook — Farmer-Herder Conflict Events (ACLED, Chad 2015–2025)

## Source
ACLED event data for Chad, 1 January 2015 – 2 October 2025 (1,593 events), downloaded via the ACLED API
with Research-tier access (12-month publication lag). Raw data are not redistributed (ACLED licence).

## Categories

| Label | Definition | Used as |
|---|---|---|
| **FH** | Direct violence (clash, attack, killing, injury, arson, cattle raid) between farmers and herders/pastoralists, where the notes explicitly identify the parties as farmers/agriculturalists and herders/pastoralists, or describe a dispute involving livestock and crop fields/farmland. | Primary target |
| **AMB** | Violence plausibly linked to farmer-herder dynamics but not explicitly framed as such: intercommunal or ethnic clashes with no stated farmer/herder roles; attacks by pastoralist-identified groups on non-farmer targets; disputes over wells or land between unspecified communities. | Sensitivity analysis (broad target = FH + AMB) |
| **REL** | Non-violent or secondary events linked to a farmer-herder conflict: protests, arrests, states of emergency, reconciliation ceremonies, displacement. | Excluded from targets |
| **NO** | All other events (insurgency, rebellion, political protest, crime, state violence, etc.). Default label. | Negative class |

## Decision rules
1. **Explicit framing required.** An ethnic label alone (e.g. "Fulani militia attacked farmers") is not sufficient for FH unless the notes also mention herding, pastoralists, livestock or crop damage. Such cases are labeled AMB. This avoids inferring conflict type from ethnicity.
2. **Cause does not matter.** Violence between farmer and herder groups is FH regardless of the stated trigger (land, crop damage, cattle theft, or an unrelated dispute), because the stated cause is often unknown or only the trigger of deeper tensions.
3. **Externally driven cross-border retaliation is excluded.** Attacks by armed groups from neighbouring countries whose stated motive lies outside Chad (e.g. the May 2023 Mbororo militia attacks in retaliation for killings in the Central African Republic) are labeled AMB, as they cannot be explained by local pasture or harvest conditions.
4. **Elite or state violence is NO.** Violence by state agents or individual notables against herders or farmers is not a farmer-herder communal conflict.
5. **Multi-day and duplicate events** are kept as coded by ACLED. The model target is binary at department-month level ("at least one FH event"), so duplicates do not inflate it.

## Procedure
1. Every one of the 1,593 event notes was read and labeled (AI-assisted first pass, applying the rules above).
2. A stratified validation sample of 60 events (30 FH, 15 AMB, 15 keyword-matching NO), with labels hidden, was independently labeled by the author.
3. Agreement was measured with Cohen's kappa; disagreements were reviewed and resolved into decision rules 2 and 3.

## Validation results (before rule refinement)

| Metric | Value |
|---|---|
| Agreement, FH vs. rest | 91.7% |
| **Cohen's kappa, FH vs. rest** | **0.83** (almost perfect, Landis & Koch 1977) |
| Agreement, 4 classes | 70.0% |
| Cohen's kappa, 4 classes | 0.53 (moderate; disagreement concentrated on AMB vs. NO) |

## Comparison with keyword filtering
A keyword filter on the notes (herder, pastoral, cattle, farmer, field, etc.) achieved 60% precision and 99% recall
against the manual labels: it captures almost all true events but includes many false positives
(Boko Haram cattle raids, refugee-camp attacks, state violence). Manual labeling was therefore necessary.

## Final label counts
FH = 154 (112 in the seven southern provinces, 73%), AMB = 80, REL = 25, NO = 1,334.
