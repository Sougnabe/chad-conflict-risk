# Model Card — Farmer-Herder Conflict Risk Outlook, Southern Chad

*Format adapted from Mitchell et al. (2019), "Model Cards for Model Reporting".*

## Model details
- **Developer:** Sougnabe Payang, undergraduate capstone project, BSc Software Engineering
  (Machine Learning), African Leadership University, 2026. Supervisor: Emmanuel Adjei.
- **Ethics approval:** ALU Research Ethics Committee, approval code SEPT26-BSE-087
  (5 November 2026, valid 6 months).
- **Model used by the dashboard:** logistic regression (`logreg_OB`, L2-regularised,
  class-weighted, standardised inputs), operational mode, quarterly horizon.
- **Output:** for each of the 26 departments of southern Chad, a risk score for the coming
  quarter, used to rank departments. Scores are **not calibrated probabilities**.
- **Code:** `scripts/train_models.py --mode operational --unit quarter --fixed-model logreg_OB`;
  monthly refresh with `scripts/update_outlook.py`.

## Intended use
- **Primary use:** help humanitarian organisations, local authorities and transhumance or
  mediation committees decide **where to focus prevention** (mediation, dialogue, monitoring of
  corridors) in the coming quarter.
- **Intended users:** practitioners who know the local context and can interpret the ranking
  alongside field information.
- **Form of use:** decision support. A ranking is one input among others; it never triggers
  action on its own.

## Out-of-scope and prohibited uses
- Predicting individual incidents, villages or dates. The model works at department-quarter level.
- Targeting, profiling or policing any community. The model contains no ethnic or religious
  information, and its outputs must not be linked to such information.
- Automated decisions about people, land, movements of herds or security deployments.
- Publishing the maps or rankings as "danger maps" of areas or communities.
- Use outside southern Chad, or with a different conflict definition, without re-validation.

## Inputs
Only information available when a forecast is issued:
- **Conflict history delayed by 12 months** (ACLED Research tier, published with a 12-month
  delay): events in the department, its neighbours and its province, months since the last
  known event.
- **Satellite indicators up to the previous month:** crop greenness and its anomaly (MODIS
  NDVI over cropland), rainfall and its anomaly (CHIRPS), pasture greenness and rainfall
  anomalies in the herds' northern departure zones (Batha, Guéra, Salamat, Ouaddaï), cropland
  share (ESA WorldCover), surface water (JRC), population density (WorldPop).
- **Season** (month of the year).

## Training and evaluation data
- **Target:** at least one farmer-herder violent event in a department during the quarter.
  Events come from ACLED (1,593 events for Chad, 2015–2025), classified by an AI assistant with a
  written codebook (`docs/labeling_codebook.md`). The author labelled a stratified sample of 60
  events blind (without seeing the AI labels): Cohen's kappa = 0.83 for farmer-herder vs. other;
  two codebook rules were added from the disagreements.
- **Period:** model selection on 2019–2023 (rolling-origin validation); single evaluation on the
  held-out period 2024 – September 2025 (182 department-quarters, 20 with a conflict,
  prevalence 0.11).
- **No look-ahead:** every feature uses only information available before the forecast period;
  this is checked by automated tests (`tests/integration/test_pipeline.py`).

## Performance (test period, quarterly)
| Model | PR-AUC [95% CI] | ROC-AUC |
|---|---|---|
| Random ranking | 0.110 | 0.50 |
| Delayed conflict history only (`logreg_OA`) | 0.110 [0.08–0.18] | 0.46 |
| Rule: conflicts in the 12 months up to t-13 | 0.189 [0.11–0.37] | 0.59 |
| **Delayed history + satellite (`logreg_OB`, dashboard)** | **0.214 [0.11–0.41]** | **0.66** |

**Priority levels.** In the test period, the 10 departments ranked highest each quarter had a
conflict in **20%** of department-quarters (14 of 70), against **5%** for the other 16 (6 of
112), and they contained **70%** of the department-quarters with a conflict. Ranks 1–5 did not
do better than ranks 6–10 (20% each), so the dashboard shows two levels only.

**Model choice.** The automatic selection on validation favoured XGBoost in 3 of the 4 runs
(including the operational quarterly run, where it chose `xgboost_OA`, a model without any
satellite feature), and XGBoost then underperformed logistic regression out of sample in the
quarterly runs: with 5 to 9 positive cases per validation year, validation cannot reliably
separate models. The dashboard model was fixed by design (it must react to
current conditions, hence satellite features; logistic regression is the most stable and
interpretable option). This decision was taken after the test results were known and is
reported as such; all models are shown in `reports/metrics_*.csv`.

## Main factors
Largest standardised coefficients among satellite features (higher risk when): crop greenness
**below** its normal level in the south; **less** surface water; pasture greenness **below**
normal in the northern departure zones; **more** rainfall in the previous month (crops still in
the fields). This matches the project's hypothesis: drier conditions where herds come from, and
standing crops where they arrive. Conflict-history features are correlated with one another and
should be interpreted together, not individually.

## Limitations
- **Small number of events:** confidence intervals are wide; the improvement from satellite
  data is suggestive, not conclusive.
- **Monthly horizon not usable:** at department-month level no model beats a random ranking.
- **Reporting bias:** ACLED relies on media and partner reports; remote areas and small
  incidents may be under-reported.
- **Herds are inferred, not observed;** local mediation, which often prevents escalation, is not
  visible in the data; the satellite sees crops drying, not the harvest itself.
- **Delay of conflict data:** with real-time conflict data, the research model (recent
  history) would also be available operationally.
- **No validation by practitioners yet:** the practitioner questionnaire
  (`docs/practitioner_feedback_form.md`) has not been sent; usefulness is not yet demonstrated.
- **Proxies:** the protected-characteristic guard checks feature names, not hidden proxies;
  population density or pasture conditions in the north may still be linked to where herders live.
- **Access:** no Arabic version, although Arabic is an official language of Chad (an unchecked
  machine translation of the risk warnings was judged worse than none); on a low-end phone with
  slow 3G the map takes about 33 s to load; internet access is required; there is no SMS, radio
  or audio channel for low-literacy users.

## Ethical considerations
- Aggregated department-level outputs only; no personal, ethnic or religious data.
- Enforced in code: `check_feature_sets` (`chadrisk/experiment.py`) stops training if a
  feature describes ethnicity, religion, language or conflict actors.
- The dashboard collects nothing about its visitors; its *Privacy & terms* page states the
  promises, prohibited uses and data licences.
- Risk rankings can stigmatise areas or communities if communicated carelessly. They should be
  shared with organisations working on prevention, with this card, and not published as
  "danger maps".
- Errors have costs both ways: a department wrongly ranked low may receive less attention.
  Rankings must be combined with local knowledge.
- Herder and farmer communities themselves have not been consulted directly.
- **AI use:** the author wrote the code and used an AI assistant to clean it up (bug fixes,
  structure, tests). The labelling codebook and the classification of all 1,593 events were
  produced by an AI assistant following the author's instructions, and validated on a blind
  60-event sample (see above), not event by event.

## Maintenance
- **Satellite data:** monthly, with `python scripts/update_outlook.py` (about 3 weeks after the
  end of each month).
- **Conflict data:** periodically; new ACLED events must be classified with the codebook before
  use, then the pipeline is re-run (README, steps 1–8).
- **Re-validation:** recompute the test metrics whenever a full new year of conflict data is
  available.
