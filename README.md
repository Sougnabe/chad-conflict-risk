# Farmer-Herder Conflict Risk in Southern Chad

Capstone project — BSc Software Engineering (Machine Learning), African Leadership University.

The project estimates, for each of the 26 departments of southern Chad and for each month,
the risk of violent farmer-herder conflict. It combines a manually validated conflict dataset
(ACLED) with satellite indicators of pasture and the crop calendar. A Streamlit dashboard
presents the results to support humanitarian and local decision-makers.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Download conflict events (ACLED API) | `scripts/download_acled.py` | `data/raw/acled_chad_2015_2025.csv` |
| 2. Apply validated labels, build target | `scripts/build_target.py` | `data/processed/acled_chad_labeled.csv` |
| 3. Exploratory figures | `scripts/eda_acled.py` | `reports/figures/*.png` |
| 4. Department x month panel | `scripts/build_panel.py` | `data/processed/panel_dept_month.csv` |
| 5. Application database | `scripts/init_db.py` | `data/app.db` |
| 6. Satellite indicators (Google Earth Engine) | `scripts/gee_extract.py` | `data/processed/satellite_features.csv` |
| 7. Feature table (no look-ahead) | `scripts/build_features.py` | `data/processed/features.csv` |
| 8. Training, evaluation, SHAP, predictions | `scripts/train_models.py` (see below) | `reports/metrics_<mode>_<unit>.csv`, predictions in `data/app.db` |
| 9. Dashboard | `streamlit run app/app.py` | — |

Step 8 is run four times:

```
python scripts/train_models.py --mode research --unit quarter
python scripts/train_models.py --mode research --unit month
python scripts/train_models.py --mode operational --unit quarter --fixed-model logreg_OB
python scripts/train_models.py --mode operational --unit month
```

**Monthly update of the outlook** (new satellite month, rebuilt features, retrained
operational models), in one command:

```
python scripts/update_outlook.py
```

The satellite extraction detects the last complete month automatically. Conflict data are
refreshed separately, because new ACLED events must first be classified with the codebook.

Steps 4 to 8 can be re-run in order at any time. Step 6 requires a Google Earth Engine account;
until it is run, only the conflict-history model (feature set A) is trained.

## Notebook

`notebooks/capstone_pipeline.ipynb` contains the whole pipeline in one place, with an
explanation and the outputs (tables, figures, test results) for each step. It is generated
from the scripts, which remain the reference implementation:

```
python scripts/make_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/capstone_pipeline.ipynb
```

## Setup

```
pip install -r requirements.txt
copy .env.example .env      # then fill in ACLED and Earth Engine credentials
earthengine authenticate    # once, for step 6
```

## Tests

```
python -m pytest -v
```
15 automated tests check label and panel consistency, the absence of look-ahead in every
feature (history, delayed history and satellite), the absence of outcome leakage, the validity of the stored
predictions and priority levels, and that the dashboard runs without errors in English
and French.

## Deployment (Streamlit Community Cloud)
1. Push the repository to GitHub. Files containing raw ACLED notes are excluded by `.gitignore`.
2. On share.streamlit.io, create an app from the repository with main file `app/app.py`.
   Dependencies are read from `app/requirements.txt`.
3. The deployed app reads only `data/app.db` and `reports/`, which hold aggregated data.

## Methodology in brief
- **Target:** at least one farmer-herder conflict event in a department-month. Events were
  classified with a written codebook (`docs/labeling_codebook.md`); blind validation on 60
  events gave Cohen's kappa = 0.83.
- **Unit:** 26 departments x months, January 2019 – September 2025 (2018 used only as burn-in).
  Two horizons are evaluated: next month, and next quarter (risk of at least one event in the
  three months starting at the forecast date).
- **Two models, two purposes.**
  - *Research model* — conflict history up to the previous month. Feature sets: A = conflict
    history; B = A + southern satellite indicators; C = B + northern herd-departure zones.
    It answers the research question: do satellite indicators add information?
  - *Operational model* — only what is really available when a forecast is issued: ACLED
    Research-tier data are published with a 12-month delay, so conflict history stops at
    month t-13; satellite data run up to the previous month. Feature sets: OA = lagged history
    + season; OB = OA + satellite. It produces the dashboard's current outlook (logistic
    regression OB, fixed by design because the outlook must react to current conditions and
    automatic selection was unstable with so few positive cases).
- **Validation:** model selection on 2019–2023 by rolling-origin validation (validate on 2021,
  2022, 2023), single evaluation on the held-out test period 2024–Sept 2025.
  Every feature uses only information available before the predicted month.
- **Metrics:** PR-AUC (primary, with month-block bootstrap CI), ROC-AUC, recall in the top 10%,
  Brier score. Baseline to beat: conflicts in the last 12 months.

## Dashboard
Available in English and French. Tabs: *Current outlook* (risk for the coming quarter, with
the factors behind each department's score and what the priority levels have meant in the
past), *Replay 2024–2025*, *Findings* (answers to the research question, read from the result
files), *Observed conflicts*, *About the models*.

## Documentation
- `docs/labeling_codebook.md` — how conflict events were classified and validated.
- `docs/model_card.md` — intended use, prohibited uses, performance, limitations, ethics.
- `docs/practitioner_feedback_form.md` — questionnaire for practitioners.

## Data and ethics
- Raw ACLED data are not redistributed (licence); `data/raw/` is excluded from Git.
- Results are aggregated at department level; no ethnic or religious variables are used.
- The dashboard is a decision-support tool, not an automated decision system.
- AI tools were used for coding assistance and the first-pass event classification;
  all labels used as ground truth were checked against a blind human validation.
