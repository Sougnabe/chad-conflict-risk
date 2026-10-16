"""
Train and evaluate the farmer-herder conflict risk models.

Unit of analysis (--unit):
    month   : risk of >= 1 conflict in the department during the month
    quarter : risk of >= 1 conflict during the quarter starting that month,
              using only information available at the start of the quarter

Temporal split (no random shuffling):
    model selection 2019-2023 | test 2024 - 2025-09 (used once, never for tuning)
Hyperparameters are selected by rolling-origin validation (validate on 2021, 2022
and 2023, each time training on the preceding years), then the model is refit on
2019-2023 and evaluated once on the test period.

Comparisons:
    rule : risk = conflicts in the last 12 months (baseline to beat)
    A    : conflict history (h_)
    B    : A + southern satellite conditions (south_)   [if available]
    C    : B + northern satellite conditions (north_)   [if available]
    x {logistic regression, XGBoost}

Metrics: PR-AUC (primary), ROC-AUC, recall and precision within the 10% of
department-months ranked most at risk, Brier score; 95% CI by month-block bootstrap.

Outputs: reports/metrics_<unit>.csv, reports/figures/*_<unit>.png, predictions in data/app.db

Usage: python scripts/train_models.py --unit month
       python scripts/train_models.py --unit quarter
"""

import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
DB = ROOT / "data" / "app.db"
SEED = 42
TOP = 0.10

parser = argparse.ArgumentParser()
parser.add_argument("--unit", choices=["month", "quarter"], default="month")
UNIT = parser.parse_args().unit
VAL_YEARS = [2021, 2022, 2023]

df = pd.read_csv(PROC / "features.csv")
if UNIT == "quarter":
    # Target = at least one conflict in months t, t+1, t+2; rows = first month of each quarter
    df = df.sort_values(["dept_id", "ym"]).reset_index(drop=True)
    for y in ["y_fh", "y_broad"]:
        g = df.groupby("dept_id")[y]
        df[y] = pd.concat([df[y], g.shift(-1), g.shift(-2)], axis=1).max(axis=1, skipna=False)
    df = df[df["month"].isin([1, 4, 7, 10])]
df = df[df["year"] >= 2019].reset_index(drop=True)
print(f"Unit of analysis: {UNIT}")
forecast = df[df["y_fh"].isna()].copy()
df = df[df["y_fh"].notna()].copy()
df["y_fh"] = df["y_fh"].astype(int)
df["y_broad"] = df["y_broad"].astype(int)

train = df[df["year"] <= 2022]
val = df[df["year"] == 2023]
trval = df[df["year"] <= 2023]
test = df[df["year"] >= 2024]

h_cols = [c for c in df.columns if c.startswith("h_")]
s_cols = [c for c in df.columns if c.startswith("south_")]
n_cols = [c for c in df.columns if c.startswith("north_")]
SETS = {"A": h_cols}
if s_cols:
    SETS["B"] = h_cols + s_cols
if n_cols:
    SETS["C"] = h_cols + s_cols + n_cols

# Leakage guard: outcome columns of the predicted month must never be features
OUTCOMES = {"n_fh", "n_broad", "n_all_events", "fatalities_fh", "y_fh", "y_broad"}
for name, cols in SETS.items():
    assert not OUTCOMES & set(cols), f"Leakage in feature set {name}: {OUTCOMES & set(cols)}"

print(f"Model selection 2019-2023: {len(trval)} rows ({trval['y_fh'].sum()} positives) | "
      f"test 2024-2025: {len(test)} rows ({test['y_fh'].sum()} positives)")
print(f"Feature sets: { {k: len(v) for k, v in SETS.items()} }")


# ---------------- Metrics ----------------
def top_k(y, p, frac=TOP):
    """Recall and precision within the top `frac` of rows ranked by risk."""
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-p, kind="stable")[:k]
    hits = y[idx].sum()
    return hits / max(y.sum(), 1), hits / k


def metrics(y, p, months, probabilistic=True, n_boot=1000):
    """Evaluation metrics, with a month-block bootstrap CI for PR-AUC."""
    y, p = np.asarray(y), np.asarray(p)
    rec, prec = top_k(y, p)
    out = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "recall_top10": rec,
        "precision_top10": prec,
        "brier": brier_score_loss(y, p) if probabilistic else np.nan,
        "prevalence": y.mean(),
    }
    # Month-block bootstrap (respects correlation between departments in the same month)
    rng = np.random.default_rng(SEED)
    um = np.unique(months)
    groups = {m: np.where(months == m)[0] for m in um}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[m] for m in rng.choice(um, len(um))])
        if y[idx].sum() > 0:
            boots.append(average_precision_score(y[idx], p[idx]))
    out["pr_auc_lo"], out["pr_auc_hi"] = np.percentile(boots, [2.5, 97.5])
    return out


# ---------------- Models ----------------
def make_lr(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    )


def make_xgb(depth, n_est, mcw, spw):
    return XGBClassifier(
        max_depth=depth, n_estimators=n_est, learning_rate=0.05, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
        random_state=SEED, n_jobs=4,
    )


def grid(kind, ytr):
    """Small hyperparameter grid for each model family."""
    if kind == "logreg":
        return [("C=%g" % C, lambda C=C: make_lr(C)) for C in (0.01, 0.1, 1.0)]
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    return [
        (f"depth={d},n={n},mcw={m}", lambda d=d, n=n, m=m: make_xgb(d, n, m, spw))
        for d in (2, 3) for n in (100, 300) for m in (1, 5)
    ]


def fit_select(kind, cols, target="y_fh"):
    """Rolling-origin selection (validate 2021, 2022, 2023), then refit on 2019-2023."""
    folds = [(df[df["year"] < vy], df[df["year"] == vy]) for vy in VAL_YEARS]
    folds = [(tr, va) for tr, va in folds if va[target].sum() > 0 and tr[target].sum() > 0]
    best = None
    for name, _ in grid(kind, trval[target]):
        aps = []
        for tr, va in folds:
            build = dict(grid(kind, tr[target]))[name]
            m = build().fit(tr[cols], tr[target])
            aps.append(average_precision_score(va[target], m.predict_proba(va[cols])[:, 1]))
        score = float(np.mean(aps))
        if best is None or score > best[1]:
            best = (name, score)
    final = dict(grid(kind, trval[target]))[best[0]]().fit(trval[cols], trval[target])
    return final, best[0], best[1]


rows, fitted = [], {}
months_test = test["ym"].to_numpy()

# Baseline rule (not probabilistic)
rule = (test["h_fh_12m"] + 0.01 * test["h_fh_cum"]).to_numpy()
rows.append({"model": "rule_history_12m", "features": "-", "hyperparams": "-",
             "val_pr_auc": np.nan,
             **metrics(test["y_fh"], rule, months_test, probabilistic=False)})

for set_name, cols in SETS.items():
    for kind in ("logreg", "xgboost"):
        model, hp, val_ap = fit_select(kind, cols)
        p = model.predict_proba(test[cols])[:, 1]
        fitted[f"{kind}_{set_name}"] = (model, cols, p)
        rows.append({"model": f"{kind}_{set_name}", "features": set_name, "hyperparams": hp,
                     "val_pr_auc": val_ap, **metrics(test["y_fh"], p, months_test)})

res = pd.DataFrame(rows)
res.to_csv(REPORTS / f"metrics_{UNIT}.csv", index=False)
pd.set_option("display.width", 200)
print(f"\nTEST RESULTS (2024 - 2025-09), unit = {UNIT}")
print(res[["model", "val_pr_auc", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc",
           "recall_top10", "precision_top10", "brier"]].round(3).to_string(index=False))

# Best model = best mean rolling-origin validation PR-AUC (never selected on the test set)
ml = res[res["model"] != "rule_history_12m"]
best_name = ml.loc[ml["val_pr_auc"].idxmax(), "model"]
best_model, best_cols, best_p = fitted[best_name]
print(f"\nSelected model (by rolling-origin validation): {best_name}")

# ---------------- Sensitivity analysis: broad target ----------------
kind, set_name = best_name.split("_")
m_b, hp_b, _ = fit_select(kind, best_cols, target="y_broad")
sens = metrics(test["y_broad"], m_b.predict_proba(test[best_cols])[:, 1], months_test)
print(f"Sensitivity (FH + AMB target): PR-AUC = {sens['pr_auc']:.3f} "
      f"(prevalence {sens['prevalence']:.3f})")
with open(REPORTS / f"sensitivity_broad_{UNIT}.json", "w", encoding="utf8") as f:
    json.dump({"model": best_name, **{k: float(v) for k, v in sens.items()}}, f, indent=2)

# ---------------- Figures ----------------
fig, ax = plt.subplots(figsize=(7, 5))
for name, (_, _, p) in fitted.items():
    pr, rc, _ = precision_recall_curve(test["y_fh"], p)
    ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(test['y_fh'], p):.2f})")
pr, rc, _ = precision_recall_curve(test["y_fh"], rule)
ax.plot(rc, pr, "k--", label=f"rule (AP={average_precision_score(test['y_fh'], rule):.2f})")
ax.axhline(test["y_fh"].mean(), color="grey", lw=0.8, label="random")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-recall curves ({UNIT}), test period 2024 - 2025-09")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGS / f"pr_curves_{UNIT}.png", dpi=150)
plt.close()

# SHAP (XGBoost trained on the same feature set as the selected model)
xgb_model, xgb_cols, _ = fitted[f"xgboost_{set_name}"]
sv = shap.TreeExplainer(xgb_model).shap_values(test[xgb_cols])
shap.summary_plot(sv, test[xgb_cols], show=False, max_display=15)
plt.title(f"SHAP values, xgboost_{set_name} ({UNIT}), test period")
plt.tight_layout()
plt.savefig(FIGS / f"shap_summary_{UNIT}.png", dpi=150, bbox_inches="tight")
plt.close()
imp = pd.Series(np.abs(sv).mean(axis=0), index=xgb_cols).sort_values(ascending=False)
imp.to_csv(REPORTS / f"shap_importance_{UNIT}.csv", header=["mean_abs_shap"])
print("\nSHAP importance (mean |value|):")
print(imp.round(4).head(10).to_string())

# ---------------- Predictions to the database ----------------
# 1) Out-of-sample on the test period; 2) forecast for the next month
final = fit_select(kind, best_cols)[0]
final_all = final.fit(df[best_cols], df["y_fh"])  # refit on all labelled data
forecast["risk"] = final_all.predict_proba(forecast[best_cols])[:, 1]
pred = pd.concat([
    test[["dept_id", "ym"]].assign(model=f"{best_name}_{UNIT}_test", risk=best_p),
    forecast[["dept_id", "ym", "risk"]].assign(model=f"{best_name}_{UNIT}_forecast"),
])
with sqlite3.connect(DB) as con:
    con.execute("DELETE FROM predictions WHERE model LIKE ?", (f"%{UNIT}%",))
    con.executemany(
        "INSERT INTO predictions (dept_id, ym, model, risk) VALUES (?, ?, ?, ?)",
        pred[["dept_id", "ym", "model", "risk"]].itertuples(index=False),
    )
print(f"\n{len(pred)} predictions saved ({forecast['ym'].iloc[0]} = forecast)")
