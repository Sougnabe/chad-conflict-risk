"""
Prototype - Early warning of farmer-herder conflict risk in southern Chad.

Run: streamlit run app/app.py
"""

import json
import sqlite3
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "app.db"

st.set_page_config(page_title="Farmer-Herder Conflict Risk - Southern Chad", layout="wide")


@st.cache_data
def load():
    with sqlite3.connect(DB) as con:
        depts = pd.read_sql("SELECT * FROM departments", con)
        panel = pd.read_sql("SELECT * FROM panel", con)
        preds = pd.read_sql("SELECT * FROM predictions", con)
    return depts, panel, preds


def department_map(frame, value_col, legend, fmt="{:.0f}"):
    """Choropleth of one value per department, with a hover tooltip."""
    fmap = folium.Map(location=[9.3, 16.8], zoom_start=7, tiles="OpenStreetMap")
    features = []
    for r in frame.itertuples():
        features.append({
            "type": "Feature",
            "id": r.dept_id,
            "geometry": json.loads(r.geometry)["features"][0]["geometry"],
            "properties": {"department": r.department, "province": r.province,
                           "value": fmt.format(getattr(r, value_col))},
        })
    gj = {"type": "FeatureCollection", "features": features}
    folium.Choropleth(
        geo_data=gj, data=frame, columns=["dept_id", value_col], key_on="feature.id",
        fill_color="YlOrRd", fill_opacity=0.75, line_opacity=0.4, legend_name=legend,
    ).add_to(fmap)
    folium.GeoJson(
        gj, style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(["department", "province", "value"],
                                      aliases=["Department", "Province", legend]),
    ).add_to(fmap)
    return fmap


depts, panel, preds = load()
months = sorted(panel["ym"].unique())

st.title("Farmer-Herder Conflict Risk — Southern Chad")
st.caption(
    "Prototype early-warning dashboard. Observed conflicts: ACLED events manually classified "
    "as farmer-herder violence (validated, Cohen's kappa = 0.83), 2019 – September 2025. "
    "Department-level resolution only."
)

tab_obs, tab_risk, tab_about = st.tabs(["Observed conflicts", "Predicted risk", "About the model"])

# --- Tab 1: observed conflicts ---
with tab_obs:
    with st.sidebar:
        st.header("Filters (observed conflicts)")
        provinces = sorted(depts["province"].unique())
        sel_prov = st.multiselect("Provinces", provinces, default=provinces)
        start, end = st.select_slider("Period", options=months, value=(months[0], months[-1]))
        target = st.radio("Conflict definition", ["Strict (FH)", "Broad (FH + ambiguous)"])
        col = "n_fh" if target.startswith("Strict") else "n_broad"

    mask = panel["ym"].between(start, end) & panel["dept_id"].isin(
        depts.loc[depts["province"].isin(sel_prov), "dept_id"]
    )
    sub = panel[mask]
    by_dept = sub.groupby("dept_id")[[col, "fatalities_fh"]].sum().reset_index()
    by_dept = depts.merge(by_dept, on="dept_id", how="left").fillna({col: 0, "fatalities_fh": 0})
    by_dept = by_dept[by_dept["province"].isin(sel_prov)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Conflict events", int(sub[col].sum()))
    c2.metric("Fatalities (strict FH)", int(sub["fatalities_fh"].sum()))
    c3.metric("Department-months with ≥1 event", int((sub[col] > 0).sum()))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Observed conflicts by department")
        st_folium(department_map(by_dept, col, "Events in period"), height=520,
                  use_container_width=True, key="obs_map")
    with right:
        st.subheader("Monthly trend")
        st.bar_chart(sub.groupby("ym")[col].sum())
        st.subheader("Most affected departments")
        st.dataframe(
            by_dept.sort_values(col, ascending=False)[["department", "province", col,
                                                        "fatalities_fh"]]
            .rename(columns={col: "events", "fatalities_fh": "fatalities"}),
            hide_index=True, width="stretch",
        )

# --- Tab 2: predicted risk ---
with tab_risk:
    if preds.empty:
        st.info("No predictions yet. Run scripts/train_models.py.")
    else:
        units = [u for u in ("quarter", "month") if preds["model"].str.contains(u).any()]
        unit = st.radio("Forecast horizon", units, horizontal=True,
                        format_func=lambda u: "Next quarter (3 months)" if u == "quarter"
                        else "Next month")
        upreds = preds[preds["model"].str.contains(unit)]
        pmonths = sorted(upreds["ym"].unique())
        sel = st.select_slider("Starting month", options=pmonths, value=pmonths[-1])
        p = upreds[upreds["ym"] == sel]
        is_forecast = p["model"].str.endswith("_forecast").any()
        st.markdown(
            f"**{sel}** — {'forecast (outcome not yet known)' if is_forecast else 'out-of-sample test month'}"
            f" · model `{p['model'].iloc[0].rsplit('_', 2)[0]}` · horizon: {unit}"
        )
        frame = depts.merge(p[["dept_id", "risk"]], on="dept_id")
        frame["risk_pct"] = 100 * frame["risk"]
        frame["rank"] = frame["risk"].rank(ascending=False).astype(int)
        if not is_forecast:
            n_months = 3 if unit == "quarter" else 1
            window = sorted(m for m in months if m >= sel)[:n_months]
            obs = panel[panel["ym"].isin(window)].groupby("dept_id")["n_fh"].sum().reset_index()
            frame = frame.merge(obs, on="dept_id", how="left")

        left, right = st.columns([3, 2])
        with left:
            st_folium(department_map(frame, "risk_pct", "Relative risk score", "{:.1f}"),
                      height=520, use_container_width=True, key="risk_map")
        with right:
            st.subheader("Departments ranked by risk")
            show = ["rank", "department", "province", "risk_pct"] + (
                [] if is_forecast else ["n_fh"])
            st.dataframe(
                frame.sort_values("rank")[show].rename(
                    columns={"risk_pct": "risk score", "n_fh": "observed events"}),
                hide_index=True, width="stretch",
            )
        st.caption(
            "Risk scores are relative rankings for prioritisation, not calibrated probabilities. "
            "They support human judgement and must not trigger automatic action."
        )

# --- Tab 3: about ---
with tab_about:
    st.subheader("Model evaluation (test period 2024 – Sept 2025)")
    unit_about = st.radio("Unit of analysis", ["quarter", "month"], horizontal=True,
                          key="about_unit")
    metrics_path = ROOT / "reports" / f"metrics_{unit_about}.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        st.dataframe(
            m[["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall_top10",
               "prevalence"]].round(3),
            hide_index=True, width="stretch",
        )
        st.markdown(
            "- **PR-AUC** is the main metric; a random model scores the prevalence.\n"
            "- **recall_top10**: share of real conflicts found in the 10% of "
            "department-months ranked most at risk.\n"
            "- Satellite indicators (pasture, crop calendar, rainfall) are added as "
            "feature sets B and C once extracted from Google Earth Engine."
        )
    figs = ROOT / "reports" / "figures"
    for name, title in [(f"pr_curves_{unit_about}.png", "Precision-recall curves"),
                        (f"shap_summary_{unit_about}.png", "Risk factors (SHAP)")]:
        if (figs / name).exists():
            st.image(str(figs / name), caption=title)

st.caption(
    "Ethics: aggregated department-level data only; no ethnic or religious variables. "
    "Decision-support tool — not an automated decision system."
)
