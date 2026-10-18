"""
Prototype - Early warning of farmer-herder conflict risk in southern Chad.

Tabs:
    Current outlook  : risk for the coming quarter (operational model: satellite data up to last
                       month + conflict history with the 12-month ACLED publication delay)
    Replay 2024-2025 : what the operational model would have flagged, quarter by quarter,
                       compared with the conflicts that actually happened
    Observed conflicts
    About the models

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
REPORTS = ROOT / "reports"
OPS = "operational_quarter"
TOP_N = 5  # departments flagged as highest priority

LABELS = {
    "south_ndvi_crop_anom": "Crop greenness vs normal (south)",
    "south_ndvi_crop": "Crop greenness (south)",
    "south_ndvi_crop_delta": "Change in crop greenness since previous month",
    "south_rain_mm": "Rainfall last month (south)",
    "south_rain_anom": "Rainfall vs normal (south)",
    "south_cropland_frac": "Share of cropland",
    "south_water_occ": "Surface water",
    "south_pop_density": "Population density",
    "north_ndvi_anom": "Pasture greenness vs normal (north, herd origin)",
    "north_rain_anom": "Rainfall vs normal (north, herd origin)",
    "o_fh_12m": "Conflicts in the department (latest known 12 months)",
    "o_fh_cum": "Conflicts in the department (all known history)",
    "o_broad_12m": "Communal violence (latest known 12 months)",
    "o_allev_12m": "All political violence (latest known 12 months)",
    "o_fatal_12m": "Conflict deaths (latest known 12 months)",
    "o_nb_fh_12m": "Conflicts in neighbouring departments (latest known 12 months)",
    "o_prov_fh_12m": "Conflicts elsewhere in the province (latest known 12 months)",
    "o_months_since_fh": "Months since last known conflict",
    "h_month_sin": "Season",
    "h_month_cos": "Season",
}

st.set_page_config(page_title="Farmer-Herder Conflict Risk - Southern Chad", layout="wide")


@st.cache_data
def load():
    with sqlite3.connect(DB) as con:
        depts = pd.read_sql("SELECT * FROM departments", con)
        panel = pd.read_sql("SELECT * FROM panel", con)
        preds = pd.read_sql("SELECT * FROM predictions", con)
        expl = pd.read_sql("SELECT * FROM features WHERE name LIKE ?", con,
                           params=(f"%|{OPS}|%",))
    return depts, panel, preds, expl


def quarter_label(ym):
    p = pd.Period(ym, freq="M")
    return f"{p.strftime('%b')}–{(p + 2).strftime('%b %Y')}"


def department_map(frame, value_col, legend, fmt="{:.0f}", key=None):
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
    return st_folium(fmap, height=520, use_container_width=True, key=key)


def ranked(frame):
    frame = frame.copy()
    frame["risk_score"] = 100 * frame["risk"]
    frame["rank"] = frame["risk"].rank(ascending=False, method="first").astype(int)
    frame["priority"] = frame["rank"].map(
        lambda r: "Highest" if r <= TOP_N else ("Elevated" if r <= 2 * TOP_N else "Lower"))
    return frame.sort_values("rank")


depts, panel, preds, expl = load()
months = sorted(panel["ym"].unique())
data_end = months[-1]
ops = preds[preds["model"].str.contains(OPS)]
fc = ops[ops["model"].str.endswith("_forecast")]
replay = ops[ops["model"].str.endswith("_test")]
model_name = ops["model"].iloc[0].split(f"_{OPS}")[0] if not ops.empty else "-"

st.title("Farmer-Herder Conflict Risk — Southern Chad")
if not fc.empty:
    fc_ym = fc["ym"].iloc[0]
    sat_end = (pd.Period(fc_ym, freq="M") - 1).strftime("%B %Y")
    st.caption(
        f"Research prototype · satellite data up to **{sat_end}** · conflict data (ACLED, "
        f"published with a 12-month delay) up to **{pd.Period(data_end, freq='M').strftime('%B %Y')}** · "
        "department-level only."
    )

tab_now, tab_replay, tab_obs, tab_about = st.tabs(
    ["Current outlook", "Replay 2024–2025", "Observed conflicts", "About the models"])

# ---------------- Current outlook ----------------
with tab_now:
    if fc.empty:
        st.info("No forecast yet. Run: python scripts/train_models.py --mode operational "
                "--unit quarter --fixed-model logreg_OB")
    else:
        st.subheader(f"Conflict risk outlook for {quarter_label(fc_ym)}")
        frame = ranked(depts.merge(fc[["dept_id", "risk"]], on="dept_id"))
        left, right = st.columns([3, 2])
        with left:
            department_map(frame, "risk_score", "Relative risk score", "{:.0f}", key="now")
        with right:
            st.markdown(f"**Highest-priority departments ({TOP_N})**")
            st.dataframe(frame[["rank", "department", "province", "priority", "risk_score"]]
                         .rename(columns={"risk_score": "risk score"}).round(0),
                         hide_index=True, width="stretch", height=420)

        st.markdown("#### Why is a department at risk?")
        sel = st.selectbox("Department", frame["department"], index=0)
        dept_id = frame.loc[frame["department"] == sel, "dept_id"].item()
        e = expl[expl["dept_id"] == dept_id].copy()
        e["kind"] = e["name"].str.split("|").str[0]
        e["feature"] = e["name"].str.split("|").str[2]
        contrib = e[e["kind"] == "contrib"].set_index("feature")["value"]
        values = e[e["kind"] == "value"].set_index("feature")["value"]
        # Correlated conflict-history variables (and the two seasonal terms) are grouped, so
        # that their combined effect is shown instead of offsetting individual terms.
        hist = contrib[contrib.index.str.startswith("o_")]
        season = contrib[contrib.index.str.startswith("h_month")]
        sat = contrib[contrib.index.str.startswith(("south_", "north_"))]
        rows = [("Conflict history (latest known, at least 13 months old)", hist.sum(),
                 f"{values.get('o_fh_12m', 0):.0f} events in the department, "
                 f"{values.get('o_nb_fh_12m', 0):.0f} in neighbours (latest known 12 months)"),
                ("Season", season.sum(), quarter_label(fc_ym))]
        rows += [(LABELS.get(f, f), v, f"{values[f]:.3f}") for f, v in sat.items()]
        table = pd.DataFrame(rows, columns=["factor", "contribution", "current value"])
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
        table["effect"] = table["contribution"].map(lambda v: "raises risk" if v > 0
                                                    else "lowers risk")
        table["strength"] = table["contribution"].abs().round(2)
        st.dataframe(table[["factor", "effect", "strength", "current value"]].head(7),
                     hide_index=True, width="stretch")
        st.caption("Effects come from the logistic model (standardised coefficient × "
                   "standardised value), for the department's current values. 'Current value' "
                   "is the latest observation available when the outlook was issued.")
        st.warning("Risk scores rank departments for prioritisation; they are not calibrated "
                   "probabilities. They must support, not replace, the judgement of people "
                   "who know the local situation.")

# ---------------- Replay ----------------
with tab_replay:
    if replay.empty:
        st.info("No replay predictions available.")
    else:
        st.markdown("If the tool had existed in 2024–2025 — using only the information that "
                    "would really have been available at the time — which departments would "
                    "it have flagged, and where did conflicts actually happen?")
        qs = sorted(replay["ym"].unique())
        q = st.select_slider("Quarter", options=qs, value=qs[-1], format_func=quarter_label)
        frame = ranked(depts.merge(replay[replay["ym"] == q][["dept_id", "risk"]],
                                   on="dept_id"))
        window = [m for m in months if m >= q][:3]
        obs = panel[panel["ym"].isin(window)].groupby("dept_id")["n_fh"].sum()
        frame["observed events"] = frame["dept_id"].map(obs).fillna(0).astype(int)

        hit = frame[(frame["observed events"] > 0)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Departments with a conflict", len(hit))
        c2.metric(f"…of which flagged in the top {TOP_N}", int((hit["rank"] <= TOP_N).sum()))
        c3.metric(f"…of which flagged in the top {2 * TOP_N}",
                  int((hit["rank"] <= 2 * TOP_N).sum()))

        left, right = st.columns([3, 2])
        with left:
            department_map(frame, "risk_score", "Relative risk score", "{:.0f}", key="replay")
        with right:
            st.dataframe(frame[["rank", "department", "province", "observed events"]],
                         hide_index=True, width="stretch", height=480)

        # Summary over the whole test period
        allq = depts[["dept_id"]].merge(replay[["dept_id", "ym", "risk"]], on="dept_id")
        allq["rank"] = allq.groupby("ym")["risk"].rank(ascending=False, method="first")
        allq["events"] = [panel[(panel["dept_id"] == d) & panel["ym"].isin(
            [m for m in months if m >= y][:3])]["n_fh"].sum()
            for d, y in zip(allq["dept_id"], allq["ym"])]
        pos = allq[allq["events"] > 0]
        share = (pos["rank"] <= TOP_N).mean()
        st.info(f"Over the {len(qs)} test quarters, {share:.0%} of department-quarters with a "
                f"conflict were in the model's top {TOP_N} (a random ranking would reach "
                f"{TOP_N / 26:.0%}).")

# ---------------- Observed conflicts ----------------
with tab_obs:
    c1, c2, c3 = st.columns(3)
    provinces = sorted(depts["province"].unique())
    sel_prov = c1.multiselect("Provinces", provinces, default=provinces)
    start, end = c2.select_slider("Period", options=months, value=(months[0], months[-1]))
    target = c3.radio("Conflict definition", ["Strict (FH)", "Broad (FH + ambiguous)"])
    col = "n_fh" if target.startswith("Strict") else "n_broad"

    mask = panel["ym"].between(start, end) & panel["dept_id"].isin(
        depts.loc[depts["province"].isin(sel_prov), "dept_id"])
    sub = panel[mask]
    by_dept = sub.groupby("dept_id")[[col, "fatalities_fh"]].sum().reset_index()
    by_dept = depts.merge(by_dept, on="dept_id", how="left").fillna({col: 0, "fatalities_fh": 0})
    by_dept = by_dept[by_dept["province"].isin(sel_prov)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Conflict events", int(sub[col].sum()))
    m2.metric("Fatalities (strict FH)", int(sub["fatalities_fh"].sum()))
    m3.metric("Department-months with ≥1 event", int((sub[col] > 0).sum()))

    left, right = st.columns([3, 2])
    with left:
        department_map(by_dept, col, "Events in period", key="obs")
    with right:
        st.bar_chart(sub.groupby("ym")[col].sum())
        st.dataframe(
            by_dept.sort_values(col, ascending=False)[["department", "province", col,
                                                        "fatalities_fh"]]
            .rename(columns={col: "events", "fatalities_fh": "fatalities"}),
            hide_index=True, width="stretch")
    st.caption("ACLED events manually classified as farmer-herder violence (blind validation: "
               f"Cohen's kappa = 0.83), January 2019 – {pd.Period(data_end, freq='M').strftime('%B %Y')}.")

# ---------------- About ----------------
with tab_about:
    st.markdown(
        "**Two models, two purposes.**\n\n"
        "- **Operational model** (used in *Current outlook* and *Replay*): uses only what is "
        "really available when a forecast is issued — satellite data up to the previous month "
        "and conflict history delayed by 12 months (ACLED publication delay).\n"
        "- **Research model**: uses conflict history up to the previous month, to answer the "
        "research question *do satellite indicators add information beyond conflict history?*\n\n"
        "All models are trained on 2019–2023 and evaluated once on 2024 – Sept 2025. "
        "PR-AUC is the main metric; a random ranking scores the prevalence.")
    c1, c2 = st.columns(2)
    mode = c1.radio("Model", ["operational", "research"], horizontal=True)
    unit = c2.radio("Horizon", ["quarter", "month"], horizontal=True)
    tag = f"{mode}_{unit}"
    mpath = REPORTS / f"metrics_{tag}.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        st.dataframe(m[["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall_top10",
                        "prevalence"]].round(3), hide_index=True, width="stretch")
    spath = REPORTS / f"selection_{tag}.json"
    if spath.exists():
        s = json.loads(spath.read_text(encoding="utf8"))
        note = (" Fixed by design: the outlook must react to current conditions (satellite "
                "data), and logistic regression is the most stable model with so few cases; "
                "automatic selection proved unstable.") if s["fixed_by_design"] else ""
        st.caption(f"Automatic selection: `{s['auto_selected']}` · used for predictions: "
                   f"`{s['used_for_predictions']}`.{note}")
    for name, title in [(f"pr_curves_{tag}.png", "Precision-recall curves"),
                        (f"shap_summary_{tag}.png", "Risk factors (SHAP, XGBoost)")]:
        if (REPORTS / "figures" / name).exists():
            st.image(str(REPORTS / "figures" / name), caption=title)

st.caption("Ethics: aggregated department-level data only; no ethnic or religious variables. "
           "Decision-support tool — not an automated decision system.")
