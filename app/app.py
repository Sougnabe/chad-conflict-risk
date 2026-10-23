"""
Early warning of farmer-herder conflict risk in southern Chad — Streamlit dashboard.

Tabs: Current outlook · Replay 2024-2025 · Findings · Observed conflicts · About the models.
Data come from the SQLite database through chadrisk.storage.RiskRepository and from the result
files in reports/. Texts are in app/i18n.py (English / French), building blocks in
app/components.py.

Run: streamlit run app/app.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):  # package and app modules
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chadrisk.storage import DatabaseNotFoundError, RiskRepository  # noqa: E402
from components import (PRIORITY_N, department_map, quarter_months,  # noqa: E402
                        rank_departments)
from i18n import Translator  # noqa: E402

REPORTS = ROOT / "reports"
DATABASE = Path(os.environ.get("CHADRISK_DB", ROOT / "data" / "app.db"))  # overridable
OPS = "operational_quarter"


# ---------------- data access ----------------
@st.cache_data
def load_data(database: str):
    """Loaded once and cached (per database path): later interactions reuse parsed data."""
    repo = RiskRepository(Path(database))
    depts = repo.departments()
    # parse the GeoJSON outlines once, not at every map rendering
    depts["geom"] = [json.loads(g)["features"][0]["geometry"] for g in depts["geometry"]]
    return (depts.drop(columns="geometry"), repo.panel(), repo.predictions(),
            repo.explanations(OPS))


def read_report(name: str):
    """A CSV or JSON result file from reports/, or None if it does not exist yet."""
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text("utf8")) if path.suffix == ".json" else pd.read_csv(path)


def quarterly_events(panel: pd.DataFrame) -> pd.Series:
    """Conflict events per (department, quarter start), computed once with a groupby."""
    period = pd.PeriodIndex(panel["ym"], freq="M")
    start = (period - (period.month - 1) % 3).astype(str)
    return panel.groupby([panel["dept_id"], start])["n_fh"].sum()


# ---------------- tabs ----------------
def tab_outlook(t, depts, forecast, explanations, meta, data_end):
    if forecast.empty:
        st.info(t("no_forecast"))
        return
    fc_ym = forecast["ym"].iloc[0]
    st.subheader(t("outlook_title", q=t.quarter(fc_ym)))
    if meta:
        st.caption(t("freshness", issued=t.date(meta["issued_on"]),
                     sat=t.month(meta["satellite_data_end"], long=True),
                     conf=t.month(data_end, long=True),
                     next=t.date(meta["next_update_expected"])))
    frame = rank_departments(depts.merge(forecast[["dept_id", "risk"]], on="dept_id"),
                             t("priority"), t("lower"))
    left, right = st.columns([3, 2])
    with left:
        department_map(frame, "risk_score", t("legend_risk"),
                       (t("dept_select"), "Province"), key="now")
    with right:
        st.dataframe(frame[["rank", "department", "province", "level", "risk_score"]]
                     .rename(columns={"rank": t("rank"), "department": t("department"),
                                      "province": t("province"), "level": t("level"),
                                      "risk_score": t("score")}).round(0),
                     hide_index=True, width="stretch", height=430)

    tiers = read_report(f"tiers_{OPS}.csv")
    if tiers is not None:
        st.markdown(f"#### {t('levels_title')}")
        for name, key in [("Priority", "priority"), ("Lower", "lower")]:
            row = tiers[tiers["tier"] == name].iloc[0]
            st.markdown("- " + t("level_line", level=t(key), ranks=row["ranks"],
                                 rate=row["conflict_rate"], n=int(row["with_conflict"]),
                                 total=int(row["periods"])))

    st.markdown(f"#### {t('why')}")
    selected = st.selectbox(t("dept_select"), frame["department"], index=0)
    dept_id = frame.loc[frame["department"] == selected, "dept_id"].item()
    st.dataframe(explanation_table(t, explanations, dept_id, fc_ym), hide_index=True,
                 width="stretch")
    st.caption(t("why_caption"))
    st.warning(t("warning"))


def explanation_table(t, explanations, dept_id, fc_ym) -> pd.DataFrame:
    """Factors behind one department's score, strongest first.

    Correlated conflict-history variables (and the two seasonal terms) are summed, so that
    their combined effect is shown instead of individual terms that offset each other.
    """
    e = explanations[explanations["dept_id"] == dept_id].copy()
    e["kind"] = e["name"].str.split("|").str[0]
    e["feature"] = e["name"].str.split("|").str[2]
    contrib = e[e["kind"] == "contrib"].set_index("feature")["value"]
    values = e[e["kind"] == "value"].set_index("feature")["value"]
    rows = [(t("hist_factor"), contrib[contrib.index.str.startswith("o_")].sum(),
             t("hist_value", a=values.get("o_fh_12m", 0), b=values.get("o_nb_fh_12m", 0))),
            (t("season"), contrib[contrib.index.str.startswith("h_month")].sum(),
             t.quarter(fc_ym))]
    satellite = contrib[contrib.index.str.startswith(("south_", "north_"))]
    rows += [(t.feature(f), v, f"{values[f]:.3f}") for f, v in satellite.items()]
    table = pd.DataFrame(rows, columns=["factor", "contribution", "current"])
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
    table["effect"] = table["contribution"].map(lambda v: t("raises") if v > 0 else t("lowers"))
    table["strength"] = table["contribution"].abs().round(2)
    return (table[["factor", "effect", "strength", "current"]].head(7)
            .rename(columns={"factor": t("factor"), "effect": t("effect"),
                             "strength": t("strength"), "current": t("current")}))


def tab_replay(t, depts, panel, replay):
    if replay.empty:
        return
    st.markdown(t("replay_intro"))
    quarters = sorted(replay["ym"].unique())
    q = st.select_slider(t("quarter"), options=quarters, value=quarters[-1],
                         format_func=t.quarter)
    frame = rank_departments(depts.merge(replay[replay["ym"] == q][["dept_id", "risk"]],
                                         on="dept_id"), t("priority"), t("lower"))
    months = sorted(panel["ym"].unique())
    observed = panel[panel["ym"].isin(quarter_months(q, months))].groupby("dept_id")["n_fh"].sum()
    frame["observed"] = frame["dept_id"].map(observed).fillna(0).astype(int)
    hit = frame[frame["observed"] > 0]
    c1, c2 = st.columns(2)
    c1.metric(t("with_conflict"), len(hit))
    c2.metric(t("flagged"), int((hit["rank"] <= PRIORITY_N).sum()))

    left, right = st.columns([3, 2])
    with left:
        department_map(frame, "risk_score", t("legend_risk"),
                       (t("dept_select"), "Province"), key="replay")
    with right:
        st.dataframe(frame[["rank", "department", "province", "level", "observed"]]
                     .rename(columns={"rank": t("rank"), "department": t("department"),
                                      "province": t("province"), "level": t("level"),
                                      "observed": t("observed")}),
                     hide_index=True, width="stretch", height=480)

    allq = replay[["dept_id", "ym", "risk"]].copy()
    allq["rank"] = allq.groupby("ym")["risk"].rank(ascending=False, method="first")
    events = quarterly_events(panel)
    allq["events"] = [events.get((d, y), 0) for d, y in zip(allq["dept_id"], allq["ym"])]
    positive = allq[allq["events"] > 0]
    st.info(t("replay_summary", nq=len(quarters), share=(positive["rank"] <= PRIORITY_N).mean(),
              n=PRIORITY_N, rand=PRIORITY_N / len(depts)))


def tab_findings(t):
    rq, rm = read_report("metrics_research_quarter.csv"), read_report("metrics_research_month.csv")
    oq, tiers = read_report(f"metrics_{OPS}.csv"), read_report(f"tiers_{OPS}.csv")
    coefs = read_report(f"coefficients_{OPS}.csv")
    if any(x is None for x in (rq, rm, oq, tiers, coefs)):
        st.info("Run the training scripts first (README, step 8).")
        return

    def score(table, model):
        return float(table.loc[table["model"] == model, "pr_auc"].iloc[0])

    def with_ci(table, model):
        r = table[table["model"] == model].iloc[0]
        return f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.2f}–{r['pr_auc_hi']:.2f}]"

    satellite = coefs[coefs.iloc[:, 0].str.startswith(("south_", "north_"))].head(3)
    drivers = "; ".join(f"{t.feature(f)} ({t.direction(c > 0)})"
                        for f, c in zip(satellite.iloc[:, 0], satellite.iloc[:, 1]))
    st.markdown(t.findings(
        prev_m=float(rm["prevalence"][0]), month_hist=score(rm, "logreg_A"),
        rule=score(rq, "rule_history_12m"), hist=score(rq, "logreg_A"),
        hist_sat=with_ci(rq, "logreg_B"), prev_q=float(rq["prevalence"][0]),
        ops_hist=with_ci(oq, "logreg_OA"), ops_sat=with_ci(oq, "logreg_OB"),
        drivers=drivers, n=PRIORITY_N,
        pri=tiers.loc[tiers["tier"] == "Priority", "conflict_rate"].iloc[0],
        low=tiers.loc[tiers["tier"] == "Lower", "conflict_rate"].iloc[0]))


def tab_observed(t, depts, panel, data_end):
    months = sorted(panel["ym"].unique())
    c1, c2, c3 = st.columns(3)
    provinces = sorted(depts["province"].unique())
    selected = c1.multiselect(t("obs_provinces"), provinces, default=provinces)
    start, end = c2.select_slider(t("obs_period"), options=months,
                                  value=(months[0], months[-1]), format_func=t.month)
    definitions = t("obs_defs")
    col = "n_fh" if c3.radio(t("obs_def"), definitions) == definitions[0] else "n_broad"

    in_scope = depts.loc[depts["province"].isin(selected), "dept_id"]
    sub = panel[panel["ym"].between(start, end) & panel["dept_id"].isin(in_scope)]
    by_dept = sub.groupby("dept_id")[[col, "fatalities_fh"]].sum().reset_index()
    by_dept = depts.merge(by_dept, on="dept_id", how="left").fillna({col: 0, "fatalities_fh": 0})
    by_dept = by_dept[by_dept["province"].isin(selected)]

    m1, m2, m3 = st.columns(3)
    m1.metric(t("obs_events"), int(sub[col].sum()))
    m2.metric(t("obs_deaths"), int(sub["fatalities_fh"].sum()))
    m3.metric(t("obs_dm"), int((sub[col] > 0).sum()))
    if by_dept.empty:  # no province selected
        return
    left, right = st.columns([3, 2])
    with left:
        department_map(by_dept, col, t("obs_legend"), (t("dept_select"), "Province"),
                       key="obs")
    with right:
        st.bar_chart(sub.groupby("ym")[col].sum())
        st.dataframe(by_dept.sort_values(col, ascending=False)[
            ["department", "province", col, "fatalities_fh"]]
            .rename(columns={"department": t("department"), "province": t("province"),
                             col: t("events"), "fatalities_fh": t("fatalities")}),
            hide_index=True, width="stretch")
    st.caption(t("obs_caption", end=t.month(data_end, long=True)))


def tab_about(t):
    st.markdown(t("about"))
    c1, c2 = st.columns(2)
    modes, units = t("modes"), t("units")
    mode = ["operational", "research"][modes.index(c1.radio(t("model_radio"), modes,
                                                             horizontal=True))]
    unit = ["quarter", "month"][units.index(c2.radio(t("horizon"), units, horizontal=True))]
    tag = f"{mode}_{unit}"
    metrics = read_report(f"metrics_{tag}.csv")
    if metrics is not None:
        st.dataframe(metrics[["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc",
                              "recall_top10", "prevalence"]].round(3),
                     hide_index=True, width="stretch")
    selection = read_report(f"selection_{tag}.json")
    if selection:
        st.caption(t("selection", auto=selection["auto_selected"],
                     used=selection["used_for_predictions"])
                   + (t("fixed") if selection["fixed_by_design"] else ""))
    for name, title in [(f"pr_curves_{tag}.png", t("pr_fig")),
                        (f"shap_summary_{tag}.png", t("shap_fig"))]:
        if (REPORTS / "figures" / name).exists():
            st.image(str(REPORTS / "figures" / name), caption=title)


# ---------------- page ----------------
def main():
    st.set_page_config(page_title="Farmer-Herder Conflict Risk - Southern Chad", layout="wide")
    # In the page, not the sidebar: Streamlit hides the sidebar on phones (found by the
    # mobile browser tests), and field users often work on mobile.
    title, switch = st.columns([5, 2])
    language = switch.radio("Language / Langue", ["English", "Français"], horizontal=True,
                            key="language")
    t = Translator("fr" if language == "Français" else "en")
    title.title(t("title"))
    try:
        depts, panel, predictions, explanations = load_data(str(DATABASE))
    except DatabaseNotFoundError as error:
        st.error(f"{error}")
        st.stop()
    if panel.empty or depts.empty:
        st.warning("The database is empty: run scripts/init_db.py.")
        st.stop()

    data_end = panel["ym"].max()
    ops = predictions[predictions["model"].str.contains(OPS)]
    forecast = ops[ops["model"].str.endswith("_forecast")]
    replay = ops[ops["model"].str.endswith("_test")]
    meta = read_report(f"outlook_{OPS}.json")
    if meta:
        st.caption(t("caption", sat=t.month(meta["satellite_data_end"], long=True),
                     conf=t.month(data_end, long=True)))

    # Only the selected page is computed (Streamlit tabs would compute all five, and their
    # three maps, at every interaction).
    pages = t("tabs")
    # the page is stored as a number, so it survives a change of language
    page = st.radio("page", range(len(pages)), format_func=lambda i: pages[i],
                    horizontal=True, key="page", label_visibility="collapsed")
    st.divider()
    if page == 0:
        tab_outlook(t, depts, forecast, explanations, meta, data_end)
    elif page == 1:
        tab_replay(t, depts, panel, replay)
    elif page == 2:
        tab_findings(t)
    elif page == 3:
        tab_observed(t, depts, panel, data_end)
    else:
        tab_about(t)
    st.caption(t("ethics"))


main()
