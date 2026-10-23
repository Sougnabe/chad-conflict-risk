"""
Prototype - Early warning of farmer-herder conflict risk in southern Chad (English / French).

Tabs:
    Current outlook  : risk for the coming quarter (operational model: satellite data up to last
                       month + conflict history with the 12-month ACLED publication delay)
    Replay 2024-2025 : what the operational model would have flagged, quarter by quarter,
                       compared with the conflicts that actually happened
    Findings         : answers to the research question, read from the result files
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
PRIORITY_N = 10  # departments flagged as priority (see reports/tiers_operational_quarter.csv)

MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                 "Dec"],
          "fr": ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.",
                 "oct.", "nov.", "déc."]}
MONTHS_LONG = {"en": ["January", "February", "March", "April", "May", "June", "July", "August",
                      "September", "October", "November", "December"],
               "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                      "septembre", "octobre", "novembre", "décembre"]}

TXT = {
    "title": ("Farmer-Herder Conflict Risk — Southern Chad",
              "Risque de conflit agriculteurs-éleveurs — Sud du Tchad"),
    "tabs": (["Current outlook", "Replay 2024–2025", "Findings", "Observed conflicts",
              "About the models"],
             ["Perspectives actuelles", "Rejeu 2024–2025", "Résultats", "Conflits observés",
              "À propos des modèles"]),
    "caption": ("Research prototype · satellite data up to **{sat}** · conflict data (ACLED, "
                "published with a 12-month delay) up to **{conf}** · department level only.",
                "Prototype de recherche · données satellite jusqu'à **{sat}** · données de "
                "conflit (ACLED, publiées avec 12 mois de retard) jusqu'à **{conf}** · "
                "échelle départementale uniquement."),
    "outlook_title": ("Conflict risk outlook for {q}", "Perspectives de risque de conflit, {q}"),
    "freshness": ("Issued on {issued} · satellite data up to {sat} · conflict data up to {conf}"
                  " · next update expected around {next}.",
                  "Émis le {issued} · données satellite jusqu'à {sat} · données de conflit "
                  "jusqu'à {conf} · prochaine mise à jour prévue vers le {next}."),
    "no_forecast": ("No forecast yet. Run: python scripts/update_outlook.py",
                    "Pas encore de prévision. Lancer : python scripts/update_outlook.py"),
    "levels_title": ("What the priority levels have meant (replay 2024–2025)",
                     "Ce que les niveaux de priorité ont signifié (rejeu 2024–2025)"),
    "level_line": ("**{level}** (ranks {ranks}): a conflict occurred in **{rate:.0%}** of "
                   "department-quarters ({n} of {total}).",
                   "**{level}** (rangs {ranks}) : un conflit a eu lieu dans **{rate:.0%}** des "
                   "trimestres-départements ({n} sur {total})."),
    "priority": ("Priority", "Prioritaire"),
    "lower": ("Lower", "Moins prioritaire"),
    "rank": ("rank", "rang"), "department": ("department", "département"),
    "province": ("province", "province"), "level": ("level", "niveau"),
    "score": ("risk score", "score de risque"),
    "legend_risk": ("Relative risk score", "Score de risque relatif"),
    "why": ("Why is a department at risk?", "Pourquoi un département est-il à risque ?"),
    "dept_select": ("Department", "Département"),
    "factor": ("factor", "facteur"), "effect": ("effect", "effet"),
    "strength": ("strength", "intensité"), "current": ("current value", "valeur actuelle"),
    "raises": ("raises risk", "augmente le risque"),
    "lowers": ("lowers risk", "diminue le risque"),
    "hist_factor": ("Conflict history (latest known, at least 13 months old)",
                    "Historique des conflits (dernier connu, au moins 13 mois)"),
    "hist_value": ("{a:.0f} events in the department, {b:.0f} in neighbours (latest known 12 "
                   "months)", "{a:.0f} événements dans le département, {b:.0f} chez les "
                   "voisins (12 derniers mois connus)"),
    "season": ("Season", "Saison"),
    "why_caption": ("Effects come from the logistic model (standardised coefficient × "
                    "standardised value), for the department's current values.",
                    "Les effets viennent du modèle logistique (coefficient standardisé × valeur "
                    "standardisée), pour les valeurs actuelles du département."),
    "warning": ("Scores rank departments for prioritisation; they are not probabilities. They "
                "must support, not replace, the judgement of people who know the local "
                "situation.",
                "Les scores classent les départements pour prioriser ; ce ne sont pas des "
                "probabilités. Ils doivent appuyer, et non remplacer, le jugement des personnes "
                "qui connaissent la situation locale."),
    "replay_intro": ("If the tool had existed in 2024–2025 — using only the information that "
                     "would really have been available at the time — which departments would "
                     "it have flagged, and where did conflicts actually happen?",
                     "Si l'outil avait existé en 2024–2025 — avec uniquement l'information "
                     "réellement disponible à l'époque — quels départements aurait-il signalés, "
                     "et où les conflits ont-ils réellement eu lieu ?"),
    "quarter": ("Quarter", "Trimestre"),
    "with_conflict": ("Departments with a conflict", "Départements avec un conflit"),
    "flagged": ("…of which flagged as priority", "…dont signalés comme prioritaires"),
    "observed": ("observed events", "événements observés"),
    "replay_summary": ("Over the {nq} test quarters, {share:.0%} of department-quarters with a "
                       "conflict were among the {n} priority departments (a random ranking would "
                       "reach {rand:.0%}).",
                       "Sur les {nq} trimestres de test, {share:.0%} des trimestres-départements "
                       "avec un conflit faisaient partie des {n} départements prioritaires (un "
                       "classement au hasard atteindrait {rand:.0%})."),
    "obs_provinces": ("Provinces", "Provinces"), "obs_period": ("Period", "Période"),
    "obs_def": ("Conflict definition", "Définition du conflit"),
    "obs_defs": (["Strict (FH)", "Broad (FH + ambiguous)"],
                 ["Stricte (FH)", "Large (FH + ambigus)"]),
    "obs_events": ("Conflict events", "Événements de conflit"),
    "obs_deaths": ("Fatalities (strict FH)", "Décès (FH strict)"),
    "obs_dm": ("Department-months with ≥1 event", "Mois-départements avec ≥ 1 événement"),
    "obs_legend": ("Events in period", "Événements sur la période"),
    "obs_caption": ("ACLED events manually classified as farmer-herder violence (blind "
                    "validation: Cohen's kappa = 0.83), January 2019 – {end}.",
                    "Événements ACLED classés manuellement comme violences agriculteurs-éleveurs"
                    " (validation à l'aveugle : kappa de Cohen = 0,83), janvier 2019 – {end}."),
    "events": ("events", "événements"), "fatalities": ("fatalities", "décès"),
    "about": ("**Two models, two purposes.**\n\n"
              "- **Operational model** (*Current outlook*, *Replay*): uses only what is really "
              "available when a forecast is issued — satellite data up to the previous month and "
              "conflict history delayed by 12 months (ACLED publication delay).\n"
              "- **Research model**: conflict history up to the previous month, to answer the "
              "research question *do satellite indicators add information beyond conflict "
              "history?*\n\n"
              "All models are trained on 2019–2023 and evaluated once on 2024 – Sept 2025. "
              "PR-AUC is the main metric; a random ranking scores the prevalence.",
              "**Deux modèles, deux usages.**\n\n"
              "- **Modèle opérationnel** (*Perspectives*, *Rejeu*) : n'utilise que ce qui est "
              "réellement disponible au moment de la prévision — données satellite jusqu'au mois "
              "précédent et historique des conflits décalé de 12 mois (délai de publication "
              "d'ACLED).\n"
              "- **Modèle de recherche** : historique des conflits jusqu'au mois précédent, pour "
              "répondre à la question de recherche *les indicateurs satellite apportent-ils une "
              "information au-delà de l'historique des conflits ?*\n\n"
              "Tous les modèles sont entraînés sur 2019–2023 et évalués une seule fois sur "
              "2024 – sept. 2025. Le PR-AUC est la métrique principale ; un classement au hasard "
              "obtient la prévalence."),
    "model_radio": ("Model", "Modèle"), "horizon": ("Horizon", "Horizon"),
    "modes": (["operational", "research"], ["opérationnel", "recherche"]),
    "units": (["quarter", "month"], ["trimestre", "mois"]),
    "selection": ("Automatic selection: `{auto}` · used for predictions: `{used}`.",
                  "Sélection automatique : `{auto}` · utilisé pour les prédictions : `{used}`."),
    "fixed": (" Fixed by design: the outlook must react to current conditions (satellite data), "
              "and logistic regression is the most stable model with so few cases; automatic "
              "selection proved unstable.",
              " Fixé par conception : les perspectives doivent réagir aux conditions actuelles "
              "(satellite), et la régression logistique est le modèle le plus stable avec si peu "
              "de cas ; la sélection automatique s'est révélée instable."),
    "pr_fig": ("Precision-recall curves", "Courbes précision-rappel"),
    "shap_fig": ("Risk factors (SHAP, XGBoost)", "Facteurs de risque (SHAP, XGBoost)"),
    "ethics": ("Ethics: aggregated department-level data only; no ethnic or religious variables."
               " Decision-support tool — not an automated decision system.",
               "Éthique : données agrégées par département uniquement ; aucune variable "
               "ethnique ou religieuse. Outil d'aide à la décision — pas un système de décision "
               "automatique."),
}

LABELS = {
    "south_ndvi_crop_anom": ("Crop greenness vs normal (south)",
                             "Verdure des cultures par rapport à la normale (sud)"),
    "south_ndvi_crop": ("Crop greenness (south)", "Verdure des cultures (sud)"),
    "south_ndvi_crop_delta": ("Change in crop greenness since previous month",
                              "Variation de la verdure des cultures sur un mois"),
    "south_rain_mm": ("Rainfall last month (south)", "Pluie du mois précédent (sud)"),
    "south_rain_anom": ("Rainfall vs normal (south)", "Pluie par rapport à la normale (sud)"),
    "south_cropland_frac": ("Share of cropland", "Part des terres cultivées"),
    "south_water_occ": ("Surface water", "Eau de surface"),
    "south_pop_density": ("Population density", "Densité de population"),
    "north_ndvi_anom": ("Pasture greenness vs normal (north, herd origin)",
                        "Verdure des pâturages par rapport à la normale (nord, origine des "
                        "troupeaux)"),
    "north_rain_anom": ("Rainfall vs normal (north, herd origin)",
                        "Pluie par rapport à la normale (nord, origine des troupeaux)"),
}

st.set_page_config(page_title="Farmer-Herder Conflict Risk - Southern Chad", layout="wide")
LANG = "fr" if st.sidebar.radio("Language / Langue", ["English", "Français"]) == "Français" \
    else "en"


def t(key, **kw):
    text = TXT[key][0 if LANG == "en" else 1]
    return text.format(**kw) if kw else text


def label(feature):
    return LABELS.get(feature, (feature, feature))[0 if LANG == "en" else 1]


def month_name(ym, long=False):
    p = pd.Period(ym, freq="M")
    names = MONTHS_LONG if long else MONTHS
    return f"{names[LANG][p.month - 1]} {p.year}"


def quarter_label(ym):
    p = pd.Period(ym, freq="M")
    return f"{MONTHS[LANG][p.month - 1]}–{month_name(p + 2)}"


def date_label(iso):
    d = pd.Timestamp(iso)
    return f"{d.day} {MONTHS_LONG[LANG][d.month - 1]} {d.year}"


@st.cache_data
def load():
    with sqlite3.connect(DB) as con:
        depts = pd.read_sql("SELECT * FROM departments", con)
        panel = pd.read_sql("SELECT * FROM panel", con)
        preds = pd.read_sql("SELECT * FROM predictions", con)
        expl = pd.read_sql("SELECT * FROM features WHERE name LIKE ?", con,
                           params=(f"%|{OPS}|%",))
    return depts, panel, preds, expl


def read_json(name):
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def read_csv(name):
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


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
                                      aliases=[t("dept_select"), "Province", legend]),
    ).add_to(fmap)
    return st_folium(fmap, height=520, use_container_width=True, key=key)


def ranked(frame):
    frame = frame.copy()
    frame["risk_score"] = 100 * frame["risk"]
    frame["rank"] = frame["risk"].rank(ascending=False, method="first").astype(int)
    frame["level"] = frame["rank"].map(lambda r: t("priority") if r <= PRIORITY_N
                                       else t("lower"))
    return frame.sort_values("rank")


def quarter_window(q, months):
    return [m for m in months if m >= q][:3]


depts, panel, preds, expl = load()
months = sorted(panel["ym"].unique())
data_end = months[-1]
ops = preds[preds["model"].str.contains(OPS)]
fc = ops[ops["model"].str.endswith("_forecast")]
replay = ops[ops["model"].str.endswith("_test")]
meta = read_json(f"outlook_{OPS}.json")

st.title(t("title"))
if meta:
    st.caption(t("caption", sat=month_name(meta["satellite_data_end"], long=True),
                 conf=month_name(data_end, long=True)))

tab_now, tab_replay, tab_find, tab_obs, tab_about = st.tabs(t("tabs"))

# ---------------- Current outlook ----------------
with tab_now:
    if fc.empty:
        st.info(t("no_forecast"))
    else:
        fc_ym = fc["ym"].iloc[0]
        st.subheader(t("outlook_title", q=quarter_label(fc_ym)))
        if meta:
            st.caption(t("freshness", issued=date_label(meta["issued_on"]),
                         sat=month_name(meta["satellite_data_end"], long=True),
                         conf=month_name(data_end, long=True),
                         next=date_label(meta["next_update_expected"])))
        frame = ranked(depts.merge(fc[["dept_id", "risk"]], on="dept_id"))
        left, right = st.columns([3, 2])
        with left:
            department_map(frame, "risk_score", t("legend_risk"), "{:.0f}", key="now")
        with right:
            st.dataframe(
                frame[["rank", "department", "province", "level", "risk_score"]]
                .rename(columns={"rank": t("rank"), "department": t("department"),
                                 "province": t("province"), "level": t("level"),
                                 "risk_score": t("score")}).round(0),
                hide_index=True, width="stretch", height=430)

        tiers = read_csv(f"tiers_{OPS}.csv")
        if tiers is not None:
            st.markdown(f"#### {t('levels_title')}")
            for name, key in [("Priority", "priority"), ("Lower", "lower")]:
                row = tiers[tiers["tier"] == name].iloc[0]
                st.markdown("- " + t("level_line", level=t(key), ranks=row["ranks"],
                                     rate=row["conflict_rate"], n=int(row["with_conflict"]),
                                     total=int(row["periods"])))

        st.markdown(f"#### {t('why')}")
        sel = st.selectbox(t("dept_select"), frame["department"], index=0)
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
        rows = [(t("hist_factor"), hist.sum(),
                 t("hist_value", a=values.get("o_fh_12m", 0), b=values.get("o_nb_fh_12m", 0))),
                (t("season"), season.sum(), quarter_label(fc_ym))]
        rows += [(label(f), v, f"{values[f]:.3f}") for f, v in sat.items()]
        table = pd.DataFrame(rows, columns=["factor", "contribution", "current"])
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
        table["effect"] = table["contribution"].map(lambda v: t("raises") if v > 0
                                                    else t("lowers"))
        table["strength"] = table["contribution"].abs().round(2)
        st.dataframe(table[["factor", "effect", "strength", "current"]].head(7)
                     .rename(columns={"factor": t("factor"), "effect": t("effect"),
                                      "strength": t("strength"), "current": t("current")}),
                     hide_index=True, width="stretch")
        st.caption(t("why_caption"))
        st.warning(t("warning"))

# ---------------- Replay ----------------
with tab_replay:
    if not replay.empty:
        st.markdown(t("replay_intro"))
        qs = sorted(replay["ym"].unique())
        q = st.select_slider(t("quarter"), options=qs, value=qs[-1], format_func=quarter_label)
        frame = ranked(depts.merge(replay[replay["ym"] == q][["dept_id", "risk"]],
                                   on="dept_id"))
        obs = panel[panel["ym"].isin(quarter_window(q, months))].groupby("dept_id")["n_fh"].sum()
        frame["observed"] = frame["dept_id"].map(obs).fillna(0).astype(int)
        hit = frame[frame["observed"] > 0]
        c1, c2 = st.columns(2)
        c1.metric(t("with_conflict"), len(hit))
        c2.metric(t("flagged"), int((hit["rank"] <= PRIORITY_N).sum()))

        left, right = st.columns([3, 2])
        with left:
            department_map(frame, "risk_score", t("legend_risk"), "{:.0f}", key="replay")
        with right:
            st.dataframe(frame[["rank", "department", "province", "level", "observed"]]
                         .rename(columns={"rank": t("rank"), "department": t("department"),
                                          "province": t("province"), "level": t("level"),
                                          "observed": t("observed")}),
                         hide_index=True, width="stretch", height=480)

        allq = replay[["dept_id", "ym", "risk"]].copy()
        allq["rank"] = allq.groupby("ym")["risk"].rank(ascending=False, method="first")
        allq["events"] = [panel[(panel["dept_id"] == d)
                                & panel["ym"].isin(quarter_window(y, months))]["n_fh"].sum()
                          for d, y in zip(allq["dept_id"], allq["ym"])]
        pos = allq[allq["events"] > 0]
        st.info(t("replay_summary", nq=len(qs), share=(pos["rank"] <= PRIORITY_N).mean(),
                  n=PRIORITY_N, rand=PRIORITY_N / 26))

# ---------------- Findings ----------------
with tab_find:
    rq = read_csv("metrics_research_quarter.csv")
    rm = read_csv("metrics_research_month.csv")
    oq = read_csv(f"metrics_{OPS}.csv")
    tiers = read_csv(f"tiers_{OPS}.csv")
    coefs = read_csv(f"coefficients_{OPS}.csv")
    if any(x is None for x in (rq, rm, oq, tiers, coefs)):
        st.info("Run the training scripts first (README, step 8).")
    else:
        def ap(m, model):
            return float(m.loc[m["model"] == model, "pr_auc"].iloc[0])

        def ci(m, model):
            r = m[m["model"] == model].iloc[0]
            return f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.2f}–{r['pr_auc_hi']:.2f}]"

        prev_q, prev_m = float(rq["prevalence"][0]), float(rm["prevalence"][0])
        pri = tiers[tiers["tier"] == "Priority"].iloc[0]
        low = tiers[tiers["tier"] == "Lower"].iloc[0]
        sat_coefs = coefs[coefs.iloc[:, 0].str.startswith(("south_", "north_"))].head(3)
        direction = {("en", True): "higher values → higher risk",
                     ("en", False): "lower values → higher risk",
                     ("fr", True): "valeurs plus élevées → risque plus élevé",
                     ("fr", False): "valeurs plus faibles → risque plus élevé"}
        drivers = "; ".join(
            f"{label(f)} ({direction[(LANG, c > 0)]})"
            for f, c in zip(sat_coefs.iloc[:, 0], sat_coefs.iloc[:, 1]))
        if LANG == "en":
            st.markdown(f"""
**Research question.** Do satellite-derived indicators of pasture conditions, rainfall and the
crop calendar improve the prediction of farmer-herder conflict risk in southern Chad, compared
with conflict history alone?

**1. Month by month, conflict is not predictable.** No model clearly beats a random ranking
(prevalence {prev_m:.3f}; history-only model {ap(rm, 'logreg_A'):.3f}). Conflicts are too rare
and scattered at that resolution, so the outlook is issued per quarter.

**2. Per quarter, satellite indicators add a modest signal to recent history.**
Rule (last 12 months) {ap(rq, 'rule_history_12m'):.3f} · history {ap(rq, 'logreg_A'):.3f} ·
history + southern satellite {ci(rq, 'logreg_B')} (random = {prev_q:.3f}).

**3. Without recent conflict data — the real situation — satellite indicators are what make
forecasting possible.** Operational model: delayed history only {ci(oq, 'logreg_OA')} →
with satellite {ci(oq, 'logreg_OB')}.

**4. The risk factors match the hypothesis.** Main satellite factors (direction of the effect):
{drivers}.

**5. The priority list is useful for targeting.** In the 2024–2025 replay, the
{PRIORITY_N} priority departments had a conflict in {pri['conflict_rate']:.0%} of quarters,
against {low['conflict_rate']:.0%} for the others.

**Limits.** Few positive cases (wide confidence intervals); ACLED's 12-month delay; herds are
inferred, not observed; local mediation, which prevents escalation, is not visible; the
satellite sees crops drying, not the harvest itself.
""")
        else:
            st.markdown(f"""
**Question de recherche.** Les indicateurs satellite de l'état des pâturages, de la pluie et du
calendrier agricole améliorent-ils la prévision du risque de conflit agriculteurs-éleveurs dans
le sud du Tchad, par rapport à l'historique des conflits seul ?

**1. Mois par mois, les conflits ne sont pas prévisibles.** Aucun modèle ne bat nettement un
classement au hasard (prévalence {prev_m:.3f} ; modèle historique seul {ap(rm, 'logreg_A'):.3f}).
Les conflits sont trop rares et dispersés à cette échelle : les perspectives sont donc
trimestrielles.

**2. Par trimestre, le satellite ajoute un signal modeste à l'historique récent.**
Règle (12 derniers mois) {ap(rq, 'rule_history_12m'):.3f} · historique {ap(rq, 'logreg_A'):.3f}
· historique + satellite du sud {ci(rq, 'logreg_B')} (hasard = {prev_q:.3f}).

**3. Sans données de conflit récentes — la situation réelle — c'est le satellite qui rend la
prévision possible.** Modèle opérationnel : historique décalé seul {ci(oq, 'logreg_OA')} →
avec satellite {ci(oq, 'logreg_OB')}.

**4. Les facteurs de risque correspondent à l'hypothèse.** Principaux facteurs satellite (sens
de l'effet) : {drivers}.

**5. La liste des départements prioritaires est utile pour cibler.** Dans le rejeu 2024–2025,
les {PRIORITY_N} départements prioritaires ont connu un conflit dans
{pri['conflict_rate']:.0%} des trimestres, contre {low['conflict_rate']:.0%} pour les autres.

**Limites.** Peu de cas positifs (intervalles de confiance larges) ; délai de 12 mois d'ACLED ;
troupeaux déduits et non observés ; la médiation locale, qui évite l'escalade, n'est pas
visible ; le satellite voit les cultures sécher, pas la récolte elle-même.
""")

# ---------------- Observed conflicts ----------------
with tab_obs:
    c1, c2, c3 = st.columns(3)
    provinces = sorted(depts["province"].unique())
    sel_prov = c1.multiselect(t("obs_provinces"), provinces, default=provinces)
    start, end = c2.select_slider(t("obs_period"), options=months, value=(months[0], months[-1]),
                                  format_func=month_name)
    defs = t("obs_defs")
    target = c3.radio(t("obs_def"), defs)
    col = "n_fh" if target == defs[0] else "n_broad"

    mask = panel["ym"].between(start, end) & panel["dept_id"].isin(
        depts.loc[depts["province"].isin(sel_prov), "dept_id"])
    sub = panel[mask]
    by_dept = sub.groupby("dept_id")[[col, "fatalities_fh"]].sum().reset_index()
    by_dept = depts.merge(by_dept, on="dept_id", how="left").fillna({col: 0, "fatalities_fh": 0})
    by_dept = by_dept[by_dept["province"].isin(sel_prov)]

    m1, m2, m3 = st.columns(3)
    m1.metric(t("obs_events"), int(sub[col].sum()))
    m2.metric(t("obs_deaths"), int(sub["fatalities_fh"].sum()))
    m3.metric(t("obs_dm"), int((sub[col] > 0).sum()))

    left, right = st.columns([3, 2])
    with left:
        department_map(by_dept, col, t("obs_legend"), key="obs")
    with right:
        st.bar_chart(sub.groupby("ym")[col].sum())
        st.dataframe(
            by_dept.sort_values(col, ascending=False)[["department", "province", col,
                                                        "fatalities_fh"]]
            .rename(columns={"department": t("department"), "province": t("province"),
                             col: t("events"), "fatalities_fh": t("fatalities")}),
            hide_index=True, width="stretch")
    st.caption(t("obs_caption", end=month_name(data_end, long=True)))

# ---------------- About ----------------
with tab_about:
    st.markdown(t("about"))
    c1, c2 = st.columns(2)
    mode_names, unit_names = t("modes"), t("units")
    mode = ["operational", "research"][mode_names.index(
        c1.radio(t("model_radio"), mode_names, horizontal=True))]
    unit = ["quarter", "month"][unit_names.index(
        c2.radio(t("horizon"), unit_names, horizontal=True))]
    tag = f"{mode}_{unit}"
    m = read_csv(f"metrics_{tag}.csv")
    if m is not None:
        st.dataframe(m[["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "recall_top10",
                        "prevalence"]].round(3), hide_index=True, width="stretch")
    s = read_json(f"selection_{tag}.json")
    if s:
        st.caption(t("selection", auto=s["auto_selected"], used=s["used_for_predictions"])
                   + (t("fixed") if s["fixed_by_design"] else ""))
    for name, title in [(f"pr_curves_{tag}.png", t("pr_fig")),
                        (f"shap_summary_{tag}.png", t("shap_fig"))]:
        if (REPORTS / "figures" / name).exists():
            st.image(str(REPORTS / "figures" / name), caption=title)

st.caption(t("ethics"))
