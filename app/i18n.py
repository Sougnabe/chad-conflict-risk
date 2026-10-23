"""Texts of the dashboard in English and French, and the Translator that serves them."""

import pandas as pd

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


FINDINGS = {
    "en": """
**Research question.** Do satellite-derived indicators of pasture conditions, rainfall and the
crop calendar improve the prediction of farmer-herder conflict risk in southern Chad, compared
with conflict history alone?

**1. Month by month, conflict is not predictable.** No model clearly beats a random ranking
(prevalence {prev_m:.3f}; history-only model {month_hist:.3f}). Conflicts are too rare and
scattered at that resolution, so the outlook is issued per quarter.

**2. Per quarter, satellite indicators add a modest signal to recent history.**
Rule (last 12 months) {rule:.3f} · history {hist:.3f} · history + southern satellite {hist_sat}
(random = {prev_q:.3f}).

**3. Without recent conflict data — the real situation — satellite indicators are what make
forecasting possible.** Operational model: delayed history only {ops_hist} → with satellite
{ops_sat}.

**4. The risk factors match the hypothesis.** Main satellite factors (direction of the effect):
{drivers}.

**5. The priority list is useful for targeting.** In the 2024–2025 replay, the {n} priority
departments had a conflict in {pri:.0%} of quarters, against {low:.0%} for the others.

**Limits.** Few positive cases (wide confidence intervals); ACLED's 12-month delay; herds are
inferred, not observed; local mediation, which prevents escalation, is not visible; the
satellite sees crops drying, not the harvest itself.
""",
    "fr": """
**Question de recherche.** Les indicateurs satellite de l'état des pâturages, de la pluie et du
calendrier agricole améliorent-ils la prévision du risque de conflit agriculteurs-éleveurs dans
le sud du Tchad, par rapport à l'historique des conflits seul ?

**1. Mois par mois, les conflits ne sont pas prévisibles.** Aucun modèle ne bat nettement un
classement au hasard (prévalence {prev_m:.3f} ; modèle historique seul {month_hist:.3f}). Les
conflits sont trop rares et dispersés à cette échelle : les perspectives sont donc trimestrielles.

**2. Par trimestre, le satellite ajoute un signal modeste à l'historique récent.**
Règle (12 derniers mois) {rule:.3f} · historique {hist:.3f} · historique + satellite du sud
{hist_sat} (hasard = {prev_q:.3f}).

**3. Sans données de conflit récentes — la situation réelle — c'est le satellite qui rend la
prévision possible.** Modèle opérationnel : historique décalé seul {ops_hist} → avec satellite
{ops_sat}.

**4. Les facteurs de risque correspondent à l'hypothèse.** Principaux facteurs satellite (sens
de l'effet) : {drivers}.

**5. La liste des départements prioritaires est utile pour cibler.** Dans le rejeu 2024–2025,
les {n} départements prioritaires ont connu un conflit dans {pri:.0%} des trimestres, contre
{low:.0%} pour les autres.

**Limites.** Peu de cas positifs (intervalles de confiance larges) ; délai de 12 mois d'ACLED ;
troupeaux déduits et non observés ; la médiation locale, qui évite l'escalade, n'est pas
visible ; le satellite voit les cultures sécher, pas la récolte elle-même.
""",
}

DIRECTION = {
    "en": {True: "higher values → higher risk", False: "lower values → higher risk"},
    "fr": {True: "valeurs plus élevées → risque plus élevé",
           False: "valeurs plus faibles → risque plus élevé"},
}


class Translator:
    """Returns texts, feature labels and dates in the selected language ("en" or "fr")."""

    LANGUAGES = ("en", "fr")

    def __init__(self, language: str = "en"):
        if language not in self.LANGUAGES:
            raise ValueError(f"language must be one of {self.LANGUAGES}")
        self.language = language
        self._i = self.LANGUAGES.index(language)

    def __call__(self, key: str, **values) -> str:
        """Text for `key`, with {placeholders} filled from `values`."""
        text = TXT[key][self._i]
        return text.format(**values) if values else text

    def feature(self, name: str) -> str:
        """Readable label of a model feature."""
        return LABELS.get(name, (name, name))[self._i]

    def month(self, ym: str, long: bool = False) -> str:
        p = pd.Period(ym, freq="M")
        names = (MONTHS_LONG if long else MONTHS)[self.language]
        return f"{names[p.month - 1]} {p.year}"

    def quarter(self, ym: str) -> str:
        """Quarter starting at `ym`, e.g. "Oct–Dec 2026" / "oct.–déc. 2026"."""
        p = pd.Period(ym, freq="M")
        return f"{MONTHS[self.language][p.month - 1]}–{self.month(str(p + 2))}"

    def findings(self, **values) -> str:
        """Answers to the research question, filled with the current results."""
        return FINDINGS[self.language].format(**values)

    def direction(self, increases: bool) -> str:
        """How a feature relates to risk, e.g. "lower values → higher risk"."""
        return DIRECTION[self.language][increases]

    def date(self, iso: str) -> str:
        d = pd.Timestamp(iso)
        return f"{d.day} {MONTHS_LONG[self.language][d.month - 1]} {d.year}"
