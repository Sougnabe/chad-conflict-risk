"""Reusable dashboard building blocks: department map and risk ranking."""

import folium
import pandas as pd
from streamlit_folium import st_folium

PRIORITY_N = 10  # departments flagged as priority (see reports/tiers_operational_quarter.csv)
MAP_CENTER = (9.3, 16.8)  # southern Chad


def department_map(frame: pd.DataFrame, value_col: str, legend: str, tooltip_names,
                   fmt: str = "{:.0f}", key: str | None = None):
    """Choropleth of one value per department, with a hover tooltip.

    `frame` needs dept_id, department, province, geom (parsed GeoJSON geometry) and
    `value_col`.
    """
    features = [{
        "type": "Feature",
        "id": r.dept_id,
        "geometry": r.geom,
        "properties": {"department": r.department, "province": r.province,
                       "value": fmt.format(getattr(r, value_col))},
    } for r in frame.itertuples()]
    geojson = {"type": "FeatureCollection", "features": features}
    fmap = folium.Map(location=MAP_CENTER, zoom_start=7, tiles="OpenStreetMap")
    folium.Choropleth(
        geo_data=geojson, data=frame, columns=["dept_id", value_col], key_on="feature.id",
        fill_color="YlOrRd", fill_opacity=0.75, line_opacity=0.4, legend_name=legend,
    ).add_to(fmap)
    folium.GeoJson(
        geojson, style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(["department", "province", "value"],
                                      aliases=[*tooltip_names, legend]),
    ).add_to(fmap)
    return st_folium(fmap, height=520, use_container_width=True, key=key)


def rank_departments(frame: pd.DataFrame, priority_label: str, lower_label: str) -> pd.DataFrame:
    """Adds risk_score (0-100), rank (1 = highest risk) and priority level, sorted by rank."""
    frame = frame.copy()
    frame["risk_score"] = 100 * frame["risk"]
    frame["rank"] = frame["risk"].rank(ascending=False, method="first").astype(int)
    frame["level"] = frame["rank"].map(lambda r: priority_label if r <= PRIORITY_N
                                       else lower_label)
    return frame.sort_values("rank")


def quarter_months(start: str, months: list[str]) -> list[str]:
    """The (up to) three months of the quarter starting at `start`, among available months."""
    return [m for m in months if m >= start][:3]
