"""Administrative geography: provinces, departments, event assignment and neighbourhoods."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from chadrisk.config import SOUTH_PROVINCES


class Geography:
    """Province (ADM1) and department (ADM2) boundaries of Chad (geoBoundaries, 2019)."""

    def __init__(self, provinces: gpd.GeoDataFrame, departments: gpd.GeoDataFrame):
        self.provinces = provinces.rename(columns={"shapeName": "province"})[
            ["province", "geometry"]]
        departments = departments.rename(columns={"shapeName": "department",
                                                  "shapeID": "dept_id"})
        self.departments = departments[["department", "dept_id", "geometry"]].copy()
        self.departments["province"] = self._province_of(self.departments)

    @classmethod
    def from_files(cls, adm1_path: Path, adm2_path: Path) -> "Geography":
        return cls(gpd.read_file(adm1_path), gpd.read_file(adm2_path))

    def _province_of(self, departments: gpd.GeoDataFrame) -> pd.Series:
        """Province containing each department (tested on an interior point)."""
        points = departments.copy()
        points["geometry"] = departments.representative_point()
        return gpd.sjoin(points, self.provinces, predicate="within")["province"]

    def region(self, provinces=SOUTH_PROVINCES) -> gpd.GeoDataFrame:
        """Departments of the given provinces."""
        region = self.departments[self.departments["province"].isin(provinces)]
        missing = set(provinces) - set(region["province"])
        if missing:
            raise ValueError(f"provinces not found in the boundaries: {sorted(missing)}")
        return region.reset_index(drop=True)

    def assign_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Adds dept_id to each event by spatial join on its coordinates.

        ACLED admin2 names and geoBoundaries names do not match, so coordinates are used.
        """
        points = gpd.GeoDataFrame(
            events, geometry=gpd.points_from_xy(events["longitude"], events["latitude"]),
            crs="EPSG:4326")
        joined = gpd.sjoin(points, self.departments[["dept_id", "geometry"]],
                           predicate="within", how="left")
        return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))

    @staticmethod
    def neighbours(region: gpd.GeoDataFrame) -> dict[str, list[str]]:
        """Adjacency graph of departments sharing a border.

        Uses the spatial index (R-tree) to test only candidate pairs whose bounding boxes
        intersect, instead of comparing every pair of polygons.
        """
        geoms = region.set_index("dept_id")["geometry"]
        index = region.sindex
        graph = {}
        for dept_id, geom in geoms.items():
            candidates = index.query(geom, predicate="touches")
            graph[dept_id] = [region["dept_id"].iloc[i] for i in sorted(candidates)
                              if region["dept_id"].iloc[i] != dept_id]
        return graph

    @staticmethod
    def same_province(region: gpd.GeoDataFrame) -> dict[str, list[str]]:
        """Other departments of the same province, for each department."""
        prov = region.set_index("dept_id")["province"]
        return {d: list(prov.index[(prov == prov[d]) & (prov.index != d)]) for d in prov.index}
