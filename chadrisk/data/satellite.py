"""Monthly satellite indicators per department, extracted with Google Earth Engine.

South (per department): crop greenness over cropland and its anomaly (MODIS MOD13Q1 NDVI),
monthly rainfall and its anomaly (CHIRPS), plus static cropland share (ESA WorldCover 2021),
surface-water occurrence (JRC) and population density (WorldPop 2020).
North (herd departure zones, one value for all departments): NDVI and rainfall anomalies.
Anomalies are deviations from the mean of the same calendar month (NDVI 2003-2017,
rainfall 1991-2017).
"""

import pandas as pd

STATIC_COLUMNS = ("south_cropland_frac", "south_water_occ", "south_pop_density")


class SatelliteExtractor:
    """Extracts indicators month by month; appends only the missing months to a file."""

    NDVI = "MODIS/061/MOD13Q1"
    RAIN = "UCSB-CHG/CHIRPS/PENTAD"
    SCALE_NDVI, SCALE_RAIN, SCALE_NORTH = 250, 5566, 1000

    def __init__(self, project: str, departments, north_region):
        """`departments`: GeoDataFrame with dept_id; `north_region`: one shapely geometry."""
        if not project:
            raise ValueError("an Earth Engine project ID is required (EE_PROJECT in .env)")
        import ee  # imported here so that the rest of the package works without Earth Engine
        self.ee = ee
        ee.Initialize(project=project)
        self.departments = self._to_feature_collection(departments)
        self.north = ee.Geometry(north_region.simplify(0.01).__geo_interface__)
        self.ndvi = ee.ImageCollection(self.NDVI).select("NDVI")
        self.rain = ee.ImageCollection(self.RAIN).select("precipitation")
        self.cropland = ee.ImageCollection("ESA/WorldCover/v200").first().eq(40)
        self._ndvi_clim = {m: self._climatology(self._monthly_ndvi, range(2003, 2018), m)
                           for m in range(1, 13)}
        self._rain_clim = {m: self._climatology(self._monthly_rain, range(1991, 2018), m)
                           for m in range(1, 13)}

    def _to_feature_collection(self, departments):
        ee = self.ee
        simplified = departments.geometry.simplify(0.005)
        return ee.FeatureCollection([
            ee.Feature(ee.Geometry(g.__geo_interface__), {"dept_id": d})
            for d, g in zip(departments["dept_id"], simplified)])

    def _monthly_ndvi(self, year, month):
        start = self.ee.Date.fromYMD(year, month, 1)
        return self.ndvi.filterDate(start, start.advance(1, "month")).mean().multiply(0.0001)

    def _monthly_rain(self, year, month):
        start = self.ee.Date.fromYMD(year, month, 1)
        return self.rain.filterDate(start, start.advance(1, "month")).sum()

    def _climatology(self, monthly, years, month):
        return self.ee.ImageCollection([monthly(y, month) for y in years]).mean()

    def last_complete_month(self) -> pd.Period:
        """Last month complete in both products: CHIRPS pentad of the 26th published, and
        MODIS second 16-day composite (starting on or after the 14th) published."""
        def last_start(name):
            c = self.ee.ImageCollection(name).filterDate("2015-01-01", "2100-01-01")
            return pd.Timestamp(self.ee.Date(c.aggregate_max("system:time_start"))
                                .format("YYYY-MM-dd").getInfo())

        def complete(last, min_day):
            p = pd.Period(last, freq="M")
            return p if last.day >= min_day else p - 1

        return min(complete(last_start(self.RAIN), 26), complete(last_start(self.NDVI), 14))

    def static_indicators(self) -> pd.DataFrame:
        ee = self.ee
        image = ee.Image.cat([
            self.cropland.rename("south_cropland_frac"),
            ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
            .divide(100).rename("south_water_occ"),
            ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "TCD"))
            .filter(ee.Filter.eq("year", 2020)).first()
            .divide(0.01).rename("south_pop_density"),  # people per 100 m pixel / 0.01 km2
        ])
        out = image.reduceRegions(self.departments, ee.Reducer.mean(), scale=250).getInfo()
        return pd.DataFrame([f["properties"] for f in out["features"]])

    def month(self, period: pd.Period) -> list[dict]:
        """Indicators of one month for every department."""
        ee, m = self.ee, period.month
        nd = self._monthly_ndvi(period.year, m)
        nd_crop = nd.updateMask(self.cropland)
        rn = self._monthly_rain(period.year, m)
        r_ndvi = ee.Image.cat([
            nd_crop.rename("south_ndvi_crop"),
            nd_crop.subtract(self._ndvi_clim[m]).rename("south_ndvi_crop_anom"),
        ]).reduceRegions(self.departments, ee.Reducer.mean(), scale=self.SCALE_NDVI).getInfo()
        r_rain = ee.Image.cat([
            rn.rename("south_rain_mm"),
            rn.subtract(self._rain_clim[m]).rename("south_rain_anom"),
        ]).reduceRegions(self.departments, ee.Reducer.mean(), scale=self.SCALE_RAIN).getInfo()
        north = ee.Image.cat([
            nd.subtract(self._ndvi_clim[m]).rename("north_ndvi_anom"),
            rn.subtract(self._rain_clim[m]).rename("north_rain_anom"),
        ]).reduceRegion(ee.Reducer.mean(), self.north, scale=self.SCALE_NORTH,
                        maxPixels=1e10).getInfo()
        return [{"ym": str(period), **a["properties"],
                 **{k: v for k, v in b["properties"].items() if k != "dept_id"}, **north}
                for a, b in zip(r_ndvi["features"], r_rain["features"])]

    def update(self, existing: pd.DataFrame | None, end: pd.Period,
               first_default: str = "2017-12", log=print) -> pd.DataFrame:
        """Existing table + the months after its last month, up to `end` (inclusive)."""
        first = (pd.Period(existing["ym"].max(), freq="M") + 1) if existing is not None \
            else pd.Period(first_default, freq="M")
        months = pd.period_range(first, end, freq="M")
        if len(months) == 0:
            return existing
        rows = []
        for period in months:
            rows += self.month(period)
            log(f"{period} done")
        new = pd.DataFrame(rows)
        static = self.static_indicators()
        table = new if existing is None else pd.concat(
            [existing.drop(columns=["south_ndvi_crop_delta", *STATIC_COLUMNS]), new],
            ignore_index=True)
        table = table.merge(static, on="dept_id")  # static values identical for every month
        table = table.sort_values(["dept_id", "ym"])
        table["south_ndvi_crop_delta"] = table.groupby("dept_id")["south_ndvi_crop"].diff()
        return table[table["ym"] >= "2018-01"]
