"""Shared loading utilities. Data are handled as (sites x hours) matrices for fast vectorised features."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
import os
DATA = Path(os.environ.get("TELECOM_DATA", ROOT.parent / "data"))   # folder holding the parquet, site_master.csv, events_ground_truth.csv
MODELS = ROOT / "models"; SERVING = ROOT / "serving"
MODELS.mkdir(exist_ok=True); SERVING.mkdir(exist_ok=True)
HOLIDAYS = {"2026-09-24"}
CAT_ORDER = {"good": 0}


class Data:
    def __init__(self, data_dir=DATA):
        d = pd.read_parquet(Path(data_dir) / "telecom_synthetic_60d_500sites.parquet")
        d = d.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
        self.df = d
        self.sites = d.site_id.unique()
        self.S = len(self.sites)
        self.T = len(d) // self.S
        ts = pd.DatetimeIndex(d.timestamp.iloc[: self.T])
        self.ts = ts
        # calendar extended 24 h past the end so we can forecast beyond the last observation
        ext = ts.append(pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=24, freq="h"))
        self.ts_ext = ext
        self.hour = ext.hour.values.astype(np.int16)
        self.dow = ext.dayofweek.values.astype(np.int16)
        self.hol = np.array([x.strftime("%Y-%m-%d") in HOLIDAYS for x in ext]).astype(np.int16)
        self.master = pd.read_csv(Path(data_dir) / "site_master.csv").set_index("site_id").loc[self.sites]
        m = self.master
        self.static = pd.DataFrame({
            "tier": m.site_demographic_tier.map({"low": 0, "medium": 1, "high": 2}).values,
            "tech": pd.Categorical(m.deployed_technologies).codes,
            "backhaul": pd.Categorical(m.backhaul_technology).codes,
            "encl": pd.Categorical(m.enclosure_type).codes,
            "cat": m.grid_reliability_category.map({"poor": 0, "low": 1, "medium": 2, "high": 3}).values,
            "lat": m.latitude.values, "lon": m.longitude.values}, index=self.sites)
        self.region = m.region_id.values

    def a(self, col, dtype=np.float32):
        """(S,T) matrix for a column."""
        v = self.df[col]
        if v.dtype == object or str(v.dtype).startswith(("str", "string")):
            raise ValueError(col)
        return v.to_numpy().reshape(self.S, self.T).astype(dtype)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    err = p - y
    return dict(MAE=float(np.abs(err).mean()), wMAPE=float(np.abs(err).sum() / np.abs(y).sum() * 100),
                R2=float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()), n=int(len(y)))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
