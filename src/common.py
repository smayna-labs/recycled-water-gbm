"""common.py: shared paths, constants, and loaders."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
RESULTS = ROOT / "results"

AF_TO_MG = 0.325851          # 1 acre-foot = 325,851 US gallons
REUSE_CATEGORY = "Recycled Water Use"
RANDOM_STATE = 42


def ensure_dirs():
    for d in (PROC, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


def utilities(include_only=True):
    u = pd.read_csv(CONFIG / "utilities.csv")
    return u[u["include"].str.upper() == "Y"] if include_only else u


MIN_DAYS = 20   # a month built from daily data needs at least this many days


def monthly_from_daily(location_key):
    """Monthly mean temperature from NOAA GHCN-Daily: daily mean = TAVG, or
    (TMAX + TMIN) / 2 when TAVG is missing; month kept only with >= MIN_DAYS days."""
    f = RAW / f"noaa_ghcnd_{location_key}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "temp_daily", "days"])
    d = pd.read_csv(f)
    if d.empty:
        return pd.DataFrame(columns=["date", "temp_daily", "days"])
    col = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    d["t"] = col("TAVG").fillna((col("TMAX") + col("TMIN")) / 2)
    d["date"] = pd.to_datetime(d["DATE"].astype(str).str[:7] + "-01")
    m = d.dropna(subset=["t"]).groupby("date").t.agg(["mean", "count"]).reset_index()
    m.columns = ["date", "temp_daily", "days"]
    return m[m.days >= MIN_DAYS]


def load_temperature(location_key):
    """Monthly mean temperature (F). Primary: NOAA Global Summary of the Month
    (TAVG, or (TMAX + TMIN) / 2). Where a month is blank, fall back to a monthly
    mean computed from NOAA GHCN-Daily (see monthly_from_daily)."""
    f = RAW / f"noaa_gsom_{location_key}.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"Missing {f.name}. Add the station ID for '{location_key}' to "
            f"config/stations.csv and run download_data.py --only noaa.")
    t = pd.read_csv(f)
    t["date"] = pd.to_datetime(t["DATE"].astype(str).str[:7] + "-01")
    col = lambda c: pd.to_numeric(t[c], errors="coerce") if c in t else pd.Series(np.nan, index=t.index)
    tavg = col("TAVG")
    est = (col("TMAX") + col("TMIN")) / 2
    t["temp_F"] = tavg.fillna(est)
    t["temp_from_max_min"] = tavg.isna() & est.notna()
    t = t[["date", "temp_F", "temp_from_max_min"]].groupby("date", as_index=False).first()
    daily = monthly_from_daily(location_key)
    t = t.merge(daily[["date", "temp_daily"]], on="date", how="outer")
    t["temp_from_daily"] = t.temp_F.isna() & t.temp_daily.notna()
    t["temp_F"] = t.temp_F.fillna(t.temp_daily)
    t["temp_from_max_min"] = t.temp_from_max_min.fillna(False).astype(bool)
    return (t.dropna(subset=["temp_F"])[["date", "temp_F", "temp_from_max_min", "temp_from_daily"]]
             .sort_values("date").reset_index(drop=True))


def load_precipitation(location_key):
    """Monthly total precipitation (inches) from the NOAA GSOM file, where reported."""
    f = RAW / f"noaa_gsom_{location_key}.csv"
    t = pd.read_csv(f)
    if "PRCP" not in t:
        return pd.DataFrame(columns=["date", "prcp_in"])
    t["date"] = pd.to_datetime(t["DATE"].astype(str).str[:7] + "-01")
    t["prcp_in"] = pd.to_numeric(t["PRCP"], errors="coerce")
    return t[["date", "prcp_in"]].dropna()


def load_singapore():
    """PUB 'Volume of NEWater sold, Annual' from data.gov.sg (million m3)."""
    f = RAW / "singapore_newater_annual.json"
    if not f.exists():
        return pd.DataFrame()
    js = json.loads(f.read_text(encoding="utf-8"))
    recs = js.get("result", {}).get("records", [])
    rows = []
    for r in recs:
        year = r.get("year")
        val = next((v for k, v in r.items() if "newater" in k.lower()), None)
        if year and val not in (None, ""):
            rows.append(dict(city="Singapore", period_type="calendar",
                             start_year=int(str(year)[:4]), value=float(val),
                             unit="million m3", value_mg=float(val) * 264.172,
                             source="singapore_newater_annual.json"))
    return pd.DataFrame(rows)


def metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    nz = actual != 0
    return dict(
        n=len(actual),
        r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        mae=np.mean(np.abs(err)),
        rmse=np.sqrt(np.mean(err ** 2)),
        mape=np.mean(np.abs(err[nz] / actual[nz])) * 100 if nz.any() else np.nan,
    )
