"""
data_prep.py: build the analysis datasets from data/raw/.

Outputs (data/processed/):
  panel.csv                 monthly recycled water use per utility, MG, with
                            actual monthly temperature and lag-1/2/12 features
  quality_report.csv        missing/zero months and temperature gaps per utility
  dsrsd_crosscheck.csv      VAR vs utility-provided DSRSD 2025, month by month
  international_annual.csv  annual totals for international cities, MG
"""
import pandas as pd
import numpy as np
from common import (RAW, PROC, AF_TO_MG, REUSE_CATEGORY, ensure_dirs,
                    utilities, load_temperature, load_singapore)


def monthly_recycled():
    eff = pd.read_csv(RAW / "var_effluent.csv", low_memory=False, encoding="utf-8-sig")
    rw = eff[eff["DISCHARGED TO"] == REUSE_CATEGORY]
    u = utilities()
    frames, quality = [], []
    for _, r in u.iterrows():
        s = rw[rw["GLOBAL ID"] == r["global_id"]]
        # several rows per month are possible (one per treatment level): sum them
        m = (s.groupby(["REPORTING YEAR", "REPORTING MONTH"])["DISCHARGE VOLUME"]
               .sum().reset_index())
        m.columns = ["year", "month", "demand_af"]
        m["date"] = pd.to_datetime(dict(year=m.year, month=m.month, day=1))
        full = pd.date_range(m.date.min(), m.date.max(), freq="MS")
        missing = len(full) - len(m)
        zeros = int((m.demand_af == 0).sum())
        m["utility"] = r["label"]
        m["location_key"] = r["location_key"]
        m["demand_mg"] = m.demand_af * AF_TO_MG
        frames.append(m)
        quality.append(dict(utility=r["label"], global_id=r["global_id"],
                            first=m.date.min().date(), last=m.date.max().date(),
                            months=len(m), missing_months=missing, zero_months=zeros))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(quality)


def add_temperature(panel, quality):
    out, temp_notes, bad = [], {}, []
    for key, g in panel.groupby("location_key"):
        t = load_temperature(key)
        if t.empty:
            bad.append(f"{key}: noaa_gsom_{key}.csv has no TAVG/TMAX/TMIN values "
                       f"(check the station ID in config/stations.csv)")
        else:
            n_miss = int(g[["date"]].merge(t, on="date", how="left").temp_F.isna().sum())
            if n_miss > 0.10 * len(g):
                bad.append(f"{key}: station reports only {len(g) - n_miss} of {len(g)} months "
                           f"(choose a station with fuller coverage)")
        if not t.empty and (t.date.max() < g.date.max() or t.date.min() > g.date.min()):
            bad.append(f"{key}: temperatures cover {t.date.min():%Y-%m} to {t.date.max():%Y-%m}, "
                       f"need {g.date.min():%Y-%m} to {g.date.max():%Y-%m}")
        g = g.merge(t, on="date", how="left")
        temp_notes[key] = dict(temp_missing=int(g.temp_F.isna().sum()),
                               temp_from_max_min=int(g.temp_from_max_min.fillna(False).sum()),
                               temp_from_daily=int(g.temp_from_daily.fillna(False).sum()))
        out.append(g)
    if bad:
        raise SystemExit("Temperature problems:\n  " + "\n  ".join(bad))
    panel = pd.concat(out, ignore_index=True)
    key_of = panel.drop_duplicates("utility").set_index("utility")["location_key"]
    quality["temp_missing"] = quality.utility.map(lambda u: temp_notes[key_of[u]]["temp_missing"])
    quality["temp_from_max_min"] = quality.utility.map(lambda u: temp_notes[key_of[u]]["temp_from_max_min"])
    quality["temp_from_daily"] = quality.utility.map(lambda u: temp_notes[key_of[u]]["temp_from_daily"])
    return panel, quality


def add_lags(panel):
    panel = panel.sort_values(["utility", "date"]).copy()
    g = panel.groupby("utility")["demand_mg"]
    for k in (1, 2, 12):
        # shift by calendar position; requires no missing months (checked in quality report)
        panel[f"lag{k}"] = g.shift(k)
    return panel


def dsrsd_crosscheck(panel):
    f = RAW / "dsrsd_2025_from_utility.csv"
    if not f.exists():
        return None
    util = pd.read_csv(f)
    var = panel[(panel.utility == "DSRSD") & (panel.year == 2025)][["month", "demand_mg"]]
    cc = util.merge(var, on="month", how="left").rename(
        columns={"recycled_mg": "utility_mg", "demand_mg": "var_mg"})
    cc["diff_pct"] = (cc.var_mg - cc.utility_mg) / cc.utility_mg * 100
    return cc[["month", "utility_mg", "var_mg", "diff_pct"]]


TO_MG = {"ML": 0.264172, "GL": 264.172, "hm3": 264.172, "million m3": 264.172,
         "1e8 m3": 26417.2}


def international():
    f = PROC.parent / "international_annual.csv"
    rows = []
    if f.exists():
        t = pd.read_csv(f)
        ok = t["value"].notna() & ~t["status"].astype(str).str.contains("TO FILL|CONFLICT|LOAD FROM FILE")
        for _, r in t[ok].iterrows():
            if r.unit not in TO_MG:
                raise ValueError(f"Unknown unit '{r.unit}' for {r.city} {r.period}")
            rows.append(dict(city=r.city, period_type=r.period_type,
                             start_year=int(str(r.period)[:4]), value=float(r.value),
                             unit=r.unit, value_mg=float(r.value) * TO_MG[r.unit],
                             source=r.source_file))
        skipped = t[~ok]
        if len(skipped):
            print(f"  international: {len(skipped)} row(s) skipped (TO FILL / CONFLICT): "
                  + ", ".join(f"{a} {b}" for a, b in zip(skipped.city, skipped.period)))
    df = pd.concat([pd.DataFrame(rows), load_singapore()], ignore_index=True)
    return df.sort_values(["city", "start_year"]) if len(df) else df


def main():
    ensure_dirs()
    panel, quality = monthly_recycled()
    panel, quality = add_temperature(panel, quality)
    panel = add_lags(panel)
    cols = ["utility", "location_key", "date", "year", "month", "temp_F",
            "demand_mg", "lag1", "lag2", "lag12"]
    panel[cols].to_csv(PROC / "panel.csv", index=False)
    quality.to_csv(PROC / "quality_report.csv", index=False)
    print("Quality report:\n" + quality.to_string(index=False))
    if (quality.missing_months > 0).any():
        print("\nWARNING: missing months break the lag features for those utilities.")

    cc = dsrsd_crosscheck(panel)
    if cc is not None:
        cc.to_csv(PROC / "dsrsd_crosscheck.csv", index=False)
        print(f"\nDSRSD 2025 cross-check: utility {cc.utility_mg.sum():.2f} MG, "
              f"VAR {cc.var_mg.sum():.2f} MG, max monthly diff {cc.diff_pct.abs().max():.2f}%")

    intl = international()
    intl.to_csv(PROC / "international_annual.csv", index=False)
    print(f"\nInternational annual rows: {len(intl)}")
    print(f"Panel: {len(panel)} rows, {panel.utility.nunique()} utilities -> {PROC/'panel.csv'}")


if __name__ == "__main__":
    main()
