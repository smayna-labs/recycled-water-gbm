"""
diagnostics.py: two checks reported in the paper.

1. Facility selection: how many VAR facilities report recycled water use, and how
   many have all 84 months (2019-2025), so readers can see the ten were drawn from
   a defined pool rather than picked for good results.
2. DSRSD autumn 2025: was the unusually early drop in recycled water use
   accompanied by unusual rainfall? Compares Sep-Nov precipitation and use by year.
"""
import pandas as pd
from common import RAW, PROC, RESULTS, REUSE_CATEGORY, ensure_dirs, utilities, load_precipitation


def selection_counts():
    e = pd.read_csv(RAW / "var_effluent.csv", low_memory=False, encoding="utf-8-sig")
    rw = e[e["DISCHARGED TO"] == REUSE_CATEGORY]
    m = rw.groupby(["GLOBAL ID", "REPORTING YEAR", "REPORTING MONTH"])["DISCHARGE VOLUME"].sum()
    months = m.groupby(level=0).size()
    sel = utilities().global_id
    out = dict(facilities_reporting_reuse=len(months),
               facilities_all_84_months=int((months == 84).sum()),
               selected=len(sel),
               selected_all_in_complete_set=bool(months.reindex(sel).eq(84).all()))
    pd.Series(out).to_csv(RESULTS / "selection_counts.csv", header=["value"])
    print("Facility selection:\n" + pd.Series(out).to_string())


def dsrsd_autumn():
    panel = pd.read_csv(PROC / "panel.csv", parse_dates=["date"])
    use = panel[panel.utility == "DSRSD"][["date", "year", "month", "demand_mg"]]
    pr = load_precipitation("dsrsd")
    t = use.merge(pr, on="date", how="left")
    t = t[t.month.isin([9, 10, 11])]
    table = t.pivot_table(index="year", columns="month", values=["demand_mg", "prcp_in"])
    table.columns = [f"{v}_{['', '', '', '', '', '', '', '', '', 'Sep', 'Oct', 'Nov'][mth]}"
                     for v, mth in table.columns]
    table.round(2).to_csv(RESULTS / "dsrsd_autumn_rain_check.csv")
    print("\nDSRSD autumn use (MG) and precipitation (in), by year:\n" + table.round(2).to_string())
    if t.prcp_in.isna().all():
        print("No precipitation values: re-run download_data.py --only noaa (it now requests PRCP).")


def main():
    ensure_dirs()
    selection_counts()
    dsrsd_autumn()


if __name__ == "__main__":
    main()
