"""
download_data.py: fetch every source file for the project into data/raw/
and record provenance (URL, UTC download time, size, SHA-256) in
data/raw/MANIFEST.csv. Commit data/raw/ and the manifest to GitHub so
any reviewer can check that the analysis used these exact files.

Usage:
    python download_data.py            # download everything
    python download_data.py --list     # show sources without downloading
    python download_data.py --only var # download one group

Standard library only. If a download fails (sites change URLs), the
script prints the source page so you can download that file by hand
into data/raw/ and re-run with --manifest-only to record it.
"""
import argparse, csv, hashlib, sys, time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("data/raw")

# group, filename, url, description
SOURCES = [
    # California Volumetric Annual Report (State Water Board), monthly 2019 onward
    ("var", "var_facilities.csv",
     "https://data.ca.gov/dataset/aec3e060-1469-44c3-8e36-5dd223b8b2d8/resource/7d6033c2-06b0-489c-8ff8-5f489dcb1455/download/avr_export_facility.csv",
     "CA VAR facility list (GLOBAL ID, names)"),
    ("var", "var_effluent.csv",
     "https://data.ca.gov/dataset/aec3e060-1469-44c3-8e36-5dd223b8b2d8/resource/fd190d10-063a-4304-8889-e99ca42e9186/download/avr_export_effluent.csv",
     "CA VAR monthly effluent by discharge category, acre-feet/month"),
    ("var", "var_reuse.csv",
     "https://data.ca.gov/dataset/aec3e060-1469-44c3-8e36-5dd223b8b2d8/resource/eca566d8-ab49-4aeb-9c33-585ccc34b48d/download/avr_export_reuse.csv",
     "CA VAR annual reuse by category, acre-feet/year (cross-check)"),

    # Singapore PUB, annual NEWater sold, million cubic metres, 2007-2025
    ("intl", "singapore_newater_annual.json",
     "https://data.gov.sg/api/action/datastore_search?resource_id=d_2eceeb792a0fca1caa74304d47b46060&limit=100",
     "PUB Volume of NEWater sold, Annual (data.gov.sg). Page: https://data.gov.sg/datasets/d_2eceeb792a0fca1caa74304d47b46060/view"),

    # Sydney Water, annual recycled water (financial year Jul-Jun), ML
    ("intl", "sydney_water_annual_report_2021_22.pdf",
     "https://www.sydneywater.com.au/content/dam/sydneywater/documents/sydney-water-annual-report-2021-22.pdf",
     "Sydney Water Annual Report 2021-22"),
    ("intl", "sydney_water_annual_report_2022_23.pdf",
     "https://www.sydneywater.com.au/content/dam/sydneywater/documents/sydney-water-annual-report-2022-23.pdf",
     "Sydney Water Annual Report 2022-23"),
    ("intl", "sydney_water_conservation_report_2023_24.pdf",
     "https://www.sydneywater.com.au/content/dam/sydneywater/documents/water-conservation-report-2023-24.pdf",
     "Sydney Water Water Conservation Report 2023-24 (Table 10)"),
    ("intl", "bom_urban_npr_2023_24.pdf",
     "https://www.bom.gov.au/water/npr/docs/2023-24/Urban_National_Performance_Report_2023-24.pdf",
     "BoM Urban National Performance Report 2023-24 (indicator W26)"),

    # Beijing Water Authority, annual reclaimed water, 100 million m3
    ("intl", "beijing_water_statistics_yearbook_2022.pdf",
     "https://swj.beijing.gov.cn/zwgk/swtjnj/202309/P020230926422430340575.pdf",
     "Beijing Water Statistical Yearbook 2022 (reclaimed water 2018-2022)"),
    ("intl", "beijing_water_resources_bulletin_2023.pdf",
     "https://swj.beijing.gov.cn/zwgk/szygb/202404/P020240430629863402602.pdf",
     "Beijing Water Resources Bulletin 2023 (reclaimed water 2023)"),
    ("intl", "beijing_2024_reclaimed_water_announcement.html",
     "https://www.beijing.gov.cn/ywdt/gzdt/202503/t20250326_4044893.html",
     "Beijing municipal government, 2024 reclaimed water total"),

]

NOAA_DAILY_API = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
                  "&stations={sid}&startDate=2018-01-01&endDate=2025-12-31"
                  "&dataTypes=TAVG,TMAX,TMIN,PRCP&units=standard&format=csv&includeStationName=true")
NOAA_API = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-summary-of-the-month"
            "&stations={sid}&startDate=2018-01-01&endDate=2025-12-31"
            "&dataTypes=TAVG,TMAX,TMIN,PRCP&units=standard&format=csv&includeStationName=true")

def noaa_sources():
    """Monthly mean/max/min temperature per station, from config/stations.csv."""
    f = Path("config/stations.csv")
    if not f.exists():
        return []
    out = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sid = str(r["noaa_station_id"] or "").strip()
            sid = sid.split(":", 1)[1] if ":" in sid else sid   # 'GHCND:USW...' -> 'USW...'
            if sid.lower() == "nan":
                sid = ""
            if sid:
                out.append(("noaa", f"noaa_gsom_{r['location_key']}.csv",
                            NOAA_API.format(sid=sid),
                            f"NOAA GSOM monthly temperature, {r['station_name']} ({sid})"))
                out.append(("noaa", f"noaa_ghcnd_{r['location_key']}.csv",
                            NOAA_DAILY_API.format(sid=sid),
                            f"NOAA GHCN-Daily temperature, {r['station_name']} ({sid})"))
    return out

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research data download)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

def read_manifest():
    path = RAW / "MANIFEST.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f)}

def write_manifest(rows):
    path = RAW / "MANIFEST.csv"
    old = read_manifest()
    for r in rows:
        prev = old.get(r["file"])
        # keep the original download time if the file content is unchanged
        if prev and prev.get("sha256") == r["sha256"]:
            r["downloaded_utc"] = prev["downloaded_utc"]
        old[r["file"]] = r
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "group", "url", "description",
                                          "downloaded_utc", "bytes", "sha256"])
        w.writeheader()
        for r in sorted(old.values(), key=lambda r: (r["group"], r["file"])):
            w.writerow(r)
    print(f"\nManifest: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", choices=["var", "intl", "noaa"])
    ap.add_argument("--manifest-only", action="store_true",
                    help="record files already in data/raw without downloading")
    a = ap.parse_args()

    todo = [s for s in SOURCES + noaa_sources() if not a.only or s[0] == a.only]
    if a.list:
        for g, fn, url, d in todo:
            print(f"[{g}] {fn}\n      {d}\n      {url}")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for g, fn, url, d in todo:
        dest = RAW / fn
        if not a.manifest_only:
            print(f"downloading {fn} ...", end=" ", flush=True)
            try:
                fetch(url, dest)
                print("ok")
            except Exception as e:
                print(f"FAILED ({e})")
                failed.append((fn, url))
                continue
            time.sleep(1)
            if g == "noaa":
                with open(dest, encoding="utf-8", errors="ignore") as fh:
                    lines = sum(1 for _ in fh)
                if lines < 2:
                    print(f"   WARNING: {fn} has no data rows. Check the station ID "
                          f"(use the ID without 'GHCND:'), or pick another station.")
                    failed.append((fn, url))
        if dest.exists():
            rows.append(dict(file=fn, group=g, url=url, description=d,
                             downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             bytes=dest.stat().st_size, sha256=sha256(dest)))
    write_manifest(rows)
    if failed:
        print("\nDownload these by hand into data/raw/, then run with --manifest-only:")
        for fn, url in failed:
            print(f"  {fn}\n    {url}")
        sys.exit(1)

if __name__ == "__main__":
    main()
