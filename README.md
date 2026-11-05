# Recycled water forecasting benchmark

Rolling-origin comparison of gradient boosting, linear regression, a monthly
average, and a seasonal naive forecast for monthly recycled water use at ten
California utilities (State Water Resources Control Board, Volumetric Annual
Report, 2019-2025).

## Environment

Tested with Python 3.12.7 on macOS and the exact versions in requirements.txt.

    pip install -r requirements.txt

## Reproduce the paper

    python download_data.py --only var     # VAR files into data/raw/ (+ MANIFEST.csv)
    python download_data.py --only noaa    # NOAA files, used for the DSRSD rainfall check
    python src/pipeline.py

Every number, table, and figure in the paper is written to `results/final/`
(evaluation) and `results/` (diagnostics). `results/final/provenance.json`
records the package versions, Git commit, parameters, and SHA-256 hashes of
the inputs. The published results correspond to release tag `v1.0`.

## Scripts

| Script | What it does |
|---|---|
| download_data.py | Downloads source files and records URL, time, size, and checksum |
| src/data_prep.py | Builds the monthly panel; quality report; DSRSD cross-check |
| src/evaluate.py | Rolling-origin evaluation, January 2022 to December 2025: all models, tests, tables, figures, provenance |
| src/diagnostics.py | Facility selection counts; DSRSD autumn rainfall check |
| src/pipeline.py | Runs the three steps above in order |

See README_DATA.md for data sources.
