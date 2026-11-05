# Data

`data/raw/MANIFEST.csv` records each downloaded file's source URL, UTC download
time, size, and SHA-256 checksum.

## Sources used in the paper

| Data | Source | Frequency | Unit |
|---|---|---|---|
| Recycled water use, ten utilities | State Water Resources Control Board, Volumetric Annual Report (data.ca.gov) | Monthly, 2019 onward | acre-feet |
| DSRSD 2025 cross-check | Provided directly by DSRSD | Monthly | million gallons |
| Precipitation, Livermore (DSRSD check) | NOAA NCEI Global Summary of the Month | Monthly | inches |

Conversion: 1 acre-foot = 0.325851 million US gallons.

## Notes

- `config/utilities.csv` lists the ten utilities and their VAR facility IDs.
  Of 298 facilities reporting recycled water use, 135 reported all 84 months
  of 2019 to 2025; the ten were chosen from those to span climate regions.
- `download_data.py` can also fetch international and temperature files used in
  earlier exploratory work. They are not used in the paper's analysis.
- An earlier version of this project used a dataset whose values could not be
  traced to official sources. It was removed and is not used anywhere here.
