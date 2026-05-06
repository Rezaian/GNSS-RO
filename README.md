# GNSS-RO Pipeline

Ground-based and satellite GNSS Radio Occultation processing suite.

## Features
- Auto-reads station position from RINEX header (no manual lat/lon entry)
- 15 configurable processing constants via `.cra` file
- Improved reference satellite selection (quality-ranked, no averaging)
- Fixed negative tangent height bug using tangent-point local curvature

## Requirements
Python 3.10+ with PyQt6, NumPy, SciPy, pandas, matplotlib, netCDF4, xarray.
```bash
pip install -r requirements.txt
```

## Run
```bash
python gnss_ro_gui.py
```

## Version
v3.4.3.2 — see `CHANGELOG_v3.4.md` for full details.