# GNSS-RO Pipeline

Ground-based and satellite GNSS Radio Occultation processing suite.

## Features
- Auto-reads station position from RINEX header (no manual lat/lon entry)
- Tunable processing constants via `metadata.cra` (`PROCESSING` block) —
  the GUI's "Advanced Settings" panel exposes them all and the file is
  preserved non-destructively across runs
- Open a previously executed `*_output` directory directly in the GUI to
  re-view its plots and results without re-running the pipeline
- Quality-ranked reference satellite selection (no averaging)
- Tangent-point local curvature for impact-parameter retrieval

## Requirements
Python 3.10+ with PyQt6 (or PyQt5), NumPy, SciPy, pandas, matplotlib, netCDF4, xarray.
```bash
pip install -r requirements.txt
```

## Run
```bash
python gnss_ro_gui.py
```

In the directory picker:
- **Pick an *input* dataset** (RINEX/UBX + SP3) → the pipeline executes as usual.
- **Pick an existing `*_output` directory** → the GUI loads the saved
  artifacts (CSVs + PNGs) and lets you browse them. The Start button becomes
  "Close Project". No re-execution.

## Version
v3.4.4.2 — see `CHANGELOG_v3.4.4.md` for full details.
