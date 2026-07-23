# GNSS-RO Pipeline

Ground-based and satellite GNSS Radio Occultation processing suite with a PyQt GUI.

GNSS Radio Occultation is a limb-sounding technique in which GNSS signal excess
phase and Doppler shift, recorded as a satellite sets below the horizon, are
inverted to retrieve vertical profiles of atmospheric bending angle, refractivity,
and thermodynamic state. This tool implements the ground-based variant (fixed
receiver, occulting GNSS satellite) alongside a satellite (LEO) path, with Abel
inversion, Smith–Weintraub refractivity retrieval, and optional ERA5 validation.

## Scientific notes

- **Impact parameter** is computed using tangent-point local Earth radius rather
  than station radius, correcting a systematic height bias that grows with
  station–tangent-point separation (up to ~13 km at mid-latitudes).
- **Reference satellite** excess Doppler is taken from a single quality-ranked
  candidate at each epoch; multi-satellite averaging is explicitly avoided to
  prevent correlated ionospheric and multipath contamination.
- **Fresnel-zone polynomial smoothing** segments the time series at data gaps
  before fitting, with a configurable gap threshold.
- Processing constants (elevation masks, Doppler thresholds, Smith–Weintraub
  coefficients, minimum RO epoch count, etc.) are exposed in the `PROCESSING`
  block of the station `.cra` file and applied without code edits.

## Requirements

Python 3.8+ · PyQt6 or PyQt5 · NumPy · SciPy · pandas · matplotlib · netCDF4 · xarray

```bash
pip install -r requirements.txt
```

## Input

| File type | Purpose |
|---|---|
| RINEX 3 `.rnx` / u-blox `.ubx` | Observation data |
| SP3 | Precise satellite orbits |
| `metadata.cra` (JSON) | Station coordinates + processing config |
| ERA5 `.nc` (optional) | Validation climatology |

Station coordinates are read automatically from the RINEX `APPROX POSITION XYZ`
header (ECEF → geodetic via iterative Bowring method on WGS84). Manual override
is available via `PROCESSING.FORCE_CRA_STATION_COORDS`.

## Output

Per-satellite CSVs and plots for bending angle, refractivity, comparison against
ERA5, and thermodynamic profiles (pressure, temperature, water vapour, relative
humidity). A `processing_summary.csv` records RO classification for all observed
satellites.

## Versioning

| Version | Notes |
|---|---|
| v3.4.7 | GUI legibility pass for HD (1366x768) Windows laptops; "Mountain" ground label |
| v3.4.5 | Fix PyInstaller module bundling on Windows 7 / Windows 10 |
| v3.4.4.2 | UI refinements: keyboard navigation, tri-state RO classification display |
| v3.4.4 | `.cra` non-destructive merge-save, `PROCESSING` block wired through, scatter profile plots |
| v3.4 | RINEX auto-position, configurable constants, ranked reference selection, tangent-point curvature |

See `CHANGELOG_v3.4.7.md`, `CHANGELOG_v3.4.4.md` and `CHANGELOG_v3.4.md` for full details.
