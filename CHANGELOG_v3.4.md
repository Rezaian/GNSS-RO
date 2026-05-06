# GNSS-RO Pipeline v3.4 — Changelog

## Feature 1: Auto-read station position from RINEX header

**Problem:** Users had to manually enter station lat/lon/alt in the `.cra` file even when the RINEX observation file already contains this information in the `APPROX POSITION XYZ` header field.

**Changes:**

- **`rinex_parser.py`** — Added `ecef_to_geodetic()` function (iterative Bowring method on WGS84) that converts the ECEF (X, Y, Z) from the RINEX header to geodetic (lat, lon, alt). The parser now auto-populates `approx_position_geodetic` and reads `MARKER NAME` from the header.

- **`ground_gnss_ro_pipeline.py`** — `parse_rnx_directory()` now extracts station position from the first RINEX file that has a valid `APPROX POSITION XYZ`. Returns it in `metadata['rinex_station']`.

- **`gnss_ro_gui.py`** — The directory scanner now accepts RINEX + SP3 **without** a `metadata.cra` file. When no `.cra` is found but RINEX files exist, the GUI calls `_try_load_rinex_station_position()` which reads just the RINEX header and auto-fills the station panel. The user can still override values before running.

**Your RINEX header example:** `APPROX POSITION XYZ = (3208200.1984, 4069606.8344, 3709505.7030)` → lat=35.778°N, lon=51.750°E, alt=2097m.

---

## Feature 2: Fine-tuning processing constants via `.cra` file

**Problem:** Key processing constants were hard-coded, requiring code edits to tune the pipeline.

**Changes:** Added 15 configurable parameters under a `"PROCESSING"` key in the `.cra` JSON. A sample `metadata_sample.cra` is included. The parameters and why each was chosen:

| Parameter | Default | Why configurable |
|---|---|---|
| `POLY_SMOOTH_WINDOW` | 150 | Controls noise vs resolution trade-off. 1 Hz data needs ~3, 50 Hz needs ~150. |
| `REF_SAT_ELEVATION_THRESHOLD` | 50° | Higher = less multipath in reference, but fewer candidates. |
| `REF_SAT_MIN_EPOCHS` | 100 | Ensures reference continuity. Lower for short sessions. |
| `REF_SAT_JUMP_THRESHOLD` | 2.0 Hz | Cycle-slip detection sensitivity. Lower = stricter. |
| `RO_ELEVATION_THRESHOLD` | 5° | Where to look for occultation signal. |
| `RO_DOPPLER_THRESHOLD` | 1.0 Hz | Minimum atmospheric Doppler to flag RO. |
| `RO_MIN_EPOCHS` | 10 | Minimum RO epochs for valid event. |
| `ELEVATION_MASK_HIGH` | 45° | Above this, satellite is reference-only. |
| `ELEVATION_MASK_LOW` | -5° | Allows sub-horizon tracking. |
| `HEIGHT_RANGE_MIN` | -10 km | Allows near-surface retrievals. |
| `HEIGHT_RANGE_MAX` | 150 km | Upper bound for output. |
| `CLIMATOLOGY_BLEND_HEIGHT` | 50 km | Statistical optimization blending height. |
| `N_COEFF_A1` | 77.6 | Smith-Weintraub dry term. |
| `N_COEFF_A2` | 3.73e5 | Smith-Weintraub wet term. |
| `USE_TANGENT_POINT_CURVATURE` | true | Use tangent-point local curvature (Feature 4). |

New functions: `load_processing_config_from_cra()`, `apply_processing_config()`, `PROCESSING_DEFAULTS` dict.

---

## Feature 3: Improved reference satellite selection

**Problem:** The fallback from a single primary reference to an **elevation-weighted average** of multiple references is risky. Each satellite has its own multipath, phase center offsets, and ionospheric path. Averaging their excess Dopplers introduces correlated errors that don't cancel, contaminating the atmospheric signal.

**Previous logic:**
1. Select best primary ref (scoring)
2. If unavailable → **weighted average of all high-elev sats** ← dangerous
3. Last resort → highest elevation sat

**New logic (sequential quality-based):**
1. Build a **ranked list** of candidate reference sats, sorted by quality score
2. Quality checks per candidate:
   - **Cycle-slip detection:** Count epoch-to-epoch jumps in excess Doppler above `REF_SAT_JUMP_THRESHOLD` Hz. Satellites with >2% jumped epochs are **rejected entirely**.
   - **Coverage continuity:** Large gap ratio penalizes intermittent sats.
   - **Doppler stability:** Lower excess_doppler variance preferred.
   - **Elevation:** Higher minimum elevation preferred.
3. At each epoch, try candidates **in rank order** until one is available.
4. **Never average** multiple references.
5. Last resort: single highest-elevation sat at that epoch.

**Key change:** Replaced `_select_primary_reference()` + `_compute_weighted_reference()` with `_build_ranked_reference_list()`. The reference type field now shows `primary`, `backup_1`, `backup_2`, etc. instead of `weighted_avg`.

---

## Feature 4: Negative impact height investigation & fix

**Problem:** Some retrievals produce negative tangent heights (impact parameter `a` < local Earth radius `R`).

**Root cause analysis** (referencing Hajj et al. 2002, Section 4.2):

The impact parameter `a = |r_t × k_t|` is the perpendicular distance from the **center of local curvature** to the ray asymptote. The tangent height is `h = a - R_local`. Previously, `R_local` was computed at the **station latitude** using `station.get_gaussian_radius()`. But the occultation tangent point can be hundreds of km away at a significantly different latitude.

The Gaussian mean radius varies from **6357 km at the equator** to **6400 km at the poles** — a 43 km range. If the station is at 35°N (R ≈ 6370 km) but the tangent point is near the equator (R ≈ 6357 km), using the station radius makes tangent heights ~13 km too low. For measurements near the surface, this easily produces negative heights.

Hajj et al. (2002): *"The center of symmetry is taken to be the center of a circle in the occultation plane which is tangent to the ellipse at the ray path tangent point with a radius equal to the ellipse's radius of curvature at the same tangent point. This center is then fixed for the entire occultation, and can be as far as 40 km from the real center of the Earth."*

**Fixes applied:**

1. **`_estimate_tangent_point_radius()`**: Estimates the tangent point location from the satellite-receiver geometry (closest approach point on the line), gets its latitude, and computes `R_local` there.

2. **`_gaussian_radius_at_lat()`**: Static method to compute Gaussian radius at any latitude (same formula as `StationConfig.get_gaussian_radius()`).

3. **Better `fsolve` initial guess**: Scales with Doppler magnitude. Large atmospheric Doppler → larger initial bending guess.

4. **Convergence validation**: After fsolve, checks that residuals are small. If not, retries with fallback guess. Returns NaN on persistent failure.

5. **Physical sanity filter**: Rejects impact parameters < `R_local - 50 km` (below surface) or > satellite orbit radius.

6. **Output includes `local_radius_km`**: The bending CSV now records which local radius was used, aiding diagnostics.

7. **Configurable via CRA**: `USE_TANGENT_POINT_CURVATURE` (default: true) controls whether to use the new tangent-point curvature or fall back to station-based radius.
