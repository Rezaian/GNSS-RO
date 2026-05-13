# GNSS-RO Pipeline v3.4.4 — Changelog

Refinement release on top of v3.4.3.x. Focuses on fixing the `.cra`
overwrite bug, making the previously-documented `PROCESSING` block
actually wired through, tightening the polyfit segmentation, switching
the retrieved-profile panels to scatter, cleaning up redundant CSVs,
and adding a load-mode for previously executed projects.

---

## Fix 1: `.cra` overwrite — advanced settings are no longer wiped

**Problem.** Until v3.4.3.2 the GUI called the destructive
`save_metadata(cra_path, station_panel.to_metadata())` at the start of
every run. `to_metadata()` returns only the four station fields, so
re-saving truncated the file: any user-edited `PROCESSING` block — and
any other custom keys the user had added — were silently discarded.

**Fix.** New helper `merge_save_metadata(filepath, station_fields,
processing_fields)` reads the existing `.cra`, updates only the keys it
owns, and writes the merged dict back. Unknown keys are preserved
forward. The destructive call site is replaced with this merge-save.

Concrete behaviour:

- Station fields are updated from the panel (`STATION_NAME`,
  `STATION_LAT`, `STATION_LON`, `STATION_HEIGHT`).
- `PROCESSING` is merged into the existing block; user values for keys
  the GUI didn't change are kept exactly as they were on disk.
- Any other top-level key (e.g. a user-added `NOTES` or `CUSTOM_X`)
  passes through untouched.
- If the picked input dir had no `.cra`, the file is created with the
  station + processing values from the GUI.

## Fix 2: `PROCESSING` block is actually used by the pipeline

**Problem.** The v3.4 changelog described a `PROCESSING` block of
configurable constants, but the constants were still hard-coded in
`ground_gnss_ro_pipeline.py` and no code path read the block. Editing
the `.cra` had no effect on processing.

**Fix.**

- New `PROCESSING_DEFAULTS` dict centralises every tunable.
- `load_processing_config_from_cra(cra_data)` overlays the user's
  `PROCESSING` block on the defaults.
- `apply_processing_config(cfg)` writes the resolved values into the
  module-level globals (`RO_MIN_EPOCHS`, `POLYNOMIAL_WINDOW`,
  `POLYFIT_GAP_THRESHOLD`, `RO_ELEVATION_THRESHOLD`,
  `RO_DOPPLER_THRESHOLD`, `N_COEFF_A1`, `N_COEFF_A2`).
- `evaluate_ro_status` and `apply_fresnel_polynomial_smoothing` now
  resolve those constants **live** at call time, so values set via
  `apply_processing_config` reach the right functions.
- The child process spawned for ground processing receives the
  `PROCESSING` dict as an argument and calls
  `apply_processing_config()` before the first pipeline step, so the
  overrides take effect there too.

The GUI gains a collapsible **Advanced Settings** panel
(`ProcessingPanel`) below the station info group. It loads from the
`.cra`'s `PROCESSING` on directory pick, falls back to defaults when
keys are missing, and writes back on Run.

## Fix 3: `RO_MIN_EPOCHS` raised from 10 → 25

The minimum number of RO-flagged epochs required to call an event a
valid occultation. Old default (10) was too lax and produced sparse
plots. New default is 25; configurable via `PROCESSING.RO_MIN_EPOCHS`.

This is wired into `evaluate_ro_status` (which gates both the per-sat
RO classification and the dual-frequency epoch check).

## Fix 4: Polyfit gap tolerance is now a named, tunable constant

The Fresnel polynomial fit segments the time series whenever it sees a
gap. The gap threshold is now `POLYFIT_GAP_THRESHOLD` (default 5.0 s)
and is exposed via `PROCESSING.POLYFIT_GAP_THRESHOLD`. The function
reads the live module value, so `.cra` overrides take effect.

(Note: the value 5.0 was already in code; v3.4.4 names and exposes it
so it can be tuned without code edits.)

## Fix 5: Tab 2 and Tab 3 profile plots are now scatter

`generate_derived_plots` (Bending / Refractivity / % Error / Specific
Humidity) and `generate_atmospheric_plots` (Pressure / Water Vapor /
RH / Temperature) previously drew their retrieved profiles as line
plots in height-space. Outliers in the retrieval connect through the
adjacent points and visually wreck the panel.

v3.4.4 switches both panels to `scatter` markers (size 8, alpha 0.7,
no edges) so outliers stay isolated and easy to spot. Tab 1
(Observations / Raw) is untouched.

## Fix 6: Redundant intermediate CSVs are deleted

`step1_observations.csv`, `step2_matched.csv`, `step3a_elevations.csv`,
and `step3b_doppler.csv` are all consumed to produce
`step4_differenced.csv`, which contains every column from the earlier
steps. The intermediates are redundant and wasted disk on large runs.

After a successful run the GUI deletes the four step1-step3 files.
The user can opt out by setting `PROCESSING.KEEP_INTERMEDIATE_CSVS` to
`true` (useful when debugging the pipeline itself).

Per-event outputs (`bending/`, `refractivity/`, `comparison/`,
`atmospheric/`, `plots/`) are never touched.

## Fix 7: Open a previously executed project in the GUI

The GUI's directory picker now distinguishes between input datasets
and previously executed `*_output` directories.

When the user picks a path that contains `ground/step4_differenced.csv`
and/or `satellite/processing_summary.csv` (or a flat layout with the
same files at the root), the GUI enters **load mode**:

- The validation strip shows a "📂 Loaded previous results" banner.
- The result list is reconstructed from disk (`evaluate_ro_status`
  applied to the saved step4 CSV for ground; the saved
  `processing_summary.csv` for satellite).
- Selecting an item loads the saved PNGs exactly as the regular flow
  would after a fresh execution.
- The **Start Processing** button becomes **Close Project** and resets
  the picker when clicked. The **Stop** button is hidden — there is
  nothing to stop.
- Station and Advanced Settings panels are hidden because no execution
  will happen.

Missing artifacts (e.g. no `step4_differenced.csv`, no `plots/`) emit
soft warnings in the validation strip rather than blocking the load.

Picking an input dataset afterwards reverts the UI to execute-mode
automatically.

## New `PROCESSING` keys introduced in v3.4.4

| Key | Default | Effect |
|---|---|---|
| `RO_MIN_EPOCHS` | 25 | Min RO epochs for a valid event. (Was 10.) |
| `POLYFIT_GAP_THRESHOLD` | 5.0 s | Restart polyfit when time gap ≥ this. |
| `KEEP_INTERMEDIATE_CSVS` | `false` | If `true`, keep step1-step3 CSVs after a successful run. |
| `FORCE_CRA_STATION_COORDS` | `false` | If `true`, `.cra` station coords override the RINEX header. Default `false` because the RINEX header is generally more accurate. |

All other `PROCESSING` keys carry the same defaults documented in the
v3.4 changelog.

## Version bump rationale

v3.4.4 is a refinement-and-bugfix release on top of v3.4.3.x. It
doesn't introduce a new module or rewrite an algorithm, so it stays on
the v3.4 line as a patch bump rather than v3.5.

---

# v3.4.4.1 — Tri-state RO classification

## Problem

A satellite could pass all of `evaluate_ro_status`'s checks (elevation
window, atmospheric Doppler threshold, minimum dual-frequency epochs)
and still produce an empty bending profile — e.g. the Abel inversion
fsolve never converged, or every retrieved row ended up NaN. In v3.4.4
that satellite appeared as a green ● row in the result list, but
clicking it on the Bending / Refractivity tab or the Atmospheric tab
revealed nothing useful.

## Fix

Ground RO status is now tri-state instead of bool:

| State | Marker | Color | Meaning | Derived tabs |
|---|---|---|---|---|
| `ro_ok` | ● | green `#2E7D32` | RO geometry + usable bending profile | Enabled |
| `ro_empty` | ◐ | amber `#F9A825` | RO geometry but no profile data | Disabled — placeholder shown |
| `False` (no RO) | ○ | gray `#757575` | Failed the RO checks | Disabled — placeholder shown |

A satellite is classified as `ro_empty` when its
`bending/<sat_id>_bending.csv` is missing, has zero rows, or has only
NaN/inf values in every recognised bending column (`bending_angle_rad`,
`bending_L1`, `bending_L2`). Otherwise it's `ro_ok`.

## Implementation

- `_has_usable_bending_data(bending_csv)` — file-level inspector.
- `classify_ground_ro_status(ground_dir, bool_status)` — upgrades a
  bool map to a tri-state map.
- `evaluate_ro_status` is unchanged (still returns booleans). The new
  classifier wraps its output. This keeps the pipeline's RO definition
  intact — yellow is purely a *display* state.
- `load_ground_ro_status_from_csv` returns the tri-state map directly
  so load-mode behaves the same as execute-mode.
- The list rows carry the tri-state on `Qt.UserRole + 3`; existing
  `UserRole + 1` keeps its semantics ("are derived tabs enabled?")
  so older bool consumers don't break.
- When a yellow row is selected, tabs 2 and 3 show:

  > Bending retrieval did not converge for *{sat_id}*<br>
  > RO geometry was detected, but no profile data is available to
  > display.

- Legend strings in both execute-mode and load-mode now reflect the
  tri-state: `● RO + profile    ◐ RO • no profile    ○ Standard`.
- Run-complete summary message reports green and yellow counts
  separately when any yellows are present.

## What didn't change

- Tab 1 (Observations / Raw) renders identically for green and yellow
  — the raw data is fine; only the retrieval failed.
- Non-RO behaviour is unchanged (still gray, still shows the
  no-occultation placeholder).
- The satellite (LEO) pipeline path is unchanged. Its `success` flag
  already covers the same situation at the event level.

---

# v3.4.4.2 — UI refinements

## Keyboard navigation in the result list

The result list now responds to up/down arrow keys (and Home/End/
PageUp/PageDown) the same way it responds to mouse clicks: as soon as
focus moves to a new row, the plots for that satellite/event load
immediately on the current tab. Previously the keyboard would move the
selection marker but the plots stayed on whatever the user had last
clicked, forcing an extra mouse click after every arrow press.

Implementation: switched the signal hookup from
`itemClicked(QListWidgetItem)` to
`currentItemChanged(current, previous)`. Both mouse selection and
keyboard navigation drive the same code path. Non-selectable
separator rows are skipped automatically by Qt, so arrow keys jump
straight from the last green row to the first non-RO row.

## "Yellow" rows blended with non-RO rows

The previous v3.4.4.1 rendering placed satellites that passed the RO
checks but produced no bending profile in their own visually
prominent block, with an `◐` marker, an amber `#F9A825` color, and a
`[RO • no profile]` suffix. That gave them too much weight in the
list: they aren't really a success category, they're just a different
flavour of "nothing to plot on tabs 2 and 3".

v3.4.4.2 merges them visually with the non-RO rows:

- **Position** — listed alongside the gray non-RO rows, after the
  separator that divides the list into "has derived data" and "doesn't".
- **Marker** — `○` (same as non-RO), no `[RO …]` suffix.
- **Color** — `#A1887F` (a faded warm gray-brown), distinguishable
  from `#757575` non-RO gray on close inspection but not competing
  with the green RO rows for the user's attention.

The data model is unchanged. Selecting a faded row still produces the
"Bending retrieval did not converge for *{sat_id}*" placeholder on
tabs 2 and 3 — the row's `ro_state` is still `'ro_empty'` and its
derived-tabs-enabled flag is still `False`. Only the visual
prominence in the sidebar changed.

Legend strings simplified accordingly:

- Ground-only: `● RO + profile    ○ No profile / no RO`
- Combined: `Ground: ● RO + profile / ○ no profile / no RO | Satellite: ●/○`
