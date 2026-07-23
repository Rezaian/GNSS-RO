# GNSS-RO Pipeline v3.4.7 — Changelog

GUI-only release on top of v3.4.6. No retrieval, inversion or file-format
behaviour changes — every edit is presentation. The trigger was a review of the
compiled Windows build running on an HD (1366x768) laptop rather than a
Full-HD screen: several controls were too small to read comfortably, the
data-directory path was truncating, and the window was being forced taller
than the desktop.

Two defects were found during the pass and are fixed here as well (Fix 6 and
Fix 7); both predate v3.4.7 and both were made more visible by the larger
fonts.

---

## Design note: sizes are now data, not scattered literals

Before this release, font sizes lived as magic numbers inside a dozen
different style-sheet strings. Re-tuning the GUI meant hunting them down.

v3.4.7 collects them into two blocks:

- `gnss_ro_gui.py` — a `v3.4.7 — UI SIZE TUNING` block near the top holds every
  main-window size as a named constant (`FS_SIDEBAR`, `FS_BROWSE`, …).
- `login_ui.py` — `LOGIN_FONT_SCALE` and `LOGIN_LOGO_SCALE` drive the whole
  login card. Setting both to `1.0` reproduces the v3.4.6 login exactly.
- `plot_style.py` (new) — `PLOT_BODY_BUMP` and `PLOT_TITLE_BUMP` drive every
  generated PNG. Setting both to `0` reproduces the v3.4.6 plots exactly.

All three are annotated with the v3.4.6 baseline value so the delta is
auditable.

---

## Fix 1: Login page — text +50%, logo 2x

**Change.** `LOGIN_FONT_SCALE = 1.5` and `LOGIN_LOGO_SCALE = 2.0`.

| Element | v3.4.6 | v3.4.7 |
|---|---|---|
| Title | 22 px | 33 px |
| Subtitle | 13 px | 20 px |
| Username / password | 14 px | 21 px |
| Sign In button | 15 px | 22 px |
| Error message | 12 px | 18 px |
| "Press Esc to exit" | 11 px | 16 px |
| Logo height | 72 px | 144 px |
| Card width | 360 px | 540 px |

Box heights (input fields, button) are scaled at *half* the font rate —
`1 + (scale - 1) * 0.5` — because a 21 px string does not need a 72 px tall
field, and the saved vertical space is what keeps the card on a short screen.
Inter-element spacings were retuned for the same reason (the gap under the
subtitle went 32 -> 24 px, the gap above the Esc hint 24 -> 14 px).

**Two guards were added.**

`_compute_scale()` reads the available screen height at construction time. If
the screen cannot host the full-scale card (~660 px plus margins), both scale
factors are reduced proportionally rather than letting the Sign In button fall
off the bottom. On a 1366x768 laptop with a normal taskbar (~720 px available)
no reduction occurs; on a 1280x720 panel it engages.

The logo guard: `scaledToHeight(144)` on a sufficiently wide logo would exceed
the card's inner width. The current asset is 500x250, which lands at 288 px
wide against 444 px of usable card — comfortable — but the code now falls back
to `scaledToWidth()` if a future logo would overflow.

**Measured result** at 1366x768: dialog 720x700, card 632 px tall, Sign In
button bottom edge at y=589.

---

## Fix 2: Main window — sidebar text +3 px

Requested: +2..4 px on the sidebar. The application base font is 12 pt, which
Windows renders at ~16 px at 96 DPI; that is the figure the deltas below are
measured against.

`FS_SIDEBAR = 19` is applied through a single object-name-scoped rule on the
sidebar container:

```
#sidebar QGroupBox   { font-size: 19px; font-weight: 600; }
#sidebar QLabel      { font-size: 19px; }
#sidebar QLineEdit   { font-size: 19px; }
#sidebar QCheckBox   { font-size: 19px; }
#sidebar QPushButton { font-size: 19px; }
```

Panels that must *not* grow set their own style sheet, which Qt resolves at
higher priority than an ancestor rule — see Fix 5.

`FS_SIDEBAR_BTN = 17` (was 14) covers Start Processing and Stop.

**Sidebar geometry.** The container was 300–350 px wide. At 19 px the longest
labels (`Altitude:`, and `Ref-sat elev thresh (°)` in Advanced Settings) no
longer fit, so it is now 340–420 px (`SIDEBAR_MIN_W` / `SIDEBAR_MAX_W`), and
the initial splitter position follows `SIDEBAR_MIN_W` instead of a hard-coded
320. Verified: the Station panel needs 286 px against 374 px available;
Advanced Settings expanded needs 378 px.

---

## Fix 3: Browse row smaller, browse notes larger

These two live in the same group box and moved in opposite directions.

**`FS_BROWSE = 13`** (was 16) — the read-only path `QLineEdit` and the Browse
button. Data-directory paths are long and were truncating mid-string on the HD
build; shrinking the field shows meaningfully more of the path. The Browse
button's max width went 70 -> 80 px so the smaller text stays centred, and it
picked up `padding: 4px 8px`.

**`FS_BROWSE_NOTE = 15`** (was 12) — the validation label underneath, which
carries the data-type tag line, the green `✓` info lines, the amber `⚠`
warnings and the red `✗` errors. This is the text an operator actually needs
to read before starting a run.

---

## Fix 4: Results box sub-notes +3 px

`FS_RESULT_NOTE = 14` (was 11) — the legend under the Results list, i.e. the
`● RO + profile  ○ No profile / no RO` line and its load-mode and
ground+satellite variants.

The list rows themselves (`FS_RESULT_LIST = 13`) are unchanged: they are
monospaced and column-aligned, and the request did not cover them.

---

## Fix 5: Processing Status and Advanced Settings pinned

"Processing panel fonts is ok" — so the Processing Status panel keeps its
v3.4.6 sizes exactly (`FS_STATUS = 16`, `FS_STATUS_DETAIL = 12`). Because it is
a child of the sidebar, leaving it alone was not sufficient: it would have
inherited the 19 px rule from Fix 2. It therefore sets its own style sheet,
and the four `setStyleSheet` calls that recolour the status label on state
changes (idle / running / complete / failed) now carry the size explicitly so
none of them silently drops back to the inherited value.

Advanced Settings is pinned the same way at `FS_ADVANCED = 16`. It is a dense
twelve-row form inside a 340–420 px sidebar and was not part of the request.
Raising the constant is a one-line change if wanted.

---

## Fix 6: Plot toolbar icons, and plot fonts

**Toolbar.** `TOOLBAR_ICON_PX = 18`, applied via
`NavigationToolbar.setIconSize()`. Matplotlib defaults to 24 px, which
dominated the plot panel on the HD build. Toolbar spacing tightened 6 -> 4 px
and padding 2 -> 1 px to match.

**Generated plots.** New `plot_style.py`, `+2 px` on body text and `+4 px` on
titles:

| | v3.4.6 | v3.4.7 |
|---|---|---|
| Figure suptitle | 14 | 18 |
| Subplot titles | 11 | 15 |
| Legends | 7–9 | 9–11 |
| In-plot annotations | 8–12 | 10–14 |
| Axis labels | *10 (implicit)* | 12 |
| Tick labels | *10 (implicit)* | 12 |

The last two rows are the important ones. Axis and tick labels had **no**
explicit `fontsize` anywhere in either pipeline, so they sat at matplotlib's
default of 10 and no amount of editing `fontsize=` arguments would have moved
them. `apply_plot_fonts()` sets them through `rcParams`.

It is called at each of the five plotting entry points rather than once at
import, because the GUI runs the pipelines under `multiprocessing` with the
`spawn` start method and `rcParams` are per-process. It is idempotent and
cheap.

---

## Fix 7: Window minimum size exceeded the desktop

**Problem.** `MainWindow.__init__` called `setMinimumSize(1200, 800)`. On a
1366x768 laptop the usable height after the taskbar is roughly 720 px, so Qt
was forced to make the window 80+ px taller than the desktop. The bottom of
the sidebar — the Results list and its legend — sat under the taskbar and
could not be reached. This is a plausible contributor to the original report.

**Fix.** The minimum is clamped to the available geometry
(`min(800, max(620, avail.height() - 40))`), the initial size is clamped the
same way, and on screens shorter than 800 px the window opens maximised. The
maximise call is deferred until after `_setup_ui()` so an empty window is
never flashed. The whole block is wrapped in `try/except` with a conservative
1100x640 fallback if the screen cannot be queried.

---

## Fix 8: Word-wrapped notes clipped their last line

**Problem.** A word-wrapped `QLabel` reports a *single-line* `minimumSizeHint`.
A `QVBoxLayout` therefore under-allocates vertical space and silently clips the
bottom of the wrapped text. Measured on the validation label: 86 px of content
in 69 px of allocated height.

This is a v3.4.6 defect, not a regression — but it was near-invisible at 11–12
px and became obvious once Fix 3 and Fix 4 enlarged the same two labels.

**Fix.** New `WrapLabel(QLabel)` overrides `minimumSizeHint` and `sizeHint` to
return `heightForWidth(width())`, and calls `updateGeometry()` on `setText` and
`resizeEvent`. Applied to the validation label and the results legend.
Measured after the fix: 69 px of content in 69 px allocated.

---

## Fix 9: Ground data labelled "Mountain" in the browse panel

New constant `GROUND_DISPLAY_NAME = "Mountain"`. Four browse-panel tag lines
now read from it:

- `📡 Mountain-based data` (was `Ground-based data`)
- `📡🛰 Mountain + Satellite data`
- `📡 Mountain project` (load mode)
- `📡🛰 Mountain + Satellite project` (load mode)

**Deliberately unchanged:** the `DataType.GROUND` constant, the
`ground_gnss_ro_pipeline` module name, the `ground_dir` output folder, the
`Station Configuration (Ground)` group-box title, and the `═══ GROUND ═══`
section header in the combined results list. Only the four browse-panel tag
lines were in scope. Changing the constant relabels all four at once if the
rename should be extended.

---

## Build changes

`qt_compat.py` now re-exports `QSize` (needed for `setIconSize`) from both the
PyQt6 and PyQt5 branches.

`.github/workflows/build.yml` gains three hidden imports on **both** jobs:

- `plot_style` — new module; the frozen build would fail at plot time without it.
- `matplotlib.backends.backend_qtagg` — **this was a latent bug.** The Windows
  job declared `backend_qt5agg`, but `gnss_ro_gui.py` calls
  `matplotlib.use('QtAgg')` and imports `backend_qtagg`. The two are different
  modules. PyInstaller's static analysis does find `backend_qtagg` from the
  top-level import, so the build was not broken, but the declared hidden import
  was pointing at the wrong module and gave no protection. Both are now listed.

---

## Files touched

| File | Change |
|---|---|
| `login_ui.py` | Scale-driven login card (Fix 1) |
| `gnss_ro_gui.py` | UI tuning block, `WrapLabel`, window sizing, Mountain rename (Fixes 2–9) |
| `plot_style.py` | **New** — plot font scaling (Fix 6) |
| `ground_gnss_ro_pipeline.py` | `apply_plot_fonts()` at 3 entry points; `fontsize=` bumps |
| `sat_gnss_ro_pipeline.py` | `apply_plot_fonts()` at 3 entry points; `fontsize=` bumps |
| `qt_compat.py` | Export `QSize` |
| `.github/workflows/build.yml` | Hidden imports |
| `README.md` | Version table |

No changes to `rinex_parser.py` or to any retrieval, inversion or I/O code
path.

---

## Verification

Rendered headless under Xvfb at exactly 1366x768 with PyQt5 (the Windows
build's binding) and Fusion style. Every size in the tables above was read back
from the live widgets via `fontInfo().pixelSize()` rather than assumed from the
style sheets. Layouts were checked with Advanced Settings both collapsed and
expanded, and with the Results list populated in tri-state ground mode.

Not covered by this verification: DPI scaling above 100%, and the appearance of
the true PyQt6 macOS build.
