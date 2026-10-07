#!/usr/bin/env python3
"""
GNSS Radio Occultation Processing GUI  3.5.2
==============================================

Supports both ground-based and satellite-based (LEO) GNSS-RO processing.

Features:
- Auto-detection of data type (ground/satellite/both)
- Ground: UBX, SP3, metadata.cra, optional ERA5 validation
- Satellite: conPhs files, optional atmPrf/wetPf2 validation
- Progress tracking with stop capability
- Results visualization per satellite/event
"""

import sys
import os

__version__ = "3.5.2"
import re
import json
import glob
from datetime import datetime
from typing import Dict, Optional, Any, List

# ============================================================================
# FROZEN-BUILD IMPORT PATH
# ============================================================================
# In a PyInstaller bundle the sibling modules (qt_compat, login_ui,
# plot_style, the two pipelines, rinex_parser) are imported through
# PyInstaller's FrozenImporter and normally resolve without help.  Putting
# _MEIPASS on sys.path explicitly is a cheap safety net for two cases where
# that has been observed to be incomplete on Windows:
#
#   - --onefile builds where a module is reached only from a lazy, function
#     level import (plot_style is imported inside the plotting functions);
#   - the 'spawn' multiprocessing children, which re-launch the executable
#     and rebuild their own sys.path from scratch.
#
# It is a no-op when running from source.
# ============================================================================

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    if sys._MEIPASS not in sys.path:
        sys.path.insert(0, sys._MEIPASS)

from qt_compat import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QTabWidget, QProgressBar,
    QSplitter, QMessageBox, Qt, QTimer, QColor, QFont, QSize,
    QCheckBox, QFormLayout, QDoubleSpinBox, QSpinBox, QScrollArea,
    QToolButton, QSizePolicy, QFrame,
    exec_app
)

import pandas as pd
import numpy as np

from ground_gnss_ro_pipeline import (
    StationConfig, PipelineConfig as GroundPipelineConfig, ProcessingResult,
    evaluate_ro_status, generate_raw_plots, generate_derived_plots,
    generate_atmospheric_plots,
    parse_gnss_directory, match_observations_with_sp3,
    calculate_accurate_elevations, calculate_geometric_doppler,
    apply_single_differencing, retrieve_bending_angles,
    retrieve_refractivity, compare_with_era5, retrieve_atmospheric_profile,
    PROCESSING_DEFAULTS, load_processing_config_from_cra, apply_processing_config,
    extract_station_info,
    session_quality, session_quality_lines, load_ro_checks,
)

from sat_gnss_ro_pipeline import (
    LEOROPipeline, PipelineConfig as SatPipelineConfig
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import multiprocessing as mp
from multiprocessing import Process, Queue
from login_ui import LoginDialog

# 3.5.2 — QMenu (Recent folders) from the SAME Qt binding qt_compat uses
QMenu = None
try:
    from qt_compat import QMenu                      # if qt_compat exports it
except ImportError:
    try:
        QMenu = __import__(QWidget.__module__, fromlist=['QMenu']).QMenu
    except Exception:
        QMenu = None

# 3.5.2 — remembered folders (last + recent), per user
PREFS_FILE = os.path.join(os.path.expanduser('~'), '.gnss_ro_gui.json')


def _load_prefs() -> Dict[str, Any]:
    try:
        with open(PREFS_FILE) as f:
            p = json.load(f)
        return p if isinstance(p, dict) else {}
    except Exception:
        return {}


def _remember_dir(directory: str):
    p = _load_prefs()
    rec = [d for d in p.get('recent', []) if d != directory and os.path.isdir(d)]
    p['recent'] = [directory] + rec[:7]
    p['last_dir'] = directory
    try:
        with open(PREFS_FILE, 'w') as f:
            json.dump(p, f, indent=1)
    except OSError:
        pass


# ============================================================================
# v3.4.7 — UI SIZE TUNING (single place to adjust every GUI font)
# ============================================================================
# All values are CSS pixels as used by Qt style sheets.  The application base
# font is 12 pt, which Windows renders as ~16 px at 96 DPI — that is the
# "current" figure the deltas below are measured against.
#
# Adjusted in v3.4.7 after review of the compiled Windows build on a 1366x768
# (non-Full-HD) laptop:
#
#   sidebar        16 -> 19  (+3)   requested +2..4
#   browse panel   16 -> 13  (-3)   requested -2..4   (path field + Browse)
#   browse notes   12 -> 15  (+3)   requested +2..4   (validation / warnings)
#   result legend  11 -> 14  (+3)   requested +2..4   (RO / profile sub-notes)
#   sidebar button 14 -> 17  (+3)   part of the sidebar
#   status panel   unchanged        "processing panel fonts is ok"
#   toolbar icons  24 -> 18         "plot panel icon size decreased"
# ============================================================================

FS_SIDEBAR        = 19   # group titles, labels and fields in the left sidebar
FS_SIDEBAR_BTN    = 17   # Start Processing / Stop buttons
FS_BROWSE         = 13   # data-directory path field and Browse button
FS_BROWSE_NOTE    = 15   # validation / warning notes under the Browse row
FS_RESULT_LIST    = 13   # rows in the Results list (unchanged)
FS_RESULT_NOTE    = 14   # legend / sub-notes under the Results list
FS_STATUS         = 16   # Processing Status panel  (unchanged by request)
FS_STATUS_DETAIL  = 12   # Processing Status detail line (unchanged by request)
FS_ADVANCED       = 16   # Advanced Settings form (unchanged; dense layout)
FS_TABS           = 13   # plot tab bar (unchanged)
FS_PLOT_PLACEHOLD = 15   # "Select an item ..." text drawn on an empty canvas

TOOLBAR_ICON_PX   = 18   # matplotlib navigation toolbar icons (default 24)

# Sidebar geometry — widened to absorb the larger sidebar font without
# clipping the "Altitude:" / "Ref-sat elev thresh" labels.
SIDEBAR_MIN_W     = 340
SIDEBAR_MAX_W     = 420

# ----------------------------------------------------------------------------
# v3.4.7 — display name for ground-based data in the browse panel.
# The processing code, file names and output folders are untouched; this only
# changes what the operator reads in the Data Directory tag line.
# ----------------------------------------------------------------------------
GROUND_DISPLAY_NAME = "Mountain"


# ============================================================================
# DATA TYPE DETECTION
# ============================================================================

class DataType:
    NONE = 0
    GROUND = 1
    SATELLITE = 2
    BOTH = 3

def _wrap_reason(reason: str, width: int = 60) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(reason or "no reason recorded", width))


def geodetic_to_ecef_gui(lat_deg: float, lon_deg: float, h_m: float):
    from ground_gnss_ro_pipeline import geodetic_to_ecef
    return geodetic_to_ecef(lat_deg, lon_deg, h_m)


def scan_input_directory(directory: str) -> Dict[str, Any]:
    """
    Scan directory to detect data type and required files.
    
    Ground data: (.ubx OR .rnx) + .sp3 + metadata.cra, optional ERA5 .nc
    Satellite data: conPhs_* files, optional atmPrf_*/wetPf2_* for validation
    """
    result = {
        'valid': False,
        'data_type': DataType.NONE,
        # Ground-specific
        'ubx_dir': None,
        'sp3_file': None,
        'era5_file': None,
        'metadata_file': None,
        'obs_source': None,  # NEW: 'UBX' or 'RNX'
        # Satellite-specific
        'conphs_files': [],
        'has_atmprf': False,
        'has_wetpf2': False,
        # Messages
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    if not os.path.isdir(directory):
        result['errors'].append("Invalid directory path")
        return result
    
    has_ground = False
    has_satellite = False
    
    # === Check for GROUND data ===
    ubx_files = glob.glob(os.path.join(directory, '*.[uU][bB][xX]'))
    
    # NEW: Check for RINEX files
    rnx_patterns = ['*.rnx', '*.RNX', '*.[0-9][0-9]o', '*.[0-9][0-9]O', 
                    '*.obs', '*.OBS', '*_MO.rnx', '*_MO.RNX']
    rnx_files = []
    for pattern in rnx_patterns:
        rnx_files.extend(glob.glob(os.path.join(directory, pattern)))
    rnx_files = list(set(rnx_files))
    
    sp3_files = glob.glob(os.path.join(directory, '*.[sS][pP]3'))
    metadata_files = glob.glob(os.path.join(directory, '[mM][eE][tT][aA][dD][aA][tT][aA].[cC][rR][aA]'))
    
    # Ground requires: (UBX or RNX) + SP3 + metadata
    has_obs_files = ubx_files or rnx_files
    
    # Ground requires: (UBX or RNX) + SP3 + optionally metadata.cra
    # metadata.cra is optional if RINEX files contain APPROX POSITION XYZ
    if has_obs_files and sp3_files:
        has_ground = True
        result['ubx_dir'] = directory
        result['sp3_file'] = sp3_files[0]
        result['metadata_file'] = metadata_files[0] if metadata_files else None
        
        # Determine observation source
        if ubx_files:
            result['obs_source'] = 'UBX'
            result['n_obs'] = len(ubx_files)
            result['info'].append(f"Ground data: {len(ubx_files)} UBX files")
        else:
            result['obs_source'] = 'RNX'
            result['n_obs'] = len(rnx_files)
            result['info'].append(f"Ground data: {len(rnx_files)} RINEX files")
        
        # If both exist, note UBX will be preferred
        if ubx_files and rnx_files:
            result['info'].append(f"Note: {len(rnx_files)} RINEX files also found (UBX preferred)")
        
        if not metadata_files:
            result['info'].append("No metadata.cra — station from the receiver (UBX 3D fix / RINEX header), "
                                  "settings from defaults; metadata.cra will be created on run")
        
        if len(sp3_files) > 1:
            result['warnings'].append(f"Multiple SP3 files — using {os.path.basename(sp3_files[0])}")
        
        # ERA5 for ground validation
        nc_files = glob.glob(os.path.join(directory, '*.[nN][cC]'))
        era5_candidates = [f for f in nc_files if 'era5' in os.path.basename(f).lower() 
                          or not any(x in os.path.basename(f).lower() for x in ['atmprf', 'wetpf', 'conphs'])]
        if era5_candidates:
            result['era5_file'] = era5_candidates[0]
            result['info'].append(f"Validation .nc: {os.path.basename(era5_candidates[0])}")
        else:
            result['warnings'].append("No .nc file — ground validation limited")
    
    # === Check for SATELLITE data ===
    conphs_files = sorted(
        glob.glob(os.path.join(directory, "conPhs_*.nc")) +
        glob.glob(os.path.join(directory, "conPhs_*_nc"))
    )
    
    if conphs_files:
        has_satellite = True
        result['conphs_files'] = conphs_files
        result['info'].append(f"Satellite data: {len(conphs_files)} conPhs files")
        
        atmprf_files = glob.glob(os.path.join(directory, "atmPrf_*"))
        wetpf2_files = glob.glob(os.path.join(directory, "wetPf2_*"))
        
        if atmprf_files:
            result['has_atmprf'] = True
            result['info'].append(f"atmPrf validation: {len(atmprf_files)} files")
        if wetpf2_files:
            result['has_wetpf2'] = True
            result['info'].append(f"wetPf2 validation: {len(wetpf2_files)} files")
        
        if not atmprf_files and not wetpf2_files:
            result['warnings'].append("No atmPrf/wetPf2 — satellite validation limited")
    
    # === Determine data type ===
    if has_ground and has_satellite:
        result['data_type'] = DataType.BOTH
        result['valid'] = True
    elif has_ground:
        result['data_type'] = DataType.GROUND
        result['valid'] = True
    elif has_satellite:
        result['data_type'] = DataType.SATELLITE
        result['valid'] = True
    else:
        result['errors'].append("No valid GNSS-RO data found")
        result['errors'].append("Ground requires: (.ubx OR .rnx) + .sp3 (metadata.cra optional: station is read from the receiver data)")
        result['errors'].append("Satellite requires: conPhs_* files")
    
    return result

CRA_PARSE_NOTES: Dict[str, str] = {}


def load_metadata(filepath: str) -> Optional[Dict]:
    """
    Read metadata.cra (JSON). v4.5: tolerant of the usual hand-editing slips -
    an emptied value ("STATION_LAT": ,) or a trailing comma - so a half-filled
    file still yields its station name and settings. A repaired read is noted
    in CRA_PARSE_NOTES[filepath] for the GUI to show.
    """
    CRA_PARSE_NOTES.pop(filepath, None)
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            text = f.read()
    except Exception:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fixed = re.sub(r':\s*(?=[,}\]])', ': null', text)          # "KEY": ,   ->  "KEY": null,
    fixed = re.sub(r',\s*(?=[}\]])', '', fixed)                # trailing commas
    try:
        data = json.loads(fixed)
        CRA_PARSE_NOTES[filepath] = "metadata.cra is not valid JSON (empty value or trailing comma) - read it after a repair"
        return data
    except Exception:
        name = re.search(r'"STATION_NAME"\s*:\s*"([^"]*)"', text)
        CRA_PARSE_NOTES[filepath] = "metadata.cra could not be read as JSON - only the station name was recovered"
        return {'STATION_NAME': name.group(1)} if name else None


def save_metadata(filepath: str, data: Dict) -> bool:
    """Write a metadata dict to .cra. Used when no prior file exists."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception:
        return False


# Keys written by GUI <= v3.4.7 under names the pipeline does not know; they
# never had any effect and are purged on the next save.
LEGACY_PROCESSING_KEYS = (
    'POLY_SMOOTH_WINDOW_S', 'POLYFIT_GAP_THRESHOLD_S', 'RO_ELEVATION_THRESHOLD_DEG',
    'RO_DOPPLER_THRESHOLD_HZ', 'REF_SAT_ELEVATION_THRESHOLD_DEG', 'REF_SAT_JUMP_THRESHOLD_HZ',
)


def merge_save_metadata(filepath: str, station_fields: Dict,
                        processing_fields: Optional[Dict] = None) -> bool:
    """
    v3.4.4 — non-destructive .cra writer.

    Reads the existing file (if any), updates only the given station and
    PROCESSING fields, and writes the merged dict back. Any other keys the
    user has in their .cra are left untouched.

    Before v3.4.4 the pipeline overwrote .cra with just the four station
    fields on every run, which wiped out user-defined advanced settings.
    """
    existing = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}

    # Update station fields (top-level)
    if station_fields:
        for k, v in station_fields.items():
            existing[k] = v

    # Merge into existing PROCESSING block — never replace it wholesale.
    if processing_fields:
        proc = existing.get('PROCESSING') or {}
        if not isinstance(proc, dict):
            proc = {}
        for k, v in processing_fields.items():
            proc[k] = v
        # v4.3: remove keys written by older GUI versions that the pipeline never
        # read (e.g. POLY_SMOOTH_WINDOW_S). Their real counterparts are kept.
        for k in LEGACY_PROCESSING_KEYS:
            proc.pop(k, None)
        existing['PROCESSING'] = proc

    try:
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=4)
        return True
    except Exception:
        return False


# ============================================================================
# v3.4.4 — Output / project detection for "Open previous results"
# ============================================================================

def detect_output_directory(directory: str) -> Dict[str, Any]:
    """
    Inspect ``directory`` to decide whether it contains the artifacts of a
    previous run (ground and/or satellite). Used by the GUI to switch into
    load-mode when the user picks an existing ``*_output`` folder.

    Layout expected (any subset may be present):
        <root>/ground/step4_differenced.csv
        <root>/ground/plots/<sat_id>_raw.png
        <root>/ground/bending/<sat_id>_bending.csv
        <root>/ground/refractivity/<sat_id>_refractivity.csv
        <root>/ground/atmospheric/<sat_id>_atmospheric.csv
        <root>/ground/comparison/<sat_id>_comparison.csv
        <root>/satellite/processing_summary.csv
        <root>/satellite/<event_id>/plots/<event_id>_panel1_raw.png

    Some installations write the ground artifacts directly into ``<root>``
    rather than ``<root>/ground``, so both layouts are accepted.
    """
    out = {
        'is_output': False,
        'data_type': DataType.NONE,
        'ground_dir': None,
        'sat_dir': None,
        'warnings': [],
        'errors': [],
    }
    if not os.path.isdir(directory):
        out['errors'].append("Invalid directory path")
        return out

    # --- ground detection ---------------------------------------------------
    ground_candidates = [
        os.path.join(directory, 'ground'),
        directory,  # flat layout
    ]
    for cand in ground_candidates:
        step4 = os.path.join(cand, 'step4_differenced.csv')
        plots_dir = os.path.join(cand, 'plots')
        if os.path.exists(step4) or os.path.isdir(plots_dir):
            out['ground_dir'] = cand
            break

    # --- satellite detection ------------------------------------------------
    # Only the canonical 'satellite/' subdir is considered when we already
    # claimed the ground layout — otherwise we'd wrongly classify the ground
    # directory's siblings as satellite events.
    if out['ground_dir'] is not None and out['ground_dir'] == directory:
        # Flat ground layout — no room for a satellite section alongside.
        sat_candidates = []
    else:
        sat_candidates = [os.path.join(directory, 'satellite')]

    for cand in sat_candidates:
        if not os.path.isdir(cand):
            continue
        summary = os.path.join(cand, 'processing_summary.csv')
        if os.path.exists(summary):
            out['sat_dir'] = cand
            break
        # Or: any conPhs-style event folders with plot subdirs
        try:
            for entry in os.listdir(cand):
                p = os.path.join(cand, entry, 'plots')
                if os.path.isdir(p):
                    out['sat_dir'] = cand
                    break
        except OSError:
            pass
        if out['sat_dir']:
            break

    if out['ground_dir'] and out['sat_dir']:
        out['data_type'] = DataType.BOTH
        out['is_output'] = True
    elif out['ground_dir']:
        out['data_type'] = DataType.GROUND
        out['is_output'] = True
    elif out['sat_dir']:
        out['data_type'] = DataType.SATELLITE
        out['is_output'] = True

    # Soft warnings about missing artifacts
    if out['ground_dir']:
        step4 = os.path.join(out['ground_dir'], 'step4_differenced.csv')
        plots_dir = os.path.join(out['ground_dir'], 'plots')
        if not os.path.exists(step4):
            out['warnings'].append("Ground: step4_differenced.csv missing — RO status cannot be reconstructed")
        if not os.path.isdir(plots_dir):
            out['warnings'].append("Ground: plots/ directory missing — only placeholders will be shown")

    if out['sat_dir']:
        summary = os.path.join(out['sat_dir'], 'processing_summary.csv')
        if not os.path.exists(summary):
            out['warnings'].append("Satellite: processing_summary.csv missing — event list cannot be reconstructed")

    return out


# v3.4.4.1 — Tri-state RO classification: green / yellow / no-RO.
# A satellite can pass the RO checks (evaluate_ro_status returns True) but
# still produce an empty bending profile (no fsolve convergence, all-NaN
# columns, etc). In that case there's nothing to plot on tabs 2 & 3, so we
# downgrade it from green to yellow and disable those tabs for it.

RO_OK = 'ro_ok'        # green — RO + usable bending data
RO_EMPTY = 'ro_empty'  # yellow — RO but bending profile is empty/diverged
NO_RO = False          # gray  — failed the RO checks (legacy False value)


def _has_usable_bending_data(bending_csv: str) -> bool:
    """
    Return True iff the per-satellite bending CSV contains at least one
    finite row in a recognised bending-angle column. Used to distinguish
    green (RO + data) from yellow (RO + diverged retrieval).
    """
    if not bending_csv or not os.path.exists(bending_csv):
        return False
    try:
        df = pd.read_csv(bending_csv)
    except Exception:
        return False
    if df.empty:
        return False
    # Any of these columns being finite is enough to call the retrieval
    # successful. We don't require all of them — single-frequency bending
    # alone is still plottable.
    candidates = ['bending_angle_rad', 'bending_L1', 'bending_L2']
    for col in candidates:
        if col in df.columns:
            try:
                vals = pd.to_numeric(df[col], errors='coerce')
                if np.isfinite(vals).any():
                    return True
            except Exception:
                continue
    return False


def classify_ground_ro_status(ground_dir: str,
                              ro_status_bool: Dict[str, bool]) -> Dict[str, Any]:
    """
    Upgrade a {sat_id: bool} RO map to a tri-state {sat_id: 'ro_ok'|'ro_empty'|False}
    map by inspecting whether each RO satellite produced usable bending data
    in <ground_dir>/bending/<sat_id>_bending.csv.

    Non-RO sats stay as False; RO sats become 'ro_ok' or 'ro_empty'.
    """
    out: Dict[str, Any] = {}
    bending_dir = os.path.join(ground_dir, 'bending')
    for sat_id, is_ro in (ro_status_bool or {}).items():
        if not is_ro:
            out[sat_id] = NO_RO
            continue
        bending_csv = os.path.join(bending_dir, f'{sat_id}_bending.csv')
        out[sat_id] = RO_OK if _has_usable_bending_data(bending_csv) else RO_EMPTY
    return out


RO_OK_1F = 'ro_ok_1f'  # amber — RO + profile, single frequency (no ionospheric correction)


def build_ground_status(ground_dir: str, step4_df: Optional[pd.DataFrame] = None):
    """
    v4.3: results-list state per item, read from what step 5 wrote.

    Returns (status, reasons):
      status  {item_id: 'ro_ok' | 'ro_ok_1f' | 'ro_empty' | False}
              item_id is the occultation event id (GPS_7 or GPS_7_e2) for events,
              the satellite id otherwise
      reasons {item_id: text} for 'ro_empty' rows (why no profile)
    Falls back to the legacy evaluation for outputs written by older versions.
    """
    status: Dict[str, Any] = {}
    reasons: Dict[str, str] = {}
    bending_dir = os.path.join(ground_dir, 'bending')
    summary_csv = os.path.join(bending_dir, 'summary.csv')
    skipped_csv = os.path.join(bending_dir, 'skipped.csv')
    if step4_df is None:
        step4 = os.path.join(ground_dir, 'step4_differenced.csv')
        step4_df = pd.read_csv(step4, usecols=lambda c: c in ('sat_id', 'gnssId', 'svId')) \
            if os.path.exists(step4) else pd.DataFrame()
    sats = []
    if not step4_df.empty:
        if 'sat_id' not in step4_df.columns:
            step4_df = step4_df.assign(sat_id=step4_df['gnssId'].astype(str) + '_' + step4_df['svId'].astype(str))
        sats = list(step4_df['sat_id'].unique())

    if not os.path.exists(skipped_csv):           # output from an older version
        try:
            return classify_ground_ro_status(ground_dir, evaluate_ro_status(step4_df)), {}
        except Exception:
            return {s: NO_RO for s in sats}, {}

    with_event = set()
    if os.path.exists(summary_csv):
        try:
            summ = pd.read_csv(summary_csv)
        except Exception:
            summ = pd.DataFrame()
        for _, r in summ.iterrows():
            ev = str(r['sat_id'])
            ok = _has_usable_bending_data(os.path.join(bending_dir, f'{ev}_bending.csv'))
            single = str(r.get('freq_mode', 'dual')) == 'single'
            status[ev] = (RO_OK_1F if single else RO_OK) if ok else RO_EMPTY
            if not ok:
                reasons[ev] = 'bending file has no finite values'
            with_event.add(ev.split('_e')[0] if '_e' in ev else ev)
    try:
        skp = pd.read_csv(skipped_csv)
    except Exception:
        skp = pd.DataFrame(columns=['sat_id', 'reason'])
    for _, r in skp.iterrows():
        sid = str(r['sat_id'])
        status[sid] = RO_EMPTY
        reasons[sid] = str(r['reason'])
        with_event.add(sid.split('_e')[0] if '_e' in sid else sid)
    for sid in sats:
        if sid not in with_event and sid not in status:
            status[sid] = NO_RO
    return status, reasons


def ground_hidden_items(ground_dir: Optional[str], status: Dict[str, Any]) -> set:
    """3.5.2: items that never went below the RO elevation threshold (test 1) -
    hidden from the Results list unless 'Show all' is ticked."""
    hidden = set()
    if not ground_dir:
        return hidden
    bdir = os.path.join(ground_dir, 'bending')
    for k, v in status.items():
        if _norm_ground_state(v) != 'no_ro':
            continue
        info = load_ro_checks(bdir, k)
        f = next((c for c in (info or {}).get('checks', []) if c.get('passed') is False), None)
        if f and f.get('name') == 'Tracked below 5° elevation':
            hidden.add(k)
    return hidden


def load_ground_ro_status_from_csv(ground_dir: str) -> Dict[str, Any]:
    """Reconstruct {sat_id: ro_state} from saved artefacts.

    v3.4.4.1: returns tri-state values ('ro_ok' | 'ro_empty' | False) instead
    of plain bools. The yellow state requires the per-sat bending CSV to be
    inspectable on disk, which is exactly what load-mode has access to.
    """
    step4 = os.path.join(ground_dir, 'step4_differenced.csv')
    if not os.path.exists(step4):
        return {}
    try:
        return build_ground_status(ground_dir)[0]
    except Exception:
        return {}


def load_sat_summary_from_csv(sat_dir: str) -> Optional[pd.DataFrame]:
    """Read a previously written satellite processing_summary.csv."""
    summary = os.path.join(sat_dir, 'processing_summary.csv')
    if not os.path.exists(summary):
        return None
    try:
        return pd.read_csv(summary)
    except Exception:
        return None


def cleanup_intermediate_csvs(ground_dir: str) -> int:
    """
    v3.4.4 — remove redundant intermediate CSVs after a successful run.

    Every column from step1/step2/step3a/step3b is preserved in
    step4_differenced.csv, so the earlier files are redundant. We keep
    step4 (the consolidated output) and remove the rest.

    Returns the number of files removed.
    """
    removed = 0
    candidates = [
        'step1_observations.csv',
        'step2_matched.csv',
        'step3a_elevations.csv',
        'step3b_doppler.csv',
    ]
    for name in candidates:
        path = os.path.join(ground_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


# ============================================================================
# GROUND PIPELINE PROCESS
# ============================================================================

def run_ground_pipeline(station_dict: dict, ubx_dir: str, sp3_file: str,
                        era5_file: str, output_dir: str, progress_queue: Queue,
                        processing_cfg: Optional[dict] = None,
                        keep_intermediate: bool = False):
    """Ground-based pipeline in a separate process.

    v4.3: runs the pipeline's own GNSSROPipeline.run_full_pipeline instead of a
    copy of the steps, so the GUI gets everything the pipeline does: receiver
    position (UBX 3D fix / RINEX header) unless FORCE_CRA_STATION_COORDS,
    station n_r and pressure from ERA5 when no met data, Fresnel-adaptive
    smoothing, occultation events, ERA5 at the occultation time, the station
    barometer as the step-7 boundary, and single-frequency events when enabled.
    ``processing_cfg`` is the complete PROCESSING dict (defaults + .cra).
    """
    import os
    from datetime import datetime
    from ground_gnss_ro_pipeline import (
        StationConfig, PipelineConfig, GNSSROPipeline,
        generate_raw_plots, generate_derived_plots, generate_atmospheric_plots,
        apply_processing_config,
    )

    if processing_cfg:
        apply_processing_config(processing_cfg)

    log_lines = []
    last = {'frac': 0.0}

    def log(message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines.append(f"[{ts}] {message}")

    def progress(message: str, fraction):
        if fraction is not None:
            last['frac'] = float(fraction)
        log(message)
        progress_queue.put(('progress', 'ground', message, last['frac'] * 0.9))

    def write_log():
        try:
            with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
                f.write('\n'.join(log_lines))
        except OSError:
            pass

    try:
        os.makedirs(output_dir, exist_ok=True)
        station = StationConfig(**station_dict)
        log("=" * 60)
        log("GROUND-BASED GNSS-RO Pipeline")
        log("=" * 60)
        log(f"Station (GUI): {station.name} {station.latitude:.6f}N {station.longitude:.6f}E GPS height {station.altitude:.1f} m")
        if processing_cfg:
            log("PROCESSING: " + ", ".join(f"{k}={v}" for k, v in sorted(processing_cfg.items())))

        pipe = GNSSROPipeline(station, PipelineConfig())
        results = pipe.run_full_pipeline(ubx_dir, sp3_file, era5_file, output_dir, progress)
        station = pipe.station                        # the station actually used
        for k, r in results.items():
            log(f"{'OK ' if r.success else 'ERR'} {k}: {r.message}")
        s5 = results.get('step5')
        if s5 is not None:
            for sid, why in (s5.metadata or {}).get('skipped', {}).items():
                log(f"    no profile {sid}: {why}")

        if not results.get('step1') or not results['step1'].success:
            write_log()
            progress_queue.put(('done', 'ground', False, "Observation file parsing failed", None))
            return
        diff_csv = os.path.join(output_dir, 'step4_differenced.csv')
        if not os.path.exists(diff_csv):
            write_log()
            msg = next((r.message for r in results.values() if not r.success), "pipeline stopped before step 4")
            progress_queue.put(('done', 'ground', False, msg, None))
            return

        # ---- plots: raw per satellite, derived/atmospheric per occultation event
        progress("Generating plots...", 0.95)
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        df_plot = pd.read_csv(diff_csv)
        if 'sat_id' not in df_plot.columns:
            df_plot['sat_id'] = df_plot['gnssId'].astype(str) + '_' + df_plot['svId'].astype(str)
        from ground_gnss_ro_pipeline import session_sample_rate
        site = {'name': station_dict.get('name') or '', 'lat': station.latitude, 'lon': station.longitude,
                'height_ell_m': station.altitude,
                'height_msl_m': station.altitude - getattr(station, 'geoid_sep_m', 0.0),
                'rate': session_sample_rate(df_plot['timestamp'])}
        if 'obs_interval_s' in df_plot.columns:                  # 3.5.2: recorded rate (RINEX thinned to 1 Hz)
            _iv = pd.to_numeric(df_plot['obs_interval_s'], errors='coerce').median()
            site['rate']['recorded_hz'] = float(1.0 / _iv) if np.isfinite(_iv) and _iv > 0 else np.nan
        log(f"Session sample rate {site['rate']['rate_hz']:.3f} Hz processed"
            + (f" ({site['rate']['recorded_hz']:.0f} Hz recorded)" if np.isfinite(site['rate'].get('recorded_hz', np.nan)) else '')
            + f", {site['rate']['n_gaps']} gap(s) excluded")
        for sat_id, sat_data in df_plot.groupby('sat_id'):
            site_sat = dict(site)
            if {'sta_lat', 'sta_lon', 'sta_h'} <= set(sat_data.columns):      # this satellite's own file fix
                med = lambda c: float(pd.to_numeric(sat_data[c], errors='coerce').median())
                geo = med('sta_geoid') if 'sta_geoid' in sat_data.columns else 0.0
                geo = 0.0 if not np.isfinite(geo) else geo
                site_sat.update({'lat': med('sta_lat'), 'lon': med('sta_lon'), 'height_msl_m': med('sta_h') - geo})
            generate_raw_plots(sat_data, sat_id, os.path.join(plots_dir, f'{sat_id}_raw.png'), site=site_sat)
        summary = s5.data if (s5 is not None and s5.data is not None) else pd.DataFrame()
        for _, row in summary.iterrows():
            ev = row['sat_id']
            sat_results = {
                'bending_csv': os.path.join(output_dir, 'bending', f'{ev}_bending.csv'),
                'refrac_csv': os.path.join(output_dir, 'refractivity', f'{ev}_refractivity.csv'),
                'comp_csv': os.path.join(output_dir, 'comparison', f'{ev}_comparison.csv'),
                'atm_csv': os.path.join(output_dir, 'atmospheric', f'{ev}_atmospheric.csv'),
            }
            generate_derived_plots(sat_results, ev, os.path.join(plots_dir, f'{ev}_derived.png'),
                                   station_altitude=station.altitude, site=site)
            generate_atmospheric_plots(sat_results, ev, os.path.join(plots_dir, f'{ev}_atmospheric.png'),
                                       station_altitude=station.altitude)

        # 3.5.2: data-collection quality of the session (GUI card + report)
        try:
            q = session_quality(df_plot)
            with open(os.path.join(output_dir, 'session_quality.json'), 'w') as f:
                json.dump(q, f, indent=1, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
        except Exception as e:
            log(f"session quality not written: {e}")

        # v4.6: every item without a profile gets a Radio Occultation page listing the RO tests
        from ground_gnss_ro_pipeline import generate_ro_checklist_plot, load_ro_checks
        try:
            status_, reasons_ = build_ground_status(output_dir, df_plot)
        except Exception:
            status_, reasons_ = {}, {}
        bdir = os.path.join(output_dir, 'bending')
        for item, state in status_.items():
            if _norm_ground_state(state) in ('ro_empty', 'no_ro'):
                generate_ro_checklist_plot(item, load_ro_checks(bdir, item),
                                           os.path.join(plots_dir, f'{item}_derived.png'), reasons_.get(item, ''))

        # 3.5.2: session report saved with the other outputs (session_report.pdf)
        try:
            from ground_gnss_ro_pipeline import generate_session_report
            st_txt = (f"{station_dict.get('name') or 'Station'} "
                      f"{station.latitude:.4f}°N {station.longitude:.4f}°E")
            generate_session_report(output_dir, os.path.join(output_dir, 'session_report.pdf'), status_,
                                    reasons_, st_txt, ground_hidden_items(output_dir, status_))
            log("Session report: session_report.pdf")
        except Exception as e:
            log(f"session report not written: {e}")

        if not keep_intermediate and results.get('step4') and results['step4'].success:
            removed = 0
            for name in ('step1_observations.csv', 'step2_matched.csv',
                         'step3a_elevations.csv', 'step3b_doppler.csv'):
                pth = os.path.join(output_dir, name)
                if os.path.exists(pth):
                    try:
                        os.remove(pth)
                        removed += 1
                    except OSError:
                        pass
            if removed:
                log(f"Cleanup: removed {removed} intermediate CSV file(s)")
        write_log()
        n_ok = sum(1 for r in results.values() if r.success)
        st_msg = results['station'].message if 'station' in results else ''
        progress_queue.put(('done', 'ground', True,
                            f"{n_ok}/{len(results)} steps completed. {st_msg}", diff_csv))
    except Exception as e:
        import traceback
        log(f"CRITICAL ERROR: {type(e).__name__}: {e}")
        log(f"TRACEBACK:\n{traceback.format_exc()}")
        write_log()
        progress_queue.put(('done', 'ground', False, f"{type(e).__name__}: {e}", None))


# ============================================================================
# SATELLITE PIPELINE PROCESS
# ============================================================================


def run_satellite_pipeline(conphs_dir: str, output_dir: str, progress_queue: Queue):
    """Satellite-based (LEO) pipeline execution in separate process."""
    import os
    from datetime import datetime
    
    from sat_gnss_ro_pipeline import LEOROPipeline, PipelineConfig, PipelinePlotter
    
    log_lines = []
    
    def log(message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines.append(f"[{ts}] {message}")
    
    def write_log():
        log_path = os.path.join(output_dir, 'log.txt')
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(log_lines))
        except:
            pass
    
    try:
        log("=" * 60)
        log("SATELLITE-BASED (LEO) GNSS-RO Pipeline")
        log("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        config = PipelineConfig()
        pipeline = LEOROPipeline(config)
        log(f"Pipeline config: rigorous={config.use_rigorous_bending}, local_curv={config.use_local_curvature}")
        
        # Find conPhs files
        conphs_files = sorted(
            glob.glob(os.path.join(conphs_dir, "conPhs_*.nc")) +
            glob.glob(os.path.join(conphs_dir, "conPhs_*_nc"))
        )
        
        if not conphs_files:
            progress_queue.put(('done', 'satellite', False, "No conPhs files found", None))
            return
        
        log(f"Found {len(conphs_files)} conPhs files")
        progress_queue.put(('progress', 'satellite', f"Processing {len(conphs_files)} events...", 0.05))
        
        summary_rows = []
        
        for i, conphs_file in enumerate(conphs_files):
            fname = os.path.basename(conphs_file)
            event_id = pipeline._extract_event_id(fname)
            
            frac = 0.05 + 0.90 * ((i + 1) / len(conphs_files))
            progress_queue.put(('progress', 'satellite', f"Event {i+1}/{len(conphs_files)}: {event_id}", frac))
            
            # Find matching validation files
            atmprf_file = pipeline._find_matching_file(conphs_dir, 'atmPrf', event_id)
            wetpf2_file = pipeline._find_matching_file(conphs_dir, 'wetPf2', event_id)
            
            log(f"Processing: {fname}")
            if atmprf_file:
                log(f"  atmPrf: {os.path.basename(atmprf_file)}")
            if wetpf2_file:
                log(f"  wetPf2: {os.path.basename(wetpf2_file)}")
            
            # Process event
            event_output = os.path.join(output_dir, event_id)
            log(f"  Calling pipeline.process_event()...")       

            try:
                results = pipeline.process_event(
                    conphs_file=conphs_file,
                    atmprf_file=atmprf_file,
                    wetpf2_file=wetpf2_file,
                    output_dir=event_output
                )
                
                # Log all messages from pipeline
                for msg in results.get('messages', []):
                    log(f"    {msg}")
                
                if not results['success']:
                    log(f"  FAILED: Pipeline returned success=False")
                    if results.get('bending_profile') is None:
                        log(f"    Bending profile is None")
                    if results.get('refractivity_profile') is None:
                        log(f"    Refractivity profile is None")
                
            except Exception as e:
                import traceback
                log(f"  EXCEPTION in process_event: {str(e)}")
                log(f"  Traceback:\n{traceback.format_exc()}")
                results = {'success': False, 'bending_profile': None}

            # Generate plots if successful
            if results['success']:
                try:
                    pipeline._generate_event_plots(
                        conphs_file=conphs_file,
                        results=results,
                        atmprf_file=atmprf_file,
                        wetpf2_file=wetpf2_file,
                        output_dir=event_output,
                        event_id=event_id
                    )
                except Exception as e:
                    log(f"  Plot generation failed: {e}")
            
            # Collect summary
            row = {
                'event_id': event_id,
                'success': results['success'],
                'has_validation': atmprf_file is not None or wetpf2_file is not None
            }
            
            if results['bending_profile'] is not None:
                bp = results['bending_profile']
                row['height_min_km'] = bp.tangent_height.min()
                row['height_max_km'] = bp.tangent_height.max()
            
            if results['validation_refractivity_atmPrf'] is not None:
                vr = results['validation_refractivity_atmPrf']
                row['refrac_rmse'] = vr.rmse
                row['refrac_corr'] = vr.correlation
            
            summary_rows.append(row)
            
            status = "OK" if results['success'] else "FAILED"
            log(f"  Status: {status}")
        
        # Save summary
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(output_dir, 'processing_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        
        write_log()
        
        success_count = summary_df['success'].sum()
        progress_queue.put(('done', 'satellite', True, 
                           f"{success_count}/{len(summary_df)} events processed", 
                           summary_csv))
        
    except Exception as e:
        import traceback
        log(f"CRITICAL ERROR: {str(e)}")
        log(f"TRACEBACK:\n{traceback.format_exc()}")
        write_log()
        progress_queue.put(('done', 'satellite', False, str(e), None))


# ============================================================================
# v3.4.7 — WORD-WRAPPING LABEL
# ============================================================================

class WrapLabel(QLabel):
    """QLabel that reports an honest height for its wrapped text.

    A plain word-wrapped QLabel returns a single-line ``minimumSizeHint``, so a
    QVBoxLayout gives it too little vertical space and silently clips the last
    line.  That was invisible at the old 11-12 px note sizes but became obvious
    in v3.4.7 once the browse-panel warnings and the results legend were
    enlarged.  Overriding ``minimumSizeHint`` to use ``heightForWidth`` makes
    the layout reserve the full wrapped height.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Minimum)

    def setText(self, text):
        super().setText(text)
        self.updateGeometry()

    def minimumSizeHint(self):
        w = self.width()
        if w <= 0:
            return super().minimumSizeHint()
        return QSize(0, self.heightForWidth(w))

    def sizeHint(self):
        w = self.width()
        if w <= 0:
            return super().sizeHint()
        return QSize(w, self.heightForWidth(w))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateGeometry()


# ============================================================================
# PLOT CANVAS
# ============================================================================

class PlotCanvas(FigureCanvas):
    """Interactive plot canvas with zoom, pan, and reset support."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#FAFAFA')
        super().__init__(self.fig)
        self.setParent(parent)
        self._is_placeholder = True
    
    def show_placeholder(self, message: str = "Select an item"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center',
                fontsize=FS_PLOT_PLACEHOLD, color='#757575', transform=ax.transAxes)
        ax.axis('off')
        self._is_placeholder = True
        self.draw()
    
    def load_from_png(self, png_path: str):
        self.fig.clear()
        if os.path.exists(png_path):
            img = matplotlib.image.imread(png_path)
            ax = self.fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self._is_placeholder = False
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Plot not available", ha='center', va='center',
                    fontsize=FS_PLOT_PLACEHOLD, color='#757575', transform=ax.transAxes)
            ax.axis('off')
            self._is_placeholder = True
        self.draw()


class InteractivePlotWidget(QWidget):
    """Wrapper that pairs a PlotCanvas with a navigation toolbar for
    zoom, pan, home (reset), and save-to-file functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # v3.4.7 — shrink the toolbar icons (matplotlib defaults to 24 px,
        # which dominated the plot panel on the 1366x768 Windows build).
        try:
            self.toolbar.setIconSize(QSize(TOOLBAR_ICON_PX, TOOLBAR_ICON_PX))
        except Exception:
            pass

        self.toolbar.setStyleSheet(f"""
            QToolBar {{
                spacing: 4px;
                padding: 1px 4px;
                background: #F5F5F5;
                border-bottom: 1px solid #CCCCCC;
            }}
            QToolButton {{
                padding: 1px;
                font-size: {FS_BROWSE}px;
            }}
            QToolBar QLabel {{ font-size: {FS_BROWSE}px; }}
        """)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    # Delegate convenience methods so the rest of the code can call them
    # the same way it called them on the old PlotCanvas.
    def show_placeholder(self, message: str = "Select an item"):
        self.canvas.show_placeholder(message)

    def load_from_png(self, png_path: str):
        self.canvas.load_from_png(png_path)


# ============================================================================
# STATION PANEL
# ============================================================================

class StationInfoPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Station Configuration (Ground)", parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("from metadata.cra (optional)")
        self.name_edit.setPlaceholderText("Station identifier")
        row1.addWidget(self.name_edit)
        layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Lat:"))
        self.lat_edit = QLineEdit()
        self.lat_edit.setPlaceholderText("°N")
        self.lat_edit.setMinimumWidth(110)
        row2.addWidget(self.lat_edit)
        row2.addWidget(QLabel("Lon:"))
        self.lon_edit = QLineEdit()
        self.lon_edit.setPlaceholderText("°E")
        self.lon_edit.setMinimumWidth(110)
        row2.addWidget(self.lon_edit)
        row2.addStretch()
        layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Altitude:"))
        self.alt_edit = QLineEdit()
        self.alt_edit.setPlaceholderText("meters")
        self.alt_edit.setMinimumWidth(90)
        row3.addWidget(self.alt_edit)
        row3.addWidget(QLabel("m (GPS height)"))
        row3.addStretch()
        layout.addLayout(row3)
    
    def _fields(self):
        return (self.name_edit, self.lat_edit, self.lon_edit, self.alt_edit)

    def _set(self, edit: QLineEdit, text: str):
        """v4.4: left-aligned and scrolled to the start (setText leaves the view at the end)."""
        edit.setText(text)
        edit.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        edit.setCursorPosition(0)

    def load_from_metadata(self, metadata: Dict):
        self._set(self.name_edit, str(metadata.get('STATION_NAME', '') or ''))
        self._set(self.lat_edit, str(metadata.get('STATION_LAT', '')))
        self._set(self.lon_edit, str(metadata.get('STATION_LON', '')))
        self._set(self.alt_edit, str(metadata.get('STATION_HEIGHT', '')))
    
    def load_from_rinex_station(self, station_info: Dict):
        """
        Station from the receiver's 3D fix (UBX NAV-PVT / RINEX header). Height is
        the GPS height (above sea level) people use; the geoid separation is kept
        so the pipeline can convert. Name: .cra only, else empty.
        """
        self._set(self.name_edit, str(station_info.get('marker_name') or ''))
        self._set(self.lat_edit, f"{station_info['latitude']:.6f}")
        self._set(self.lon_edit, f"{station_info['longitude']:.6f}")
        msl = station_info.get('altitude_msl')
        self._set(self.alt_edit, f"{(msl if msl is not None else station_info['altitude']):.1f}")
        self.geoid_sep_m = float(station_info.get('geoid_sep_m') or 0.0)
    
    def has_valid_coords(self) -> bool:
        """Check if station coordinates are filled in and valid."""
        try:
            float(self.lat_edit.text())
            float(self.lon_edit.text())
            float(self.alt_edit.text())
            return True
        except (ValueError, AttributeError):
            return False
    
    def get_station_config(self) -> Optional[StationConfig]:
        try:
            return StationConfig(
                latitude=float(self.lat_edit.text()),
                longitude=float(self.lon_edit.text()),
                altitude=float(self.alt_edit.text()),          # GPS height (above sea level)
                name=self.name_edit.text(),
                height_ref='msl',
                geoid_sep_m=getattr(self, 'geoid_sep_m', 0.0),
            )
        except ValueError:
            return None
    
    def to_metadata(self) -> Dict:
        try:
            return {
                'STATION_NAME': self.name_edit.text(),
                'STATION_LAT': float(self.lat_edit.text()),
                'STATION_LON': float(self.lon_edit.text()),
                'STATION_HEIGHT': float(self.alt_edit.text()),
                'STATION_HEIGHT_REF': 'GPS height above sea level (m)',
            }
        except ValueError:
            return {}


# ============================================================================
# v3.4.4 — ADVANCED PROCESSING SETTINGS PANEL
# ============================================================================

class ProcessingPanel(QGroupBox):
    """
    Collapsible Advanced Settings panel exposing the tunable PROCESSING
    constants from the .cra file.

    v3.4.5 layout: wrapped in a QScrollArea so the form never overflows on
    low-resolution screens (HD 720p on Windows).  Height is capped at 55% of
    the available screen height; spin-box minimum widths prevent compression.
    """

    # Field spec: (key, label, kind, min, max, decimals, tooltip)
    # v4.3: keys are the pipeline's own PROCESSING names (older versions used
    # *_S / *_DEG / *_HZ names the pipeline never read).
    _SPEC = [
        ('POLY_SMOOTH_WINDOW',          'Max smooth window (s)',     'float', 1.0,   10000.0, 1,    "Upper limit of the Fresnel-adaptive Doppler smoothing window (s)."),
        ('POLY_MIN_WINDOW',             'Min smooth window (s)',     'float', 1.0,   1000.0,  1,    "Lower limit of the smoothing window (s)."),
        ('POLYFIT_GAP_THRESHOLD',       'Polyfit gap threshold (s)', 'float', 0.1,   600.0,   1,    "Restart the smoothing fit when the time gap is at least this long (s)."),
        ('SMOOTH_ELEV_MAX_DEG',         'Smooth below elev (°)',     'float', 0.0,   90.0,    1,    "Smooth the atmospheric Doppler only below this elevation. 90 = all (fit line on every raw plot)."),
        ('RO_ELEVATION_THRESHOLD',      'RO elevation thresh (°)',   'float', -10.0, 90.0,    2,    "Only rays below this elevation are used (both sides of the horizon)."),
        ('RO_DOPPLER_THRESHOLD',        'RO Doppler thresh (Hz)',    'float', 0.0,   100.0,   2,    "Minimum |atmospheric Doppler| for RO epochs. 0 = off (real values are only 0.2–3 Hz)."),
        ('RO_MIN_EPOCHS',               'RO min epochs',             'int',   1,     100000,  1,    "Minimum low-elevation epochs for an RO candidate."),
        ('RO_MIN_NEG_ELEV_EPOCHS',      'RO min epochs below 0°',    'int',   0,     100000,  1,    "Minimum epochs at negative geometric elevation."),
        ('EVENT_GAP_S',                 'Event gap (s)',             'float', 1.0,   86400.0, 1,    "A gap longer than this splits a satellite into separate occultation events."),
        ('ALLOW_SINGLE_FREQ',           'Allow single-frequency',    'bool',  None,  None,    None, "Use L1-only events when dual-frequency data is missing.\nNO ionospheric correction (~1–2% in N). Results are flagged ◐ [RO·1F] in amber."),
        ('REF_SAT_ELEVATION_THRESHOLD', 'Ref-sat elev thresh (°)',   'float', 0.0,   90.0,    1,    "Reference satellites must stay above this elevation; epochs without one are dropped."),
        ('REF_SAT_MIN_EPOCHS',          'Ref-sat min epochs',        'int',   1,     100000,  1,    "Minimum coverage for the primary reference satellite."),
        ('REF_SAT_JUMP_THRESHOLD',      'Ref-sat jump thresh (Hz)',  'float', 0.0,   100.0,   2,    "Excess-Doppler jump counted as a cycle slip when scoring references."),
        ('N_COEFF_A1',                  'Smith–Weintraub a1',        'float', 0.0,   1e6,     3,    "Refractivity dry term coefficient (K/hPa)."),
        ('N_COEFF_A2',                  'Smith–Weintraub a2',        'float', 0.0,   1e9,     2,    "Refractivity wet term coefficient (K²/hPa)."),
        ('KEEP_INTERMEDIATE_CSVS',      'Keep step1–3 CSVs',         'bool',  None,  None,    None, "Keep intermediate step1/2/3 CSVs after a successful run."),
        ('FORCE_CRA_STATION_COORDS',    'Force .cra station coords', 'bool',  None,  None,    None, "Use the station coordinates typed here / in .cra instead of the receiver's own position\n(UBX 3D fix or RINEX header). Leave off unless the receiver position is known to be wrong."),
    ]

    def __init__(self, parent=None):
        super().__init__("Advanced Settings", parent)
        # v3.4.7 — pinned: this is a dense 12-row form inside a 340-420 px
        # sidebar, so it keeps the v3.4.6 size rather than following FS_SIDEBAR.
        self.setStyleSheet(f"""
            QGroupBox {{ font-size: {FS_ADVANCED}px; }}
            QGroupBox QLabel {{ font-size: {FS_ADVANCED}px; }}
            QGroupBox QCheckBox {{ font-size: {FS_ADVANCED}px; }}
            QGroupBox QSpinBox, QGroupBox QDoubleSpinBox {{
                font-size: {FS_ADVANCED}px;
            }}
        """)
        self.setCheckable(True)
        self.setChecked(False)           # collapsed by default
        self._widgets: Dict[str, QWidget] = {}
        self._labels: Dict[str, QLabel] = {}
        self._defaults = load_processing_config_from_cra({})

        # Build form inside a plain widget -----------------------------------
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setContentsMargins(4, 2, 4, 6)
        form.setSpacing(3)

        for key, label, kind, mn, mx, step, tip in self._SPEC:
            w: QWidget
            if kind == 'float':
                w = QDoubleSpinBox()
                w.setDecimals(int(step) if isinstance(step, int) else 3)
                w.setRange(float(mn), float(mx))
                w.setSingleStep(0.1)
                w.setMinimumWidth(90)
            elif kind == 'int':
                w = QSpinBox()
                w.setRange(int(mn), int(mx))
                w.setMinimumWidth(90)
            elif kind in ('bool', 'apmodel'):
                w = QCheckBox()
            else:
                continue
            w.setToolTip(tip)
            self._widgets[key] = w
            lbl = QLabel(label)
            self._labels[key] = lbl
            form.addRow(lbl, w)
            # 3.5.2 — highlight values that differ from the defaults
            if isinstance(w, QCheckBox):
                w.toggled.connect(lambda _=None: self._mark_changed())
            else:
                w.valueChanged.connect(lambda _=None: self._mark_changed())

        self.reset_btn = QPushButton("Reset to defaults")
        self.reset_btn.setToolTip("Set every value back to the pipeline default (saved to .cra on the next run).")
        self.reset_btn.clicked.connect(lambda: self.load_from_cra({}))
        form.addRow(self.reset_btn)

        # Scroll area — caps height to 55% of screen so the panel never
        # pushes below the visible area on 720p / HD displays.
        self._scroll = QScrollArea()
        self._scroll.setWidget(form_widget)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        try:
            screen_h = QApplication.primaryScreen().availableGeometry().height()
        except Exception:
            screen_h = 768
        self._scroll.setMaximumHeight(max(200, int(screen_h * 0.55)))

        # GroupBox layout ----------------------------------------------------
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(4, 4, 4, 4)
        wrap.setSpacing(0)
        wrap.addWidget(self._scroll)

        # Wire toggle and set initial state ----------------------------------
        self.toggled.connect(self._on_toggled)
        self._on_toggled(False)          # start collapsed

        self.load_from_cra({})

    def _on_toggled(self, checked: bool):
        """Show/hide the scroll area; reflow the sidebar so no gap is left."""
        self._scroll.setVisible(checked)
        self.updateGeometry()
        p = self.parent()
        if p is not None:
            p.updateGeometry()

    def load_from_cra(self, cra_data: Optional[Dict]):
        """Populate the widgets from a parsed .cra dict (or defaults)."""
        cfg = load_processing_config_from_cra(cra_data or {})
        for key, w in self._widgets.items():
            if key not in cfg:
                continue
            val = cfg[key]
            try:
                if key == 'ALPHA_P_MODEL':
                    v = str(val).lower()
                    self._ap_always = v == 'always'
                    w.setChecked(v in ('fill', 'always', 'true', '1'))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
            except (TypeError, ValueError):
                pass
        self._mark_changed()

    def _mark_changed(self):
        """3.5.2 — amber, bold label where the value differs from the default."""
        cur = self.to_processing_dict()
        n = 0
        for key, lbl in self._labels.items():
            d, v = self._defaults.get(key), cur.get(key)
            try:
                changed = (abs(float(v) - float(d)) > 1e-9 * max(1.0, abs(float(d))))
            except (TypeError, ValueError):
                changed = v != d
            n += bool(changed)
            lbl.setStyleSheet("QLabel { color: #B35C00; font-weight: 600; }" if changed else "")
            lbl.setToolTip(f"Default: {d}")
        self.setTitle("Advanced Settings" + (f"  ({n} changed)" if n else ""))

    def to_processing_dict(self) -> Dict[str, Any]:
        """Read widget values back into a PROCESSING dict."""
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            if key == 'ALPHA_P_MODEL':
                out[key] = (('always' if getattr(self, '_ap_always', False) else 'fill')
                            if w.isChecked() else 'off')
            elif isinstance(w, QCheckBox):
                out[key] = w.isChecked()
            elif isinstance(w, QSpinBox):
                out[key] = int(w.value())
            elif isinstance(w, QDoubleSpinBox):
                out[key] = float(w.value())
        return out

    def force_cra_coords(self) -> bool:
        w = self._widgets.get('FORCE_CRA_STATION_COORDS')
        return bool(w.isChecked()) if isinstance(w, QCheckBox) else False

    def keep_intermediate(self) -> bool:
        w = self._widgets.get('KEEP_INTERMEDIATE_CSVS')
        return bool(w.isChecked()) if isinstance(w, QCheckBox) else False


# ============================================================================
# RESULT LIST WIDGET
# ============================================================================

GROUND_LEGEND = "● RO profile    ◐ RO profile, single-freq (no iono corr.)    ○ no profile / no RO"


GROUND_ROW_STYLE = {
    # state: (marker, suffix, color, derived tabs enabled)
    'ro_ok':    ('●', '  [RO]',    '#2E7D32', True),
    'ro_ok_1f': ('◐', '  [RO·1F]', '#E65100', True),    # single frequency: no iono correction
    'ro_empty': ('○', '',          '#A1887F', False),
    'no_ro':    ('○', '',          '#757575', False),
}


def _norm_ground_state(v) -> str:
    if v == 'ro_ok' or v is True:
        return 'ro_ok'
    if v in ('ro_ok_1f', 'ro_empty'):
        return v
    return 'no_ro'


class ResultListWidget(QListWidget):
    """Unified list for both ground satellites and satellite events."""

    def _add_ground_rows(self, ro_status: Dict[str, Any], indent: str = "", hidden: Optional[set] = None):
        """v4.3: rows with a profile (dual first, then single-frequency), separator, the rest.
        3.5.2: items in ``hidden`` (never below the RO elevation threshold) are left out."""
        hidden = hidden or set()
        groups = {k: sorted(s for s, v in ro_status.items() if _norm_ground_state(v) == k and s not in hidden)
                  for k in GROUND_ROW_STYLE}
        with_profile = groups['ro_ok'] + groups['ro_ok_1f']
        without = groups['ro_empty'] + groups['no_ro']
        for state in ('ro_ok', 'ro_ok_1f'):
            for sat_id in groups[state]:
                self._ground_item(sat_id, state, indent)
        if with_profile and without:
            sep = QListWidgetItem("─" * 24)
            sep.setFlags(Qt.ItemFlag.NoItemFlags)
            sep.setForeground(QColor('#BDBDBD'))
            self.addItem(sep)
        for state in ('ro_empty', 'no_ro'):
            for sat_id in groups[state]:
                self._ground_item(sat_id, state, indent)

    def _ground_item(self, sat_id: str, state: str, indent: str = ""):
        marker, suffix, color, enabled = GROUND_ROW_STYLE[state]
        item = QListWidgetItem(f"{indent}{marker} {sat_id}{suffix}")
        item.setForeground(QColor(color))
        if state == 'ro_ok_1f':
            item.setToolTip("Single-frequency occultation: no ionospheric correction (~1–2% in N).")
        item.setData(Qt.ItemDataRole.UserRole, sat_id)
        item.setData(Qt.ItemDataRole.UserRole + 1, enabled)
        item.setData(Qt.ItemDataRole.UserRole + 2, 'ground')
        item.setData(Qt.ItemDataRole.UserRole + 3, state)
        self.addItem(item)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(f"""
            QListWidget {{
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {FS_RESULT_LIST}px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }}
            QListWidget::item {{ padding: 4px 8px; }}
            QListWidget::item:selected {{
                background-color: #1976D2;
                color: white;
            }}
        """)
        self.current_mode = None  # 'ground' or 'satellite'
    
    def populate_ground(self, ro_status: Dict[str, Any], hidden: Optional[set] = None):
        """Populate with ground-based satellite results.

        v3.4.4.1: ``ro_status`` values are tri-state:
            'ro_ok'    → green  (RO + bending data available)
            'ro_empty' → faded  (RO but retrieval produced no usable profile)
            False      → gray   (no RO geometry detected)

        v3.4.4.2: yellow rows are placed together with the gray rows
        (both signify "no derived data") and rendered with a subdued
        color so they don't compete with the green rows visually.

        For backward compatibility, plain True is treated as 'ro_ok'.
        """
        self.clear()
        self.current_mode = 'ground'
        self._add_ground_rows(ro_status, hidden=hidden)

    def populate_satellite(self, summary_df: pd.DataFrame):
        """Populate with satellite event results."""
        self.clear()
        self.current_mode = 'satellite'
        
        if summary_df is None or summary_df.empty:
            return
        
        success_events = summary_df[summary_df['success'] == True]
        failed_events = summary_df[summary_df['success'] == False]
        
        for _, row in success_events.iterrows():
            event_id = row['event_id']
            has_val = row.get('has_validation', False)
            suffix = " [VAL]" if has_val else ""
            item = QListWidgetItem(f"● {event_id}{suffix}")
            item.setForeground(QColor('#2E7D32'))
            item.setData(Qt.ItemDataRole.UserRole, event_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, True)  # success
            item.setData(Qt.ItemDataRole.UserRole + 2, 'satellite')
            self.addItem(item)
        
        if len(success_events) > 0 and len(failed_events) > 0:
            sep = QListWidgetItem("─" * 24)
            sep.setFlags(Qt.ItemFlag.NoItemFlags)
            sep.setForeground(QColor('#BDBDBD'))
            self.addItem(sep)
        
        for _, row in failed_events.iterrows():
            event_id = row['event_id']
            item = QListWidgetItem(f"○ {event_id}  [FAILED]")
            item.setForeground(QColor('#D32F2F'))
            item.setData(Qt.ItemDataRole.UserRole, event_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, False)
            item.setData(Qt.ItemDataRole.UserRole + 2, 'satellite')
            self.addItem(item)
    
    def populate_both(self, ground_ro_status: Dict[str, Any], sat_summary_df: pd.DataFrame,
                      hidden: Optional[set] = None):
        """Populate with both ground and satellite results."""
        self.clear()
        self.current_mode = 'both'
        
        # Ground section header
        header_g = QListWidgetItem("═══ GROUND ═══")
        header_g.setFlags(Qt.ItemFlag.NoItemFlags)
        header_g.setForeground(QColor('#1976D2'))
        font = header_g.font()
        font.setBold(True)
        header_g.setFont(font)
        self.addItem(header_g)

        self._add_ground_rows(ground_ro_status, indent="  ", hidden=hidden)
        
        # Satellite section header
        header_s = QListWidgetItem("═══ SATELLITE ═══")
        header_s.setFlags(Qt.ItemFlag.NoItemFlags)
        header_s.setForeground(QColor('#1976D2'))
        header_s.setFont(font)
        self.addItem(header_s)
        
        if sat_summary_df is not None and not sat_summary_df.empty:
            for _, row in sat_summary_df.iterrows():
                event_id = row['event_id']
                success = row['success']
                has_val = row.get('has_validation', False)
                
                if success:
                    suffix = " [VAL]" if has_val else ""
                    item = QListWidgetItem(f"  ● {event_id}{suffix}")
                    item.setForeground(QColor('#2E7D32'))
                else:
                    item = QListWidgetItem(f"  ○ {event_id}  [FAILED]")
                    item.setForeground(QColor('#D32F2F'))
                
                item.setData(Qt.ItemDataRole.UserRole, event_id)
                item.setData(Qt.ItemDataRole.UserRole + 1, success)
                item.setData(Qt.ItemDataRole.UserRole + 2, 'satellite')
                self.addItem(item)


# ============================================================================
# PROGRESS PANEL
# ============================================================================

class ProgressPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Processing Status", parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        
        # v3.4.7 — "processing panel fonts is ok": pinned to the v3.4.6
        # effective sizes so the enlarged sidebar rule does not touch them.
        self.setStyleSheet(f"""
            QGroupBox {{ font-size: {FS_STATUS}px; }}
            QGroupBox QLabel {{ font-size: {FS_STATUS}px; }}
        """)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet(
            f"font-weight: bold; color: #424242; font-size: {FS_STATUS}px;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #1976D2;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet(
            f"color: #616161; font-size: {FS_STATUS_DETAIL}px;")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
    
    def set_status(self, status: str, detail: str = "", progress: float = 0):
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"font-weight: bold; color: #1976D2; font-size: {FS_STATUS}px;")
        self.detail_label.setText(detail)
        self.progress_bar.setValue(int(progress * 100))
    
    def set_complete(self, success: bool, message: str):
        if success:
            self.status_label.setText("Completed")
            self.status_label.setStyleSheet(
            f"font-weight: bold; color: #2E7D32; font-size: {FS_STATUS}px;")
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText("Stopped" if "cancel" in message.lower() else "Failed")
            self.status_label.setStyleSheet(
            f"font-weight: bold; color: #D32F2F; font-size: {FS_STATUS}px;")
        self.detail_label.setText(message)
    
    def reset(self):
        self.status_label.setText("Idle")
        self.status_label.setStyleSheet(
            f"font-weight: bold; color: #424242; font-size: {FS_STATUS}px;")
        self.detail_label.setText("")
        self.progress_bar.setValue(0)


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"GNSS Radio Occultation Processor {__version__}")

        # v3.4.7 — a 1200x800 minimum does not fit a 1366x768 laptop: the
        # window was forced taller than the desktop and the bottom of the
        # sidebar (Results list) fell below the taskbar.  Clamp the minimum
        # to the available geometry and open maximised on short screens.
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            min_w = min(1200, max(1000, avail.width() - 40))
            min_h = min(800, max(620, avail.height() - 40))
            self.setMinimumSize(min_w, min_h)
            self.resize(min(1400, avail.width()), min(900, avail.height()))
            # Applied after _setup_ui() so we never flash an empty window.
            self._start_maximized = avail.height() < 800
        except Exception:
            self.setMinimumSize(1100, 640)
            self._start_maximized = False
        
        self.input_dir = None
        self.scan_result = None
        self.output_dir = None
        
        # Ground results
        self.ground_intermediate_data = None
        self.ground_ro_status = {}
        self.ground_reasons = {}
        self.ground_message = ''
        self.ground_output_dir = None
        
        # Satellite results
        self.sat_summary_df = None
        self.sat_output_dir = None
        
        # Multiprocessing
        self.ground_process = None
        self.sat_process = None
        self.progress_queue = None
        self.poll_timer = None
        
        # Track completion
        self.ground_done = False
        self.sat_done = False
        self.ground_success = False
        self.sat_success = False

        # v3.4.4 — load-mode (showing previously executed results)
        self.load_mode = False
        self.load_info = None  # dict from detect_output_directory()
        
        self._setup_ui()
        self._connect_signals()

        if getattr(self, '_start_maximized', False):
            self.showMaximized()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # SIDEBAR
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMaximumWidth(SIDEBAR_MAX_W)
        sidebar.setMinimumWidth(SIDEBAR_MIN_W)

        # v3.4.7 — one scoped rule enlarges every sidebar control.  Panels that
        # must NOT grow (Processing Status, Advanced Settings, the Results list
        # and the Browse row) set their own style sheet, which Qt resolves with
        # higher priority than this ancestor rule.
        sidebar.setStyleSheet(f"""
            #sidebar QGroupBox {{
                font-size: {FS_SIDEBAR}px;
                font-weight: 600;
            }}
            #sidebar QLabel     {{ font-size: {FS_SIDEBAR}px; }}
            #sidebar QLineEdit  {{ font-size: {FS_SIDEBAR}px; }}
            #sidebar QCheckBox  {{ font-size: {FS_SIDEBAR}px; }}
            #sidebar QPushButton {{ font-size: {FS_SIDEBAR}px; }}
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        # ------------------------------------------------------------------
        # Input directory
        # v3.4.7 — the path field and the Browse button are the one place that
        # gets SMALLER: the path string is long and was truncating on the HD
        # laptop.  The validation / warning notes underneath get LARGER.
        # ------------------------------------------------------------------
        input_group = QGroupBox("Data Directory")
        input_layout = QVBoxLayout(input_group)

        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Ground: .ubx/.sp3 | Satellite: conPhs_*")
        self.dir_edit.setReadOnly(True)
        self.dir_edit.setStyleSheet(f"QLineEdit {{ font-size: {FS_BROWSE}px; }}")
        dir_layout.addWidget(self.dir_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMaximumWidth(80)
        self.browse_btn.setStyleSheet(
            f"QPushButton {{ font-size: {FS_BROWSE}px; padding: 4px 8px; }}")
        dir_layout.addWidget(self.browse_btn)

        # 3.5.2 — recent folders
        self.recent_btn = QPushButton("Recent")
        self.recent_btn.setMaximumWidth(72)
        self.recent_btn.setStyleSheet(
            f"QPushButton {{ font-size: {FS_BROWSE}px; padding: 4px 6px; }}")
        self.recent_btn.setVisible(QMenu is not None)
        dir_layout.addWidget(self.recent_btn)
        input_layout.addLayout(dir_layout)

        self.validation_label = WrapLabel("")
        self.validation_label.setStyleSheet(
            f"QLabel {{ font-size: {FS_BROWSE_NOTE}px; }}")
        input_layout.addWidget(self.validation_label)

        sidebar_layout.addWidget(input_group)
        
        # Station (only for ground)
        self.station_panel = StationInfoPanel()
        self.station_panel.setVisible(False)
        sidebar_layout.addWidget(self.station_panel)

        # v3.4.4 — Advanced processing settings (only for ground; collapsible)
        self.processing_panel = ProcessingPanel()
        self.processing_panel.setVisible(False)
        sidebar_layout.addWidget(self.processing_panel)

        # Run and Stop buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
        self.run_btn = QPushButton("Start Processing")
        self.run_btn.setEnabled(False)
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #1976D2;
                color: white;
                font-weight: bold;
                font-size: {FS_SIDEBAR_BTN}px;
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #1565C0; }}
            QPushButton:pressed {{ background-color: #0D47A1; }}
            QPushButton:disabled {{ background-color: #BDBDBD; }}
        """)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #D32F2F;
                color: white;
                font-weight: bold;
                font-size: {FS_SIDEBAR_BTN}px;
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #C62828; }}
            QPushButton:pressed {{ background-color: #B71C1C; }}
            QPushButton:disabled {{ background-color: #BDBDBD; }}
        """)
        
        btn_layout.addWidget(self.run_btn, 2)
        btn_layout.addWidget(self.stop_btn, 1)
        sidebar_layout.addLayout(btn_layout)
        
        # Progress
        self.progress_panel = ProgressPanel()
        sidebar_layout.addWidget(self.progress_panel)
        
        # Result list
        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout(result_group)

        # 3.5.2 — one-line summary + 'show all' toggle above the list
        self.summary_label = WrapLabel("")
        self.summary_label.setStyleSheet(f"QLabel {{ color: #424242; font-size: {FS_RESULT_NOTE}px; }}")
        result_layout.addWidget(self.summary_label)
        self.show_all_cb = QCheckBox("Show all satellites")
        self.show_all_cb.setToolTip("Also list satellites that never went below the RO elevation threshold.")
        self.show_all_cb.setStyleSheet(f"QCheckBox {{ font-size: {FS_RESULT_NOTE}px; }}")
        self.show_all_cb.setVisible(False)
        result_layout.addWidget(self.show_all_cb)

        self.result_list = ResultListWidget()
        result_layout.addWidget(self.result_list)
        
        # v3.4.7 — "results box sub notes" (RO / profile / no profile ...)
        self.legend_label = WrapLabel("● Success/RO    ○ Failed/No RO")
        self.legend_label.setStyleSheet(
            f"QLabel {{ color: #757575; font-size: {FS_RESULT_NOTE}px; }}")
        result_layout.addWidget(self.legend_label)

        # 3.5.2 — session quality card + session report
        self.quality_label = WrapLabel("")
        self.quality_label.setStyleSheet(
            f"QLabel {{ color: #424242; font-size: {FS_RESULT_NOTE}px; background: #F5F5F2;"
            f" border: 1px solid #E0E0DC; border-radius: 4px; padding: 4px 6px; }}")
        self.quality_label.setVisible(False)
        result_layout.addWidget(self.quality_label)

        sidebar_layout.addWidget(result_group, 1)
        
        splitter.addWidget(sidebar)
        
        # MAIN PANEL
        main_panel = QWidget()
        main_panel_layout = QVBoxLayout(main_panel)
        main_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }}
            QTabBar::tab {{
                padding: 8px 16px;
                font-size: {FS_TABS}px;
            }}
            QTabBar::tab:selected {{ font-weight: bold; }}
        """)
        
        self.raw_canvas = InteractivePlotWidget()
        self.raw_canvas.show_placeholder("Select an item to view observations")
        self.tab_widget.addTab(self.raw_canvas, "GNSS Raw Observations")
        
        self.derived_canvas = InteractivePlotWidget()
        self.derived_canvas.show_placeholder("Select a successful item to view profiles")
        self.tab_widget.addTab(self.derived_canvas, "Radio Occultation")
        
        self.atm_canvas = InteractivePlotWidget()
        self.atm_canvas.show_placeholder("Select a successful item to view atmospheric profiles")
        self.tab_widget.addTab(self.atm_canvas, "Atmospheric Profiles")
        
        main_panel_layout.addWidget(self.tab_widget)
        splitter.addWidget(main_panel)
        
        splitter.setSizes([SIDEBAR_MIN_W, 880])
    
    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._browse_directory)
        self.recent_btn.clicked.connect(self._show_recent_menu)
        self.show_all_cb.toggled.connect(lambda _=None: self._populate_results())
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.stop_btn.clicked.connect(self._stop_pipeline)
        # v3.4.4.2 — react to BOTH mouse clicks AND keyboard navigation.
        # itemClicked only fires on mouse; currentItemChanged fires on
        # up/down arrow keys, Home/End/PageUp/PageDown, and clicks. Hooking
        # the latter alone is enough and gives keyboard parity with the mouse.
        self.result_list.currentItemChanged.connect(self._on_current_item_changed)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_run_clicked(self):
        """Dispatch the Run button depending on current mode."""
        if self.load_mode:
            self._close_project()
        else:
            self._run_pipeline()
    
    def _browse_directory(self):
        last = _load_prefs().get('last_dir', '')
        start = os.path.dirname(last) if last and os.path.isdir(os.path.dirname(last)) else ""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", start,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self._validate_directory(directory)

    def _show_recent_menu(self):
        """3.5.2 — recently opened data / result folders."""
        if QMenu is None:
            return
        menu = QMenu(self)
        rec = [d for d in _load_prefs().get('recent', []) if os.path.isdir(d)]
        if not rec:
            a = menu.addAction("(no recent folders)")
            a.setEnabled(False)
        for d in rec:
            a = menu.addAction(d)
            a.triggered.connect(lambda _=False, d=d: self._validate_directory(d))
        menu.exec(self.recent_btn.mapToGlobal(self.recent_btn.rect().bottomLeft()))
    
    def _validate_directory(self, directory: str):
        self.input_dir = directory
        self.dir_edit.setText(directory)

        # v3.4.4 — Detect whether this is a previously executed *output* dir.
        # If so, switch into load-mode and skip the input-data validation path.
        out_info = detect_output_directory(directory)
        if out_info['is_output']:
            self._enter_load_mode(directory, out_info)
            return

        # Normal path: this is an *input* dataset directory.
        self._exit_load_mode()
        self.scan_result = scan_input_directory(directory)

        _remember_dir(directory)
        sr = self.scan_result
        ok_ = lambda b: "✓" if b else "✗"
        msgs = []
        if sr['data_type'] in (DataType.GROUND, DataType.BOTH):
            parts = [f"📡 {GROUND_DISPLAY_NAME}",
                     f"{sr.get('n_obs', 0)} {sr.get('obs_source') or ''}",
                     f"SP3 {ok_(sr['sp3_file'])}",
                     f".nc {ok_(sr['era5_file'])}",
                     ".cra ✓" if sr['metadata_file'] else ".cra: new"]
            msgs.append("<span style='color:#1976D2; font-weight:bold;'>" + " · ".join(parts) + "</span>")
        if sr['data_type'] in (DataType.SATELLITE, DataType.BOTH):
            msgs.append("<span style='color:#7B1FA2; font-weight:bold;'>🛰 "
                        f"{len(sr['conphs_files'])} conPhs · validation "
                        f"{ok_(sr['has_atmprf'] or sr['has_wetpf2'])}</span>")
        for w in sr['warnings']:
            if w.startswith('No .nc') or w.startswith('No atmPrf'):
                continue                                   # already shown as ✗ in the header line
            msgs.append(f"<span style='color:#F57C00'>⚠ {w}</span>")
        for e in sr['errors']:
            msgs.append(f"<span style='color:#D32F2F'>✗ {e}</span>")

        self.validation_label.setText("<br>".join(msgs))

        # Show station / processing panels only for ground data
        has_ground = self.scan_result['data_type'] in (DataType.GROUND, DataType.BOTH)
        self.station_panel.setVisible(has_ground)
        self.processing_panel.setVisible(has_ground)

        # Load metadata if ground data present
        cra_data = None
        if has_ground and self.scan_result['metadata_file']:
            cra_data = load_metadata(self.scan_result['metadata_file'])
            note_ = CRA_PARSE_NOTES.get(self.scan_result['metadata_file'])
            if note_:
                self.validation_label.setText(self.validation_label.text() +
                                              f"<br><span style='color:#F57C00'>⚠ {note_}</span>")
            if cra_data:
                self.station_panel.load_from_metadata(cra_data)

        # Always load PROCESSING into the panel (falls back to defaults if no .cra).
        if has_ground:
            self.processing_panel.load_from_cra(cra_data or {})

        # v4.3 — Resolve station coords:
        #   - FORCE_CRA_STATION_COORDS on and .cra has coords -> use the .cra.
        #   - Otherwise use the receiver's own position (UBX 3D fix or RINEX
        #     header) — the same rule the pipeline applies — and warn if the
        #     .cra disagrees. Without a .cra this fills the panel from the data.
        if has_ground:
            self._resolve_station_for_display(cra_data)
            if self.processing_panel.to_processing_dict().get('ALLOW_SINGLE_FREQ'):
                self._append_note("◐ Single-frequency allowed (flagged 1F)", '#E65100')

        self.run_btn.setEnabled(self.scan_result['valid'])
    
    def _append_note(self, text: str, color: str = '#1B5E20'):
        self.validation_label.setText(self.validation_label.text() +
                                      f"<br><span style='color:{color}'>{text}</span>")

    def _resolve_station_for_display(self, cra_data: Optional[Dict]):
        """Fill the station panel with the position the pipeline will actually use."""
        force_cra = self.processing_panel.force_cra_coords()
        cra_has_coords = bool(cra_data and all(
            cra_data.get(k) not in (None, '') for k in ('STATION_LAT', 'STATION_LON', 'STATION_HEIGHT')))
        if force_cra and cra_has_coords:
            self._append_note("✓ Station from .cra (forced)")
            return
        info = None
        try:
            info = extract_station_info(self.scan_result.get('ubx_dir'))
        except Exception as e:
            self._append_note(f"⚠ Receiver position unreadable: {e}", '#F57C00')
        if info is None:
            if cra_has_coords:
                self._append_note("⚠ No receiver fix — .cra position used", '#F57C00')
            else:
                self._append_note("✗ No station position — enter it below", '#D32F2F')
            return
        name = (cra_data or {}).get('STATION_NAME') or ''         # site name only from the .cra
        self.station_panel.load_from_rinex_station({**info, 'marker_name': name})
        h_show = info.get('altitude_msl') if info.get('altitude_msl') is not None else info['altitude']
        detail = f"{info['latitude']:.4f}°N {info['longitude']:.4f}°E · {h_show:.0f} m · receiver fix"
        if info.get('altitude_msl') is None:
            detail += " (ellipsoidal h)"
        if (info.get('max_file_spread_m') or 0) > 5.0:
            self._append_note(f"⚠ Files up to {info['max_file_spread_m']:.0f} m apart — each uses its own fix",
                              '#F57C00')
        self._append_note(f"✓ Station {detail}")
        if cra_has_coords:
            try:
                h_rec = info.get('altitude_msl') if info.get('altitude_msl') is not None else info['altitude']
                a = np.array(geodetic_to_ecef_gui(float(cra_data['STATION_LAT']), float(cra_data['STATION_LON']),
                                                  float(cra_data['STATION_HEIGHT'])))      # GPS heights both
                b = np.array(geodetic_to_ecef_gui(info['latitude'], info['longitude'], h_rec))
                off = float(np.linalg.norm(a - b))
                warn_m = float(load_processing_config_from_cra(cra_data).get('STATION_MISMATCH_WARN_M', 100.0))
                if off > warn_m:
                    self._append_note(f"⚠ .cra position {off / 1000:.2f} km from the receiver — receiver used",
                                      '#F57C00')
            except (TypeError, ValueError, KeyError):
                pass

    # ------------------------------------------------------------------
    # v3.4.4 — Load-mode (open a previously executed *_output project)
    # ------------------------------------------------------------------
    def _enter_load_mode(self, directory: str, out_info: Dict[str, Any]):
        """Switch the GUI from execute-mode to load-mode and display prior results."""
        self.load_mode = True
        self.load_info = out_info
        self.output_dir = directory

        # Hide controls that don't make sense in load-mode.
        self.station_panel.setVisible(False)
        self.processing_panel.setVisible(False)
        self.progress_panel.setVisible(False)

        # Button morph: Start Processing → Close Project; hide Stop.
        self.run_btn.setText("Close Project")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)

        # Compose status message.
        _remember_dir(directory)
        kind = {DataType.GROUND: f"📡 {GROUND_DISPLAY_NAME}", DataType.SATELLITE: "🛰 Satellite",
                DataType.BOTH: f"📡🛰 {GROUND_DISPLAY_NAME} + Satellite"}.get(out_info['data_type'], '')
        msgs = [f"<span style='color:#00695C; font-weight:bold;'>📂 Loaded results · {kind}</span>"]
        for w in out_info.get('warnings', []):
            msgs.append(f"<span style='color:#F57C00'>⚠ {w}</span>")
        for e in out_info.get('errors', []):
            msgs.append(f"<span style='color:#D32F2F'>✗ {e}</span>")
        self.validation_label.setText("<br>".join(msgs))

        # Reconstruct in-memory state from on-disk artifacts.
        self.ground_ro_status = {}
        self.sat_summary_df = None
        self.ground_output_dir = out_info.get('ground_dir')
        self.sat_output_dir = out_info.get('sat_dir')

        self.ground_reasons = {}
        if self.ground_output_dir:
            try:
                self.ground_ro_status, self.ground_reasons = build_ground_status(self.ground_output_dir)
            except Exception:
                self.ground_ro_status = load_ground_ro_status_from_csv(self.ground_output_dir)
        if self.sat_output_dir:
            self.sat_summary_df = load_sat_summary_from_csv(self.sat_output_dir)

        # Build a fake scan_result so other code paths (e.g. _on_item_selected)
        # see the right data_type and don't NPE.
        self.scan_result = {
            'valid': True,
            'data_type': out_info['data_type'],
            'ubx_dir': None, 'sp3_file': None, 'era5_file': None,
            'metadata_file': None, 'obs_source': None,
            'conphs_files': [], 'has_atmprf': False, 'has_wetpf2': False,
            'errors': [], 'warnings': [], 'info': [],
        }

        # Populate the result list using the same widgets as execute-mode.
        self._populate_results()

        self.raw_canvas.show_placeholder("Select an item to view observations")
        self.derived_canvas.show_placeholder("Select an item to view profiles")
        self.atm_canvas.show_placeholder("Select an item to view atmospheric profiles")

    def _exit_load_mode(self):
        """Restore execute-mode UI state (called when a fresh dir is picked)."""
        if not self.load_mode:
            # Even on the very first call, ensure controls are in execute-mode state.
            self.run_btn.setText("Start Processing")
            self.stop_btn.setVisible(True)
            self.progress_panel.setVisible(True)
            return
        self.load_mode = False
        self.load_info = None
        self._clear_results_extras()
        self.run_btn.setText("Start Processing")
        self.stop_btn.setVisible(True)
        self.progress_panel.setVisible(True)
        self.result_list.clear()
        self.raw_canvas.show_placeholder("Select an item to view observations")
        self.derived_canvas.show_placeholder("Select an item to view profiles")
        self.atm_canvas.show_placeholder("Select an item to view atmospheric profiles")

    def _close_project(self):
        """In load-mode, the Run button acts as 'Close Project' — reset UI."""
        self._exit_load_mode()
        self.dir_edit.clear()
        self.validation_label.clear()
        self.input_dir = None
        self.output_dir = None
        self.run_btn.setEnabled(False)
        self.result_list.clear()
        self._clear_results_extras()
        self.station_panel.setVisible(False)
        self.processing_panel.setVisible(False)

    def _run_pipeline(self):
        data_type = self.scan_result['data_type']
        has_ground = data_type in (DataType.GROUND, DataType.BOTH)
        has_satellite = data_type in (DataType.SATELLITE, DataType.BOTH)
        
        # Validate ground config if needed
        if has_ground:
            station = self.station_panel.get_station_config()
            if not station:
                QMessageBox.warning(
                    self, "No station position",
                    "No station position could be found:\n"
                    "• no metadata.cra with coordinates, and\n"
                    "• no usable receiver position in the data (UBX 3D fix / RINEX APPROX POSITION).\n\n"
                    "Enter Lat / Lon / Altitude in the Station panel and run again.")
                return

            # v3.4.4 — Persist the user's settings non-destructively.
            #   1. Always write station fields (the user may have edited them).
            #   2. Merge PROCESSING into the existing .cra without wiping it.
            #   3. If no .cra existed before, create one alongside the input.
            processing_to_save = self.processing_panel.to_processing_dict()
            cra_path = self.scan_result['metadata_file']
            if not cra_path:
                # v4.3: no .cra yet -> create it with every default, so the file
                # documents all settings the run used.
                cra_path = os.path.join(self.input_dir, 'metadata.cra')
                processing_to_save = {**load_processing_config_from_cra({}), **processing_to_save}
            merge_save_metadata(
                cra_path,
                station_fields=self.station_panel.to_metadata(),
                processing_fields=processing_to_save,
            )
            self.scan_result['metadata_file'] = cra_path
            # v4.3: the child gets the COMPLETE PROCESSING (defaults + whole .cra),
            # not only the keys shown in the panel.
            self._current_processing_cfg = load_processing_config_from_cra(load_metadata(cra_path) or {})
            apply_processing_config(self._current_processing_cfg)       # same rules in this process
            self._keep_intermediate = self.processing_panel.keep_intermediate()
        else:
            self._current_processing_cfg = dict(PROCESSING_DEFAULTS)
            self._keep_intermediate = False
        
        # Setup output directories
        self.output_dir = os.path.join(os.path.dirname(self.input_dir), 
                               os.path.basename(self.input_dir) + '_output')

        # Setup output directories - BESIDE the input folder, not inside
        parent_dir = os.path.dirname(self.input_dir)
        input_name = os.path.basename(self.input_dir.rstrip(os.sep))
        self.output_dir = os.path.join(parent_dir, f'{input_name}_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        if has_ground:
            self.ground_output_dir = os.path.join(self.output_dir, 'ground')
            os.makedirs(self.ground_output_dir, exist_ok=True)
        
        if has_satellite:
            self.sat_output_dir = os.path.join(self.output_dir, 'satellite')
            os.makedirs(self.sat_output_dir, exist_ok=True)




        os.makedirs(self.output_dir, exist_ok=True)
        
        if has_ground:
            self.ground_output_dir = os.path.join(self.output_dir, 'ground')
            os.makedirs(self.ground_output_dir, exist_ok=True)
        
        if has_satellite:
            self.sat_output_dir = os.path.join(self.output_dir, 'satellite')
            os.makedirs(self.sat_output_dir, exist_ok=True)
        
        # Reset state
        self.run_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_panel.reset()
        self.result_list.clear()
        self.raw_canvas.show_placeholder("Processing...")
        self.derived_canvas.show_placeholder("Processing...")
        
        self.ground_done = not has_ground
        self.sat_done = not has_satellite
        self.ground_success = False
        self.sat_success = False
        self.ground_intermediate_data = None
        self.ground_ro_status = {}
        self.ground_reasons = {}
        self.ground_message = ''
        self.sat_summary_df = None
        
        # Setup multiprocessing
        self.progress_queue = mp.Queue()
        
        # Start ground pipeline
        if has_ground:
            station = self.station_panel.get_station_config()
            station_dict = {
                'latitude': station.latitude,
                'longitude': station.longitude,
                'altitude': station.altitude,              # GPS height; converted in the pipeline
                'name': station.name,
                'height_ref': station.height_ref,
                'geoid_sep_m': station.geoid_sep_m,
            }
            
            self.ground_process = mp.Process(
                target=run_ground_pipeline,
                args=(
                    station_dict,
                    self.scan_result['ubx_dir'],
                    self.scan_result['sp3_file'],
                    self.scan_result['era5_file'],
                    self.ground_output_dir,
                    self.progress_queue,
                    self._current_processing_cfg,   # v3.4.4
                    self._keep_intermediate,        # v3.4.4
                )
            )
            self.ground_process.start()
        
        # Start satellite pipeline
        if has_satellite:
            self.sat_process = mp.Process(
                target=run_satellite_pipeline,
                args=(
                    self.input_dir,
                    self.sat_output_dir,
                    self.progress_queue
                )
            )
            self.sat_process.start()
        
        # Timer to poll queue
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_progress)
        self.poll_timer.start(100)
    
    def _poll_progress(self):
        try:
            while not self.progress_queue.empty():
                msg = self.progress_queue.get_nowait()
                
                if msg[0] == 'progress':
                    # ('progress', pipeline_type, message, fraction)
                    pipeline_type = msg[1]
                    message = msg[2]
                    fraction = msg[3]
                    
                    prefix = "[Ground] " if pipeline_type == 'ground' else "[Satellite] "
                    self.progress_panel.set_status("Processing", prefix + message, fraction)
                
                elif msg[0] == 'done':
                    # ('done', pipeline_type, success, message, result_path)
                    pipeline_type = msg[1]
                    success = msg[2]
                    message = msg[3]
                    result_path = msg[4]
                    
                    if pipeline_type == 'ground':
                        self.ground_done = True
                        self.ground_success = success
                        self.ground_message = message
                        if success and result_path and os.path.exists(result_path):
                            self.ground_intermediate_data = pd.read_csv(result_path)
                            # v4.3: events, single-frequency flag and skip reasons
                            self.ground_ro_status, self.ground_reasons = build_ground_status(
                                self.ground_output_dir, self.ground_intermediate_data)
                    
                    elif pipeline_type == 'satellite':
                        self.sat_done = True
                        self.sat_success = success
                        if success and result_path and os.path.exists(result_path):
                            self.sat_summary_df = pd.read_csv(result_path)
                    
                    # Check if all pipelines done
                    if self.ground_done and self.sat_done:
                        self.poll_timer.stop()
                        self._on_all_pipelines_finished()
        except:
            pass
    
    def _stop_pipeline(self):
        if self.poll_timer:
            self.poll_timer.stop()
            self.poll_timer = None
        
        if self.ground_process and self.ground_process.is_alive():
            self.ground_process.kill()
            self.ground_process.join(timeout=0.5)
        
        if self.sat_process and self.sat_process.is_alive():
            self.sat_process.kill()
            self.sat_process.join(timeout=0.5)
        
        self.ground_process = None
        self.sat_process = None
        self.progress_queue = None
        
        self.progress_panel.set_complete(False, "Processing cancelled")
        self.run_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.raw_canvas.show_placeholder("Processing cancelled")
        self.derived_canvas.show_placeholder("Processing cancelled")
        self.atm_canvas.show_placeholder("Processing cancelled")
    
    def _on_all_pipelines_finished(self):
        # Cleanup
        self.ground_process = None
        self.sat_process = None
        self.progress_queue = None
        
        overall_success = self.ground_success or self.sat_success
        
        self.progress_panel.set_complete(overall_success, "Processing complete")
        self.run_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Populate result list (3.5.2: summary, hidden items, quality card, report)
        self._populate_results()
        
        self.raw_canvas.show_placeholder("Select an item to view")
        self.derived_canvas.show_placeholder("Select an item to view profiles")
        self.atm_canvas.show_placeholder("Select an item to view atmospheric profiles")
        
        # Summary message
        msg_parts = []
        if self.ground_success:
            st = [_norm_ground_state(v) for v in self.ground_ro_status.values()]
            dual, single, empty = st.count('ro_ok'), st.count('ro_ok_1f'), st.count('ro_empty')
            line = f"Ground: {len(st)} items — {dual} profiles (dual-frequency)"
            if single:
                line += f", {single} SINGLE-FREQUENCY profiles (no ionospheric correction)"
            if empty:
                line += f", {empty} RO candidates without profile"
            msg_parts.append(line)
            if getattr(self, 'ground_message', ''):
                msg_parts.append(self.ground_message)
        elif self.scan_result['data_type'] in (DataType.GROUND, DataType.BOTH):
            msg_parts.append(f"Ground FAILED: {getattr(self, 'ground_message', '')}")
        if self.sat_success and self.sat_summary_df is not None:
            success_count = self.sat_summary_df['success'].sum()
            msg_parts.append(f"Satellite: {success_count}/{len(self.sat_summary_df)} events")
        
        if msg_parts:
            QMessageBox.information(
                self, "Processing Complete",
                "\n".join(msg_parts) + f"\n\nResults saved to:\n{self.output_dir}"
            )
    
    # ------------------------------------------------------------------
    # 3.5.2 — results summary, hidden items, session quality, report
    # ------------------------------------------------------------------
    def _clear_results_extras(self):
        self.summary_label.setText("")
        self.show_all_cb.setVisible(False)
        self.quality_label.setVisible(False)

    def _populate_results(self):
        dt = (self.load_info or {}).get('data_type') if self.load_mode else (self.scan_result or {}).get('data_type')
        status = self.ground_ro_status or {}
        hidden = ground_hidden_items(self.ground_output_dir, status) if status else set()
        show = hidden if self.show_all_cb.isChecked() else set()
        hide = hidden - show
        if dt == DataType.GROUND:
            self.result_list.populate_ground(status, hidden=hide)
            self.legend_label.setText(GROUND_LEGEND)
        elif dt == DataType.SATELLITE:
            self.result_list.populate_satellite(self.sat_summary_df)
            self.legend_label.setText("● Success    ○ Failed    [VAL] Has validation")
        elif dt == DataType.BOTH:
            self.result_list.populate_both(status, self.sat_summary_df, hidden=hide)
            self.legend_label.setText("Ground: " + GROUND_LEGEND + " | Satellite: ●/○")
        has_ground = bool(status) and dt in (DataType.GROUND, DataType.BOTH)
        if has_ground:
            st = [_norm_ground_state(v) for v in status.values()]
            prof = st.count('ro_ok') + st.count('ro_ok_1f')
            self.summary_label.setText(
                f"<b>{prof} profile{'s' if prof != 1 else ''}</b> · {st.count('ro_empty')} RO, no profile · "
                f"{st.count('no_ro') - len(hidden)} not RO" + (f" · {len(hidden)} never below 5°" if hidden else ''))
            self.show_all_cb.setVisible(bool(hidden))
            q = {}
            qf = os.path.join(self.ground_output_dir or '', 'session_quality.json')
            try:
                if os.path.exists(qf):
                    q = json.load(open(qf))
                else:
                    s4 = os.path.join(self.ground_output_dir or '', 'step4_differenced.csv')
                    if os.path.exists(s4):
                        q = session_quality(pd.read_csv(s4, usecols=lambda c: c in (
                            'sat_id', 'gnssId', 'svId', 'sigID', 'timestamp', 'utc',
                            'accurate_elevation', 'accurate_azimuth')))
            except Exception:
                q = {}
            lines = session_quality_lines(q)
            self.quality_label.setText("<b>Session</b><br>" + "<br>".join(
                (f"<span style='color:#B35C00'>{l}</span>" if l.startswith('⚠') else l) for l in lines))
            self.quality_label.setVisible(bool(lines))
        else:
            self._clear_results_extras()

    def _on_item_selected(self, item: QListWidgetItem):
        if item is None:
            return
        item_id = item.data(Qt.ItemDataRole.UserRole)
        if not item_id:
            return

        is_success = item.data(Qt.ItemDataRole.UserRole + 1)
        item_type = item.data(Qt.ItemDataRole.UserRole + 2)
        # v3.4.4.1 — tri-state for ground: 'ro_ok' | 'ro_empty' | 'no_ro' | None
        ro_state = item.data(Qt.ItemDataRole.UserRole + 3)

        self._load_plots(item_id, is_success, item_type, ro_state)

    def _on_current_item_changed(self, current, previous):
        """v3.4.4.2 — fires on both mouse click and arrow-key navigation.

        Qt skips non-selectable rows (the section separators) on arrow keys
        automatically, so this handler only ever receives real result rows.
        We still null-check ``current`` for the brief moment after the list
        is cleared.
        """
        if current is None:
            return
        # Reuse the existing single-item handler.
        self._on_item_selected(current)

    def _on_tab_changed(self, index: int):
        current_item = self.result_list.currentItem()
        if current_item:
            item_id = current_item.data(Qt.ItemDataRole.UserRole)
            is_success = current_item.data(Qt.ItemDataRole.UserRole + 1)
            item_type = current_item.data(Qt.ItemDataRole.UserRole + 2)
            ro_state = current_item.data(Qt.ItemDataRole.UserRole + 3)
            if item_id:
                self._load_plots(item_id, is_success, item_type, ro_state)

    def _load_plots(self, item_id: str, is_success: bool, item_type: str,
                    ro_state: Optional[str] = None):
        current_tab = self.tab_widget.currentIndex()

        if item_type == 'ground':
            plots_dir = os.path.join(self.ground_output_dir, 'plots')
            reason = (getattr(self, 'ground_reasons', {}) or {}).get(item_id, '')
            sat_of_item = re.sub(r'_e\d+$', '', item_id)       # GPS_7_e2 -> GPS_7

            if current_tab == 0:
                raw_path = os.path.join(plots_dir, f'{sat_of_item}_raw.png')
                self.raw_canvas.load_from_png(raw_path)
            elif current_tab == 1:
                if is_success:
                    derived_path = os.path.join(plots_dir, f'{item_id}_derived.png')
                    self.derived_canvas.load_from_png(derived_path)
                elif os.path.exists(os.path.join(plots_dir, f'{item_id}_derived.png')):
                    # v4.6: page with every RO test, pass / fail, values
                    self.derived_canvas.load_from_png(os.path.join(plots_dir, f'{item_id}_derived.png'))
                elif ro_state == 'ro_empty':
                    self.derived_canvas.show_placeholder(
                        f"No profile for {item_id}\n\n"
                        "RO candidate, but the retrieval was stopped:\n" + _wrap_reason(reason))
                else:
                    self.derived_canvas.show_placeholder(
                        f"No radio occultation for {item_id}\n\n"
                        "An RO candidate needs:\n"
                        "• epochs below the RO elevation threshold, some below 0°\n"
                        "• both frequencies of its pair (or 'Allow single-frequency')\n"
                        "• a reference satellite of the same constellation\n"
                        "• the track crossing the apparent horizon (≈ −0.5° on a mountain)"
                    )
            elif current_tab == 2:
                if is_success:
                    atm_path = os.path.join(plots_dir, f'{item_id}_atmospheric.png')
                    self.atm_canvas.load_from_png(atm_path)
                elif ro_state == 'ro_empty':
                    self.atm_canvas.show_placeholder(
                        f"No profile for {item_id}\n\n" + _wrap_reason(reason) +
                        "\n\nThe Radio Occultation tab lists every RO test.")
                else:
                    self.atm_canvas.show_placeholder(
                        f"No radio occultation detected for {item_id}\n\n"
                        "Atmospheric retrieval requires successful RO processing."
                    )

        elif item_type == 'satellite':
            event_plots_dir = os.path.join(self.sat_output_dir, item_id, 'plots')

            if current_tab == 0:
                # Panel 1: raw observations
                raw_path = os.path.join(event_plots_dir, f'{item_id}_panel1_raw.png')
                self.raw_canvas.load_from_png(raw_path)
            elif current_tab == 1:
                if is_success:
                    derived_path = os.path.join(event_plots_dir, f'{item_id}_panel2_derived.png')
                    self.derived_canvas.load_from_png(derived_path)
                else:
                    self.derived_canvas.show_placeholder(
                        f"Processing failed for event {item_id}"
                    )
            elif current_tab == 2:
                if is_success:
                    atm_path = os.path.join(event_plots_dir, f'{item_id}_panel3_atmospheric.png')
                    self.atm_canvas.load_from_png(atm_path)
                else:
                    self.atm_canvas.show_placeholder(
                        f"Processing failed for event {item_id}"
                    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    if not LoginDialog.authenticate(app):
        sys.exit(0)
    
    font = QFont("Segoe UI", 12)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(exec_app(app))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
