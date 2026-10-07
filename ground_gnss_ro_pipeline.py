"""
GNSS Radio Occultation Processing Pipeline 3.5.2 (mountain-top / ground-based)
==============================================================================

Release 3.5.2. Change notes in the code tagged v4.x are the pipeline's internal
revisions leading to this release (v4.7 = 3.5.1; 3.5.2 = UI and report additions).

Receiver: stationary, INSIDE the atmosphere (n_r != 1). Retrieves the profile
between the lowest tangent point and the station.

Pipeline Steps:
    1. UBX/RINEX Parsing: raw GNSS observations
    2. SP3 Matching: precise orbits interpolated to observation times
    3a. Elevation/azimuth: geodetic (ENU) normal
    3b. Geometric Doppler: expected Doppler from satellite-receiver geometry
    4. Single Differencing + Fresnel-adaptive 2nd-order smoothing
    5. Bending: closed-form stationary-receiver solve; two branches split at the
       apparent horizon (alpha_N: tangent below station, alpha_P: above);
       iono-free per branch; PARTIAL bending alpha' = alpha_N - alpha_P
    6. Abel Inversion with a finite top at the station (boundary n_r)
    7. Atmospheric Retrieval: hydrostatic from the station barometer, ERA5 T

v4.0 Changes (from v3.4.4.2):
    - Step 3a: geodetic up instead of geocentric (was ~0.2 deg off at 36N)
    - Step 5: vectors relative to the centre of curvature (was ECEF origin
      while subtracting the Gaussian radius: ~0.5 km height bias)
    - Step 5: fsolve + abs(dt)+abs(dr) replaced by an explicit solve with
      branch selection; partial bending alpha_N - alpha_P
    - Step 6: finite-top Abel, ln n(x_r) = ln n_r (v3 forced n = 1 at the top);
      exact piecewise-linear segment integrals; climatology blend removed
    - Step 7: top boundary from the station barometer; heights sorted;
      hypsometric layers with virtual temperature; ERA5 nearest time
    - Step 4: smoothing window derived from the Fresnel crossing time;
      2nd-order fit as documented; .cra window override now honoured
    - Step 1: sub-second epochs kept (rcvTow was truncated to int);
      NAV-PVT UTC built from datetime + nano; derived-Doppler time units fixed
    - Step 2: OBS_TIME_IS_GPS option for RINEX in GPS time; debug file removed
    - RO selection: requires negative-elevation epochs; Doppler-magnitude
      threshold off by default (it favoured multipath)
"""

from __future__ import annotations
import os
import struct
import glob
import math
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

__version__ = "3.5.2"

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

SPEED_OF_LIGHT = 299792458.0
EARTH_ROTATION_RATE = 7.2921159e-5
GPS_LEAP_SECONDS = 18.0

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2
R_EARTH = 6371000.0

SIGNAL_FREQUENCIES = {
    # GPS L1 (1575.42 MHz)
    'L1C/A': 1575.420e6, 'L1 C/A': 1575.420e6, 'L1C': 1575.420e6,
    'L1 P': 1575.420e6, 'L1 P(Y)': 1575.420e6,
    'L1C(D)': 1575.420e6, 'L1C(P)': 1575.420e6, 'L1C(D+P)': 1575.420e6,
    
    # GPS L2 (1227.60 MHz)
    'L2CL': 1227.600e6, 'L2CM': 1227.600e6, 'L2C(L)': 1227.600e6, 'L2C(M)': 1227.600e6,
    'L2C(M+L)': 1227.600e6, 'L2 C/A': 1227.600e6,
    'L2 P': 1227.600e6, 'L2 P(Y)': 1227.600e6, 'L2 semi-codeless': 1227.600e6,
    
    # GPS L5 (1176.45 MHz)
    'L5I': 1176.450e6, 'L5Q': 1176.450e6, 'L5 I': 1176.450e6, 'L5 Q': 1176.450e6,
    'L5 I+Q': 1176.450e6,
    
    # Galileo E1 (1575.42 MHz)
    'E1C': 1575.420e6, 'E1B': 1575.420e6, 'E1B+C': 1575.420e6,
    'E1 PRS': 1575.420e6, 'E1A+B+C': 1575.420e6,
    
    # Galileo E5a (1176.45 MHz)
    'E5a': 1176.450e6, 'E5aI': 1176.450e6, 'E5aQ': 1176.450e6,
    'E5a I+Q': 1176.450e6,
    
    # Galileo E5b (1207.14 MHz)
    'E5bI': 1207.140e6, 'E5bQ': 1207.140e6, 'E5b I+Q': 1207.140e6,
    
    # Galileo E5 AltBOC (1191.795 MHz)
    'E5(a+b)I': 1191.795e6, 'E5(a+b)Q': 1191.795e6, 'E5 AltBOC': 1191.795e6,
    
    # Galileo E6 (1278.75 MHz)
    'E6A PRS': 1278.750e6, 'E6B': 1278.750e6, 'E6C': 1278.750e6, 'E6B+C': 1278.750e6,
    
    # GLONASS G1 (~1602 MHz, varies by channel)
    'L1OF': 1602.000e6, 'G1 C/A': 1602.000e6, 'G1 P': 1602.000e6,
    
    # GLONASS G2 (~1246 MHz, varies by channel)
    'L2OF': 1246.000e6, 'G2 C/A': 1246.000e6, 'G2 P': 1246.000e6,
    
    # GLONASS G3 (1202.025 MHz)
    'G3 I': 1202.025e6, 'G3 Q': 1202.025e6, 'G3 I+Q': 1202.025e6,
    
    # BeiDou B1I (1561.098 MHz)
    'B1I': 1561.098e6, 'B1I D1': 1561.098e6, 'B1I D2': 1561.098e6,
    'B1Q': 1561.098e6, 'B1 I+Q': 1561.098e6,
    
    # BeiDou B1C (1575.42 MHz)
    'B1C': 1575.420e6, 'B1C Data': 1575.420e6, 'B1C Pilot': 1575.420e6, 'B1C D+P': 1575.420e6,
    
    # BeiDou B2I (1207.14 MHz)
    'B2I': 1207.140e6, 'B2I D1': 1207.140e6, 'B2I D2': 1207.140e6,
    'B2Q': 1207.140e6, 'B2 I+Q': 1207.140e6,
    
    # BeiDou B2a (1176.45 MHz)
    'B2a': 1176.450e6, 'B2a Data': 1176.450e6, 'B2a Pilot': 1176.450e6,
    
    # BeiDou B3 (1268.52 MHz)
    'B3I': 1268.520e6, 'B3Q': 1268.520e6, 'B3 I+Q': 1268.520e6,
    
    # QZSS (same as GPS)
    'L1-SAIF': 1575.420e6,
    'LEX(S)': 1278.750e6, 'LEX(L)': 1278.750e6, 'LEX(S+L)': 1278.750e6,
    
    # SBAS
    'L1 SBAS': 1575.420e6,
}

# Frequency band patterns for fallback inference
FREQ_BAND_PATTERNS = {
    # GPS/QZSS/SBAS
    'L1': 1575.420e6,
    'L2': 1227.600e6,
    'L5': 1176.450e6,
    # Galileo
    'E1': 1575.420e6,
    'E5a': 1176.450e6,
    'E5b': 1207.140e6,
    'E6': 1278.750e6,
    # GLONASS
    'G1': 1602.000e6,
    'G2': 1246.000e6,
    'G3': 1202.025e6,
    # BeiDou
    'B1': 1561.098e6,  # B1I default
    'B2': 1207.140e6,  # B2I default
    'B3': 1268.520e6,
}



RINEX_TO_UBX_SIGNAL_MAP = {
    # GPS
    'L1 C/A': 'L1C/A', 'L1C': 'L1C/A', 'L1 P': 'L1C/A', 'L1 P(Y)': 'L1C/A',
    'L1C(D)': 'L1C', 'L1C(P)': 'L1C', 'L1C(D+P)': 'L1C',
    'L2 C/A': 'L2CL', 'L2C(L)': 'L2CL', 'L2C(M)': 'L2CM', 'L2C(M+L)': 'L2CL',
    'L2 P': 'L2CL', 'L2 P(Y)': 'L2CL', 'L2 semi-codeless': 'L2CL',
    'L5 I': 'L5I', 'L5 Q': 'L5Q', 'L5 I+Q': 'L5I',
    # Galileo
    'E1C': 'E1C', 'E1B': 'E1B', 'E1B+C': 'E1C', 'E1 PRS': 'E1C',
    'E5aI': 'E5a', 'E5aQ': 'E5a', 'E5a I+Q': 'E5a',
    'E5bI': 'E5bI', 'E5bQ': 'E5bQ', 'E5b I+Q': 'E5bQ',
    'E5(a+b)I': 'E5a', 'E5(a+b)Q': 'E5a', 'E5 AltBOC': 'E5a',
    # BeiDou
    'B1I': 'B1I D1', 'B1Q': 'B1I D1', 'B1 I+Q': 'B1I D1',
    'B1C Data': 'B1C', 'B1C Pilot': 'B1C', 'B1C D+P': 'B1C',
    'B2I': 'B2I D1', 'B2Q': 'B2I D1', 'B2 I+Q': 'B2I D1',
    'B2a Data': 'B2a', 'B2a Pilot': 'B2a',
    'B3I': 'B2I D1', 'B3Q': 'B2I D1', 'B3 I+Q': 'B2I D1',
    # GLONASS
    'G1 C/A': 'L1OF', 'G1 P': 'L1OF',
    'G2 C/A': 'L2OF', 'G2 P': 'L2OF',
    'G3 I': 'L2OF', 'G3 Q': 'L2OF', 'G3 I+Q': 'L2OF',
    # QZSS (map to GPS equivalents)
    'L1-SAIF': 'L1C/A', 'LEX(S)': 'L5I', 'LEX(L)': 'L5I', 'LEX(S+L)': 'L5I',
}

DOPPLER_MISSING_THRESHOLD = 0.5  # If >50% of doppler values missing, use carrier phase

# Primary dual-frequency pairs for ionospheric correction
FREQ_PAIRS = {
    'GPS': ('L1C/A', 'L2CL'),
    'BDS': ('B1I D1', 'B2I D1'),
    'GAL': ('E1C', 'E5bQ'),
    'GLO': ('L1OF', 'L2OF'),
}

FREQ_PAIRS_EXTENDED = {
    'GPS': {
        'L1': ['L1C/A', 'L1 C/A', 'L1C', 'L1 P', 'L1 P(Y)', 'L1C(D)', 'L1C(P)', 'L1C(D+P)'],
        'L2': ['L2CL', 'L2CM', 'L2C(L)', 'L2C(M)', 'L2C(M+L)', 'L2 C/A', 'L2 P', 'L2 P(Y)', 'L2 semi-codeless'],
        'L5': ['L5I', 'L5Q', 'L5 I', 'L5 Q', 'L5 I+Q'],
    },
    'GAL': {
        'E1': ['E1C', 'E1B', 'E1B+C', 'E1 PRS', 'E1A+B+C'],
        'E5a': ['E5a', 'E5aI', 'E5aQ', 'E5a I+Q'],
        'E5b': ['E5bI', 'E5bQ', 'E5b I+Q'],
    },
    'BDS': {
        'B1': ['B1I', 'B1I D1', 'B1I D2', 'B1Q', 'B1 I+Q', 'B1C', 'B1C Data', 'B1C Pilot', 'B1C D+P'],
        'B2': ['B2I', 'B2I D1', 'B2I D2', 'B2Q', 'B2 I+Q', 'B2a', 'B2a Data', 'B2a Pilot'],
    },
    'GLO': {
        'G1': ['L1OF', 'G1 C/A', 'G1 P'],
        'G2': ['L2OF', 'G2 C/A', 'G2 P'],
    },
}

N_COEFF_A1 = 77.6
N_COEFF_A2 = 3.73e5

RO_ELEVATION_THRESHOLD = 5.0     # deg: only rays below this are used (both branches)
RO_DOPPLER_THRESHOLD = 0.0       # Hz: 0 disables the |Doppler| selection (it favours multipath)
RO_MIN_EPOCHS = 25               # minimum low-elevation dual-frequency epochs
RO_MIN_NEG_ELEV_EPOCHS = 10      # minimum epochs at negative geometric elevation

POLYNOMIAL_WINDOW = 150.0        # s: MAXIMUM smoothing window (Fresnel-adaptive below this)
POLY_MIN_WINDOW = 10.0           # s: minimum smoothing window

# Polyfit segmentation: gap (sec) above which the rolling polynomial fit restarts.
POLYFIT_GAP_THRESHOLD = 5.0

# Mountain-top bending / inversion
A_BIN_M = 20.0                   # m: impact-parameter bin for averaging bending
IONO_SMOOTH_BINS = 9             # bins: smoothing of the L1-L2 bending difference (Hajj Eq. 18)
BRANCH_SMOOTH = 15               # epochs: median filter on a(t) before locating the apparent horizon
BRANCH_AMBIGUOUS_DEG = 1.0       # deg: below -this the ray always arrives going up (receiver bending < 1 deg)
BRANCH_GUARD_DEG = 0.15          # deg: epochs this close to the apparent horizon are left out of both branches
MIN_VP = 50.0                    # m/s: min in-plane transmitter velocity normal to the LOS
EVENT_GAP_S = 300.0              # s: a gap longer than this starts a new occultation event
SMOOTH_ELEV_MAX_DEG = 90.0       # deg: smooth atmos Doppler only below this elevation (90 = all; plots need all)
ALPHA_P_MODEL = 'off'            # 3.5.1: model alpha_P dropped (always 'off'); kept for research use only.
# alpha_P from a model atmosphere: 'off' | 'fill' (only where not measured) | 'always'
ALPHA_P_WET_H_M = 2000.0         # m: water-vapour scale height of the model atmosphere when ERA5 is not available
ALLOW_SINGLE_FREQ = True         # use single-frequency events (no ionospheric correction) when dual is missing
SINGLE_FREQ_SIGNALS = {'GPS': 'L1C/A', 'GAL': 'E1C', 'BDS': 'B1I D1', 'GLO': 'L1OF', 'QZSS': 'L1C/A'}
REF_SAT_ELEVATION_THRESHOLD = 30.0  # deg: reference candidates must stay above this
                                    # (own atmospheric Doppler ~0.006 Hz at 30 deg; 50 deg dropped 20% of epochs)
REF_SAT_MIN_EPOCHS = 100            # epochs a primary reference must cover
REF_SAT_JUMP_THRESHOLD = 2.0        # Hz: excess-Doppler jump counted as a slip when scoring references
REF_MODE = 'epoch_mean'             # 'epoch_mean': per-epoch weighted mean of satellites above the
                                    # threshold (depends only on that epoch); 'primary': one satellite
                                    # chosen over the whole session (v4.4 and earlier)
OBS_TIME_IS_GPS = False          # True if observation times are GPS time (no +18 s shift)
FORCE_CRA_STATION_COORDS = False # True: trust .cra coordinates over the receiver's own position
STATION_MISMATCH_WARN_M = 100.0  # m: warn when .cra and receiver position differ by more
STATION_SPLIT_WARN_M = 5.0       # m: note when files in one folder come from positions further apart

# ----------------------------------------------------------------------------
# v3.4.4 — Configurable processing constants exposed via the .cra "PROCESSING" key.
# Anything the user puts under "PROCESSING" overrides the corresponding default
# below. Missing keys fall back to the defaults — i.e. the .cra is additive.
# ----------------------------------------------------------------------------
PROCESSING_DEFAULTS = {
    # Smoothing
    'POLY_SMOOTH_WINDOW': 150.0,      # Max smoothing window (s); Fresnel-adaptive below it.
    'POLY_MIN_WINDOW': 10.0,          # Min smoothing window (s).
    'POLYFIT_GAP_THRESHOLD': 5.0,     # Restart polyfit when gap >= this many seconds.

    # RO detection
    'RO_ELEVATION_THRESHOLD': 5.0,    # deg
    'RO_DOPPLER_THRESHOLD': 0.0,      # Hz (0 = off)
    'RO_MIN_EPOCHS': 25,              # Minimum RO epochs for a valid event.
    'RO_MIN_NEG_ELEV_EPOCHS': 10,     # Minimum epochs below 0 deg geometric elevation.

    # Mountain-top bending / inversion
    'A_BIN_M': 20.0,
    'IONO_SMOOTH_BINS': 9,
    'BRANCH_SMOOTH': 15,
    'MIN_VP': 50.0,
    'OBS_TIME_IS_GPS': False,
    'EVENT_GAP_S': 300.0,
    'REF_MODE': 'epoch_mean',         # 'epoch_mean' (independent of other data) or 'primary'.
    'STATION_SPLIT_WARN_M': 5.0,      # Note when the folder's files come from positions further apart (m).
    'BRANCH_GUARD_DEG': 0.15,         # Epochs this close to the apparent horizon are not used (deg).
    'BRANCH_AMBIGUOUS_DEG': 1.0,      # Below -this elevation a ray always has its tangent point below the station (deg).
    'ALLOW_SINGLE_FREQ': True,        # Single-frequency events when dual is missing (no ionospheric correction; flagged).
    'SMOOTH_ELEV_MAX_DEG': 90.0,       # Smooth atmos Doppler below this elevation (90 = all; lower = faster).

    # Reference satellite selection (kept for forward compatibility w/ ref-rework)
    'REF_SAT_ELEVATION_THRESHOLD': 30.0,
    'REF_SAT_MIN_EPOCHS': 100,
    'REF_SAT_JUMP_THRESHOLD': 2.0,

    # Smith-Weintraub refractivity coefficients
    'N_COEFF_A1': 77.6,
    'N_COEFF_A2': 3.73e5,

    # Pipeline behaviour
    'KEEP_INTERMEDIATE_CSVS': False,   # If true, keep step1/step2/step3 CSVs after run.
    'FORCE_CRA_STATION_COORDS': False, # If true, use .cra station coords instead of the receiver's own (UBX NAV-PVT / RINEX header).
    'STATION_MISMATCH_WARN_M': 100.0,  # Warn when .cra and receiver position differ by more (m).
    'UBX_POS_MAX_HACC_M': 5.0,         # NAV-PVT fixes with worse horizontal accuracy are ignored (m).
    'UBX_POS_MAX_VACC_M': 10.0,        # ... vertical accuracy (m).
    'UBX_POS_MIN_FIXES': 10,           # Minimum good NAV-PVT fixes to trust the receiver position.
}


def load_processing_config_from_cra(cra_data: Optional[Dict]) -> Dict:
    """
    Extract the PROCESSING section from a parsed .cra dict and merge over
    PROCESSING_DEFAULTS. Returns a complete config dict.

    Unknown keys in the user's PROCESSING block are kept (forward compat),
    missing keys fall through to defaults.
    """
    cfg = dict(PROCESSING_DEFAULTS)
    if not cra_data:
        return cfg
    user_proc = cra_data.get('PROCESSING') or {}
    if not isinstance(user_proc, dict):
        return cfg
    for k, v in user_proc.items():
        cfg[k] = v
    return cfg


def apply_processing_config(cfg: Dict) -> None:
    """
    Apply a processing config dict to the module-level constants so the rest
    of the pipeline picks them up. Safe to call multiple times.
    """
    global RO_ELEVATION_THRESHOLD, RO_DOPPLER_THRESHOLD, RO_MIN_EPOCHS, RO_MIN_NEG_ELEV_EPOCHS
    global POLYNOMIAL_WINDOW, POLY_MIN_WINDOW, POLYFIT_GAP_THRESHOLD
    global N_COEFF_A1, N_COEFF_A2
    global A_BIN_M, IONO_SMOOTH_BINS, BRANCH_SMOOTH, MIN_VP, OBS_TIME_IS_GPS
    global FORCE_CRA_STATION_COORDS
    global REF_MODE, STATION_SPLIT_WARN_M
    global ALPHA_P_MODEL, ALPHA_P_WET_H_M, DERIVED_DOPPLER_OUT_S
    global SMOOTH_ELEV_MAX_DEG, ALLOW_SINGLE_FREQ, BRANCH_GUARD_DEG, BRANCH_AMBIGUOUS_DEG
    global REF_SAT_ELEVATION_THRESHOLD, REF_SAT_MIN_EPOCHS, REF_SAT_JUMP_THRESHOLD
    global EVENT_GAP_S, STATION_MISMATCH_WARN_M, UBX_POS_MAX_HACC_M, UBX_POS_MAX_VACC_M, UBX_POS_MIN_FIXES

    if not cfg:
        return

    RO_ELEVATION_THRESHOLD = float(cfg.get('RO_ELEVATION_THRESHOLD', RO_ELEVATION_THRESHOLD))
    RO_DOPPLER_THRESHOLD   = float(cfg.get('RO_DOPPLER_THRESHOLD', RO_DOPPLER_THRESHOLD))
    RO_MIN_EPOCHS          = int(cfg.get('RO_MIN_EPOCHS', RO_MIN_EPOCHS))
    POLYNOMIAL_WINDOW      = float(cfg.get('POLY_SMOOTH_WINDOW', POLYNOMIAL_WINDOW))
    POLYFIT_GAP_THRESHOLD  = float(cfg.get('POLYFIT_GAP_THRESHOLD', POLYFIT_GAP_THRESHOLD))
    N_COEFF_A1             = float(cfg.get('N_COEFF_A1', N_COEFF_A1))
    N_COEFF_A2             = float(cfg.get('N_COEFF_A2', N_COEFF_A2))
    RO_MIN_NEG_ELEV_EPOCHS = int(cfg.get('RO_MIN_NEG_ELEV_EPOCHS', RO_MIN_NEG_ELEV_EPOCHS))
    POLY_MIN_WINDOW        = float(cfg.get('POLY_MIN_WINDOW', POLY_MIN_WINDOW))
    A_BIN_M                = float(cfg.get('A_BIN_M', A_BIN_M))
    IONO_SMOOTH_BINS       = int(cfg.get('IONO_SMOOTH_BINS', IONO_SMOOTH_BINS))
    BRANCH_SMOOTH          = int(cfg.get('BRANCH_SMOOTH', BRANCH_SMOOTH))
    MIN_VP                 = float(cfg.get('MIN_VP', MIN_VP))
    OBS_TIME_IS_GPS        = bool(cfg.get('OBS_TIME_IS_GPS', OBS_TIME_IS_GPS))
    FORCE_CRA_STATION_COORDS = bool(cfg.get('FORCE_CRA_STATION_COORDS', FORCE_CRA_STATION_COORDS))
    EVENT_GAP_S            = float(cfg.get('EVENT_GAP_S', EVENT_GAP_S))
    SMOOTH_ELEV_MAX_DEG    = float(cfg.get('SMOOTH_ELEV_MAX_DEG', SMOOTH_ELEV_MAX_DEG))
    ALLOW_SINGLE_FREQ      = bool(cfg.get('ALLOW_SINGLE_FREQ', ALLOW_SINGLE_FREQ))
    ALPHA_P_MODEL          = 'off'      # 3.5.1: feature dropped; old .cra values are ignored
    DERIVED_DOPPLER_OUT_S  = 0.0        # 3.5.2: always the recorded rate (no thinning); old .cra values ignored
    ALPHA_P_WET_H_M        = float(cfg.get('ALPHA_P_WET_H_M', ALPHA_P_WET_H_M))
    REF_MODE               = str(cfg.get('REF_MODE', REF_MODE))
    STATION_SPLIT_WARN_M   = float(cfg.get('STATION_SPLIT_WARN_M', STATION_SPLIT_WARN_M))
    BRANCH_GUARD_DEG       = float(cfg.get('BRANCH_GUARD_DEG', BRANCH_GUARD_DEG))
    BRANCH_AMBIGUOUS_DEG   = float(cfg.get('BRANCH_AMBIGUOUS_DEG', BRANCH_AMBIGUOUS_DEG))
    REF_SAT_ELEVATION_THRESHOLD = float(cfg.get('REF_SAT_ELEVATION_THRESHOLD', REF_SAT_ELEVATION_THRESHOLD))
    REF_SAT_MIN_EPOCHS     = int(cfg.get('REF_SAT_MIN_EPOCHS', REF_SAT_MIN_EPOCHS))
    REF_SAT_JUMP_THRESHOLD = float(cfg.get('REF_SAT_JUMP_THRESHOLD', REF_SAT_JUMP_THRESHOLD))
    STATION_MISMATCH_WARN_M = float(cfg.get('STATION_MISMATCH_WARN_M', STATION_MISMATCH_WARN_M))
    UBX_POS_MAX_HACC_M     = float(cfg.get('UBX_POS_MAX_HACC_M', UBX_POS_MAX_HACC_M))
    UBX_POS_MAX_VACC_M     = float(cfg.get('UBX_POS_MAX_VACC_M', UBX_POS_MAX_VACC_M))
    UBX_POS_MIN_FIXES      = int(cfg.get('UBX_POS_MIN_FIXES', UBX_POS_MIN_FIXES))


def infer_signal_frequency(sig_id: str, gnss_id: str = None) -> Optional[float]:
    """
    Infer carrier frequency from signal ID using pattern matching.
    
    Args:
        sig_id: Signal identifier (e.g., 'L1 C/A', 'E5a I+Q')
        gnss_id: Optional GNSS system ID for disambiguation
    
    Returns:
        Frequency in Hz, or None if cannot determine
    """
    if not sig_id or pd.isna(sig_id):
        return None
    
    sig_id = str(sig_id).strip()
    
    # Direct lookup first
    if sig_id in SIGNAL_FREQUENCIES:
        return SIGNAL_FREQUENCIES[sig_id]
    
    # Pattern matching on frequency band
    sig_upper = sig_id.upper()
    
    # Check each band pattern
    for band, freq in FREQ_BAND_PATTERNS.items():
        # Match band at start of signal name
        if sig_upper.startswith(band.upper()):
            return freq
        # Match band anywhere in signal name (e.g., "L2C(M+L)" contains "L2")
        if band.upper() in sig_upper:
            return freq
    
    # System-specific fallbacks
    if gnss_id:
        if gnss_id == 'GPS' and '1' in sig_id:
            return 1575.420e6  # Assume L1
        elif gnss_id == 'GPS' and '2' in sig_id:
            return 1227.600e6  # Assume L2
        elif gnss_id == 'GPS' and '5' in sig_id:
            return 1176.450e6  # Assume L5
        elif gnss_id == 'GAL' and '1' in sig_id:
            return 1575.420e6  # E1
        elif gnss_id == 'GAL' and '5' in sig_id:
            return 1176.450e6  # E5a default
        elif gnss_id == 'GAL' and '7' in sig_id:
            return 1207.140e6  # E5b
        elif gnss_id == 'BDS' and '2' in sig_id:
            return 1561.098e6  # B1I (RINEX uses '2' for B1)
        elif gnss_id == 'BDS' and '7' in sig_id:
            return 1207.140e6  # B2I
        elif gnss_id == 'GLO':
            if '1' in sig_id:
                return 1602.000e6
            elif '2' in sig_id:
                return 1246.000e6
    
    return None

def get_signal_frequency(sig_id: str, gnss_id: str = None) -> float:
    """
    Get carrier frequency for signal, with fallback inference.
    Returns NaN if frequency cannot be determined.
    """
    freq = SIGNAL_FREQUENCIES.get(sig_id)
    if freq is not None:
        return freq
    
    freq = infer_signal_frequency(sig_id, gnss_id)
    if freq is not None:
        return freq
    
    return np.nan

GLO_FDMA_STEP = {'L1OF': 0.5625e6, 'G1 C/A': 0.5625e6, 'G1 P': 0.5625e6,
                 'L2OF': 0.4375e6, 'G2 C/A': 0.4375e6, 'G2 P': 0.4375e6}


def glonass_frequency(sig_id: str, k: Optional[float]) -> float:
    """Carrier frequency of a GLONASS FDMA signal on channel k (G1: 1602 + 0.5625k MHz, G2: 1246 + 0.4375k MHz)."""
    base = get_signal_frequency(sig_id, 'GLO')
    step = GLO_FDMA_STEP.get(sig_id)
    if step is None or k is None or not np.isfinite(k):
        return base
    return base + float(k) * step


def estimate_glonass_channels(df: pd.DataFrame) -> Dict[str, int]:
    """
    Fallback when the receiver did not report the channel (e.g. RINEX without the
    'GLONASS SLOT / FRQ #' header): carrier phase (cycles) vs pseudorange (m) has
    slope f/c, because range, clock and troposphere enter both identically.
    Returns {sat_id: k}. Uses every GLONASS signal with >= 30 epochs; L1 and L2 must agree.
    """
    out = {}
    g = df[df['gnssId'] == 'GLO']
    for sat, sd in g.groupby(g['gnssId'] + '_' + g['svId'].astype(str)):
        ks = []
        for sig, ss in sd.groupby('sigID'):
            step = GLO_FDMA_STEP.get(sig)
            ph = pd.to_numeric(ss['carrierPhase'], errors='coerce')
            pr = pd.to_numeric(ss['pseudorange'], errors='coerce')
            ok = ph.notna() & pr.notna()
            if step is None or ok.sum() < 30 or np.ptp(pr[ok]) < 1000.0:
                continue
            f = SPEED_OF_LIGHT * np.polyfit(pr[ok].values, ph[ok].values, 1)[0]
            k = (f - get_signal_frequency(sig, 'GLO')) / step
            if abs(k - round(k)) < 0.3 and -7 <= round(k) <= 6:
                ks.append(int(round(k)))
        if ks and all(k == ks[0] for k in ks):
            out[sat] = ks[0]
    return out


def get_frequency_band(sig_id: str) -> Optional[str]:
    """
    Extract frequency band from signal ID.
    Returns band name like 'L1', 'L2', 'E1', 'B1', etc.
    """
    if not sig_id or pd.isna(sig_id):
        return None
    
    sig_id = str(sig_id).upper()
    
    # Check standard bands
    for band in ['L1', 'L2', 'L5', 'E1', 'E5A', 'E5B', 'E6', 'G1', 'G2', 'G3', 'B1', 'B2', 'B3']:
        if band in sig_id or sig_id.startswith(band):
            return band
    
    return None

def find_dual_freq_signals(df: pd.DataFrame, gnss_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find best dual-frequency signal pair for a constellation in the data.
    
    Returns:
        (sig1, sig2) tuple or (None, None) if no valid pair found
    """
    if gnss_id not in FREQ_PAIRS_EXTENDED:
        return None, None
    
    bands = FREQ_PAIRS_EXTENDED[gnss_id]
    available_sigs = df['sigID'].unique().tolist()
    
    # Find signals in each band
    signals_by_band = {}
    for band_name, band_sigs in bands.items():
        for sig in available_sigs:
            if sig in band_sigs:
                if band_name not in signals_by_band:
                    signals_by_band[band_name] = []
                signals_by_band[band_name].append(sig)
    
    # Try to find L1/L2 or E1/E5 pair
    band_names = list(signals_by_band.keys())
    
    if len(band_names) < 2:
        return None, None
    
    # Prefer L1/E1/B1/G1 as first frequency
    primary_bands = ['L1', 'E1', 'B1', 'G1']
    secondary_bands = ['L2', 'L5', 'E5a', 'E5b', 'B2', 'G2']
    
    sig1, sig2 = None, None
    
    for pb in primary_bands:
        if pb in signals_by_band:
            sig1 = signals_by_band[pb][0]
            break
    
    for sb in secondary_bands:
        if sb in signals_by_band:
            sig2 = signals_by_band[sb][0]
            break
    
    if sig1 and sig2:
        return sig1, sig2
    
    # Fallback: just use first two different bands
    if len(band_names) >= 2:
        return signals_by_band[band_names[0]][0], signals_by_band[band_names[1]][0]
    
    return None, None



# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class StationConfig:
    latitude: float
    longitude: float
    altitude: float
    name: str = "Station"
    # Surface meteorological data for accurate refractivity.
    # If provided, N_surface is computed from P, T, e using Smith-Weintraub.
    # If not provided, falls back to standard atmosphere N=315 * exp(-h/7km).
    surface_pressure_hPa: Optional[float] = None    # station-level pressure (hPa)
    surface_temp_K: Optional[float] = None           # station-level temperature (K)
    surface_humidity_hPa: Optional[float] = None     # water vapor pressure (hPa)
    surface_N: Optional[float] = None                # direct override: N-units at station
    # v4.4: geoid separation N = h_ellipsoid - h_MSL at the station (m), from the
    # receiver (UBX NAV-PVT height - hMSL); 0 when unknown (RINEX, .cra only).
    # Only used to report heights above mean sea level.
    geoid_sep_m: float = 0.0
    # v4.5: 'msl' when `altitude` is the GPS height the user sees (above mean sea
    # level, as typed in the GUI / .cra); converted to ellipsoidal once the
    # receiver's geoid separation is known (resolve_station). 'ellipsoid' otherwise.
    height_ref: str = 'ellipsoid'

    def to_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.latitude, self.longitude, self.altitude)

    def get_gaussian_radius(self) -> float:
        lat_r = np.radians(self.latitude)
        M = (WGS84_A * (1 - WGS84_E2)) / (1 - WGS84_E2 * np.sin(lat_r) ** 2) ** 1.5
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_r) ** 2)
        return np.sqrt(M * N)

    def get_surface_refractivity(self) -> float:
        """
        Return surface refractivity in N-units at station level.
        
        Priority:
        1. Direct override (surface_N)
        2. Smith-Weintraub from met data: N = 77.6·P/T + 3.73e5·e/T²
        3. Standard atmosphere fallback: N = 315·exp(-h/7km)
        """
        if self.surface_N is not None:
            return self.surface_N
        if (self.surface_pressure_hPa is not None and 
            self.surface_temp_K is not None and
            self.surface_humidity_hPa is not None):
            P = self.surface_pressure_hPa
            T = self.surface_temp_K
            e = self.surface_humidity_hPa
            return 77.6 * P / T + 3.73e5 * e / T**2
        # Fallback: standard atmosphere
        return 315.0 * np.exp(-self.altitude / 7000.0)


@dataclass
class PipelineConfig:
    elevation_mask_high: float = 45.0
    elevation_mask_low: float = -5.0
    height_range_min: float = -1.0
    height_range_max: float = -1.0  # Sentinel: computed dynamically from Bouguer bound
    # For ground-based RO, the maximum physical impact height is:
    #   h_max = (n_r - 1)*R_c + n_r*h_station ≈ h_station + 2 km
    # Set to -1 to auto-compute from station altitude (recommended).
    # Override with a positive value to use a fixed upper bound (km).
    climatology_blend_height: float = 50.0
    min_epochs_for_bending: int = 10
    bending_angle_threshold: float = 1e-6


@dataclass
class ProcessingResult:
    success: bool
    data: Optional[pd.DataFrame] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad) ** 2)
    x = (N + height_m) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + height_m) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + height_m) * math.sin(lat_rad)
    return np.array([x, y, z])


def geodetic_up(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Unit ellipsoid normal (geodetic 'up') in ECEF."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    """ECEF -> (lat_deg, lon_deg, h_m), Bowring + 2 Newton refinements."""
    x, y, z = map(float, xyz)
    b = WGS84_A * (1 - WGS84_F)
    ep2 = (WGS84_A ** 2 - b ** 2) / b ** 2
    p = math.hypot(x, y)
    th = math.atan2(z * WGS84_A, p * b)
    lat = math.atan2(z + ep2 * b * math.sin(th) ** 3, p - WGS84_E2 * WGS84_A * math.cos(th) ** 3)
    for _ in range(2):
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - WGS84_E2 * N / (N + h)))
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(math.atan2(y, x)), h


def elevation_azimuth(sat_xyz: np.ndarray, sta_xyz: np.ndarray,
                      lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """Geometric elevation and azimuth (deg) in the local geodetic ENU frame."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    up = geodetic_up(lat_deg, lon_deg)
    d = np.asarray(sat_xyz, float) - np.asarray(sta_xyz, float)
    d = d / np.linalg.norm(d)
    el = float(np.degrees(np.arcsin(np.clip(d @ up, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(d @ east, d @ north)) % 360.0)
    return el, az


def calculate_elevation_angle(sat_xyz: np.ndarray, station_xyz: np.ndarray,
                              lat_deg: Optional[float] = None,
                              lon_deg: Optional[float] = None) -> float:
    """
    Geometric elevation (deg) w.r.t. the GEODETIC horizon.
    v4.0: v3 used the geocentric radial (~0.19 deg error at 36N), which mattered
    because the whole mountain-top window is only ~0 to -2 deg.
    """
    if lat_deg is None or lon_deg is None:
        lat_deg, lon_deg, _ = ecef_to_geodetic(station_xyz)
    return elevation_azimuth(sat_xyz, station_xyz, lat_deg, lon_deg)[0]


def compute_gravity(h_m: float, lat_deg: Optional[float] = None) -> float:
    g0 = 9.80665
    if lat_deg is None:
        return g0 * (R_EARTH / (R_EARTH + h_m)) ** 2
    lat_rad = np.radians(lat_deg)
    sin2 = np.sin(lat_rad) ** 2
    sin22 = np.sin(2 * lat_rad) ** 2
    g_surf = 9.780327 * (1 + 0.0053024 * sin2 - 0.0000058 * sin22)
    return g_surf - 3.086e-6 * h_m


def ro_selection_checks(sat_data: pd.DataFrame,
                        elevation_threshold: Optional[float] = None,
                        doppler_threshold: Optional[float] = None,
                        min_epochs: Optional[int] = None,
                        min_dual_freq_epochs: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    v4.6: the RO candidate tests per satellite, with their values - one source
    for evaluate_ro_status() and for the checklist shown in the GUI.
    Returns {sat_id: {'candidate': bool, 'freq_mode': 'dual'|'single'|None,
                      'checks': [{'name', 'passed', 'value', 'need'}, ...]}}.
    'passed' is True / False, or None when the test could not be evaluated.
    """
    if elevation_threshold is None:
        elevation_threshold = RO_ELEVATION_THRESHOLD
    if doppler_threshold is None:
        doppler_threshold = RO_DOPPLER_THRESHOLD
    if min_epochs is None:
        min_epochs = RO_MIN_EPOCHS
    if min_dual_freq_epochs is None:
        min_dual_freq_epochs = RO_MIN_EPOCHS
    out: Dict[str, Dict[str, Any]] = {}
    if sat_data.empty:
        return out
    df = sat_data.copy()
    if 'sat_id' not in df.columns:
        if 'gnssId' in df.columns and 'svId' in df.columns:
            df['sat_id'] = df['gnssId'].astype(str) + '_' + df['svId'].astype(str)
        else:
            return out
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    t_col = 'timestamp' if 'timestamp' in df.columns else 'utc'

    for sat_id, group in df.groupby('sat_id'):
        checks = []
        if 'atmos_dopp_poli' not in group.columns or elev_col not in group.columns:
            out[sat_id] = {'candidate': False, 'freq_mode': None, 'checks': [
                {'name': 'Processed data', 'passed': False, 'value': 'no elevation / atmospheric Doppler', 'need': ''}]}
            continue
        el = pd.to_numeric(group[elev_col], errors='coerce')
        low = group[el < elevation_threshold]
        n_low = low[t_col].nunique()
        el_min = float(el.min()) if el.notna().any() else np.nan
        checks.append({'name': f'Tracked below {elevation_threshold:g}° elevation',
                       'passed': n_low >= min_epochs,
                       'value': f'{n_low} epochs, lowest {el_min:+.2f}°',
                       'need': f'≥ {min_epochs} epochs'})
        # (original test 1) atmospheric Doppler available at low elevation
        dop_ok = (el < elevation_threshold) & (pd.to_numeric(group['atmos_dopp_poli'], errors='coerce').abs()
                                               > doppler_threshold)
        ro_count = int(dop_ok.sum())
        n_dop_ep = group.loc[dop_ok, t_col].nunique()
        checks.append({'name': 'Atmospheric Doppler (reference satellite found)',
                       'passed': ro_count >= min_epochs,
                       'value': f'{n_dop_ep} of {n_low} low epochs'
                                + (f', |Doppler| > {doppler_threshold:g} Hz' if doppler_threshold > 0 else ''),
                       'need': f'≥ {min_epochs} signal-epochs'})
        # (original test 2) rays from below the horizon
        n_neg = float((el < 0.0).sum())
        if 'sigID' in group.columns and group['sigID'].nunique() > 0:
            n_neg = n_neg / group['sigID'].nunique()
        checks.append({'name': 'Below the horizon (0°)', 'passed': n_neg >= RO_MIN_NEG_ELEV_EPOCHS,
                       'value': f'{int(round(n_neg))} epochs', 'need': f'≥ {RO_MIN_NEG_ELEV_EPOCHS} epochs'})
        # (original test 3) dual frequency, or the single-frequency fallback
        gnss_id = group['gnssId'].iloc[0] if 'gnssId' in group.columns else None
        single_ok, n_single = False, 0
        if gnss_id in SINGLE_FREQ_SIGNALS and t_col in low.columns:
            n_single = low.loc[low['sigID'] == SINGLE_FREQ_SIGNALS[gnss_id], t_col].nunique()
            single_ok = ALLOW_SINGLE_FREQ and n_single >= min_epochs
        dual_count = 0
        if gnss_id in FREQ_PAIRS and t_col in low.columns:
            s1, s2 = FREQ_PAIRS[gnss_id]
            dual_count = len(set(low.loc[low['sigID'] == s1, t_col]) & set(low.loc[low['sigID'] == s2, t_col]))
            pair_txt = f'{s1} + {s2}'
        else:
            pair_txt = 'no pair defined'
        dual_ok = dual_count >= min_dual_freq_epochs
        if dual_ok:
            checks.append({'name': f'Dual frequency ({pair_txt})', 'passed': True,
                           'value': f'{dual_count} common epochs', 'need': f'≥ {min_dual_freq_epochs}'})
        elif single_ok:
            checks.append({'name': f'Dual frequency ({pair_txt})', 'passed': None,
                           'value': f'only {dual_count} common epochs - SINGLE-FREQUENCY used '
                                    f'({SINGLE_FREQ_SIGNALS[gnss_id]}, {n_single} epochs, no iono correction)',
                           'need': f'≥ {min_dual_freq_epochs}'})
        else:
            checks.append({'name': f'Dual frequency ({pair_txt})', 'passed': False,
                           'value': f'{dual_count} common epochs'
                                    + ('' if ALLOW_SINGLE_FREQ else f' (single-frequency mode off; {n_single} L1 epochs)'),
                           'need': f'≥ {min_dual_freq_epochs}'})
        candidate = (ro_count >= min_epochs) and (n_neg >= RO_MIN_NEG_ELEV_EPOCHS) and (dual_ok or single_ok)
        out[sat_id] = {'candidate': bool(candidate), 'freq_mode': 'dual' if dual_ok else ('single' if single_ok else None),
                       'checks': checks}
    return out


def evaluate_ro_status(
    sat_data: pd.DataFrame,
    elevation_threshold: Optional[float] = None,
    doppler_threshold: Optional[float] = None,
    min_epochs: Optional[int] = None,
    min_dual_freq_epochs: Optional[int] = None
) -> Dict[str, bool]:
    """RO candidate per satellite (thresholds default to the live .cra values). See ro_selection_checks()."""
    res = ro_selection_checks(sat_data, elevation_threshold, doppler_threshold, min_epochs, min_dual_freq_epochs)
    return {k: v['candidate'] for k, v in res.items()}


# ============================================================================
# STEP 1: UBX OR RNX PARSING
# ============================================================================

class UBXParser:
    GNSS_ID_MAP = {
        0: 'GPS', 1: 'SBAS', 2: 'GAL', 3: 'BDS',
        4: 'IMES', 5: 'QZSS', 6: 'GLO', 7: 'NavIC'
    }

    SIGNAL_MAP = {
        (0, 0): "L1C/A", (0, 3): "L2CL", (0, 4): "L2CM", (0, 6): "L5I", (0, 7): "L5Q",
        (1, 0): "L1C/A",
        (2, 0): "E1C", (2, 1): "E1B", (2, 3): "E5aI", (2, 4): "E5aQ",
        (2, 5): "E5bI", (2, 6): "E5bQ",
        (3, 0): "B1I D1", (3, 1): "B1I D2", (3, 2): "B2I D1", (3, 3): "B2I D2",
        (3, 5): "B1C", (3, 7): "B2a",
        (5, 0): "L1C/A", (5, 1): "L1S", (5, 4): "L2CM", (5, 5): "L2CL",
        (5, 8): "L5I", (5, 9): "L5Q",
        (6, 0): "L1OF", (6, 2): "L2OF",
        (7, 0): "L5A"
    }

    def __init__(self):
        self.rawx_data = {}
        self.measx_data = {}
        self.navsat_data = {}
        self.navpvt_data = {}
        self.exception_count = 0

    def parse_file(self, file_path: str) -> None:
        messages = self._read_ubx_messages(file_path)
        self._process_messages(messages)

    def parse_directory(self, directory: str, progress_callback: Optional[Callable] = None) -> ProcessingResult:
        ubx_files = sorted(glob.glob(os.path.join(directory, '*.ubx')))
        if not ubx_files:
            return ProcessingResult(False, message=f"No .ubx files in {directory}")

        for i, ubx_file in enumerate(ubx_files):
            self.parse_file(ubx_file)
            if progress_callback:
                progress_callback(f"Parsing UBX {i+1}/{len(ubx_files)}", (i+1)/len(ubx_files) * 0.1)

        rows = self._merge_data()
        df = pd.DataFrame(rows)

        return ProcessingResult(
            success=True,
            data=df,
            message=f"Parsed {len(ubx_files)} files, {len(rows)} observations",
            metadata={'file_count': len(ubx_files), 'exception_count': self.exception_count}
        )

    def _read_ubx_messages(self, file_path: str) -> List[Tuple[int, int, bytes]]:
        with open(file_path, 'rb') as f:
            data = f.read()
        messages = []
        idx = 0
        while idx < len(data) - 8:
            if data[idx] == 0xB5 and data[idx + 1] == 0x62:
                msg_class = data[idx + 2]
                msg_id = data[idx + 3]
                length = struct.unpack_from('<H', data, idx + 4)[0]
                payload = data[idx + 6:idx + 6 + length]
                messages.append((msg_class, msg_id, payload))
                idx += 6 + length + 2
            else:
                idx += 1
        return messages

    def _process_messages(self, messages: List[Tuple[int, int, bytes]]) -> None:
        for msg_class, msg_id, payload in messages:
            try:
                if (msg_class, msg_id) == (0x02, 0x14):
                    for sat in self._parse_rxm_measx(payload):
                        key = (round(sat['iTOW'], 3), sat['gnss'], sat['svId'])
                        self.measx_data[key] = sat
                elif (msg_class, msg_id) == (0x02, 0x15):
                    for sat in self._parse_rxm_rawx(payload):
                        key = (sat['rcvTow'], sat['gnss'], sat['svId'], sat['sigID'])
                        self.rawx_data[key] = sat
                elif (msg_class, msg_id) == (0x01, 0x35):
                    for sat in self._parse_nav_sat(payload):
                        key = (round(sat['iTOW'], 3), sat['gnss'], sat['svId'])
                        self.navsat_data[key] = sat
                elif (msg_class, msg_id) == (0x01, 0x07):
                    navpvt = self._parse_nav_pvt(payload)
                    self.navpvt_data[round(navpvt['iTOW'], 3)] = navpvt
            except Exception:
                self.exception_count += 1

    def _parse_rxm_measx(self, payload: bytes) -> List[Dict]:
        sats = []
        numSV = struct.unpack_from('<B', payload, 34)[0]
        iTOW = struct.unpack_from('<I', payload, 4)[0]
        for i in range(numSV):
            offset = 44 + i * 24
            gnssId, svId, cNo, _ = struct.unpack_from('<BBBB', payload, offset)
            _, dopplerHz = struct.unpack_from('<ii', payload, offset + 4)
            codePhase = struct.unpack_from('<I', payload, offset + 16)[0]
            sats.append({
                'iTOW': iTOW * 1e-3, 'gnss': self.GNSS_ID_MAP.get(gnssId, '?'),
                'svId': svId, 'cno': cNo, 'dopplerHz': dopplerHz,
                'codePhase': codePhase * 2 ** -21,
            })
        return sats

    def _parse_rxm_rawx(self, payload: bytes) -> List[Dict]:
        sats = []
        rcvTow = struct.unpack_from('<d', payload, 0)[0]
        numMeas = struct.unpack_from('<B', payload, 11)[0]
        for i in range(numMeas):
            offset = 16 + i * 32
            prMes, cpMes, doMes = struct.unpack_from('<ddf', payload, offset)
            gnssId, svId, sigID, freqId = struct.unpack_from('<BBBB', payload, offset + 20)
            cno = struct.unpack_from('<B', payload, offset + 26)[0]
            trkStat = struct.unpack_from('<B', payload, offset + 30)[0]
            sats.append({
                'rcvTow': round(rcvTow, 3), 'gnss': self.GNSS_ID_MAP.get(gnssId, '?'),
                'svId': svId, 'prMes': prMes, 'cpMes': cpMes, 'doppler': doMes, 'cno': cno,
                'sigID': self.SIGNAL_MAP.get((gnssId, sigID), f"unknown({sigID})"),
                # v4.1: GLONASS FDMA channel number (only meaningful for GLONASS)
                'glo_k': (freqId - 7) if gnssId == 6 else None,
                'trkStat': trkStat,
            })
        return sats

    def _parse_nav_sat(self, payload: bytes) -> List[Dict]:
        sats = []
        iTOW = struct.unpack_from('<I', payload, 0)[0]
        numSvs = struct.unpack_from('<B', payload, 5)[0]
        for i in range(numSvs):
            offset = 8 + i * 12
            gnssId, svId, cno, elev, azim, _, _ = struct.unpack_from('<BBBbhhI', payload, offset)
            sats.append({
                'iTOW': iTOW * 1e-3, 'gnss': self.GNSS_ID_MAP.get(gnssId, '?'),
                'svId': svId, 'elev': elev, 'azim': azim, 'cno': cno
            })
        return sats

    def _parse_nav_pvt(self, payload: bytes) -> Dict:
        iTOW = struct.unpack_from('<I', payload, 0)[0]
        year, month, day, hour, minute, second = struct.unpack_from('<HBBBBB', payload, 4)
        nano = struct.unpack_from('<i', payload, 16)[0]
        lon, lat, height = struct.unpack_from('<iii', payload, 24)
        fixType = struct.unpack_from('<B', payload, 20)[0]
        # v4.0: nano can be negative; build the time arithmetically
        try:
            utc_dt = datetime(year, month, day, hour, minute, min(second, 59)) + \
                timedelta(seconds=max(second - 59, 0), microseconds=nano / 1000.0)
            utc = utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            utc = ''
        return {
            'iTOW': iTOW * 1e-3, 'lat': lat * 1e-7, 'lon': lon * 1e-7,
            'height': height * 1e-3, 'fixType': fixType, 'utc': utc
        }

    def _nearest_pvt(self, rcvTow: float) -> Dict:
        """UTC for an epoch without NAV-PVT, from the nearest NAV-PVT (<= 2 s away)."""
        if not hasattr(self, '_pvt_keys'):
            self._pvt_keys = np.array(sorted(self.navpvt_data.keys()))
        if len(self._pvt_keys) == 0:
            return {}
        k = self._pvt_keys[np.argmin(np.abs(self._pvt_keys - rcvTow))]
        dt = rcvTow - k
        if abs(dt) > 2.0:
            return {}
        pvt = dict(self.navpvt_data[k])
        try:
            t = pd.Timestamp(pvt['utc']) + pd.Timedelta(seconds=dt)
            pvt['utc'] = t.strftime('%Y-%m-%dT%H:%M:%S.%f')
        except Exception:
            return {}
        return pvt

    def _merge_data(self) -> List[Dict]:
        rows = []
        for key in sorted(self.rawx_data):
            rcvTow, gnss, svId, sigID = key
            rawx = self.rawx_data[key]
            iTOW_key = (rcvTow, gnss, svId)
            measx = self.measx_data.get(iTOW_key, {})
            navsat = self.navsat_data.get(iTOW_key, {})
            pvt = self.navpvt_data.get(round(rcvTow, 3), {})
            if not pvt:
                # RAWX epochs not aligned with NAV-PVT: shift the nearest PVT UTC
                pvt = self._nearest_pvt(rcvTow)

            if sigID in ['L1C/A', 'L1OF', 'L1C', 'E1C', 'B1I', 'B1C']:
                codePhase = measx.get('codePhase', '')
            else:
                codePhase = ''

            rows.append({
                'timestamp': rcvTow, 'utc': pvt.get('utc', ''), 'gnssId': gnss,
                'svId': svId, 'sigID': sigID, 'elevation': navsat.get('elev', ''),
                'azimuth': navsat.get('azim', ''), 'carrierPhase': rawx.get('cpMes', ''),
                'pseudorange': rawx.get('prMes', ''), 'doppler': rawx.get('doppler', ''),
                'codePhase': codePhase, 'cno': rawx.get('cno', ''),
                'glo_k': rawx.get('glo_k', None), 'trkStat': rawx.get('trkStat', None)
            })
        return rows


def parse_ubx_directory(
    input_dir: str, output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> ProcessingResult:
    parser = UBXParser()
    result = parser.parse_directory(input_dir, progress_callback)
    if result.success:
        # receiver's own position: checksum-verified, quality-filtered NAV-PVT fixes
        info = extract_ubx_station_info(input_dir)
        if info is not None:
            result.metadata['ubx_station'] = info
    if result.success and output_csv and result.data is not None:
        result.data.to_csv(output_csv, index=False)
    return result


def resolve_station(station: 'StationConfig', receiver_pos: Optional[Dict[str, float]],
                    force_cra: Optional[bool] = None) -> Tuple['StationConfig', str]:
    """
    v4.1: pick the station position. The receiver's own fix (UBX NAV-PVT or RINEX
    APPROX POSITION) wins unless FORCE_CRA_STATION_COORDS is set. Every 1 km of
    position error costs ~0.5 Hz of geometric-Doppler error at the horizon, i.e.
    ~0.8 km of retrieved height, so rounded .cra coordinates are not usable.
    Met fields (surface P/T/e/N) are kept from the .cra station.
    """
    if force_cra is None:
        force_cra = FORCE_CRA_STATION_COORDS
    if receiver_pos and receiver_pos.get('geoid_sep_m') is not None:
        station.geoid_sep_m = float(receiver_pos['geoid_sep_m'])       # a property of the place
    if station.height_ref == 'msl':                                   # GPS height -> ellipsoidal
        station.altitude = station.altitude + station.geoid_sep_m
        station.height_ref = 'ellipsoid'
    if not receiver_pos or force_cra:
        return station, 'cra'
    a = geodetic_to_ecef(station.latitude, station.longitude, station.altitude)
    b = geodetic_to_ecef(receiver_pos['latitude'], receiver_pos['longitude'], receiver_pos['altitude'])
    off = float(np.linalg.norm(a - b))
    new = StationConfig(latitude=receiver_pos['latitude'], longitude=receiver_pos['longitude'],
                        altitude=receiver_pos['altitude'], name=station.name,
                        surface_pressure_hPa=station.surface_pressure_hPa,
                        surface_temp_K=station.surface_temp_K,
                        surface_humidity_hPa=station.surface_humidity_hPa,
                        surface_N=station.surface_N,
                        geoid_sep_m=station.geoid_sep_m, height_ref='ellipsoid')
    note = f"receiver position, {receiver_pos.get('source', 'receiver')} ({off:.0f} m from .cra)"
    if off > STATION_MISMATCH_WARN_M:
        note += " WARNING: .cra coordinates are off"
    return new, note


def era5_station_met(era5_file: str, lat: float, lon: float, h_m: float,
                     when: Optional[str] = None) -> Optional[Dict[str, float]]:
    """P (hPa), T (K), e (hPa) at the station height from ERA5 (log-P interpolation in height)."""
    try:
        import xarray as xr
        with xr.open_dataset(era5_file) as ds:
            T_da, q_da, z_da, _ = _era5_extract(ds, lat, lon, when)
            P = ds['pressure_level'].values.astype(float)
            T, q, z = T_da.values.astype(float), q_da.values.astype(float), z_da.values / 9.80665
        o = np.argsort(z)
        z, P, T, q = z[o], P[o], T[o], q[o]
        Ps = float(np.exp(np.interp(h_m, z, np.log(P))))
        Ts = float(np.interp(h_m, z, T))
        qs = float(np.interp(h_m, z, q))
        es = qs * Ps / (0.622 + 0.378 * qs)
        return {'P': Ps, 'T': Ts, 'e': es}
    except Exception:
        return None


UBX_POS_MAX_HACC_M = 5.0     # m: reject NAV-PVT fixes with worse reported horizontal accuracy
UBX_POS_MAX_VACC_M = 10.0    # m: ... and vertical accuracy
UBX_POS_MIN_FIXES = 10       # minimum good fixes for a station position


def _ubx_checksum_ok(data: bytes, idx: int, length: int) -> bool:
    """8-bit Fletcher checksum over class, id, length and payload."""
    ck_a = ck_b = 0
    for byte in data[idx + 2: idx + 6 + length]:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    end = idx + 6 + length
    return end + 1 < len(data) and data[end] == ck_a and data[end + 1] == ck_b


def _ubx_good_fixes(fpath: str, max_hacc_m: float, max_vacc_m: float) -> Dict[str, Any]:
    """Checksum-verified 3D NAV-PVT fixes of one .ubx file (iTOW in s, heights in m)."""
    out = {'itow': [], 'lat': [], 'lon': [], 'h': [], 'hmsl': [], 'rejected': 0}
    with open(fpath, 'rb') as f:
        data = f.read()
    idx, n = 0, len(data)
    while idx < n - 8:
        if data[idx] != 0xB5 or data[idx + 1] != 0x62:
            idx += 1
            continue
        length = struct.unpack_from('<H', data, idx + 4)[0]
        if idx + 8 + length > n:
            break
        if data[idx + 2] == 0x01 and data[idx + 3] == 0x07 and length >= 92:
            if not _ubx_checksum_ok(data, idx, length):
                out['rejected'] += 1
                idx += 1                          # resync: this was not a real message
                continue
            pl = idx + 6
            itow = struct.unpack_from('<I', data, pl)[0] * 1e-3
            fix_type, flags = data[pl + 20], data[pl + 21]
            lon_i, lat_i, h_i, hm_i = struct.unpack_from('<iiii', data, pl + 24)
            hacc, vacc = struct.unpack_from('<II', data, pl + 40)
            if fix_type == 3 and (flags & 0x01) and hacc / 1e3 <= max_hacc_m and vacc / 1e3 <= max_vacc_m:
                out['itow'].append(itow)
                out['lat'].append(lat_i * 1e-7)
                out['lon'].append(lon_i * 1e-7)
                out['h'].append(h_i * 1e-3)
                out['hmsl'].append(hm_i * 1e-3)
            else:
                out['rejected'] += 1
        idx += 8 + length
    return out


def _summarise_fixes(fx: Dict[str, Any], name: str) -> Dict[str, Any]:
    lat, lon, hgt, hmsl = (np.asarray(fx[k], float) for k in ('lat', 'lon', 'h', 'hmsl'))
    la, lo, h = float(np.median(lat)), float(np.median(lon)), float(np.median(hgt))
    xyz = geodetic_to_ecef(la, lo, h)
    lat_m = (lat - la) * 111_132.0
    lon_m = (lon - lo) * 111_320.0 * np.cos(np.radians(la))
    return {
        'latitude': la, 'longitude': lo, 'altitude': h,
        'ecef_x': float(xyz[0]), 'ecef_y': float(xyz[1]), 'ecef_z': float(xyz[2]),
        'marker_name': name, 'n_fixes': len(lat), 'n_rejected': int(fx.get('rejected', 0)),
        'altitude_msl': float(np.median(hmsl)),
        'geoid_sep_m': float(np.median(hgt - hmsl)),
        'horizontal_scatter_m': float(np.std(np.hypot(lat_m, lon_m))),
        'vertical_scatter_m': float(np.std(hgt)),
    }


def extract_ubx_station_info(input_dir: str,
                             max_hacc_m: Optional[float] = None,
                             max_vacc_m: Optional[float] = None,
                             min_fixes: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Station position from the receiver's own NAV-PVT solutions in the .ubx files
    of a directory - the UBX counterpart of extract_rinex_station_info().

    A fix is used only if its UBX checksum is valid, fixType == 3 (3D) with
    gnssFixOK, and hAcc <= max_hacc_m, vAcc <= max_vacc_m. Returns the median of
    the good fixes (latitude, longitude, altitude [ellipsoidal], altitude_msl
    [the GPS height people use], geoid_sep_m, ecef_x/y/z, ...); None if too few.
    v4.5: also 'files' - the same summary per file - and 'max_file_spread_m',
    the largest distance between two files' positions (several antenna set-ups
    in one folder).
    """
    max_hacc_m = UBX_POS_MAX_HACC_M if max_hacc_m is None else max_hacc_m
    max_vacc_m = UBX_POS_MAX_VACC_M if max_vacc_m is None else max_vacc_m
    min_fixes = UBX_POS_MIN_FIXES if min_fixes is None else min_fixes
    files = sorted(glob.glob(os.path.join(input_dir, '*.[uU][bB][xX]')))
    allfx = {'itow': [], 'lat': [], 'lon': [], 'h': [], 'hmsl': [], 'rejected': 0}
    per_file = []
    for fpath in files:
        fx = _ubx_good_fixes(fpath, max_hacc_m, max_vacc_m)
        for k in ('itow', 'lat', 'lon', 'h', 'hmsl'):
            allfx[k] += fx[k]
        allfx['rejected'] += fx['rejected']
        if len(fx['lat']) >= min_fixes:
            summ = _summarise_fixes(fx, os.path.basename(fpath))
            summ.update({'file': os.path.basename(fpath), 't0': min(fx['itow']), 't1': max(fx['itow'])})
            per_file.append(summ)
    if len(allfx['lat']) < min_fixes:
        return None
    info = _summarise_fixes(allfx, os.path.splitext(os.path.basename(files[0]))[0] if files else 'Unknown')
    info['files'] = per_file
    info['max_file_spread_m'] = _max_spread_m(per_file)
    return info


def _max_spread_m(per_file: List[Dict[str, Any]]) -> float:
    pts = [geodetic_to_ecef(f['latitude'], f['longitude'], f['altitude']) for f in per_file]
    return float(max((np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:]), default=0.0))


def rinex_file_stations(input_dir: str) -> List[Dict[str, Any]]:
    """Per-file receiver position (APPROX POSITION XYZ) and time span (GPS seconds of week)."""
    from rinex_parser import RINEXParser
    out = []
    pats = ['*.rnx', '*.RNX', '*.[0-9][0-9]o', '*.[0-9][0-9]O', '*.obs', '*.OBS', '*_MO.rnx', '*_MO.RNX']
    files = sorted(set(f for pt in pats for f in glob.glob(os.path.join(input_dir, pt))))
    for fpath in files:
        try:
            pr = RINEXParser(fpath)
            pr.parse_header_only()
            info = pr.get_station_geodetic()
            if info is None or pr.first_obs_time is None:
                continue
            tow = lambda t: ((t.weekday() + 1) % 7) * 86400 + t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            t1 = pr.last_obs_time or pr.first_obs_time
            info.update({'file': os.path.basename(fpath), 't0': tow(pr.first_obs_time), 't1': tow(t1) + 1.0,
                         'geoid_sep_m': None, 'altitude_msl': None})
            out.append(info)
        except Exception:
            continue
    return out


def assign_station_columns(df: pd.DataFrame, per_file: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    v4.5: tag every observation with the receiver position of the file it came
    from (sta_lat, sta_lon, sta_h [ellipsoidal], sta_geoid, sta_file), matched by
    time. Observations outside every file's span keep NaN (the session station
    is used for them).
    """
    df = df.copy()
    t = pd.to_numeric(df['timestamp'], errors='coerce').values.astype(float)
    cols = {c: np.full(len(df), np.nan) for c in ('sta_lat', 'sta_lon', 'sta_h', 'sta_geoid')}
    fname = np.full(len(df), '', dtype=object)
    for f in sorted(per_file, key=lambda f: f.get('n_fixes', 0)):      # larger files win overlaps
        m = (t >= f['t0'] - 1.5) & (t <= f['t1'] + 1.5)
        cols['sta_lat'][m] = f['latitude']
        cols['sta_lon'][m] = f['longitude']
        cols['sta_h'][m] = f['altitude']
        g = f.get('geoid_sep_m')
        cols['sta_geoid'][m] = np.nan if g is None else g
        fname[m] = f['file']
    for c, v in cols.items():
        df[c] = v
    df['sta_file'] = fname
    return df


def extract_station_info(input_dir: str) -> Optional[Dict[str, Any]]:
    """Receiver position from UBX (NAV-PVT) or, failing that, RINEX (APPROX POSITION XYZ)."""
    info = extract_ubx_station_info(input_dir)
    if info is not None:
        info['source'] = 'UBX NAV-PVT'
        return info
    info = extract_rinex_station_info(input_dir)
    if info is not None:
        info['source'] = 'RINEX header'
    return info


def extract_rinex_station_info(input_dir: str) -> Optional[Dict[str, Any]]:
    """
    Extract station position from the first valid RINEX file header in directory.
    Returns dict with keys: latitude, longitude, altitude, ecef_x, ecef_y, ecef_z, marker_name
    or None if no valid station position found.
    """
    from rinex_parser import RINEXParser
    
    rnx_patterns = ['*.rnx', '*.RNX', '*.[0-9][0-9]o', '*.[0-9][0-9]O',
                    '*.obs', '*.OBS', '*_MO.rnx', '*_MO.RNX']
    rnx_files = []
    for pattern in rnx_patterns:
        rnx_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    rnx_files = sorted(set(rnx_files))
    
    for rnx_file in rnx_files:
        try:
            parser = RINEXParser(rnx_file)
            parser.parse_header_only()
            station_info = parser.get_station_geodetic()
            if station_info is not None:
                return station_info
        except Exception:
            continue
    return None


def parse_rnx_directory(
    input_dir: str,
    output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> ProcessingResult:
    """
    Parse RINEX observation files from directory.
    Output format matches parse_ubx_directory().
    """
    from rinex_parser import RINEXParser
    
    # Find RNX files (various extensions)
    rnx_patterns = ['*.rnx', '*.RNX', '*.[0-9][0-9]o', '*.[0-9][0-9]O', 
                    '*.obs', '*.OBS', '*_MO.rnx', '*_MO.RNX']
    rnx_files = []
    for pattern in rnx_patterns:
        rnx_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    rnx_files = sorted(set(rnx_files))
    
    if not rnx_files:
        return ProcessingResult(False, message=f"No RINEX files in {input_dir}")
    
    all_observations = []
    station_info = None
    
    for i, rnx_file in enumerate(rnx_files):
        try:
            parser = RINEXParser(rnx_file)
            observations = parser.parse()
            all_observations.extend(observations)
            
            # Extract station position from first file that has it
            if station_info is None:
                station_info = parser.get_station_geodetic()
            
            if progress_callback:
                progress_callback(f"Parsing RNX {i+1}/{len(rnx_files)}", (i+1)/len(rnx_files) * 0.1)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Failed to parse {os.path.basename(rnx_file)}: {e}", None)
    
    if not all_observations:
        return ProcessingResult(False, message="No observations extracted from RINEX files")
    
    df = pd.DataFrame(all_observations)
    
    # Normalize column names to match UBX output
    column_map = {
        'sigId': 'sigID',  # RINEX uses lowercase 'i'
    }
    df.rename(columns=column_map, inplace=True)
    
    # Map RINEX signal names to UBX signal names
    if 'sigID' in df.columns:
        df['sigID'] = df['sigID'].map(lambda x: RINEX_TO_UBX_SIGNAL_MAP.get(x, x))
    
    # Ensure all expected columns exist
    expected_cols = ['timestamp', 'utc', 'gnssId', 'svId', 'sigID', 'elevation',
                     'azimuth', 'carrierPhase', 'pseudorange', 'doppler', 'codePhase', 'cno',
                     'lli', 'glo_k']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder columns
    df = df[expected_cols]
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    return ProcessingResult(
        success=True,
        data=df,
        message=f"Parsed {len(rnx_files)} RINEX files, {len(df)} observations",
        metadata={'file_count': len(rnx_files), 'source': 'RINEX', 'rinex_station': station_info}
    )


def check_doppler_availability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if Doppler data is available and valid.
    
    Returns:
        dict with keys:
            - has_doppler: bool - True if sufficient doppler data exists
            - missing_ratio: float - ratio of missing/invalid doppler values
            - total_rows: int
            - valid_doppler_rows: int
    """
    if 'doppler' not in df.columns:
        return {
            'has_doppler': False,
            'missing_ratio': 1.0,
            'total_rows': len(df),
            'valid_doppler_rows': 0
        }
    
    # Check for valid numeric doppler values
    doppler_valid = pd.to_numeric(df['doppler'], errors='coerce')
    valid_count = doppler_valid.notna().sum()
    total_count = len(df)
    
    missing_ratio = 1.0 - (valid_count / total_count) if total_count > 0 else 1.0
    
    return {
        'has_doppler': missing_ratio < DOPPLER_MISSING_THRESHOLD,
        'missing_ratio': missing_ratio,
        'total_rows': total_count,
        'valid_doppler_rows': int(valid_count)
    }


DERIVED_DOPPLER_WINDOW_S = 1.0   # s: quadratic Savitzky-Golay window for d(phase)/dt
DERIVED_DOPPLER_OUT_S = 0.0      # s: output interval after derivation (0 = keep the recorded rate; 3.5.2 default)
PHASE_JUMP_TOL_CYC = 0.5         # cycles: per-epoch phase-step anomaly treated as a jump/slip


def _repair_phase_steps(phase: np.ndarray, tol: float, k: int = 11) -> Tuple[np.ndarray, int]:
    """
    Repair receiver clock jumps (Septentrio ms jumps) and cycle slips inside one
    continuous segment. Each epoch-to-epoch phase step is compared with the
    rolling median of its neighbours; steps off by more than `tol` cycles are
    replaced by that median. Smooth Doppler changes the step by ~0.01 cycle at
    20 Hz, so real dynamics are untouched.
    Returns (repaired phase, number of repaired steps).
    """
    if len(phase) < 3:
        return phase, 0
    dph = np.diff(phase)
    med = pd.Series(dph).rolling(k, center=True, min_periods=1).median().values
    bad = np.abs(dph - med) > tol
    if bad.any():
        dph = np.where(bad, med, dph)
    return np.concatenate([[phase[0]], phase[0] + np.cumsum(dph)]), int(bad.sum())


def derive_doppler_from_carrier_phase(
    df: pd.DataFrame,
    window_s: Optional[float] = None,
    out_interval_s: Optional[float] = None,
    progress_callback: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Doppler from carrier phase (RINEX without D observables). v4.1 rewrite.

    Per satellite and signal:
      1. split into continuous segments at missing phase, time gaps
         (> 2.5 x sample interval) and loss-of-lock flags (LLI bit 0);
         nothing is ever interpolated across a break (v3 extrapolated a constant
         phase over leading/trailing gaps, which differentiated to a false 0 Hz);
      2. repair clock jumps / cycle slips inside each segment (_repair_phase_steps);
      3. D = -d(phase)/dt from a quadratic Savitzky-Golay derivative over window_s;
         segments shorter than the window get NaN, never a fallback value;
      4. optionally keep one sample per out_interval_s (the fit already averages
         over window_s, so 20 Hz -> 1 Hz loses nothing at Fresnel time scales).
    Timestamps must be in seconds.
    """
    from scipy.signal import savgol_filter
    if window_s is None:
        window_s = DERIVED_DOPPLER_WINDOW_S
    if out_interval_s is None:
        out_interval_s = DERIVED_DOPPLER_OUT_S

    df = df.copy()
    df['carrierPhase'] = pd.to_numeric(df['carrierPhase'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    lli = pd.to_numeric(df['lli'], errors='coerce').fillna(0).astype(int) if 'lli' in df.columns \
        else pd.Series(0, index=df.index)
    df['doppler'] = np.nan
    df['doppler_derived'] = True
    df['phase_repairs'] = 0

    group_cols = [c for c in ('gnssId', 'svId', 'sigID') if c in df.columns]
    groups = df.groupby(group_cols, sort=False)
    total_repairs = 0
    for gi, (_, g) in enumerate(groups):
        if progress_callback and gi % 50 == 0:
            progress_callback(f"Deriving Doppler: {gi}/{len(groups)} signals", gi / max(len(groups), 1) * 0.1)
        g = g.sort_values('timestamp')
        t = g['timestamp'].values.astype(float)
        ph = g['carrierPhase'].values.astype(float)
        if len(t) < 5:
            continue
        dt0 = float(np.median(np.diff(t)))
        if not np.isfinite(dt0) or dt0 <= 0:
            continue
        win = int(round(window_s / dt0))
        win = max(5, win + (1 - win % 2))                       # odd, >= 5
        brk = (~np.isfinite(ph)) | ((lli.loc[g.index].values & 1) == 1)
        gap = np.concatenate([[True], np.diff(t) > 2.5 * dt0])
        seg_id = np.cumsum(gap | brk)
        D = np.full(len(t), np.nan)
        rep = np.zeros(len(t), dtype=int)
        for sid in np.unique(seg_id):
            m = np.where((seg_id == sid) & np.isfinite(ph))[0]
            if len(m) < win:
                continue
            ph_r, n_rep = _repair_phase_steps(ph[m], PHASE_JUMP_TOL_CYC)
            rep[m[0]] = n_rep
            total_repairs += n_rep
            D[m] = -savgol_filter(ph_r - ph_r[0], win, 2, deriv=1, delta=dt0, mode='interp')
        df.loc[g.index, 'doppler'] = D
        df.loc[g.index, 'phase_repairs'] = rep

    # 3.5.2: keep the recorded (native) interval for display before thinning
    _tu = np.unique(df['timestamp'].dropna().values.astype(float))
    df['obs_interval_s'] = float(np.median(np.diff(_tu))) if len(_tu) > 1 else np.nan
    if out_interval_s and out_interval_s > 0:
        t_all = df['timestamp'].values.astype(float)
        on_grid = np.abs(t_all / out_interval_s - np.round(t_all / out_interval_s)) < 1e-3
        if on_grid.sum() < len(df):
            df = df[on_grid].copy()
    df.attrs['phase_repairs'] = int(total_repairs)
    return df


def _compute_doppler_polynomial(
    t_sec: np.ndarray,
    phases: np.ndarray,
    window_size: int,
    poly_order: int,
    max_gap_sec: float = 120.0  # NEW: configurable gap threshold
) -> np.ndarray:
    """
    Compute Doppler from carrier phase using polynomial derivative.
    """
    n = len(t_sec)
    doppler = np.full(n, np.nan)
    
    # Determine actual sample interval
    if n > 1:
        median_dt = np.median(np.diff(t_sec))
    else:
        median_dt = 1.0
    
    # Adaptive window: use ~3 seconds worth of samples
    target_window_sec = 3.0
    adaptive_half_window = max(3, int(target_window_sec / median_dt / 2))
    
    for i in range(n):
        start_idx = max(0, i - adaptive_half_window)
        end_idx = min(n, i + adaptive_half_window + 1)
        
        if end_idx - start_idx < poly_order + 1:
            # Simple difference fallback
            if i > 0:
                dt = t_sec[i] - t_sec[i-1]
                if dt > 0 and dt < max_gap_sec:  # CHANGED: was 1.0
                    doppler[i] = -(phases[i] - phases[i-1]) / dt
            continue
        
        t_window = t_sec[start_idx:end_idx]
        p_window = phases[start_idx:end_idx]
        
        # Check for gaps exceeding threshold
        dt_max = np.max(np.diff(t_window))
        if dt_max > max_gap_sec:
            # Large gap - use simple difference
            if i > 0:
                dt = t_sec[i] - t_sec[i-1]
                if dt > 0 and dt < max_gap_sec:  # CHANGED: was 1.0
                    doppler[i] = -(phases[i] - phases[i-1]) / dt
            continue
        
        t_center = t_sec[i]
        t_norm = t_window - t_center
        
        try:
            coeffs = np.polyfit(t_norm, p_window, poly_order)
            doppler[i] = -coeffs[-2]  # derivative at center
        except (np.linalg.LinAlgError, ValueError):
            if i > 0:
                dt = t_sec[i] - t_sec[i-1]
                if dt > 0 and dt < max_gap_sec:
                    doppler[i] = -(phases[i] - phases[i-1]) / dt
    
    return doppler


def _compute_doppler_simple(
    t_sec: np.ndarray,
    phases: np.ndarray
) -> np.ndarray:
    """
    Simple Doppler derivation using forward difference.
    
    Doppler[i] = (phase[i+1] - phase[i]) / (t[i+1] - t[i])
    
    Faster but noisier than polynomial method.
    """
    n = len(t_sec)
    doppler = np.full(n, np.nan)
    
    for i in range(n - 1):
        dt = t_sec[i+1] - t_sec[i]
        if dt > 0 and dt < 1.0:  # Valid time step
            doppler[i] = (phases[i+1] - phases[i]) / dt
    
    # Last point: use backward difference
    if n > 1:
        dt = t_sec[-1] - t_sec[-2]
        if dt > 0 and dt < 1.0:
            doppler[-1] = (phases[-1] - phases[-2]) / dt
    
    return doppler


def ensure_doppler_data(
    df: pd.DataFrame,
    progress_callback: Optional[Callable] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ensure Doppler data is available, deriving from carrier phase if needed.
    
    This is the main function to call at pipeline start (Step 0).
    
    Args:
        df: Observation DataFrame
        progress_callback: Optional callback
    
    Returns:
        Tuple of (processed_df, status_dict)
        status_dict contains:
            - doppler_source: 'measured' or 'derived'
            - original_valid_ratio: ratio of valid measured doppler
            - derived_count: number of derived values (if any)
    """
    status = check_doppler_availability(df)
    
    result_status = {
        'doppler_source': 'measured',
        'original_valid_ratio': 1.0 - status['missing_ratio'],
        'derived_count': 0
    }
    
    if status['has_doppler']:
        # Sufficient measured Doppler data exists
        if progress_callback:
            progress_callback(
                f"Using measured Doppler ({status['valid_doppler_rows']}/{status['total_rows']} valid)", 
                None
            )
        return df, result_status
    
    # Need to derive Doppler from carrier phase
    if progress_callback:
        progress_callback(
            f"Doppler missing ({status['missing_ratio']*100:.1f}%), deriving from carrier phase...", 
            0.0
        )
    
    # Check carrier phase availability
    if 'carrierPhase' not in df.columns:
        if progress_callback:
            progress_callback("ERROR: No carrier phase data for Doppler derivation", None)
        return df, {
            'doppler_source': 'none',
            'original_valid_ratio': 0,
            'derived_count': 0,
            'error': 'No carrier phase data'
        }
    
    cp_valid = pd.to_numeric(df['carrierPhase'], errors='coerce').notna().sum()
    if cp_valid < 10:
        if progress_callback:
            progress_callback("ERROR: Insufficient carrier phase data", None)
        return df, {
            'doppler_source': 'none',
            'original_valid_ratio': 0,
            'derived_count': 0,
            'error': 'Insufficient carrier phase data'
        }
    
    # Derive Doppler (v4.1: clock-jump / slip repair, no interpolation across breaks)
    df_processed = derive_doppler_from_carrier_phase(df, progress_callback=progress_callback)
    
    # Count derived values
    derived_count = df_processed['doppler'].notna().sum()
    result_status_repairs = df_processed.attrs.get('phase_repairs', 0)
    
    if progress_callback:
        progress_callback(f"Derived {derived_count} Doppler values from carrier phase", 0.1)
    
    result_status = {
        'doppler_source': 'derived',
        'original_valid_ratio': 1.0 - status['missing_ratio'],
        'derived_count': int(derived_count),
        'phase_repairs': int(result_status_repairs),
    }
    
    return df_processed, result_status

def parse_gnss_directory(
    input_dir: str,
    output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> ProcessingResult:
    """
    Parse GNSS observation files with UBX-first, RNX-fallback logic.
    Ensures Doppler data is available (derives from carrier phase if needed).
    """
    # Check for UBX files
    ubx_files = glob.glob(os.path.join(input_dir, '*.[uU][bB][xX]'))
    
    result = None
    
    if ubx_files:
        if progress_callback:
            progress_callback("Found UBX files, parsing...", 0.0)
        result = parse_ubx_directory(input_dir, None, progress_callback)  # Don't save yet
        if result.success:
            result.metadata['source'] = 'UBX'
    
    # Try RINEX if UBX not found or failed
    if result is None or not result.success:
        rnx_patterns = ['*.rnx', '*.RNX', '*.[0-9][0-9]o', '*.[0-9][0-9]O', 
                        '*.obs', '*.OBS', '*_MO.rnx', '*_MO.RNX']
        rnx_files = []
        for pattern in rnx_patterns:
            rnx_files.extend(glob.glob(os.path.join(input_dir, pattern)))
        
        if rnx_files:
            if progress_callback:
                progress_callback("Found RINEX files, parsing...", 0.0)
            result = parse_rnx_directory(input_dir, None, progress_callback)  # Don't save yet
    
    if result is None or not result.success:
        return ProcessingResult(False, message=f"No UBX or RINEX files found in {input_dir}")

    # v4.5: each observation carries its own file's receiver position
    if result.data is not None and 'timestamp' in result.data.columns:
        if result.metadata.get('source') == 'UBX':
            info = result.metadata.get('ubx_station') or extract_ubx_station_info(input_dir)
            per_file = info['files'] if info else []
        else:
            per_file = rinex_file_stations(input_dir)
            if per_file:
                result.metadata['rinex_station'] = per_file[0]
                result.metadata['rinex_station']['max_file_spread_m'] = _max_spread_m(per_file)
        if per_file:
            result.data = assign_station_columns(result.data, per_file)
            result.metadata['station_files'] = per_file
    
    # === DOPPLER FALLBACK: Ensure Doppler data exists ===
    if result.data is not None:
        df, doppler_status = ensure_doppler_data(result.data, progress_callback)
        result.data = df
        result.metadata['doppler_source'] = doppler_status['doppler_source']
        result.metadata['doppler_derived_count'] = doppler_status.get('derived_count', 0)
        
        if doppler_status['doppler_source'] == 'derived':
            result.message += (f" | Doppler derived from carrier phase ({doppler_status['derived_count']} values, "
                               f"{doppler_status.get('phase_repairs', 0)} phase jumps/slips repaired)")
        elif doppler_status['doppler_source'] == 'none':
            return ProcessingResult(
                False, 
                message=f"No Doppler or carrier phase data available: {doppler_status.get('error', 'unknown')}"
            )
    
    # Save to CSV
    if output_csv and result.data is not None:
        result.data.to_csv(output_csv, index=False)
    
    return result







# ============================================================================
# STEP 2: SP3 MATCHING
# ============================================================================
"""
GNSS Observation SP3 Matcher - CORRECTED VERSION
Fixed based on Document 2's working implementation
"""

# GPS TIME CORRECTION
# As of 2017, GPS Time is ahead of UTC by 18 seconds.
# SP3 files are in GPS Time. CSV is in UTC.
GPS_LEAP_SECONDS = 18.0


class SP3Parser:
    """High-precision SP3 parser with microsecond-level interpolation support"""
    
    CONSTELLATION_MAP = {
        'GPS': 'G', 'GLO': 'R', 'GLONASS': 'R',
        'GAL': 'E', 'BDS': 'C', 'QZS': 'J', 'QZSS': 'J', 'IRN': 'I', 'IRNSS': 'I',
        'SBAS': 'S',
    }

    def __init__(self, sp3_file: str):
        self.epochs: Dict[datetime, Dict[str, Dict]] = {}
        self.satellites: set = set()
        self._parse(sp3_file)

    def _parse(self, filename: str) -> None:
        """Parse SP3 file. Note: SP3 timestamps are natively GPS Time."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        current_epoch = None
        
        for line in lines:
            line = line.strip()
            
            # Parse epoch line: * 2025  7 24  9  0  0.00000000
            if line.startswith('*'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        year = int(parts[1])
                        month = int(parts[2])
                        day = int(parts[3])
                        hour = int(parts[4])
                        minute = int(parts[5])
                        second = float(parts[6])
                        
                        # Parse as standard datetime representing GPS Time
                        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                        dt += timedelta(seconds=second)
                        # Store as NAIVE datetime (this represents GPS time)
                        current_epoch = dt.replace(tzinfo=None)
                        
                        if current_epoch not in self.epochs:
                            self.epochs[current_epoch] = {}
                            
                    except (ValueError, IndexError):
                        continue
            
            # Parse position line
            elif line.startswith('P') and current_epoch is not None:
                try:
                    sat_id = line[1:4]
                    parts = line[4:].split()
                    
                    if len(parts) >= 4:
                        x = np.float64(parts[0]) * 1000.0  # km -> m
                        y = np.float64(parts[1]) * 1000.0
                        z = np.float64(parts[2]) * 1000.0
                        clk = np.float64(parts[3]) * 1e-6
                        
                        if abs(x) < 50000000 and abs(y) < 50000000:
                            self.epochs[current_epoch][sat_id] = {
                                'x': x, 'y': y, 'z': z, 'clk': clk
                            }
                            self.satellites.add(sat_id)
                            
                except (ValueError, IndexError):
                    continue

    def interpolate(self, sat_id: str, target_time_utc: datetime) -> Optional[Dict]:
        """
        Get satellite position with GPS-UTC Time Sync
        target_time_utc: The UTC time from the Observation CSV
        """
        if pd.isna(target_time_utc) or target_time_utc is pd.NaT:
            return None
        
        # Convert to naive datetime if needed
        if hasattr(target_time_utc, 'to_pydatetime'):
            target_time_utc = target_time_utc.to_pydatetime()
        
        # Ensure naive UTC
        if hasattr(target_time_utc, 'tzinfo') and target_time_utc.tzinfo is not None:
            target_time_utc = target_time_utc.astimezone(timezone.utc).replace(tzinfo=None)

        # ### GPS TIME CORRECTION ###
        # To find the satellite position at 12:00:00 UTC, we must look up 
        # 12:00:18 in the SP3 file (because SP3 is GPS time).
        target_time_gps = target_time_utc + timedelta(
            seconds=0.0 if OBS_TIME_IS_GPS else GPS_LEAP_SECONDS)

        # Convert satellite ID format
        sp3_sat_id = self._convert_sat_id(sat_id)
        if not sp3_sat_id:
            return None

        # Collect available epochs for this satellite
        available_epochs = []
        positions = []
        clocks = []
        
        for epoch_time, epoch_data in self.epochs.items():
            if sp3_sat_id in epoch_data:
                available_epochs.append(epoch_time)
                sat_data = epoch_data[sp3_sat_id]
                positions.append([sat_data['x'], sat_data['y'], sat_data['z']])
                clocks.append(sat_data['clk'])

        if len(available_epochs) < 4:
            return None

        # Sort by time (all naive datetimes)
        sorted_indices = np.argsort([t.timestamp() for t in available_epochs])
        available_epochs = [available_epochs[i] for i in sorted_indices]
        positions = np.array([positions[i] for i in sorted_indices], dtype=np.float64)
        clocks = np.array([clocks[i] for i in sorted_indices], dtype=np.float64)
        
        # Convert to timestamps CONSISTENTLY (all from naive datetimes)
        available_timestamps = np.array([t.timestamp() for t in available_epochs], dtype=np.float64)
        
        # Use GPS Time for the interpolation target (also naive datetime)
        target_timestamp_gps = target_time_gps.timestamp()
        
        # *** CRITICAL RANGE CHECK ***
        # This check ensures we only interpolate within the SP3 file's time range
        # If observation is from a different day than SP3, this will correctly return None
        if target_timestamp_gps < available_timestamps[0] or target_timestamp_gps > available_timestamps[-1]:
            return None

        # Find interpolation window
        window_size = min(12, len(available_timestamps))
        center_idx = np.argmin(np.abs(available_timestamps - target_timestamp_gps))
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(available_timestamps), start_idx + window_size)

        t_window = available_timestamps[start_idx:end_idx]
        pos_window = positions[start_idx:end_idx]
        clk_window = clocks[start_idx:end_idx]

        # Perform cubic spline interpolation
        try:
            cs_x = CubicSpline(t_window, pos_window[:, 0], bc_type='natural')
            cs_y = CubicSpline(t_window, pos_window[:, 1], bc_type='natural')
            cs_z = CubicSpline(t_window, pos_window[:, 2], bc_type='natural')
            cs_clk = CubicSpline(t_window, clk_window, bc_type='natural')

            # Evaluate at GPS Time
            interp_x = np.float64(cs_x(target_timestamp_gps))
            interp_y = np.float64(cs_y(target_timestamp_gps))
            interp_z = np.float64(cs_z(target_timestamp_gps))
            interp_clk = np.float64(cs_clk(target_timestamp_gps))
            
            vel_x = np.float64(cs_x.derivative()(target_timestamp_gps))
            vel_y = np.float64(cs_y.derivative()(target_timestamp_gps))
            vel_z = np.float64(cs_z.derivative()(target_timestamp_gps))

            return {
                'gps_time_used': target_time_gps.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'interp_x': interp_x,
                'interp_y': interp_y,
                'interp_z': interp_z,
                'interp_vel_x': vel_x,
                'interp_vel_y': vel_y,
                'interp_vel_z': vel_z,
                'interp_speed': np.sqrt(vel_x**2 + vel_y**2 + vel_z**2),
                'interp_clk': interp_clk,
                'interp_clk_rate': np.float64(cs_clk.derivative()(target_timestamp_gps))
            }
        except Exception:
            return None

    def _convert_sat_id(self, gnss_sv_string: str) -> Optional[str]:
        """Convert satellite ID from CSV format to SP3 format"""
        if not isinstance(gnss_sv_string, str):
            return None
        
        gnss_sv_string = gnss_sv_string.strip()
        
        # Already in SP3 format (e.g., "G01", "E12")
        if len(gnss_sv_string) == 3 and gnss_sv_string[0].isalpha() and gnss_sv_string[1:].isdigit():
            return gnss_sv_string.upper()
        
        # CSV format (e.g., "GPS 1", "GAL 12")
        parts = gnss_sv_string.split()
        if len(parts) == 2:
            try:
                constellation = parts[0].strip().upper()
                sv_id = int(parts[1])
                prefix = self.CONSTELLATION_MAP.get(constellation)
                if prefix:
                    return f"{prefix}{sv_id:02d}"
            except ValueError:
                pass
        
        return None


def _sp3_series(sp3: 'SP3Parser', sp3_sat_id: str):
    """(epoch seconds, positions, clocks) for one satellite, time-sorted. Cached on the parser."""
    cache = sp3.__dict__.setdefault('_series_cache', {})
    if sp3_sat_id not in cache:
        ep, pos, clk = [], [], []
        for t, ed in sp3.epochs.items():
            if sp3_sat_id in ed:
                ep.append((pd.Timestamp(t) - pd.Timestamp('1970-01-01')).total_seconds())
                v = ed[sp3_sat_id]
                pos.append((v['x'], v['y'], v['z']))
                clk.append(v['clk'])
        o = np.argsort(ep)
        cache[sp3_sat_id] = (np.asarray(ep, float)[o], np.asarray(pos, float)[o], np.asarray(clk, float)[o])
    return cache[sp3_sat_id]


def match_observations_with_sp3(
    obs_csv: str,
    sp3_file: str,
    output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    batch_size: int = 2000
) -> ProcessingResult:
    """
    Match observations (UTC) with SP3 orbits (GPS time).

    v4.2: vectorised. Same interpolation as SP3Parser.interpolate (natural cubic
    spline over the 12 SP3 epochs around the target, chosen by nearest epoch),
    but each spline is built once per window and evaluated for every observation
    that uses it, instead of once per row.
    """
    df = pd.read_csv(obs_csv)
    df['parsed_utc'] = pd.to_datetime(df['utc'], format='mixed', errors='coerce')
    if getattr(df['parsed_utc'].dt, 'tz', None) is not None:
        df['parsed_utc'] = df['parsed_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
    df = df.dropna(subset=['parsed_utc']).reset_index(drop=True)
    df['sat_identifier'] = df['gnssId'].astype(str) + ' ' + df['svId'].astype(str)

    sp3 = SP3Parser(sp3_file)
    shift = pd.Timedelta(seconds=0.0 if OBS_TIME_IS_GPS else GPS_LEAP_SECONDS)
    t_gps = (df['parsed_utc'] + shift)
    tsec = (t_gps - pd.Timestamp('1970-01-01')).dt.total_seconds().values   # unit-safe

    n = len(df)
    cols = {k: np.full(n, np.nan) for k in ('interp_x', 'interp_y', 'interp_z', 'interp_vel_x',
                                            'interp_vel_y', 'interp_vel_z', 'interp_clk', 'interp_clk_rate')}
    matched = np.zeros(n, dtype=bool)

    sats = df['sat_identifier'].unique()
    for si, sat in enumerate(sats):
        sp3_id = sp3._convert_sat_id(sat)
        if not sp3_id:
            continue
        ep, pos, clk = _sp3_series(sp3, sp3_id)
        if len(ep) < 4:
            continue
        rows = np.where(df['sat_identifier'].values == sat)[0]
        tt = tsec[rows]
        inside = (tt >= ep[0]) & (tt <= ep[-1])
        rows, tt = rows[inside], tt[inside]
        if len(rows) == 0:
            continue
        w = min(12, len(ep))
        j = np.searchsorted(ep, tt)
        j0 = np.clip(j - 1, 0, len(ep) - 1)
        j1 = np.clip(j, 0, len(ep) - 1)
        center = np.where(np.abs(ep[j0] - tt) <= np.abs(ep[j1] - tt), j0, j1)
        start = np.maximum(0, center - w // 2)
        start = np.minimum(start, len(ep) - w)      # same window as min(len, start+w) with full length
        for st0 in np.unique(start):
            sel = start == st0
            sl = slice(st0, st0 + w)
            try:
                splines = [CubicSpline(ep[sl], pos[sl, k], bc_type='natural') for k in range(3)]
                cs_clk = CubicSpline(ep[sl], clk[sl], bc_type='natural')
            except Exception:
                continue
            x = tt[sel]
            r = rows[sel]
            for k, nm in enumerate(('x', 'y', 'z')):
                cols[f'interp_{nm}'][r] = splines[k](x)
                cols[f'interp_vel_{nm}'][r] = splines[k].derivative()(x)
            cols['interp_clk'][r] = cs_clk(x)
            cols['interp_clk_rate'][r] = cs_clk.derivative()(x)
            matched[r] = True
        if progress_callback and si % 10 == 0:
            progress_callback(f"SP3 matching: {si + 1}/{len(sats)} satellites", 0.1 + 0.25 * (si + 1) / len(sats))

    for k, v in cols.items():
        df[k] = v
    df['interp_speed'] = np.sqrt(df['interp_vel_x'] ** 2 + df['interp_vel_y'] ** 2 + df['interp_vel_z'] ** 2)
    df['gps_time_used'] = t_gps.dt.strftime('%Y-%m-%d %H:%M:%S.%f').where(matched, None)
    df['sp3_match_status'] = np.where(matched, 'matched', 'no_match')

    df_matched = df[matched].copy()
    if output_csv and not df_matched.empty:
        df_matched.to_csv(output_csv, index=False)
    return ProcessingResult(
        success=True,
        data=df_matched,
        message=f"Matched {int(matched.sum())}/{n} observations",
        metadata={'total': n, 'matched': int(matched.sum())}
    )


# ============================================================================
# STEP 3A & 3B: ELEVATION AND DOPPLER
# ============================================================================


def _row_station(df: pd.DataFrame, station: 'StationConfig'):
    """Per-row receiver position (sta_* columns from step 1), the session station where missing."""
    n = len(df)
    def col(c, v):
        if c in df.columns:
            return pd.to_numeric(df[c], errors='coerce').fillna(v).values.astype(float)
        return np.full(n, float(v))
    return col('sta_lat', station.latitude), col('sta_lon', station.longitude), col('sta_h', station.altitude)


def row_station_ecef(lat_d, lon_d, h_m) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat_d, float)), np.radians(np.asarray(lon_d, float))
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    h = np.asarray(h_m, float)
    return np.column_stack([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo),
                            (N * (1 - WGS84_E2) + h) * np.sin(la)])


def calculate_accurate_elevations(
    input_csv: str, 
    station: StationConfig, 
    output_csv: Optional[str] = None
) -> ProcessingResult:
    """Geometric elevation/azimuth from SP3 positions in the geodetic ENU frame (vectorised)."""
    df = pd.read_csv(input_csv)
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    if df['elevation'].notna().any():
        df = df[~(df['elevation'] < -5)]          # keep NaN (computed below)
    if df.empty:
        return ProcessingResult(False, None, "No observations after elevation filter", {'filtered_count': 0})

    lat_d, lon_d, h_m = _row_station(df, station)          # v4.5: each row's own receiver fix
    sta = row_station_ecef(lat_d, lon_d, h_m)
    la, lo = np.radians(lat_d), np.radians(lon_d)
    east = np.column_stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)])
    north = np.column_stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    up = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    S = df[['interp_x', 'interp_y', 'interp_z']].apply(pd.to_numeric, errors='coerce').values
    d = S - sta
    d = d / np.linalg.norm(d, axis=1)[:, None]
    dot = lambda a, b: np.einsum('ij,ij->i', a, b)
    df['accurate_elevation'] = np.degrees(np.arcsin(np.clip(dot(d, up), -1.0, 1.0)))
    df['accurate_azimuth'] = np.degrees(np.arctan2(dot(d, east), dot(d, north))) % 360.0
    failed = int(df['accurate_elevation'].isna().sum())
    before = len(df)
    df = df[df['accurate_elevation'] >= -5]
    if output_csv:
        df.to_csv(output_csv, index=False)
    return ProcessingResult(
        success=True, data=df,
        message=f"Calculated elevations for {len(df)} observations",
        metadata={'total_processed': before, 'passed_filter': len(df), 'failed_calculations': failed})


def calculate_geometric_doppler(
    input_csv: str, 
    station: StationConfig, 
    output_csv: Optional[str] = None
) -> ProcessingResult:
    """
    Calculate geometric Doppler from satellite motion.
    Robust frequency lookup with fallback inference.
    """
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    required_cols = ['pseudorange', 'sigID', 'interp_x', 'interp_y', 'interp_z',
                     'interp_vel_x', 'interp_vel_y', 'interp_vel_z']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return ProcessingResult(
            success=False,
            data=None,
            message=f"Missing columns: {missing}",
            metadata={'missing_columns': missing}  # FIXED: Added metadata
        )
    
    # Convert pseudorange to numeric
    df['pseudorange'] = pd.to_numeric(df['pseudorange'], errors='coerce')

    # Keep rows that have a sigID; pseudorange is optional.
    # RINEX carrier-only rows (L observable, no C) have pseudorange=NaN but
    # still carry valid Doppler/carrier phase for atmospheric differencing.
    initial_count = len(df)
    df = df.dropna(subset=['sigID']).copy()

    if df.empty:
        return ProcessingResult(
            success=False,
            data=None,
            message="No valid observations after filtering",
            metadata={'initial_count': initial_count, 'valid_count': 0}
        )
    
    station_ecef = row_station_ecef(*_row_station(df, station))     # v4.5: per-row receiver fix
    
    # Time delay: use pseudorange when available; fall back to geometric range.
    # RINEX carrier-only rows have NaN pseudorange; using geometric range
    # introduces < 0.01 Hz error in geometric Doppler — well within thresholds.
    station_ecef_tmp = station_ecef
    sat_pos_tmp = df[['interp_x', 'interp_y', 'interp_z']].values
    geom_range = np.linalg.norm(sat_pos_tmp - station_ecef_tmp, axis=1)
    pr_numeric = df['pseudorange'].values
    time_delay_values = np.where(
        np.isfinite(pr_numeric),
        pr_numeric / SPEED_OF_LIGHT,
        geom_range  / SPEED_OF_LIGHT
    )
    df['time_delay_s'] = time_delay_values

    # Get satellite positions and velocities
    sat_pos = df[['interp_x', 'interp_y', 'interp_z']].values
    sat_vel_ecef = df[['interp_vel_x', 'interp_vel_y', 'interp_vel_z']].values

    # Calculate satellite position at transmission time
    sat_pos_tx = sat_pos - sat_vel_ecef * df['time_delay_s'].values[:, np.newaxis]
    
    # Line-of-sight calculations
    los_vector = sat_pos_tx - station_ecef
    los_dist = np.linalg.norm(los_vector, axis=1)
    los_unit = los_vector / los_dist[:, np.newaxis]
    
    # Range rate
    range_rate = np.einsum('ij,ij->i', sat_vel_ecef, los_unit)
    
    # Sagnac effect
    sagnac_rate = EARTH_ROTATION_RATE * (
        sat_vel_ecef[:, 0] * station_ecef[:, 1] - sat_vel_ecef[:, 1] * station_ecef[:, 0]
    ) / SPEED_OF_LIGHT
    
    # ROBUST FREQUENCY LOOKUP with fallback
    gnss_col = 'gnssId' if 'gnssId' in df.columns else None
    
    carrier_freq = np.zeros(len(df))
    freq_missing_count = 0

    # v4.1: GLONASS FDMA channel per satellite (receiver-reported, else estimated)
    glo_k_by_sat: Dict[str, int] = {}
    if gnss_col and (df[gnss_col] == 'GLO').any():
        if 'glo_k' in df.columns:
            kk = pd.to_numeric(df['glo_k'], errors='coerce')
            glo_rows = (df[gnss_col] == 'GLO') & kk.notna()
            for (g_, sv), kv in kk[glo_rows].groupby([df.loc[glo_rows, gnss_col], df.loc[glo_rows, 'svId']]):
                glo_k_by_sat[f"{g_}_{sv}"] = int(kv.mode().iloc[0])
        missing = set(f"GLO_{sv}" for sv in df.loc[df[gnss_col] == 'GLO', 'svId'].unique()) - set(glo_k_by_sat)
        if missing:
            est = estimate_glonass_channels(df)
            glo_k_by_sat.update({k: v for k, v in est.items() if k in missing})
    glo_k_col = np.full(len(df), np.nan)

    # v4.2: one lookup per (constellation, satellite, signal) instead of per row
    keys = df[[gnss_col, 'svId', 'sigID']].astype(str).agg('|'.join, axis=1) if gnss_col \
        else df['sigID'].astype(str)
    fmap, kmap = {}, {}
    for key in keys.unique():
        if gnss_col:
            gnss_id, sv, sig_id = key.split('|', 2)
        else:
            gnss_id, sv, sig_id = None, None, key
        if gnss_id == 'GLO':
            k = glo_k_by_sat.get(f"GLO_{sv}")
            freq = glonass_frequency(sig_id, k)
            kmap[key] = np.nan if k is None else k
        else:
            freq = get_signal_frequency(sig_id, gnss_id)
        if np.isnan(freq):
            freq_missing_count += int((keys == key).sum())
            freq = 1575.420e6                                    # last resort: L1
        fmap[key] = freq
    carrier_freq = keys.map(fmap).values.astype(float)
    glo_k_col = keys.map(kmap).astype(float).values if kmap else glo_k_col

    # Clock drift contribution
    clk_rate = df['interp_clk_rate'].values if 'interp_clk_rate' in df.columns else 0
    clock_doppler = carrier_freq * clk_rate
    
    # Calculate geometric Doppler
    df['geometric_doppler'] = -carrier_freq * (range_rate + sagnac_rate) / SPEED_OF_LIGHT + clock_doppler
    df['carrier_freq_hz'] = carrier_freq  # true carrier (GLONASS: per FDMA channel)
    df['glo_k_used'] = glo_k_col
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    msg = f"Calculated geometric Doppler for {len(df)} observations"
    if freq_missing_count > 0:
        msg += f" ({freq_missing_count} used fallback frequency)"
    
    return ProcessingResult(
        success=True,
        data=df,
        message=msg,
        metadata={  # FIXED: Added metadata
            'total_observations': len(df),
            'freq_fallback_count': freq_missing_count,
            'initial_count': initial_count,
            'dropped_count': initial_count - len(df)
        }
    )


def diagnose_high_elevation_bias(input_csv: str, elevation_threshold: float = 45.0):
    """Check excess_doppler statistics at high elevation before single differencing."""
    df = pd.read_csv(input_csv)
    
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    df['excess_doppler'] = df['doppler'] - df['geometric_doppler']
    
    high_elev = df[df[elev_col] >= elevation_threshold].copy()
    
    print(f"\nHigh elevation (>{elevation_threshold}°) excess_doppler statistics:")
    print(f"  Count: {len(high_elev)}")
    
    if len(high_elev) > 0:
        print(f"  Mean:  {high_elev['excess_doppler'].mean():.2f} Hz")
        print(f"  Std:   {high_elev['excess_doppler'].std():.2f} Hz")
        print(f"  Min:   {high_elev['excess_doppler'].min():.2f} Hz")
        print(f"  Max:   {high_elev['excess_doppler'].max():.2f} Hz")
        
        print("\nPer-satellite breakdown:")
        for sat_id, group in high_elev.groupby(['gnssId', 'svId']):
            mean = group['excess_doppler'].mean()
            std = group['excess_doppler'].std()
            print(f"  {sat_id[0]}_{sat_id[1]:02d}: mean={mean:+.2f} Hz, std={std:.2f} Hz, n={len(group)}")
    else:
        print("  No observations found at this elevation threshold")
    
    return high_elev

# ============================================================================
# STEP 4: SINGLE DIFFERENCING & 2ND ORDER POLYNOMIAL FIT
# ============================================================================
def apply_single_differencing(
    input_csv: str, 
    config: PipelineConfig = PipelineConfig(), 
    output_csv: Optional[str] = None, 
    fresnel_window_sec: Optional[float] = None,
    reference_elevation_threshold: Optional[float] = None,
    min_reference_epochs: Optional[int] = None,
    station_alt_m: Optional[float] = None
) -> ProcessingResult:
    """
    Single differencing against a reference from the same constellation and signal.

    Per (constellation, signal) and epoch, the reference is:
      1. the primary reference satellite (best-scored, always above the threshold), else
      2. the sin(elevation)-weighted mean excess Doppler of the satellites above the threshold.
    v4.2: if neither exists, or the only candidate is the target satellite itself,
    the epoch is DROPPED (atmos_doppler = NaN). The old 'highest elevation'
    fallback is gone: a low reference carries its own atmospheric Doppler.

    Differencing is done in velocity-equivalent units (see v4.1) and vectorised
    with groupby instead of a per-epoch loop.
    """
    if reference_elevation_threshold is None:
        reference_elevation_threshold = REF_SAT_ELEVATION_THRESHOLD
    if min_reference_epochs is None:
        min_reference_epochs = REF_SAT_MIN_EPOCHS
    df = pd.read_csv(input_csv)
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'

    f_sat = pd.to_numeric(df['carrier_freq_hz'], errors='coerce') if 'carrier_freq_hz' in df.columns \
        else pd.Series(np.nan, index=df.index)
    _pairs = df[['sigID', 'gnssId']].drop_duplicates()
    _fb = {(a, b): get_signal_frequency(a, b) for a, b in zip(_pairs['sigID'], _pairs['gnssId'])}
    f_band = pd.Series([_fb[(a, b)] for a, b in zip(df['sigID'], df['gnssId'])], index=df.index)
    fscale = np.where(np.isfinite(f_sat) & (f_sat > 0), f_band / f_sat, 1.0)
    df['excess_doppler'] = (pd.to_numeric(df['doppler'], errors='coerce') - df['geometric_doppler']) * fscale
    df['sat_id'] = df['gnssId'] + '_' + df['svId'].astype(str)

    atmos = np.full(len(df), np.nan)
    ref_sat = np.full(len(df), '', dtype=object)
    ref_type = np.full(len(df), '', dtype=object)
    thr = reference_elevation_threshold
    pos = pd.Series(np.arange(len(df)), index=df.index)

    for (gnss_id, sig_id), g in df.groupby(['gnssId', 'sigID']):
        primary = (_select_primary_reference(g, elev_col, thr, min_reference_epochs)
                   if REF_MODE == 'primary' else None)
        ex = g['excess_doppler']
        el = g[elev_col]
        ok = np.isfinite(ex) & np.isfinite(el)
        hi = ok & (el >= thr)
        key = g['utc']

        # 1) primary reference value per epoch
        if primary is not None:
            pv = g[hi & (g['sat_id'] == primary)].groupby('utc')['excess_doppler'].first()
        else:
            pv = pd.Series(dtype=float)
        # 2) weighted mean of high satellites per epoch
        h = g[hi]
        w = np.sin(np.radians(h[elev_col]))
        wsum = w.groupby(h['utc']).sum()
        wval = (w * h['excess_doppler']).groupby(h['utc']).sum() / wsum
        nhi = h.groupby('utc').size()
        sats = h.groupby('utc')['sat_id'].agg('+'.join)

        prim_e = key.map(pv)
        wv_e = key.map(wval)
        n_e = key.map(nhi).fillna(0)
        use_p = prim_e.notna() & (g['sat_id'] != primary)
        # weighted reference must contain at least one satellite other than the target
        self_only = hi & (n_e == 1)
        use_w = ~use_p & wv_e.notna() & ~self_only
        ref = np.where(use_p, prim_e, np.where(use_w, wv_e, np.nan))

        ix = pos.loc[g.index].values
        atmos[ix] = np.where(ok, ex.values - ref, np.nan)
        ref_sat[ix] = np.where(use_p, primary or '', np.where(use_w, key.map(sats).fillna(''), ''))
        ref_type[ix] = np.where(use_p, 'primary', np.where(use_w, 'weighted_avg', 'dropped'))

    df['atmos_doppler'] = atmos / fscale                       # back to each satellite's own carrier
    df['reference_sat'] = ref_sat
    df['reference_type'] = ref_type
    df.loc[df[elev_col] >= config.elevation_mask_high, 'atmos_doppler'] = np.nan

    df = apply_fresnel_polynomial_smoothing(df, fresnel_window_sec, station_alt_m=station_alt_m)

    ref_stats = df.groupby('reference_type').size().to_dict()
    primary_refs = df.loc[df['reference_type'] == 'primary', 'reference_sat'].unique()
    if output_csv:
        df.to_csv(output_csv, index=False)
    return ProcessingResult(
        success=True,
        data=df,
        message=f"Single differencing complete. Primary refs: {list(primary_refs)}. Stats: {ref_stats}",
        metadata={'reference_stats': ref_stats, 'primary_references': list(primary_refs)}
    )


def _select_primary_reference(
    sig_group: pd.DataFrame, 
    elev_col: str,
    elevation_threshold: float,
    min_epochs: int
) -> Optional[str]:
    """
    Select the best primary reference satellite for a signal group.
    
    Criteria (in order of importance):
    1. Minimum elevation stays above threshold
    2. Maximum number of epochs (coverage)
    3. Lowest excess_doppler variance (stability)
    """
    candidates = []
    
    for sat_id, sat_data in sig_group.groupby('sat_id'):
        min_elev = sat_data[elev_col].min()
        max_elev = sat_data[elev_col].max()
        n_epochs = len(sat_data)
        
        # Must stay above threshold for entire session
        if min_elev < elevation_threshold:
            continue
        
        # Must have sufficient epochs
        if n_epochs < min_epochs:
            continue

        # v3.4.9: must have enough FINITE excess Doppler to be scored at all
        if np.isfinite(sat_data['excess_doppler'].values).sum() < min_epochs:
            continue
        
        # Compute stability metrics
        excess_std = sat_data['excess_doppler'].std()
        excess_range = sat_data['excess_doppler'].max() - sat_data['excess_doppler'].min()
        
        # Check for gaps (discontinuous coverage)
        timestamps = sat_data['timestamp'].sort_values().values
        if len(timestamps) > 1:
            max_gap = np.max(np.diff(timestamps))
        else:
            max_gap = 0
        
        # Detect potential cycle slips via excess_doppler jumps
        excess_diff = sat_data.sort_values('timestamp')['excess_doppler'].diff().abs()
        n_jumps = (excess_diff > REF_SAT_JUMP_THRESHOLD).sum()  # jump = suspect slip
        
        candidates.append({
            'sat_id': sat_id,
            'min_elev': min_elev,
            'max_elev': max_elev,
            'n_epochs': n_epochs,
            'excess_std': excess_std,
            'excess_range': excess_range,
            'max_gap': max_gap,
            'n_jumps': n_jumps,
        })
    
    if not candidates:
        return None
    
    # Score candidates
    cdf = pd.DataFrame(candidates)
    
    # Normalize metrics (lower is better for std, range, gap, jumps; higher is better for n_epochs, min_elev)
    cdf['score'] = (
        cdf['min_elev'] / 90.0 * 2.0 +           # Weight: 2 (higher elevation better)
        cdf['n_epochs'] / cdf['n_epochs'].max() * 1.5 +  # Weight: 1.5 (more coverage better)
        (1 - cdf['excess_std'] / (cdf['excess_std'].max() + 0.1)) * 1.0 +  # Weight: 1 (lower variance better)
        (1 - cdf['n_jumps'] / (cdf['n_jumps'].max() + 1)) * 2.0  # Weight: 2 (fewer jumps better)
    )
    
    cdf = cdf[np.isfinite(cdf['score'].values)]
    if cdf.empty:
        return None
    best = cdf.loc[cdf['score'].idxmax()]
    return best['sat_id']


def _compute_weighted_reference(
    epoch_data: pd.DataFrame, 
    elev_col: str, 
    elevation_threshold: float
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute elevation-weighted average of excess_doppler from high-elevation satellites.
    
    Returns:
        (weighted_doppler, satellite_list_string) or (None, None)
    """
    high_elev = epoch_data[epoch_data[elev_col] >= elevation_threshold].copy()
    
    if high_elev.empty:
        return None, None
    
    if len(high_elev) == 1:
        return high_elev['excess_doppler'].iloc[0], high_elev['sat_id'].iloc[0]
    
    # Elevation-based weights (higher = more weight)
    # Use sin(elevation) as weight - physical basis: lower multipath, better geometry
    elevations = high_elev[elev_col].values
    weights = np.sin(np.radians(elevations))
    weights = weights / weights.sum()
    
    weighted_doppler = np.sum(high_elev['excess_doppler'].values * weights)
    sat_list = '+'.join(high_elev['sat_id'].tolist())
    
    return weighted_doppler, sat_list
    
def fresnel_window_seconds(elev_deg: np.ndarray, elev_rate_deg_s: np.ndarray,
                           station_alt_m: float, wavelength: float = 0.1903) -> np.ndarray:
    """
    Time for the tangent point to cross one Fresnel diameter (mountain-top geometry).

        D   ~ sqrt(2 R h)              receiver -> tangent point distance
        F   ~ sqrt(lam D)              first Fresnel radius
        v_h ~ R |e| |de/dt|            tangent-point vertical speed (impact height ~ R e^2 / 2)
        T   = 2 F / v_h                clipped to [POLY_MIN_WINDOW, POLYNOMIAL_WINDOW]
    """
    R = R_EARTH
    D = np.sqrt(2.0 * R * max(station_alt_m, 50.0))
    F = np.sqrt(wavelength * D)
    e = np.radians(np.abs(np.asarray(elev_deg, float)))
    edot = np.radians(np.abs(np.asarray(elev_rate_deg_s, float)))
    v_h = R * np.maximum(e, 1e-4) * np.maximum(edot, 1e-6)
    return np.clip(2.0 * F / v_h, POLY_MIN_WINDOW, POLYNOMIAL_WINDOW)


def apply_fresnel_polynomial_smoothing(df: pd.DataFrame, fresnel_window_sec: Optional[float] = None,
                                       station_alt_m: Optional[float] = None,
                                       max_elev_deg: Optional[float] = None) -> pd.DataFrame:
    """
    2nd-order polynomial fit to atmos_doppler, evaluated at the window centre.
    Respects gaps >= POLYFIT_GAP_THRESHOLD seconds.

    Window:
      - fresnel_window_sec given      -> fixed window (s)
      - else station_alt_m given      -> Fresnel crossing time per epoch
      - else                          -> POLYNOMIAL_WINDOW

    v4.2: window bounds come from searchsorted, so cost grows linearly with length.
    v4.3: only epochs below max_elev_deg (default SMOOTH_ELEV_MAX_DEG = 90, i.e. all)
    are smoothed. Lower it to speed up very long sessions; the fit line in the raw
    plots then only appears below that elevation.
    """
    df = df.copy()
    df['atmos_dopp_poli'] = np.nan
    df['smooth_window_s'] = np.nan
    if 'atmos_doppler' not in df.columns or 'timestamp' not in df.columns:
        return df

    gap_threshold = POLYFIT_GAP_THRESHOLD
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    adaptive = fresnel_window_sec is None and station_alt_m is not None and elev_col in df.columns
    fixed_window = fresnel_window_sec if fresnel_window_sec is not None else POLYNOMIAL_WINDOW
    el_lim = SMOOTH_ELEV_MAX_DEG if max_elev_deg is None else max_elev_deg
    el_lim = max(el_lim, RO_ELEVATION_THRESHOLD + 2.0)       # never below what RO needs
    group_cols = ['sat_id', 'sigID'] if 'sat_id' in df.columns else ['gnssId', 'svId', 'sigID']
    out_all = np.full(len(df), np.nan)
    win_all = np.full(len(df), np.nan)
    pos = pd.Series(np.arange(len(df)), index=df.index)

    for _, group in df.groupby(group_cols):
        el = pd.to_numeric(group[elev_col], errors='coerce').values.astype(float) \
            if elev_col in group.columns else np.full(len(group), -90.0)
        if not (el < el_lim).any():
            continue
        group = group.sort_values('timestamp')
        el = pd.to_numeric(group[elev_col], errors='coerce').values.astype(float) \
            if elev_col in group.columns else np.full(len(group), -90.0)
        t = group['timestamp'].values.astype(float)
        v = group['atmos_doppler'].values.astype(float)
        if adaptive:
            edot = np.gradient(el, t) if len(el) > 2 else np.full_like(el, np.nan)
            win = fresnel_window_seconds(el, edot, station_alt_m)
            win = np.where(np.isfinite(win), win, POLYNOMIAL_WINDOW)
        else:
            win = np.full(len(t), float(fixed_window))
        seg = np.concatenate([[0], np.cumsum(np.diff(t) >= gap_threshold)])
        fin = np.isfinite(v)
        out = np.full(len(t), np.nan)
        for k in np.where(fin & (el < el_lim))[0]:
            lo = np.searchsorted(t, t[k] - win[k] / 2.0, side='left')
            hi = np.searchsorted(t, t[k] + win[k] / 2.0, side='right')
            m = np.arange(lo, hi)
            m = m[(seg[m] == seg[k]) & fin[m]]
            if len(m) < 2:
                out[k] = v[k]
                continue
            order = 2 if len(m) >= 5 else 1
            try:
                out[k] = np.polyval(np.polyfit(t[m] - t[k], v[m], order), 0.0)
            except (np.linalg.LinAlgError, ValueError):
                out[k] = v[k]
        ix = pos.loc[group.index].values
        out_all[ix] = out
        win_all[ix] = win
    df['atmos_dopp_poli'] = out_all
    df['smooth_window_s'] = win_all
    return df


# ============================================================================
# STEP 5: BENDING ANGLE RETRIEVAL  (stationary receiver inside the atmosphere)
# ============================================================================
#
# Geometry. All vectors are relative to the centre of curvature c0, fixed for
# one occultation (Hajj et al. 2002 §4.2):
#   k0 : unit chord direction, transmitter -> receiver
#   p^ : unit vector from c0 to the chord's closest point (outward)
#   A refracted ray bulges OUTWARD of the chord:
#       k_t = cos(dt) k0 + sin(dt) p^          (leaves transmitter tilted out)
#       k_r = cos(dr) k0 - sin(dr) p^          (arrives at receiver tilted in)
#       alpha = dt + dr  (> 0 for normal refraction)
#
#   Doppler, receiver fixed (v_r = 0):  f*lam = v_t . (k_t - k0)      -> dt
#   Bouguer at transmitter (n_t = 1):   a = d cos dt - s_t sin dt      -> a
#   Bouguer at receiver:                a = n_r |r_r| cos(dr - phi)    -> dr
#   The receiver equation has two roots:
#     branch -1 : ray arrives going UP   (tangent point below station) -> alpha_N
#     branch +1 : ray arrives going DOWN (no tangent below station)    -> alpha_P
#   They meet where a peaks (a -> n_r r_r), i.e. at the apparent horizon.
#   For a setting satellite: before the peak = +1, after = -1 (reverse if rising).
#
#   Partial bending  alpha'(a) = alpha_N(a) - alpha_P(a)
#   removes the air above the station, which both rays cross exactly once.


def curvature_centre(sta_xyz: np.ndarray, lat_deg: float, lon_deg: float,
                     h_ell_m: float, az_deg: float) -> Tuple[np.ndarray, float]:
    """
    Centre and radius of the circle osculating the WGS84 ellipsoid along
    azimuth az_deg at the station (Euler's formula).
    With this origin, |r_r| = R_c + h exactly.
    """
    s2 = np.sin(np.radians(lat_deg)) ** 2
    w = 1.0 - WGS84_E2 * s2
    M = WGS84_A * (1 - WGS84_E2) / w ** 1.5     # meridional radius
    N = WGS84_A / np.sqrt(w)                     # prime-vertical radius
    A = np.radians(az_deg)
    R_c = 1.0 / (np.cos(A) ** 2 / M + np.sin(A) ** 2 / N)
    c0 = np.asarray(sta_xyz, float) - (R_c + h_ell_m) * geodetic_up(lat_deg, lon_deg)
    return c0, float(R_c)


def bending_geometry(f_atm: float, lam: float, r_t: np.ndarray, v_t: np.ndarray,
                     r_r: np.ndarray, n_r: float,
                     min_vp: Optional[float] = None) -> Optional[Tuple[float, float, float, float]]:
    """
    Solve the stationary-receiver geometry for one epoch and one frequency.

    Args:
        f_atm: atmospheric Doppler (Hz), convention f = v_t.(k_t - k0)/lam
        lam:   wavelength (m)
        r_t, v_t: transmitter position / velocity relative to c0 (Earth-fixed)
        r_r:   receiver position relative to c0
        n_r:   refractive index at the receiver

    Returns:
        (dt, a, phi, acos_c) or None. Bending for a branch b is
        alpha = dt + phi + b * acos_c   (b = -1 for alpha_N, +1 for alpha_P).
    """
    if min_vp is None:
        min_vp = MIN_VP
    if not np.isfinite(f_atm):
        return None
    L = r_r - r_t
    k0 = L / np.linalg.norm(L)
    s_t = float(r_t @ k0)                      # < 0: transmitter is before the closest point
    p = r_t - s_t * k0
    d = float(np.linalg.norm(p))
    ph = p / d
    vk, vp = float(v_t @ k0), float(v_t @ ph)
    if abs(vp) < min_vp:                       # motion mostly out of plane / along LOS
        return None

    # 1) Doppler -> dt (exact: f*lam = vk (cos dt - 1) + vp sin dt), Newton
    rhs = f_atm * lam
    dt = rhs / vp
    for _ in range(5):
        g = vk * (np.cos(dt) - 1.0) + vp * np.sin(dt) - rhs
        dt -= g / (vp * np.cos(dt) - vk * np.sin(dt))

    # 2) Bouguer at the transmitter -> impact parameter
    a = d * np.cos(dt) - s_t * np.sin(dt)

    # 3) Bouguer at the receiver
    s_r = float(r_r @ k0)
    c = a / (n_r * np.hypot(s_r, d))
    if not (0.0 < c <= 1.0):
        return None
    return float(dt), float(a), float(np.arctan2(s_r, d)), float(np.arccos(c))


def apparent_horizon_prior_deg(n_r: float, r_r: float, scale_height_m: float = 7500.0) -> float:
    """
    Geometric elevation of the apparent horizon (the ray arriving horizontally),
    i.e. minus the one-sided bending of a ray tangent at the station:
        alpha ~ (n_r - 1) * sqrt(pi * r_r / (2 H))      (exponential atmosphere)
    ~ -0.4 deg for a 2.5 km station. Only used as a fallback / starting point.
    """
    return -float(np.degrees((n_r - 1.0) * np.sqrt(np.pi * r_r / (2.0 * scale_height_m))))


def label_branches(a: np.ndarray, elevation: np.ndarray,
                   smooth: Optional[int] = None,
                   el_h_prior: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Branch per epoch: -1 = ray arrives going UP (tangent point below the station,
    alpha_N), +1 = ray arrives going DOWN (alpha_P). Epochs in time order.

    v4.4: physically constrained instead of 'argmax of a over the whole track'
    (on noisy GLO_21, 2025-09-28 that put the horizon at +0.36 deg):
      elevation >= 0                      -> +1 (never a tangent point below)
      elevation <= -BRANCH_AMBIGUOUS_DEG  -> -1 (always arrives going up)
    In between, the apparent horizon el_h is found by fitting the impact
    parameter on both sides of a trial el_h with
        a = c - k1*d - q1*d^2  (d = el - el_h > 0),   a = c + k2*d - q2*d^2  (d < 0)
    k, q >= 0 (a peaks at the horizon), over all epochs from
    -BRANCH_AMBIGUOUS_DEG-0.3 to +0.3 deg, and keeping the el_h with the
    smallest residual. A data gap around the horizon is harmless: every trial
    inside the gap gives the same split. With too few epochs, el_h_prior
    (physical estimate, apparent_horizon_prior_deg) is used.
    Returns (branch array, index of the epoch nearest the apparent horizon).
    """
    from scipy.optimize import lsq_linear
    a = np.asarray(a, float)
    el = np.asarray(elevation, float)
    n = len(a)
    if n == 0:
        return np.array([], dtype=int), 0
    lo, hi = -BRANCH_AMBIGUOUS_DEG, 0.0
    fit_m = np.isfinite(a) & np.isfinite(el) & (el > lo - 0.3) & (el < hi + 0.3)
    el_h = el_h_prior if el_h_prior is not None else 0.5 * lo
    amb_pts = np.isfinite(a) & (el > lo) & (el < hi)
    if fit_m.sum() >= 8 and amb_pts.sum() >= 3:
        ef = np.radians(el[fit_m])
        af = a[fit_m] - np.nanmedian(a[fit_m])
        u = np.unique(np.round(el[fit_m], 4))
        cand = np.concatenate([[lo], (u[1:] + u[:-1]) / 2.0, [hi]])
        cand = cand[(cand >= lo) & (cand <= hi)]
        best = (np.inf, el_h)
        for c_deg in cand:
            d = ef - np.radians(c_deg)
            up = d > 0
            A = np.column_stack([np.ones_like(d),
                                 np.where(up, -d, 0.0), np.where(up, -d * d, 0.0),
                                 np.where(~up, d, 0.0), np.where(~up, -d * d, 0.0)])
            r = lsq_linear(A, af, bounds=([-np.inf, 0, 0, 0, 0], [np.inf] * 5))
            sse = float(np.sum((A @ r.x - af) ** 2))
            if sse < best[0] - 1e-9:
                best = (sse, float(c_deg))
        el_h = best[1]
    # v4.7: the fit needs data on BOTH sides of the horizon it finds. A track that stays
    # below (or above) the horizon has no peak in a, and the fit then lands on an edge of
    # the search range (GAL_21, 2025-09-28: whole track at -1.02..-0.91 deg, "horizon"
    # -1.00 deg, every epoch inside the guard band). Then use the physical estimate,
    # with a wider guard band for its uncertainty.
    guard = BRANCH_GUARD_DEG
    fin = np.isfinite(el)
    n_lo = int(np.sum(fin & (el < el_h - BRANCH_GUARD_DEG)))
    n_hi = int(np.sum(fin & (el > el_h + BRANCH_GUARD_DEG)))
    if (n_lo < 3 or n_hi < 3) and el_h_prior is not None:
        el_h = float(el_h_prior)
        guard = BRANCH_GUARD_DEG + 0.10
    br = np.where(el >= 0.0, 1, np.where(el <= lo, -1, np.where(el > el_h, 1, -1)))
    # Guard band: epochs this close to the horizon carry little information and
    # most of the noise (tested over 8 noise seeds: 0.15 deg cut the mean profile
    # error from 4.8 to 3.7 N and the worst case from 8.2 to 5.0 N). Branch 0 =
    # used in neither alpha_N nor alpha_P.
    br = np.where(np.abs(el - el_h) < guard, 0, br)
    k = int(np.argmin(np.abs(np.where(np.isfinite(el), el, np.inf) - el_h)))
    return br.astype(int), k


def bin_by_impact(a: np.ndarray, alpha: np.ndarray,
                  bin_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average alpha in bins of impact parameter. Returns (a_mean, alpha_mean, count), a ascending."""
    a = np.asarray(a, float)
    alpha = np.asarray(alpha, float)
    ok = np.isfinite(a) & np.isfinite(alpha)
    if not ok.any():
        return np.array([]), np.array([]), np.array([], dtype=int)
    k = np.floor(a[ok] / bin_m).astype(np.int64)
    _, inv = np.unique(k, return_inverse=True)
    cnt = np.bincount(inv)
    return (np.bincount(inv, weights=a[ok]) / cnt,
            np.bincount(inv, weights=alpha[ok]) / cnt,
            cnt)


def iono_free_on_bins(a1, al1, a2, al2, f1, f2, bin_m, smooth_bins):
    """
    Dual-frequency combination on common impact-parameter bins (Hajj Eq. 18):
        alpha_neut = alpha1 + c2 * <alpha1 - alpha2>,   c2 = f2^2/(f1^2 - f2^2)
    The L1-L2 difference is smoothed over `smooth_bins` bins to limit L2 noise.
    v4.7: bins with no L2 nearby (L2 tracking gap) fall back to L1 alone instead
    of being dropped; the 5th return value flags them (l1_only).
    Returns (a_bins, alpha, alpha1_bins, alpha2_on_bins, l1_only) or None.
    """
    ca1, b1, _ = bin_by_impact(a1, al1, bin_m)
    if len(ca1) < 2:
        return None
    ca2, b2, _ = bin_by_impact(a2, al2, bin_m)
    if len(ca2) >= 2:
        b2_on1 = np.interp(ca1, ca2, b2, left=np.nan, right=np.nan)
        near = np.abs(ca2[np.clip(np.searchsorted(ca2, ca1), 0, len(ca2) - 1)] - ca1)
        near = np.minimum(near, np.abs(ca2[np.clip(np.searchsorted(ca2, ca1) - 1, 0, len(ca2) - 1)] - ca1))
        b2_on1 = np.where(near <= smooth_bins * bin_m, b2_on1, np.nan)   # no bridging across L2 gaps
    else:
        b2_on1 = np.full_like(b1, np.nan)
    c2 = f2 ** 2 / (f1 ** 2 - f2 ** 2)
    diff = pd.Series(b1 - b2_on1).rolling(int(smooth_bins), center=True,
                                          min_periods=1).mean().values
    l1_only = ~np.isfinite(diff)
    return ca1, np.where(l1_only, b1, b1 + c2 * diff), b1, b2_on1, l1_only


def refractivity_table_above_station(N_r: float, h_tab_m: Optional[np.ndarray] = None,
                                     N_tab: Optional[np.ndarray] = None,
                                     T_r: Optional[float] = None, e_r: Optional[float] = None,
                                     station_h_m: float = 0.0, wet_h_m: float = 2000.0,
                                     top_m: float = 150e3):
    """
    v4.7: N(h) above the station for the model alpha_P. h = height above the
    station (m). With an ERA5 column (h_tab_m, N_tab) it is scaled so that
    N(0) = N_r (the same boundary value the Abel inversion uses) and extended
    exponentially above its top. Otherwise a physical model tied to N_r:
    dry part hydrostatic with a 6.5 K/km lapse rate (isothermal above 11 km),
    wet part N_w(0) exp(-h / wet_h_m); T_r defaults to the standard atmosphere.
    """
    grid = np.concatenate([np.linspace(0.0, 2000.0, 201), np.linspace(2010.0, top_m, 3000)])
    if h_tab_m is not None and N_tab is not None:
        h = np.asarray(h_tab_m, float); N = np.asarray(N_tab, float)
        ok = np.isfinite(h) & np.isfinite(N) & (N > 0) & (h >= 0)
        h, N = h[ok], N[ok]
        if len(h) >= 5 and h[0] < 200.0 and h[-1] > 5000.0:
            o = np.argsort(h); h, N = h[o], N[o]
            N = N * N_r / np.interp(0.0, h, N)
            k = max(1, len(h) // 5)
            Hs = -(h[-1] - h[-k - 1]) / np.log(N[-1] / N[-k - 1]) if N[-1] < N[-k - 1] else scale_height_m
            lo = grid[grid <= h[-1]]
            hi = grid[grid > h[-1]]
            Nlo = np.exp(np.interp(lo, h, np.log(N)))
            return grid, np.concatenate([Nlo, N[-1] * np.exp(-(hi - h[-1]) / Hs)]), '.nc'
    T0 = float(T_r) if T_r else 288.15 - 6.5e-3 * station_h_m
    Nw0 = N_COEFF_A2 * float(e_r) / T0 ** 2 if e_r else 0.0
    Nd0 = max(N_r - Nw0, 1.0)
    gam, gMR = 6.5e-3, 9.80665 * 28.9644e-3 / 8.314462
    h_tp = max(11000.0 - station_h_m, 0.0)
    T = np.maximum(T0 - gam * grid, T0 - gam * h_tp)
    lnP = np.where(grid <= h_tp, (gMR / gam) * np.log(T / T0),
                   (gMR / gam) * np.log((T0 - gam * h_tp) / T0) - gMR * (grid - h_tp) / (T0 - gam * h_tp))
    Nd = Nd0 * np.exp(lnP) * T0 / T
    return grid, Nd + Nw0 * np.exp(-grid / wet_h_m), 'standard-lapse'



def alpha_P_model(a: np.ndarray, r_r: float, h_tab: np.ndarray, N_tab: np.ndarray) -> np.ndarray:
    """
    v4.7: bending of the above-horizon ray with impact parameter a (a < n_r r_r),
    from the station out to space, in a spherical model atmosphere N(h):
        alpha_P(a) = -int_{r_r}^{inf} a (dn/dr) / (n sqrt(n^2 r^2 - a^2)) dr
    With r = r_r + t^2 the integrand stays finite as a -> n_r r_r.
    Checked on GLO_21 (2025-09-28) against the measured alpha_P: median ratio 1.02-1.08.
    """
    t = np.linspace(0.0, np.sqrt(float(h_tab[-1])), 20001)
    hh = t * t
    lnN = np.interp(hh, h_tab, np.log(N_tab))
    N = np.exp(lnN)
    n = 1.0 + N * 1e-6
    dndr = N * np.gradient(lnN, hh) * 1e-6
    r = r_r + hh
    out = np.full(len(np.atleast_1d(a)), np.nan)
    for i, ai in enumerate(np.atleast_1d(a)):
        q = (n * r) ** 2 - ai * ai
        g = np.where(q > 0, ai * (-dndr) / (n * np.sqrt(np.maximum(q, 1e-30))) * 2.0 * t, 0.0)
        out[i] = np.trapezoid(g, t)
    return out


class BendingAngleRetriever:
    """
    Bending angles for a stationary receiver inside the atmosphere
    (mountain-top / ground-based RO). Produces the PARTIAL bending
    alpha'(a) = alpha_N(a) - alpha_P(a) needed by the finite-top Abel inversion.
    """

    def __init__(self, station: StationConfig, met_fn: Optional[Callable] = None,
                 nprof_fn: Optional[Callable] = None):
        """
        met_fn(event_station, utc) -> {'P','T','e'} or None: station met at the
        occultation's own time and place (ERA5). None: use the station's met.
        nprof_fn(event_station, utc) -> (h above station m, N) or None: refractivity
        column above the station for the model alpha_P (v4.7). None: exponential.
        """
        self.station = station
        self.met_fn = met_fn
        self.nprof_fn = nprof_fn
        self.r_rec_ecef = station.to_ecef()
        self.R_local = station.get_gaussian_radius()
        self.c = SPEED_OF_LIGHT
        self.N_r = station.get_surface_refractivity()
        self.n_r = 1.0 + self.N_r * 1e-6

    def _event_station(self, merged: pd.DataFrame, utc_mid: Optional[str]) -> 'StationConfig':
        """v4.5: receiver fix of the event's own epochs; met at the event time."""
        base = self.station
        def med(c, v):
            if c in merged.columns:
                x = pd.to_numeric(merged[c], errors='coerce').dropna()
                if not x.empty:
                    return float(x.median())
            return v
        ev = StationConfig(latitude=med('sta_lat_f1', base.latitude), longitude=med('sta_lon_f1', base.longitude),
                           altitude=med('sta_h_f1', base.altitude), name=base.name,
                           surface_pressure_hPa=base.surface_pressure_hPa, surface_temp_K=base.surface_temp_K,
                           surface_humidity_hPa=base.surface_humidity_hPa, surface_N=base.surface_N,
                           geoid_sep_m=med('sta_geoid_f1', base.geoid_sep_m))
        if self.met_fn is not None:
            try:
                met = self.met_fn(ev, utc_mid)
                if met:
                    ev.surface_pressure_hPa, ev.surface_temp_K, ev.surface_humidity_hPa = met['P'], met['T'], met['e']
                    ev.surface_N = None
            except Exception:
                pass
        return ev

    @staticmethod
    def _impact_param_to_tangent_height(a_m, R_c_m, N_surface=315.0, H_scale=7000.0,
                                        tol=0.01, max_iter=20):
        """
        Display-only tangent height from a = n(r_tp) r_tp with an exponential
        n-model. The retrieved height comes from the Abel step (r = a/n).
        """
        a_m = np.asarray(a_m, dtype=float)
        R_c_m = np.asarray(R_c_m, dtype=float)
        t = a_m - R_c_m
        t_new = t
        for _ in range(max_iter):
            t_safe = np.clip(t, -50000.0, 200000.0)
            n = 1.0 + N_surface * 1e-6 * np.exp(-t_safe / H_scale)
            t_new = a_m / n - R_c_m
            if np.all(np.abs(t_new - t) < tol):
                break
            t = t_new
        return t_new

    def process(
        self,
        input_csv: str,
        config: PipelineConfig = PipelineConfig(),
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        df = pd.read_csv(input_csv)
        if 'atmos_doppler' not in df.columns:
            return ProcessingResult(False, message="Missing atmos_doppler column")
        df['sat_id'] = df['gnssId'] + '_' + df['svId'].astype(str)
        if 'atmos_dopp_poli' not in df.columns:
            df['atmos_dopp_poli'] = np.nan
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        sel = ro_selection_checks(df)                       # v4.6: tests + values, per satellite
        ro_sats = [sat for sat, v in sel.items() if v['candidate']]
        c5: Dict[str, List[Dict[str, Any]]] = {}             # step-5 tests per event / satellite

        def _ck(item, name, passed, value, need=''):
            c5.setdefault(item, []).append({'name': name, 'passed': passed, 'value': str(value), 'need': need})

        def _write_checks():
            if not output_dir:
                return
            os.makedirs(output_dir, exist_ok=True)
            items = {}
            for sid, v in sel.items():
                evs = [k for k in c5 if k == sid or k.startswith(sid + '_e')]
                for item in (evs or [sid]):
                    items[item] = {'sat_id': sid, 'candidate': v['candidate'], 'freq_mode': v['freq_mode'],
                                   'checks': v['checks'] + c5.get(item, [])}
            with open(f"{output_dir}/ro_checks.json", 'w') as f:
                json.dump(items, f, indent=1, default=str)

        if not ro_sats:
            _write_checks()
            return ProcessingResult(success=True, data=pd.DataFrame(),
                                    message="No RO satellites found")
        df = df[df['sat_id'].isin(ro_sats)]
        elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
        st = self.station

        generated_files, summary_stats, skipped = [], [], {}
        # v4.2: split each satellite's low-elevation dual-frequency data into
        # separate occultation EVENTS (a long session can hold several per satellite)
        events = []
        for sat_id, sat_data in df.groupby('sat_id'):
            gnss_id = sat_data['gnssId'].iloc[0]

            def _sig(sig):
                return (sat_data[sat_data['sigID'] == sig]
                        .drop_duplicates('timestamp').set_index('timestamp').sort_index())

            mode = None
            if gnss_id in FREQ_PAIRS:
                s1, s2 = FREQ_PAIRS[gnss_id]
                merged_all = _sig(s1).join(_sig(s2), lsuffix='_f1', rsuffix='_f2', how='inner')
                merged_all = merged_all[merged_all[f'{elev_col}_f1'] < RO_ELEVATION_THRESHOLD]
                if len(merged_all) >= config.min_epochs_for_bending:
                    mode = 'dual'
            if mode is None and ALLOW_SINGLE_FREQ and gnss_id in SINGLE_FREQ_SIGNALS:
                # v4.3: single frequency - the signal is joined with itself so the
                # rest of the step is unchanged; only the iono combination is skipped.
                s1 = s2 = SINGLE_FREQ_SIGNALS[gnss_id]
                d1 = _sig(s1)
                merged_all = d1.join(d1, lsuffix='_f1', rsuffix='_f2', how='inner')
                merged_all = merged_all[merged_all[f'{elev_col}_f1'] < RO_ELEVATION_THRESHOLD]
                if len(merged_all) >= config.min_epochs_for_bending:
                    mode = 'single'
            if mode is None:
                skipped[sat_id] = ('too few low-elevation dual-frequency epochs'
                                   + ('' if ALLOW_SINGLE_FREQ else ' (single-frequency mode is off)'))
                _ck(sat_id, 'Usable epochs for bending', False, f'{len(merged_all)} epochs',
                    f'≥ {config.min_epochs_for_bending}')
                continue
            t_idx = merged_all.index.values.astype(float)
            ev_no = np.concatenate([[0], np.cumsum(np.diff(t_idx) > EVENT_GAP_S)])
            n_ev = int(ev_no.max()) + 1
            for e in range(n_ev):
                ev_id = sat_id if n_ev == 1 else f"{sat_id}_e{e + 1}"
                events.append((ev_id, gnss_id, s1, s2, merged_all[ev_no == e], mode))
        total_sats = len(events)

        for idx, (sat_id, gnss_id, s1, s2, merged, freq_mode) in enumerate(events):
            if progress_callback:
                progress_callback(f"Bending angles: {sat_id} ({idx+1}/{total_sats})",
                                  0.55 + 0.25 * ((idx + 1) / max(total_sats, 1)))
                if hasattr(progress_callback, '__self__') and \
                        getattr(progress_callback.__self__, '_stopped', False):
                    return ProcessingResult(False, message="Cancelled")

            if len(merged) < config.min_epochs_for_bending:
                skipped[sat_id] = 'too few low-elevation dual-frequency epochs'
                _ck(sat_id, 'Epochs in this occultation', False, f'{len(merged)}', f'≥ {config.min_epochs_for_bending}')
                continue
            _ck(sat_id, 'Epochs in this occultation', True,
                f"{len(merged)} ({'dual' if freq_mode == 'dual' else 'SINGLE'}-frequency)",
                f'≥ {config.min_epochs_for_bending}')

            f1 = get_signal_frequency(s1, gnss_id)
            f2 = get_signal_frequency(s2, gnss_id)
            if 'carrier_freq_hz_f1' in merged.columns:      # v4.1: true carrier (GLONASS FDMA)
                f1 = float(pd.to_numeric(merged['carrier_freq_hz_f1'], errors='coerce').median()) or f1
                f2 = float(pd.to_numeric(merged['carrier_freq_hz_f2'], errors='coerce').median()) or f2
            lam1, lam2 = self.c / f1, self.c / f2

            r_sat = merged[['interp_x_f1', 'interp_y_f1', 'interp_z_f1']].values.astype(float)
            v_sat = merged[['interp_vel_x_f1', 'interp_vel_y_f1', 'interp_vel_z_f1']].values.astype(float)
            el = merged[f'{elev_col}_f1'].values.astype(float)
            utc = merged['utc_f1'].values

            # v4.5: this occultation's own receiver fix and station met (its own time)
            _ut = pd.to_datetime(pd.Series(utc), format='mixed', errors='coerce').dropna()
            ev_utc = str(_ut.iloc[len(_ut) // 2]) if not _ut.empty else None
            st = self._event_station(merged, ev_utc)
            r_rec = st.to_ecef()
            N_ev = st.get_surface_refractivity()
            n_r = 1.0 + N_ev * 1e-6

            # One fixed centre of curvature per occultation, from the mean azimuth
            az = np.array([elevation_azimuth(r, r_rec, st.latitude, st.longitude)[1]
                           for r in r_sat])
            az0 = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(az))))) % 360.0)
            c0, R_c = curvature_centre(r_rec, st.latitude, st.longitude,
                                       st.altitude, az0)
            r_r = r_rec - c0
            x_r = n_r * float(np.linalg.norm(r_r))

            d1 = merged['atmos_dopp_poli_f1'].fillna(merged['atmos_doppler_f1']).values.astype(float)
            d2 = merged['atmos_dopp_poli_f2'].fillna(merged['atmos_doppler_f2']).values.astype(float)

            n_ep = len(merged)
            G1 = np.full((n_ep, 4), np.nan)
            G2 = np.full((n_ep, 4), np.nan)
            for i in range(n_ep):
                r_t = r_sat[i] - c0
                g = bending_geometry(d1[i], lam1, r_t, v_sat[i], r_r, n_r)
                if g is not None:
                    G1[i] = g
                g = bending_geometry(d2[i], lam2, r_t, v_sat[i], r_r, n_r)
                if g is not None:
                    G2[i] = g

            ok = np.isfinite(G1[:, 1])          # v4.7: L2 may be missing (L1 fallback per bin)
            if ok.sum() < config.min_epochs_for_bending:
                skipped[sat_id] = 'geometry solve failed on most epochs'
                _ck(sat_id, 'Ray geometry solved', False, f'{int(ok.sum())}/{len(ok)} epochs',
                    f'≥ {config.min_epochs_for_bending}')
                continue
            _ck(sat_id, 'Ray geometry solved', True, f'{int(ok.sum())}/{len(ok)} epochs',
                f'≥ {config.min_epochs_for_bending}')
            G1, G2, el_ok, utc_ok, az_ok = G1[ok], G2[ok], el[ok], utc[ok], az[ok]

            branch, i_pk = label_branches(
                G1[:, 1], el_ok, el_h_prior=apparent_horizon_prior_deg(n_r, float(np.linalg.norm(r_r))))
            alpha1 = G1[:, 0] + G1[:, 2] + branch * G1[:, 3]
            alpha2 = G2[:, 0] + G2[:, 2] + branch * G2[:, 3]
            alpha1 = np.where(branch == 0, np.nan, alpha1)     # guard band
            alpha2 = np.where(branch == 0, np.nan, alpha2)

            # v4.1: refraction bends rays toward the Earth, so alpha <= 0 is unphysical
            # (almost always a Doppler bias). Reject those epochs and report them.
            neg = (alpha1 <= 0) | (alpha2 <= 0)
            n_neg = int(neg.sum())
            if n_neg:
                alpha1 = np.where(neg, np.nan, alpha1)
                alpha2 = np.where(neg, np.nan, alpha2)
            _ck(sat_id, 'Bending > 0 (no Doppler bias)', bool((~neg).sum() >= config.min_epochs_for_bending),
                f'{n_neg}/{len(neg)} epochs rejected', f'≥ {config.min_epochs_for_bending} left')
            if (~neg).sum() < config.min_epochs_for_bending:
                skipped[sat_id] = (f'{n_neg}/{len(neg)} epochs have alpha <= 0 '
                                   f'(Doppler bias? median L1 atmos Doppler {np.nanmedian(d1):+.2f} Hz)')
                if output_dir:
                    pd.DataFrame({'utc': utc_ok, 'elevation_deg': el_ok, 'branch': branch,
                                  'a_L1_m': G1[:, 1], 'alpha_L1_rad': G1[:, 0] + G1[:, 2] + branch * G1[:, 3]}
                                 ).to_csv(f"{output_dir}/{sat_id}_epochs.csv", index=False)
                continue

            if output_dir:
                pd.DataFrame({
                    'utc': utc_ok, 'elevation_deg': el_ok, 'azimuth_deg': az_ok,
                    'branch': branch, 'a_L1_m': G1[:, 1], 'a_L2_m': G2[:, 1],
                    'dt_L1_rad': G1[:, 0], 'dr_L1_rad': alpha1 - G1[:, 0],
                    'alpha_L1_rad': alpha1, 'alpha_L2_rad': alpha2,
                }).to_csv(f"{output_dir}/{sat_id}_epochs.csv", index=False)

            res = {}
            for b in (-1, 1):
                m = branch == b
                if m.sum() < 3:
                    res[b] = None
                elif freq_mode == 'single':
                    ca, cb, _ = bin_by_impact(G1[m, 1], alpha1[m], A_BIN_M)
                    res[b] = (ca, cb, cb, np.full_like(cb, np.nan), np.ones(len(cb), bool)) if len(ca) >= 2 else None
                else:
                    res[b] = iono_free_on_bins(G1[m, 1], alpha1[m], G2[m, 1], alpha2[m],
                                               f1, f2, A_BIN_M, IONO_SMOOTH_BINS)
                if res[b] is not None and np.isfinite(res[b][1]).sum() < 2:
                    res[b] = None
            nN = int(np.sum((branch == -1) & np.isfinite(alpha1)))       # usable epochs per side
            nP = int(np.sum((branch == 1) & np.isfinite(alpha1)))
            _eN, _eP = el_ok[branch == -1], el_ok[branch == 1]
            el_hor = (0.5 * (np.nanmax(_eN) + np.nanmin(_eP)) if len(_eN) and len(_eP)
                      else apparent_horizon_prior_deg(n_r, float(np.linalg.norm(r_r))))   # v4.7: estimate when one-sided
            use_model = ALPHA_P_MODEL in ('fill', 'always')
            if res[-1] is None or (res[1] is None and not use_model):
                skipped[sat_id] = ('no epochs past the apparent horizon (alpha_N)'
                                   if res[-1] is None else
                                   'no epochs before the apparent horizon (alpha_P); model alpha_P is off')
                _ck(sat_id, 'Crosses the apparent horizon', False,
                    f'{nN} epochs below / {nP} above (horizon {el_hor:+.2f}°)',
                    '≥ 3 below' + ('' if use_model else ' and ≥ 3 above'))
                continue

            aN, alN, alN_L1, alN_L2, l1N = res[-1]
            # alpha_P: measured (above-horizon rays) and/or model atmosphere (v4.7)
            alP_meas = np.full_like(aN, np.nan)
            alP_L1m = np.full_like(aN, np.nan)
            alP_L2m = np.full_like(aN, np.nan)
            l1P = np.zeros(len(aN), bool)
            if res[1] is not None:
                aP = res[1][0]
                def _on_N(v):
                    okc = np.isfinite(v)
                    return (np.interp(aN, aP[okc], v[okc], left=np.nan, right=np.nan) if okc.sum() >= 2
                            else np.full_like(aN, np.nan))
                alP_meas, alP_L1m, alP_L2m = _on_N(res[1][1]), _on_N(res[1][2]), _on_N(res[1][3])
                l1P = _on_N(res[1][4].astype(float)) > 0.5
            alP_mod = np.full_like(aN, np.nan)
            nprof_src = ''
            if use_model:
                h_tab = N_tab = None
                if self.nprof_fn is not None:
                    try:
                        prof = self.nprof_fn(st, ev_utc)
                        if prof is not None:
                            h_tab, N_tab = prof
                    except Exception:
                        h_tab = N_tab = None
                grid, Ng, nprof_src = refractivity_table_above_station(
                    N_ev, h_tab, N_tab, T_r=st.surface_temp_K, e_r=st.surface_humidity_hPa,
                    station_h_m=st.altitude - st.geoid_sep_m, wet_h_m=ALPHA_P_WET_H_M)
                inside = aN < x_r
                alP_mod[inside] = alpha_P_model(aN[inside], float(np.linalg.norm(r_r)), grid, Ng)
            if ALPHA_P_MODEL == 'always':
                p_model = np.isfinite(alP_mod)
            elif ALPHA_P_MODEL == 'fill':
                p_model = ~np.isfinite(alP_meas) & np.isfinite(alP_mod)
            else:
                p_model = np.zeros(len(aN), bool)
            alP_on_N = np.where(p_model, alP_mod, alP_meas)
            n_meas_lv = int(np.sum(np.isfinite(alP_meas) & ~p_model & (aN < x_r)))
            n_mod_lv = int(np.sum(p_model & (aN < x_r)))
            hor_txt = f'{nN} epochs below / {nP} above (horizon {el_hor:+.2f}°)'
            if n_mod_lv and n_meas_lv == 0:
                _ck(sat_id, 'Crosses the apparent horizon', None,
                    hor_txt + f' - alpha_P from the {nprof_src} model', '≥ 3 below')
            else:
                _ck(sat_id, 'Crosses the apparent horizon', True,
                    hor_txt + (f'; {n_mod_lv} levels with model alpha_P' if n_mod_lv else ''),
                    '≥ 3 each side' if not use_model else '≥ 3 below')
            partial = alN - alP_on_N
            # v4.5: per-channel partial bending (Hajj et al. 2002, Fig. 9 shows L1, L2, iono-free)
            partial_L1 = alN_L1 - np.where(p_model, alP_mod, alP_L1m)
            partial_L2 = alN_L2 - np.where(p_model, alP_mod, alP_L2m)
            l1_only_lv = l1N | (l1P & ~p_model) if freq_mode == 'dual' else np.ones(len(aN), bool)
            keep = np.isfinite(partial) & (aN < x_r)
            if keep.sum() < 3:
                skipped[sat_id] = 'alpha_N and alpha_P do not overlap in impact parameter'
                _ck(sat_id, 'Below/above rays overlap', False, f'{int(keep.sum())} levels', '≥ 3 levels')
                continue
            _ck(sat_id, 'Below/above rays overlap', True, f'{int(keep.sum())} levels', '≥ 3 levels')

            # v4.7: model alpha_P does not cancel a Doppler bias the way a measured alpha_P
            # does (it is common to alpha_N and alpha_P). Physical check on model levels:
            # alpha' > 0, and for an all-model event alpha' must grow with depth.
            if p_model[keep].any():
                bad = keep & p_model & ~(partial > 0)
                keep = keep & ~bad
                depth = x_r - aN[keep]
                rho = (float(pd.Series(depth).corr(pd.Series(partial[keep]), method='spearman'))
                       if keep.sum() >= 3 else np.nan)
                all_model = not (keep & ~p_model).any()
                ok_phys = keep.sum() >= 3 and (not all_model or (np.isfinite(rho) and rho > 0.3))
                val = (f"{int(bad.sum())} model levels with alpha' <= 0 dropped"
                       + (f"; alpha' vs depth rank corr. {rho:+.2f}" if all_model and np.isfinite(rho) else ''))
                _ck(sat_id, "Partial bending physical (model alpha_P)", ok_phys, val,
                    "α′ > 0" + (", grows with depth" if all_model else ''))
                if not ok_phys:
                    skipped[sat_id] = ("partial bending with the model alpha_P is not physical (negative or "
                                       "shrinking with depth): a Doppler bias does not cancel without a measured alpha_P")
                    continue

            N_sea = N_ev * np.exp(st.altitude / 7000.0)
            tan_h = self._impact_param_to_tangent_height(aN[keep], R_c, N_surface=N_sea) / 1000.0
            out_df = pd.DataFrame({
                'impact_parameter_m': aN[keep],
                'impact_height_km': (aN[keep] - R_c) / 1000.0,
                'tangent_height_est_km': tan_h,          # rough model estimate; exact value from step 6
                'local_radius_km': R_c / 1000.0,
                'bending_angle_rad': partial[keep],
                'bending_angle_deg': np.degrees(partial[keep]),
                'bending_N_rad': alN[keep],
                'bending_P_rad': alP_on_N[keep],
                'bending_L1': alN_L1[keep],
                'bending_L2': alN_L2[keep],
                'partial_L1_rad': partial_L1[keep],
                'partial_L2_rad': partial_L2[keep],
                'x_r_m': x_r,
                'n_r': n_r,
                'station_P_hPa': st.surface_pressure_hPa,
                'station_T_K': st.surface_temp_K,
                'station_e_hPa': st.surface_humidity_hPa,
                'event_utc': ev_utc,
                'station_height_km': st.altitude / 1000.0,
                'station_lat': st.latitude,
                'station_lon': st.longitude,
                'geoid_sep_m': st.geoid_sep_m,
                'freq_mode': freq_mode,                     # 'dual' | 'single'
                'iono_corrected': ~l1_only_lv[keep],          # v4.7: per level (L1 fallback in L2 gaps)
                'alpha_P_src': np.where(p_model[keep], 'model', 'measured'),
                'alpha_P_model_rad': alP_mod[keep],
                'alpha_P_measured_rad': alP_meas[keep],
                'nprof_src': nprof_src,
                'sig1': s1, 'sig2': s2 if freq_mode == 'dual' else '',
            })
            h_max = (config.height_range_max if config.height_range_max > 0
                     else (x_r - R_c) / 1000.0 + 0.05)
            out_df = out_df[(out_df['impact_height_km'] > config.height_range_min) &
                            (out_df['impact_height_km'] < h_max)]
            if out_df.empty:
                skipped[sat_id] = 'no levels inside the height range'
                _ck(sat_id, 'Profile levels', False, '0 inside the height range', '≥ 1')
                continue
            _ck(sat_id, 'Profile levels', True, f'{len(out_df)}', '≥ 1')

            utc_N = pd.to_datetime(pd.Series(utc_ok[branch == -1]), errors='coerce').dropna()
            utc_mid = str(utc_N.iloc[len(utc_N) // 2]) if not utc_N.empty else ''

            if output_dir:
                fname = f"{output_dir}/{sat_id}_bending.csv"
                out_df.to_csv(fname, index=False)
                generated_files.append(fname)
            summary_stats.append({
                'sat_id': sat_id,
                'gnss_system': gnss_id,
                'freq_mode': freq_mode,
                'signals': s1 if freq_mode == 'single' else f"{s1}+{s2}",
                'utc_mid': utc_mid,
                'azimuth_deg': az0,
                'n_epochs_N': int((branch == -1).sum()),
                'n_epochs_P': int((branch == 1).sum()),
                'n_rejected_alpha_le_0': n_neg,
                'valid_epochs': len(out_df),
                'min_height_est_km': out_df['tangent_height_est_km'].min(),
                'max_height_est_km': out_df['tangent_height_est_km'].max(),
                'min_impact_height_km': out_df['impact_height_km'].min(),
                'max_impact_height_km': out_df['impact_height_km'].max(),
                'max_bending_rad': out_df['bending_angle_rad'].max(),
            })

        summary_df = pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()
        if output_dir:
            if not summary_df.empty:
                summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
            # v4.3: why each RO candidate produced no profile (shown by the GUI)
            pd.DataFrame({'sat_id': list(skipped.keys()), 'reason': list(skipped.values())}
                         ).to_csv(f"{output_dir}/skipped.csv", index=False)
        _write_checks()
        return ProcessingResult(
            success=True,
            data=summary_df,
            message=f"Generated bending angles for {len(summary_stats)} satellites"
                    + (f" ({len(skipped)} skipped)" if skipped else ""),
            metadata={'files': generated_files, 'skipped': skipped}
        )


def retrieve_bending_angles(
    input_csv: str,
    station: StationConfig,
    config: PipelineConfig = PipelineConfig(),
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    met_fn: Optional[Callable] = None,
    nprof_fn: Optional[Callable] = None
) -> ProcessingResult:
    return BendingAngleRetriever(station, met_fn, nprof_fn).process(input_csv, config, output_dir, progress_callback)


def era5_column_above_station(era5_file: str, ev_st: 'StationConfig', when: Optional[str],
                              top_m: float = 40e3):
    """v4.7: ERA5 refractivity column above the station (for the model alpha_P).
    Returns (height above the station m, N) or None."""
    try:
        h_msl = ev_st.altitude - ev_st.geoid_sep_m
        dh = np.concatenate([np.linspace(0.0, 2000.0, 41), np.linspace(2250.0, top_m, 160)])
        E = era5_at_levels(era5_file, h_msl + dh, None, None, ev_st.latitude, ev_st.longitude, when=when)
        if E is None:
            return None
        return dh, np.asarray(E['N'], float)
    except Exception:
        return None

# ============================================================================
# STEP 6: ABEL INVERSION  (finite top at the receiver)
# ============================================================================


def abel_finite_top(a: np.ndarray, alpha_p: np.ndarray, x_r: float, n_r: float):
    """
    ln n(x) = ln n_r + (1/pi) * Int_x^{x_r} alpha'(a) / sqrt(a^2 - x^2) da

    alpha' is taken piecewise linear and each segment is integrated exactly:
        Int (p + q a)/sqrt(a^2 - x^2) da = p*arccosh(a/x) + q*sqrt(a^2 - x^2)
    which removes the endpoint singularity. Boundary: alpha'(x_r) = 0, since a
    ray grazing the station level crosses no layer below it.

    Returns (x, n, r, alpha_used) with r = x/n measured from the centre of curvature.
    """
    a = np.asarray(a, float)
    alpha_p = np.asarray(alpha_p, float)
    ok = np.isfinite(a) & np.isfinite(alpha_p) & (a < x_r)
    A, B, _ = bin_by_impact(a[ok], alpha_p[ok], 1e-3)   # sort + merge duplicate a
    A = np.append(A, x_r)
    B = np.append(B, 0.0)
    ln_n = np.empty(len(A) - 1)
    for i in range(len(A) - 1):
        x = A[i]
        a0, a1, b0, b1 = A[i:-1], A[i + 1:], B[i:-1], B[i + 1:]
        q = (b1 - b0) / (a1 - a0)
        p = b0 - q * a0
        ach1 = np.arccosh(np.maximum(a1 / x, 1.0))
        ach0 = np.arccosh(np.maximum(a0 / x, 1.0))
        sq1 = np.sqrt(np.maximum(a1 * a1 - x * x, 0.0))
        sq0 = np.sqrt(np.maximum(a0 * a0 - x * x, 0.0))
        ln_n[i] = np.log(n_r) + np.sum(p * (ach1 - ach0) + q * (sq1 - sq0)) / np.pi
    n = np.exp(ln_n)
    x = A[:-1]
    return x, n, x / n, B[:-1]


def tangent_point_central_angle(a_vals: np.ndarray, x_tab: np.ndarray, r_tab: np.ndarray,
                                n_u: int = 1500) -> np.ndarray:
    """
    Earth-central angle between the receiver and the tangent point of each ray.

    Along a ray in a spherically symmetric atmosphere (Bouguer, r n sin(psi) = a):
        d(theta)/dr = a / (r * sqrt(n^2 r^2 - a^2))
    With x = n r and u = sqrt(x^2 - a^2) (which removes the endpoint singularity):
        theta(a) = Int_0^U  a * (dr/dx) / (r * x) du,   U = sqrt(x_r^2 - a^2)
    r(x) and dr/dx come from the retrieved profile (Abel output, including the
    station level as the last row). The straight-line value arccos(a/x_r) would
    be ~15% too short, because the ray hugs the Earth (checked against an RK4 ray trace).
    Returns theta in radians (NaN outside the retrieved range).
    """
    x_tab = np.asarray(x_tab, float)
    r_tab = np.asarray(r_tab, float)
    o = np.argsort(x_tab)
    x_tab, r_tab = x_tab[o], r_tab[o]
    drdx = np.gradient(r_tab, x_tab)
    x_r = x_tab[-1]
    out = np.full(len(np.atleast_1d(a_vals)), np.nan)
    for i, a in enumerate(np.atleast_1d(a_vals).astype(float)):
        if not np.isfinite(a) or a < x_tab[0] - 1.0 or a > x_r:
            continue
        U = np.sqrt(max(x_r * x_r - a * a, 0.0))
        if U == 0.0:
            out[i] = 0.0
            continue
        u = np.linspace(0.0, U, n_u)
        xx = np.sqrt(a * a + u * u)
        rr = np.interp(xx, x_tab, r_tab)
        dd = np.interp(xx, x_tab, drdx)
        out[i] = np.trapezoid(a * dd / (rr * xx), u)
    return out


def destination_point(lat_deg: float, lon_deg: float, az_deg, dist_m, radius_m: float):
    """Point at great-circle distance dist_m along azimuth az_deg (sphere of radius radius_m)."""
    la1, lo1 = np.radians(lat_deg), np.radians(lon_deg)
    az = np.radians(np.asarray(az_deg, float))
    d = np.asarray(dist_m, float) / radius_m
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(az))
    lo2 = lo1 + np.arctan2(np.sin(az) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 540.0) % 360.0 - 180.0


class AbelInversion:
    """
    Finite-top Abel inversion for a receiver inside the atmosphere.
    The top boundary is the measured refractive index at the station (n_r),
    so no climatology is needed. `climatology_blend_km` is kept for API
    compatibility and ignored.

    v4.4 output columns (refractivity CSV):
      height_km                tangent-point height above mean sea level (km)
                               = (r - R_c)/1000 - geoid_sep/1000 ; r = x/n from Abel.
                               Same reference as ERA5 geopotential height.
      height_ellipsoid_km      same, above the WGS84 ellipsoid
      height_above_station_km  negative below the antenna
      tp_distance_km           ground distance station -> tangent point
      tp_azimuth_deg, tp_lat, tp_lon   tangent-point location
    The bending CSV gets the same tp_* columns (tp_height_km = height_km), and
    <event>_tangent_points.csv lists the tangent point of every epoch used.
    """

    REQUIRED = ('impact_parameter_m', 'bending_angle_rad', 'x_r_m', 'n_r', 'local_radius_km')

    def __init__(self, climatology_blend_km: Optional[float] = None):
        self.climatology_blend_km = climatology_blend_km

    def run(self, bending_csv: str, output_csv: Optional[str] = None) -> ProcessingResult:
        df = pd.read_csv(bending_csv)
        if df.empty:
            return ProcessingResult(False, None, "Empty bending angle file", {'input_rows': 0})
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            return ProcessingResult(False, None,
                                    f"Bending file lacks {missing}; re-run step 5 (v4 format)", {})

        x_r = float(df['x_r_m'].iloc[0])
        n_r = float(df['n_r'].iloc[0])
        R_c = float(df['local_radius_km'].iloc[0]) * 1000.0
        geoid = float(df['geoid_sep_m'].iloc[0]) if 'geoid_sep_m' in df.columns else 0.0
        h_sta = float(df['station_height_km'].iloc[0]) * 1000.0 if 'station_height_km' in df.columns \
            else x_r / n_r - R_c

        x, n, r, alpha_used = abel_finite_top(df['impact_parameter_m'].values,
                                              df['bending_angle_rad'].values, x_r, n_r)
        if len(x) < 2:
            return ProcessingResult(False, None, "Fewer than 2 levels below the station", {})

        # levels + the station itself (measured n_r) as the top row
        X = np.append(x, x_r)
        Nn = np.append(n, n_r)
        Rr = X / Nn
        A = np.append(alpha_used, 0.0)
        h_ell = Rr - R_c
        result_df = pd.DataFrame({
            'height_km': (h_ell - geoid) / 1000.0,
            'height_ellipsoid_km': h_ell / 1000.0,
            'height_above_station_km': (h_ell - h_sta) / 1000.0,
            'refractivity_N': (Nn - 1.0) * 1e6,
            'impact_parameter_m': X,
            'bending_optimized': A,
        })

        # ---- tangent-point position
        theta = tangent_point_central_angle(X, X, Rr)
        dist_m = R_c * theta
        result_df['tp_distance_km'] = dist_m / 1000.0
        ep_csv = bending_csv.replace('_bending.csv', '_epochs.csv')
        ep = pd.read_csv(ep_csv) if os.path.exists(ep_csv) else pd.DataFrame()
        az_lv = np.full(len(X), np.nan)
        if not ep.empty and {'branch', 'a_L1_m', 'azimuth_deg'} <= set(ep.columns):
            en = ep[(ep['branch'] == -1) & np.isfinite(ep['a_L1_m'])].sort_values('a_L1_m')
            if len(en) >= 2:
                az_u = np.degrees(np.unwrap(np.radians(en['azimuth_deg'].values)))
                az_lv = np.interp(X, en['a_L1_m'].values, az_u) % 360.0
            elif len(en) == 1:
                az_lv[:] = en['azimuth_deg'].iloc[0]
        if np.isnan(az_lv).all() and not ep.empty and 'azimuth_deg' in ep.columns:
            az_lv[:] = float(np.nanmedian(ep['azimuth_deg']))
        result_df['tp_azimuth_deg'] = az_lv
        if 'station_lat' in df.columns and np.isfinite(az_lv).any():
            la0, lo0 = float(df['station_lat'].iloc[0]), float(df['station_lon'].iloc[0])
            la, lo = destination_point(la0, lo0, az_lv, dist_m, R_c)
            result_df['tp_lat'], result_df['tp_lon'] = la, lo
        else:
            result_df['tp_lat'] = result_df['tp_lon'] = np.nan
        result_df = result_df.sort_values('height_km').reset_index(drop=True)

        # ---- write the same tangent-point columns into the bending CSV
        o = np.argsort(X)
        for col in ('height_km', 'tp_distance_km', 'tp_azimuth_deg', 'tp_lat', 'tp_lon'):
            src = result_df.set_index('impact_parameter_m')[col].reindex(X[o]).values
            tgt = 'tp_height_km' if col == 'height_km' else col
            df[tgt] = np.interp(df['impact_parameter_m'].values, X[o], src, left=np.nan, right=np.nan)
        df['tp_height_above_station_km'] = df['tp_height_km'] - (h_sta - geoid) / 1000.0
        df.to_csv(bending_csv, index=False)

        # ---- tangent point of every epoch used (map)
        if not ep.empty and {'branch', 'a_L1_m'} <= set(ep.columns):
            en = ep[ep['branch'] == -1].copy()
            a_e = en['a_L1_m'].values
            ok = (a_e >= X.min()) & (a_e <= x_r)
            th_e = np.where(ok, tangent_point_central_angle(np.where(ok, a_e, np.nan), X, Rr), np.nan)
            en['tp_height_km'] = np.interp(a_e, X[o], ((Rr - R_c - geoid) / 1000.0)[o], left=np.nan, right=np.nan)
            en['tp_distance_km'] = R_c * th_e / 1000.0
            if 'station_lat' in df.columns and 'azimuth_deg' in en.columns:
                la, lo = destination_point(float(df['station_lat'].iloc[0]), float(df['station_lon'].iloc[0]),
                                           en['azimuth_deg'].values, R_c * th_e, R_c)
                en['tp_lat'], en['tp_lon'] = la, lo
            en = en[np.isfinite(en['tp_distance_km'])]
            cols = [c for c in ('utc', 'elevation_deg', 'azimuth_deg', 'a_L1_m', 'tp_height_km',
                                'tp_distance_km', 'tp_lat', 'tp_lon') if c in en.columns]
            en[cols].to_csv(bending_csv.replace('_bending.csv', '_tangent_points.csv'), index=False)

        if output_csv:
            result_df.to_csv(output_csv, index=False)

        return ProcessingResult(
            success=True,
            data=result_df,
            message=f"Retrieved refractivity: {result_df['height_km'].min():.2f}-"
                    f"{result_df['height_km'].max():.2f} km MSL, tangent points "
                    f"{np.nanmin(result_df['tp_distance_km']):.0f}-{np.nanmax(result_df['tp_distance_km']):.0f} km away",
            metadata={
                'input_levels': len(df),
                'output_levels': len(result_df),
                'height_range_km': (result_df['height_km'].min(), result_df['height_km'].max()),
                'N_station': (n_r - 1.0) * 1e6,
                'R_c_km': R_c / 1000.0,
            }
        )


def retrieve_refractivity(
    bending_csv: str,
    output_csv: Optional[str] = None,
    climatology_blend_km: Optional[float] = None
) -> ProcessingResult:
    """Convenience wrapper for the finite-top Abel inversion."""
    return AbelInversion(climatology_blend_km).run(bending_csv, output_csv)


# ============================================================================
# STEP 6B: ERA5 COMPARISON
# ============================================================================

TP_COLUMNS = ('height_ellipsoid_km', 'height_above_station_km', 'tp_distance_km',
              'tp_azimuth_deg', 'tp_lat', 'tp_lon')


def _era5_extract(ds, lat, lon, when=None):
    """
    Extract ERA5 T, q, z profiles.

    Time: the field nearest to `when` (occultation time, UTC) if given,
    otherwise the mean over the file's time axis.
    Space: interpolation to (lat, lon) -> nearest grid point -> spatial mean.
    """
    fallback_used = 'interp'
    when64 = None
    if when is not None and str(when) != '':
        ts = pd.Timestamp(when)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        when64 = ts.to_datetime64()

    def _time_reduce(da):
        for dim in ('valid_time', 'time'):
            if dim in da.dims:
                if when64 is not None:
                    return da.sel({dim: when64}, method='nearest')
                return da.mean(dim=dim)
        return da

    def _spatial(sel):
        return (_time_reduce(sel(ds['t'])), _time_reduce(sel(ds['q'])),
                _time_reduce(sel(ds['z'])))

    if lat is not None and lon is not None:
        try:
            T, q, z = _spatial(lambda v: v.interp(latitude=lat, longitude=lon))
            if not (np.isfinite(T.values).any() and np.isfinite(z.values).any()):
                raise ValueError("interp returned all-NaN (point outside grid)")
        except Exception:
            fallback_used = 'nearest'
            try:
                T, q, z = _spatial(lambda v: v.sel(latitude=lat, longitude=lon, method='nearest'))
                if not np.isfinite(T.values).any():
                    raise ValueError("nearest also returned all-NaN")
            except Exception:
                fallback_used = 'spatial_mean'
                T, q, z = _spatial(lambda v: v.mean(dim=['latitude', 'longitude']))
    else:
        fallback_used = 'spatial_mean'
        T, q, z = _spatial(lambda v: v.mean(dim=['latitude', 'longitude']))

    if when64 is not None:
        fallback_used += '+nearest_time'
    return T, q, z, fallback_used


def era5_at_levels(era5_file: str, h_m: np.ndarray, lat_lv, lon_lv, sta_lat: float, sta_lon: float,
                   when: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    v4.5: ERA5 at each retrieved level, at that level's tangent point.

    Each level i gets the ERA5 column at (lat_lv[i], lon_lv[i]) - interpolated
    horizontally, then vertically to h_m[i] (m above MSL; ERA5 geopotential
    height z/g0 is also relative to MSL). Levels whose tangent point lies outside
    the file's grid (or has no position) use the column above the station and are
    flagged colocated=False. Time: the field nearest to `when`.
    Returns arrays T (K), q (kg/kg), P (hPa), e (hPa), N, colocated, lat, lon.
    """
    try:
        import xarray as xr
    except ImportError:
        return None
    h_m = np.asarray(h_m, float)
    n = len(h_m)
    lat_lv = np.asarray(lat_lv if lat_lv is not None else np.full(n, np.nan), float)
    lon_lv = np.asarray(lon_lv if lon_lv is not None else np.full(n, np.nan), float)
    with xr.open_dataset(era5_file) as ds:
        for dim in ('valid_time', 'time'):
            if dim in ds.dims:
                if when:
                    ts = pd.Timestamp(when)
                    if ts.tzinfo is not None:
                        ts = ts.tz_convert('UTC').tz_localize(None)
                    ds = ds.sel({dim: ts.to_datetime64()}, method='nearest')
                else:
                    ds = ds.mean(dim=dim)
        la_g, lo_g = ds['latitude'].values, ds['longitude'].values
        tol = 1e-6
        inside = (np.isfinite(lat_lv) & np.isfinite(lon_lv)
                  & (lat_lv >= la_g.min() - tol) & (lat_lv <= la_g.max() + tol)
                  & (lon_lv >= lo_g.min() - tol) & (lon_lv <= lo_g.max() + tol))
        la_use = np.clip(np.where(inside, lat_lv, sta_lat), la_g.min(), la_g.max())
        lo_use = np.clip(np.where(inside, lon_lv, sta_lon), lo_g.min(), lo_g.max())
        pts = ds[['t', 'q', 'z']].interp(latitude=xr.DataArray(la_use, dims='lv'),
                                         longitude=xr.DataArray(lo_use, dims='lv'))
        T = pts['t'].transpose('lv', 'pressure_level').values.astype(float)
        Q = pts['q'].transpose('lv', 'pressure_level').values.astype(float)
        Z = pts['z'].transpose('lv', 'pressure_level').values.astype(float) / 9.80665
        Pl = ds['pressure_level'].values.astype(float)
    out = {k: np.full(n, np.nan) for k in ('T', 'q', 'P')}
    for i in range(n):
        o = np.argsort(Z[i])
        z, t, q, p = Z[i][o], T[i][o], Q[i][o], Pl[o]
        if not np.isfinite(z).any():
            continue
        out['T'][i] = np.interp(h_m[i], z, t)
        out['q'][i] = np.interp(h_m[i], z, q)
        out['P'][i] = float(np.exp(np.interp(h_m[i], z, np.log(p))))
    out['e'] = out['q'] * out['P'] / (0.622 + 0.378 * out['q'])
    out['N'] = N_COEFF_A1 * out['P'] / out['T'] + N_COEFF_A2 * out['e'] / out['T'] ** 2
    out['colocated'] = inside
    out['lat'], out['lon'] = la_use, lo_use
    return out


def compare_with_era5(refractivity_csv: str, era5_file: str,
                      lat: Optional[float] = None, lon: Optional[float] = None,
                      output_csv: Optional[str] = None,
                      when: Optional[str] = None) -> ProcessingResult:
    """
    Compare retrieved refractivity with ERA5 at the occultation time and, per
    level, at that level's tangent point (v4.5; station column where the ERA5
    file does not cover the tangent point - flagged in 'era5_colocated').
    lat/lon: the station (fallback column).
    """
    ro_df = pd.read_csv(refractivity_csv)
    if ro_df.empty or 'height_km' not in ro_df.columns:
        return ProcessingResult(False, None, "Empty or invalid refractivity CSV", {})
    ro_df = ro_df[np.isfinite(ro_df['height_km']) & np.isfinite(ro_df['refractivity_N'])
                  & (ro_df['height_km'] > -1.0)].reset_index(drop=True)
    if ro_df.empty:
        return ProcessingResult(False, None, "No valid RO refractivity points", {})
    try:
        e5 = era5_at_levels(era5_file, ro_df['height_km'].values * 1000.0,
                            ro_df.get('tp_lat'), ro_df.get('tp_lon'), lat, lon, when)
    except Exception as e:
        return ProcessingResult(False, None, f".nc profile extraction failed: {e}", {})
    if e5 is None:
        return ProcessingResult(False, None, "xarray required", {})
    ok = np.isfinite(e5['N'])
    if not ok.any():
        return ProcessingResult(False, None, ".nc refractivity all-NaN at the retrieved levels", {})
    comparison_df = pd.DataFrame({'height_km': ro_df['height_km'].values, 'N_RO': ro_df['refractivity_N'].values,
                                  'N_ERA5': e5['N'], 'error': ro_df['refractivity_N'].values - e5['N'],
                                  'era5_colocated': e5['colocated'], 'era5_lat': e5['lat'], 'era5_lon': e5['lon']})
    for c in TP_COLUMNS:
        if c in ro_df.columns:
            comparison_df[c] = ro_df[c].values
    comparison_df = comparison_df[ok].reset_index(drop=True)
    if output_csv:
        comparison_df.to_csv(output_csv, index=False)
    err = comparison_df['error']
    below = (comparison_df['tp_distance_km'] > 1.0) if 'tp_distance_km' in comparison_df.columns \
        else np.ones(len(comparison_df), bool)                 # leave out the station row itself
    n_co = int((comparison_df['era5_colocated'] & below).sum())
    where = (f"at the tangent points ({n_co}/{int(below.sum())} levels below the station)" if n_co
             else "above the station only - the .nc file does not cover the tangent points")
    return ProcessingResult(
        success=True, data=comparison_df,
        message=f".nc comparison {where}: RMSE={np.sqrt(np.mean(err ** 2)):.2f}, bias={err.mean():+.2f} N",
        metadata={'rmse': float(np.sqrt(np.mean(err ** 2))), 'bias': float(err.mean()),
                  'n_colocated': n_co, 'n_levels': len(comparison_df)})


# ============================================================================
# STEP 7: ATMOSPHERIC RETRIEVAL  (top boundary at the station)
# ============================================================================

def retrieve_atmospheric_profile(refractivity_csv: str, era5_file: str,
                                 lat: Optional[float] = None, lon: Optional[float] = None,
                                 output_csv: Optional[str] = None,
                                 when: Optional[str] = None,
                                 station: Optional[StationConfig] = None) -> ProcessingResult:
    """
    Retrieve P, Pw, q below the station with ERA5 temperature as the constraint.

    - ERA5 (T, and P/q for comparison) per level at the level's tangent point
      when the file covers it (v4.5), else above the station; nearest time.
    - Top boundary: the station pressure of this occultation (barometer / .cra,
      or ERA5 at the event time and station height) when the profile's top is the
      station level; otherwise ERA5 pressure at the top.
    - Hydrostatic layers integrated with the hypsometric equation using virtual
      temperature, iterated with Pw from the refractivity equation.
    """
    ro_df = pd.read_csv(refractivity_csv)
    if ro_df.empty or 'height_km' not in ro_df.columns:
        return ProcessingResult(False, None, "Empty or invalid refractivity CSV", {})
    ro_df = ro_df[np.isfinite(ro_df['height_km']) & np.isfinite(ro_df['refractivity_N'])]
    ro_df = ro_df.sort_values('height_km').drop_duplicates('height_km').reset_index(drop=True)
    if len(ro_df) < 2:
        return ProcessingResult(False, None, "Fewer than 2 finite refractivity rows", {})
    tp_extra = {c: ro_df[c].values for c in TP_COLUMNS if c in ro_df.columns}
    h_m = ro_df['height_km'].values * 1000.0
    N = ro_df['refractivity_N'].values
    try:
        e5 = era5_at_levels(era5_file, h_m, ro_df.get('tp_lat'), ro_df.get('tp_lon'), lat, lon, when)
    except Exception as e:
        return ProcessingResult(False, None, f".nc extraction failed: {e}", {})
    if e5 is None or not np.isfinite(e5['T']).any():
        return ProcessingResult(False, None, ".nc temperature unavailable", {})
    T, P_era5, q_era5 = e5['T'], e5['P'], e5['q']

    n = len(h_m)
    i_top = n - 1
    if (station is not None and station.surface_pressure_hPa is not None
            and abs(h_m[i_top] - (station.altitude - station.geoid_sep_m)) < 100.0):   # heights are MSL
        P_top, P_src = float(station.surface_pressure_hPa), 'station'
    else:
        P_top, P_src = float(P_era5[i_top]), 'era5'

    R_gas, m_dry, m_water = 8.314462, 28.97e-3, 18.015e-3
    eps = m_water / m_dry
    P = np.full(n, P_top)
    Pw = np.zeros(n)
    n_capped = 0
    for _ in range(20):
        Pw_old = Pw.copy()
        for i in range(i_top - 1, -1, -1):
            Tv_hi = T[i + 1] / (1.0 - (1.0 - eps) * Pw[i + 1] / P[i + 1])
            Tv_lo = T[i] / (1.0 - (1.0 - eps) * Pw[i] / max(P[i], 1e-6))
            g = compute_gravity(0.5 * (h_m[i] + h_m[i + 1]), lat)
            P[i] = P[i + 1] * np.exp(g * m_dry * (h_m[i + 1] - h_m[i]) /
                                     (R_gas * 0.5 * (Tv_hi + Tv_lo)))
        n_capped = 0
        for i in range(n):
            Pw[i] = max(0.0, (T[i] ** 2 / N_COEFF_A2) * (N[i] - N_COEFF_A1 * P[i] / T[i]))
            T_c = T[i] - 273.15
            P_sat = 6.1094 * np.exp(17.625 * T_c / (T_c + 243.04))
            if Pw[i] > P_sat:
                Pw[i] = P_sat
                n_capped += 1
        if np.max(np.abs(Pw - Pw_old)) < 0.001:
            break

    q = eps * Pw / (P - Pw + eps * Pw) * 1000.0
    profile = pd.DataFrame({
        'height_km': h_m / 1000.0,
        'temperature_K': T,
        'pressure_hPa': P,
        'water_vapor_hPa': Pw,
        'specific_humidity_g_kg': q,
        'refractivity_N': N,
        'T_era5': T,
        'P_era5': P_era5,
        'Pw_era5': e5['e'],
        'q_era5': q_era5 * 1000.0,
        'era5_colocated': e5['colocated'],
    })
    for c, v in tp_extra.items():
        profile[c] = v
    if output_csv:
        profile.to_csv(output_csv, index=False)
    below = (np.asarray(tp_extra['tp_distance_km']) > 1.0) if 'tp_distance_km' in tp_extra else np.ones(n, bool)
    n_co = int(np.sum(e5['colocated'] & below))
    msg = (f"Atmospheric retrieval OK (P boundary={P_src}; .nc "
           + (f"at the tangent points for {n_co}/{int(below.sum())} levels)" if n_co
              else "above the station only - the .nc file does not cover the tangent points)"))
    if n_capped:
        msg += f"; {n_capped} levels capped at saturation"
    return ProcessingResult(
        success=True, data=profile, message=msg,
        metadata={'P_rmse': float(np.sqrt(np.nanmean((P - P_era5) ** 2))), 'P_boundary_source': P_src,
                  'n_saturation_capped': n_capped, 'n_colocated': n_co})


# ============================================================================
# PLOTTING FUNCTIONS  (v4.4 layout)
# ============================================================================
# Colours: validated categorical slots (blue, orange, aqua: colour-blind safe
# in all pairs); each series also has its own marker shape. Sequential blue
# ramp for tangent-point height on the map. Text stays in neutral ink.
PLOT_BLUE, PLOT_ORANGE, PLOT_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
PLOT_INK, PLOT_INK2, PLOT_GRID = '#0b0b0b', '#52514e', '#d9d8d4'
PLOT_SEQ_BLUE = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']

MAP_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"   # '' disables the basemap
MAP_TILE_CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'gnss_ro_tiles')
MAP_USER_AGENT = "GNSS-RO-Processor/4.4 (ground-based radio occultation research tool)"
_TILES_OFFLINE = {'flag': False}                 # stop retrying after the first failure


def session_sample_rate(timestamps, gap_factor: float = 5.0) -> Dict[str, float]:
    """
    Average sampling rate of a session, excluding data drops: intervals longer
    than gap_factor x the median interval are gaps and are left out.
    rate = 1 / mean(normal intervals).
    """
    t = np.unique(np.asarray(pd.to_numeric(pd.Series(timestamps), errors='coerce').dropna(), float))
    if len(t) < 2:
        return {'rate_hz': np.nan, 'gap_threshold_s': np.nan, 'n_gaps': 0, 'gap_time_s': 0.0}
    dt = np.diff(t)
    thr = gap_factor * float(np.median(dt))
    ok = dt <= thr
    return {'rate_hz': 1.0 / float(np.mean(dt[ok])) if ok.any() else np.nan,
            'gap_threshold_s': thr, 'n_gaps': int((~ok).sum()), 'gap_time_s': float(dt[~ok].sum())}



# ============================================================================
# 3.5.2 — SESSION QUALITY, ERA5 COVERAGE, SESSION REPORT
# ============================================================================

AZ_SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
APPARENT_HORIZON_TYPICAL_DEG = -0.5      # ~2.5 km station; only used for the quality summary


def session_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Data-collection quality of one session (step-4 table): sampling, satellites
    reaching below the horizon, L2 tracking near the horizon and the lowest
    elevation tracked per azimuth sector (where the horizon is open).
    """
    q: Dict[str, Any] = {}
    if df is None or df.empty:
        return q
    el_c = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    az_c = 'accurate_azimuth' if 'accurate_azimuth' in df.columns else 'azimuth'
    d = df.copy()
    if 'sat_id' not in d.columns:
        d['sat_id'] = d['gnssId'].astype(str) + '_' + d['svId'].astype(str)
    d[el_c] = pd.to_numeric(d[el_c], errors='coerce')
    t = pd.to_numeric(d['timestamp'], errors='coerce')
    rate = session_sample_rate(t)
    if 'obs_interval_s' in d.columns:
        _iv = pd.to_numeric(d['obs_interval_s'], errors='coerce').median()
        q['recorded_hz'] = float(1.0 / _iv) if np.isfinite(_iv) and _iv > 0 else np.nan
    q.update({'rate_hz': rate['rate_hz'], 'n_gaps': rate['n_gaps'], 'gap_time_s': rate['gap_time_s'],
              'duration_min': float((t.max() - t.min()) / 60.0) if t.notna().any() else np.nan})
    if 'utc' in d.columns:
        u = pd.to_datetime(d['utc'], format='mixed', errors='coerce').dropna()
        if not u.empty:
            q['utc_start'], q['utc_end'] = str(u.min())[:19], str(u.max())[:19]
    mins = d.groupby('sat_id')[el_c].min()
    q['n_sats'] = int(len(mins))
    q['n_below_0'] = int((mins < 0).sum())
    q['n_below_app'] = int((mins < APPARENT_HORIZON_TYPICAL_DEG).sum())
    if az_c in d.columns:
        low = d[d[el_c] < 5.0]
        az = pd.to_numeric(low[az_c], errors='coerce')
        sec = (((az + 22.5) // 45) % 8).astype('Int64')
        q['sector_min_el'] = {AZ_SECTORS[int(k)]: float(v) for k, v in
                              low.groupby(sec)[el_c].min().items() if pd.notna(k)}
    # L2 tracking near the horizon: share of L1 epochs (el < 1 deg) that also have L2
    if 'sigID' in d.columns:
        near = d[d[el_c] < 1.0]
        n1 = n12 = 0
        for sid, g in near.groupby('sat_id'):
            pair = FREQ_PAIRS.get(str(sid).split('_')[0])
            if not pair:
                continue
            ep1 = set(g.loc[g['sigID'] == pair[0], 'timestamp'])
            ep2 = set(g.loc[g['sigID'] == pair[1], 'timestamp'])
            n1 += len(ep1)
            n12 += len(ep1 & ep2)
        q['l2_share_near_horizon'] = (n12 / n1) if n1 else np.nan
    hints = []
    if q['n_below_app'] == 0:
        hints.append('no satellite went below the apparent horizon (≈ −0.5°): no profile possible')
    if np.isfinite(q.get('l2_share_near_horizon', np.nan)) and q['l2_share_near_horizon'] < 0.7:
        hints.append(f"L2 tracked on only {q['l2_share_near_horizon'] * 100:.0f}% of near-horizon epochs")
    if q.get('gap_time_s', 0) > 60:
        hints.append(f"data gaps total {q['gap_time_s']:.0f} s")
    open_dirs = [k for k in AZ_SECTORS if q.get('sector_min_el', {}).get(k, 99) < APPARENT_HORIZON_TYPICAL_DEG]
    q['open_directions'] = open_dirs
    q['hints'] = hints
    return q


def session_quality_lines(q: Dict[str, Any]) -> List[str]:
    """Short text lines for the GUI card and the report."""
    if not q:
        return []
    lines = []
    when = (f"{q['utc_start'][11:16]}–{q['utc_end'][11:16]} UTC · " if q.get('utc_start') else '')
    gaps = f"{q['n_gaps']} gaps" if q.get('n_gaps') else 'no gaps'
    rec = q.get('recorded_hz', np.nan)
    rate_s = (f"{rec:.0f} Hz recorded → {q.get('rate_hz', np.nan):.2f} Hz processed"
              if rec is not None and np.isfinite(rec) and rec > 1.5 * q.get('rate_hz', np.inf)
              else f"{q.get('rate_hz', np.nan):.2f} Hz")
    lines.append(f"{when}{q.get('duration_min', np.nan):.1f} min · {rate_s} ({gaps})")
    lines.append(f"{q.get('n_sats', 0)} satellites · {q.get('n_below_0', 0)} below 0° · "
                 f"{q.get('n_below_app', 0)} below −0.5°")
    l2 = q.get('l2_share_near_horizon', np.nan)
    if np.isfinite(l2):
        lines.append(f"L2 near horizon: {l2 * 100:.0f}% of L1 epochs")
    sec = q.get('sector_min_el', {})
    if sec:
        lines.append('Lowest elev.: ' + ' · '.join(f"{k} {sec[k]:+.1f}°" for k in AZ_SECTORS if k in sec))
    for h in q.get('hints', []):
        lines.append('⚠ ' + h)
    return lines


def era5_coverage(era5_file: str, lat: float, lon: float, reach_km: float = 250.0) -> Optional[Dict[str, Any]]:
    """
    Does the ERA5 file cover where tangent points can fall (up to reach_km from
    the station)? Returns {'covered', 'have': (la0, la1, lo0, lo1), 'need': (...)}.
    """
    try:
        import xarray as xr
        with xr.open_dataset(era5_file) as ds:
            la, lo = ds['latitude'].values, ds['longitude'].values
        dla = reach_km / 111.0
        dlo = reach_km / (111.0 * max(np.cos(np.radians(lat)), 0.1))
        need = (lat - dla, lat + dla, lon - dlo, lon + dlo)
        have = (float(np.min(la)), float(np.max(la)), float(np.min(lo)), float(np.max(lo)))
        tol = 0.26
        covered = (have[0] <= need[0] + tol and have[1] >= need[1] - tol
                   and have[2] <= need[2] + tol and have[3] >= need[3] - tol)
        return {'covered': bool(covered), 'have': have, 'need': need}
    except Exception:
        return None


def generate_session_report(ground_dir: str, pdf_path: str, status: Dict[str, Any],
                            reasons: Optional[Dict[str, str]] = None, station_text: str = '',
                            hidden: Optional[set] = None) -> bool:
    """
    One PDF per session: summary page (station, session quality, results table),
    then Panel 2 and Panel 3 of every satellite with a profile.
    status values: 'ro_ok' | 'ro_ok_1f' | 'ro_empty' | other (no RO).
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from plot_style import apply_plot_fonts
        apply_plot_fonts()
    except ImportError:
        pass
    reasons = reasons or {}
    hidden = hidden or set()
    plots = os.path.join(ground_dir, 'plots')
    step4 = os.path.join(ground_dir, 'step4_differenced.csv')
    qfile = os.path.join(ground_dir, 'session_quality.json')
    q = {}
    if os.path.exists(qfile):
        try:
            q = json.load(open(qfile))
        except Exception:
            q = {}
    elif os.path.exists(step4):
        q = session_quality(pd.read_csv(step4))

    def st(v):
        return v if v in ('ro_ok', 'ro_ok_1f', 'ro_empty') else 'no_ro'
    order = {'ro_ok': 0, 'ro_ok_1f': 1, 'ro_empty': 2, 'no_ro': 3}
    items = sorted((k for k in status if k not in hidden), key=lambda k: (order[st(status[k])], k))
    rows = []
    for k in items:
        s_ = st(status[k])
        lev = hr = err = note = ''
        if s_ in ('ro_ok', 'ro_ok_1f'):
            try:
                rf = pd.read_csv(os.path.join(ground_dir, 'refractivity', f'{k}_refractivity.csv'))
                rf = rf[rf.get('height_above_station_km', rf['height_km'] * 0 - 1) < -0.005]
                lev = str(len(rf))
                hr = f"{rf['height_km'].min():.2f}–{rf['height_km'].max():.2f}"
            except Exception:
                pass
            try:
                c = pd.read_csv(os.path.join(ground_dir, 'comparison', f'{k}_comparison.csv'))
                c = c[c['height_above_station_km'] < -0.005] if 'height_above_station_km' in c else c
                p = (c['N_RO'] - c['N_ERA5']) / c['N_ERA5'] * 100
                err = f"{p.mean():+.1f} / {np.sqrt((p ** 2).mean()):.1f}"
            except Exception:
                pass
            try:
                b = pd.read_csv(os.path.join(ground_dir, 'bending', f'{k}_bending.csv'))
                if s_ == 'ro_ok_1f':
                    note = 'single-frequency'
                elif 'iono_corrected' in b and (~b['iono_corrected'].astype(bool)).any():
                    note = f"{int((~b['iono_corrected'].astype(bool)).sum())} L1-only levels"
            except Exception:
                pass
        else:
            info = load_ro_checks(os.path.join(ground_dir, 'bending'), k)
            f = next((c_ for c_ in (info or {}).get('checks', []) if c_.get('passed') is False), None)
            note = f"{_short_check_name(f['name'])}: {f['value']}" if f else reasons.get(k, '')
            note = note if len(note) <= 46 else note[:45] + '…'
        label = {'ro_ok': 'profile', 'ro_ok_1f': 'profile (1F)', 'ro_empty': 'RO, no profile',
                 'no_ro': 'not RO'}[s_]
        rows.append([k, label, lev, hr, err, note])

    os.makedirs(os.path.dirname(pdf_path) or '.', exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, 'GNSS-RO session report', fontsize=18, fontweight='bold', color=PLOT_INK)
        fig.text(0.07, 0.925, f"{station_text}   ·   pipeline {__version__}   ·   "
                              f"{datetime.now():%Y-%m-%d %H:%M}", fontsize=9, color=PLOT_INK2)
        y = 0.89
        fig.text(0.07, y, 'Session', fontsize=11.5, fontweight='bold', color=PLOT_INK)
        for ln in session_quality_lines(q):
            y -= 0.022
            fig.text(0.07, y, ln, fontsize=9, color='#7a2e00' if ln.startswith('⚠') else PLOT_INK)
        y -= 0.04
        fig.text(0.07, y, 'Results' + (f"   ({len(hidden)} satellites never below 5° not listed)" if hidden else ''),
                 fontsize=11.5, fontweight='bold', color=PLOT_INK)
        if rows:
            h = min(0.03 * (len(rows) + 1), y - 0.05)
            ax = fig.add_axes([0.07, y - 0.01 - h, 0.86, h]); ax.axis('off')
            tb = ax.table(cellText=rows, colLabels=['Satellite', 'Result', 'Levels', 'Height (km)',
                                                    'vs .nc mean/RMS %', 'Note'],
                          colWidths=[0.11, 0.13, 0.07, 0.13, 0.16, 0.40], loc='upper left', cellLoc='left')
            tb.auto_set_font_size(False); tb.set_fontsize(7.6); tb.scale(1, 1.25)
            for (r_, c_), cell in tb.get_celld().items():
                cell.set_edgecolor('#d9d8d3')
                if r_ == 0:
                    cell.set_facecolor('#f2f1ee'); cell.set_text_props(fontweight='bold')
        pdf.savefig(fig); plt.close(fig)
        for k in items:
            if st(status[k]) not in ('ro_ok', 'ro_ok_1f'):
                continue
            for suffix in ('derived', 'atmospheric'):
                png = os.path.join(plots, f'{k}_{suffix}.png')
                if not os.path.exists(png):
                    continue
                img = plt.imread(png)
                fig = plt.figure(figsize=(8.27, 11.69))
                hh, ww = img.shape[:2]
                fw = 0.92; fh = fw * (hh / ww) * (8.27 / 11.69)
                ax = fig.add_axes([0.04, max(0.02, 0.97 - fh), fw, min(fh, 0.95)]); ax.imshow(img); ax.axis('off')
                pdf.savefig(fig, dpi=200); plt.close(fig)
    return True


def _style_axes(ax, title: str):
    ax.set_title(title, fontsize=13, fontweight='bold', color=PLOT_INK, loc='left')
    ax.grid(True, color=PLOT_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#9a9993')
    ax.tick_params(colors=PLOT_INK2)


def _header_box(fig, lines, y0=0.915, h=0.07):
    """
    Wide rounded box at the top of a figure; the first item of lines[0] bold.
    v4.5: the font is reduced until the longest line fits inside the box
    (plot_style can make the base font large).
    """
    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_axes([0.015, y0, 0.97, h])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.003, 0.04), 0.994, 0.92, boxstyle='round,pad=0,rounding_size=0.06',
                                transform=ax.transAxes, facecolor='#f4f4f2', edgecolor='#c9c8c3', linewidth=1.0,
                                mutation_aspect=0.12))
    renderer = fig.canvas.get_renderer()
    box_w = ax.get_window_extent(renderer=renderer).width
    gap = 0.012
    for scale in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6):
        texts, fits = [], True
        n = len(lines)
        for i, parts in enumerate(lines):
            y = 0.70 - i * (0.42 if n > 1 else 0)
            x = 0.015
            for j, (txt, bold) in enumerate(parts):
                t = ax.text(x, y, txt, transform=ax.transAxes, fontsize=(12 if bold else 10.5) * scale,
                            fontweight='bold' if bold else 'normal',
                            color=PLOT_INK if (bold or j == 0) else PLOT_INK2, va='center', ha='left')
                texts.append(t)
                x += t.get_window_extent(renderer=renderer).width / box_w + gap
            if x - gap > 0.985:
                fits = False
        if fits:
            break
        for t in texts:
            t.remove()
    return ax


def _signal_colors(sat_id: str, sigs) -> Dict[str, str]:
    """Fixed order: first signal of the constellation pair blue, second orange, others aqua."""
    gnss = sat_id.split('_')[0]
    pair = FREQ_PAIRS.get(gnss, ())
    out = {}
    for s in sigs:
        out[s] = PLOT_BLUE if (pair and s == pair[0]) else PLOT_ORANGE if (len(pair) > 1 and s == pair[1]) else PLOT_AQUA
    return out


def generate_raw_plots(sat_data: pd.DataFrame, sat_id: str, output_path: str, dpi: int = 150,
                       site: Optional[Dict[str, Any]] = None) -> bool:
    """
    Panel 1 - GNSS raw observations (v4.4).
    Header box: satellite, site name / lat / lon / height, session sample rate
    (gaps excluded) and track summary. 2x2: C/N0, elevation, measured Doppler,
    atmospheric Doppler (+ smoothed fit), all against UTC.
    site keys: name, lat, lon, height_msl_m, height_ell_m, rate (session_sample_rate dict)
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from plot_style import apply_plot_fonts
    except ImportError:
        return False
    apply_plot_fonts()
    if sat_data.empty:
        return False
    site = site or {}
    df = sat_data.copy()
    df['utc_parsed'] = pd.to_datetime(df['utc'], format='mixed', errors='coerce') if 'utc' in df.columns else pd.NaT
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    sigs = list(pd.unique(df['sigID'])) if 'sigID' in df.columns else []
    pair = list(FREQ_PAIRS.get(sat_id.split('_')[0], ()))
    sigs.sort(key=lambda sg: pair.index(sg) if sg in pair else 99)
    colors = _signal_colors(sat_id, sigs)

    fig = plt.figure(figsize=(13, 10.6))
    # ---- header
    rate = site.get('rate') or {}
    rec = rate.get('recorded_hz', np.nan)
    gaps_txt = (f" ({rate['n_gaps']} gap{'s' if rate['n_gaps'] != 1 else ''} > {rate['gap_threshold_s']:.1f} s excluded)"
                if rate.get('n_gaps') else " (no data gaps)")
    if np.isfinite(rate.get('rate_hz', np.nan)) and np.isfinite(rec) and rec > 1.5 * rate['rate_hz']:
        rate_txt = f"Sample rate {rec:.0f} Hz recorded · {rate['rate_hz']:.2f} Hz processed" + gaps_txt
    elif np.isfinite(rate.get('rate_hz', np.nan)):
        rate_txt = f"Session sample rate {rate['rate_hz']:.2f} Hz" + gaps_txt
    else:
        rate_txt = ""
    h_txt = f"H {site['height_msl_m']:.1f} m" if site.get('height_msl_m') is not None else ""
    pos_txt = (f"{site['lat']:.6f}°N  {site['lon']:.6f}°E" if site.get('lat') is not None else "")
    t = df['utc_parsed'].dropna()
    el = pd.to_numeric(df[elev_col], errors='coerce') if elev_col in df.columns else pd.Series(dtype=float)
    track = ""
    if not t.empty:
        order = df.dropna(subset=['utc_parsed']).sort_values('utc_parsed')
        e0 = pd.to_numeric(order[elev_col], errors='coerce').iloc[0] if elev_col in order else np.nan
        e1 = pd.to_numeric(order[elev_col], errors='coerce').iloc[-1] if elev_col in order else np.nan
        az = pd.to_numeric(df.get('accurate_azimuth', pd.Series(dtype=float)), errors='coerce').median()
        track = (f"Track {t.min():%H:%M:%S}–{t.max():%H:%M:%S} UTC  ·  {df['utc_parsed'].nunique()} epochs  ·  "
                 f"elevation {e0:+.2f}° → {e1:+.2f}°" + (f"  ·  azimuth {az:.0f}°" if np.isfinite(az) else "")
                 + f"  ·  signals {', '.join(map(str, sigs))}")
    line1 = [(sat_id, True), (f"Site: {site.get('name') or '—'}", False)]
    if pos_txt:
        line1.append((pos_txt, False))
    if h_txt:
        line1.append((h_txt, False))
    line1.append((rate_txt or "Sample rate n/a", False))
    line2 = [(track, False)]
    _header_box(fig, [line1, line2])

    gs = fig.add_gridspec(2, 2, left=0.065, right=0.985, top=0.875, bottom=0.065, hspace=0.34, wspace=0.2)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    t_all = df['utc_parsed'].dropna()
    t_pad = (t_all.max() - t_all.min()) * 0.03 if len(t_all) > 1 else pd.Timedelta(seconds=30)
    t_lim = (t_all.min() - t_pad, t_all.max() + t_pad) if len(t_all) else None

    def time_axis(ax):
        # one locator per axis (a shared locator binds to the last axis) and
        # one common time range for all four subplots
        loc = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        if t_lim is not None:
            ax.set_xlim(*t_lim)
        ax.set_xlabel('UTC', color=PLOT_INK2)

    def per_signal(ax, col, s=6, alpha=0.75):
        any_ = False
        for sg in sigs:
            sub = df[(df['sigID'] == sg)].dropna(subset=[col, 'utc_parsed'])
            if sub.empty:
                continue
            ax.scatter(sub['utc_parsed'], pd.to_numeric(sub[col], errors='coerce'), s=s, alpha=alpha,
                       color=colors[sg], edgecolors='none', label=sg)
            any_ = True
        if any_ and len(sigs) > 1:
            ax.legend(fontsize=9, markerscale=2.2, frameon=False, loc='best')
        return any_

    # (a) C/N0
    ax = axes[0][0]
    if 'cno' in df.columns:
        per_signal(ax, 'cno')
    _style_axes(ax, '(a) Signal strength C/N₀'); ax.set_ylabel('C/N₀ (dB-Hz)'); time_axis(ax)

    # (b) elevation
    ax = axes[0][1]
    if elev_col in df.columns:
        e = df.dropna(subset=[elev_col, 'utc_parsed']).drop_duplicates('utc_parsed')
        ax.scatter(e['utc_parsed'], e[elev_col], s=6, color=PLOT_BLUE, edgecolors='none')
        ax.axhline(0.0, color=PLOT_INK2, linewidth=1.0)
        ax.axhline(RO_ELEVATION_THRESHOLD, color=PLOT_INK2, linewidth=1.0, linestyle='--')
        ax.annotate(f'RO threshold {RO_ELEVATION_THRESHOLD:g}°', xy=(1.0, RO_ELEVATION_THRESHOLD), xycoords=('axes fraction', 'data'),
                    xytext=(-4, 3), textcoords='offset points', ha='right', va='bottom', fontsize=9, color=PLOT_INK2)
        ax.annotate('horizon 0°', xy=(1.0, 0.0), xycoords=('axes fraction', 'data'),
                    xytext=(-4, 3), textcoords='offset points', ha='right', va='bottom', fontsize=9, color=PLOT_INK2)
    _style_axes(ax, '(b) Elevation (geometric)'); ax.set_ylabel('Elevation (°)'); time_axis(ax)

    # (c) measured Doppler
    ax = axes[1][0]
    if 'doppler' in df.columns:
        per_signal(ax, 'doppler')
    _style_axes(ax, '(c) Measured Doppler'); ax.set_ylabel('Doppler (Hz)'); time_axis(ax)

    # (d) atmospheric Doppler + fit
    ax = axes[1][1]
    if 'atmos_doppler' in df.columns:
        per_signal(ax, 'atmos_doppler', s=6, alpha=0.35)
        if 'atmos_dopp_poli' in df.columns and 'timestamp' in df.columns:
            for sg in sigs:
                sd = df[df['sigID'] == sg].dropna(subset=['atmos_dopp_poli', 'utc_parsed']).sort_values('timestamp')
                if len(sd) < 2:
                    continue
                seg = np.concatenate([[0], np.cumsum(np.diff(sd['timestamp'].values) >= POLYFIT_GAP_THRESHOLD)])
                first = True
                for k in np.unique(seg):
                    part = sd[seg == k]
                    if len(part) < 2:
                        continue
                    ax.plot(part['utc_parsed'], part['atmos_dopp_poli'], color=colors[sg], linewidth=2.0,
                            label=f'{sg} fit' if first else None)
                    first = False
            ax.legend(fontsize=9, markerscale=2.2, frameon=False, loc='best')
        ax.axhline(0.0, color=PLOT_INK2, linewidth=0.8)
    _style_axes(ax, '(d) Atmospheric Doppler'); ax.set_ylabel('Atmospheric Doppler (Hz)'); time_axis(ax)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return True


CONSTELLATION_FREQ_LABELS = {
    'GPS': ('L1', 'L2'), 'GAL': ('E1', 'E5b'), 'BDS': ('B1', 'B2'),
    'GLO': ('G1', 'G2'), 'SBAS': ('L1', 'L5'), 'QZSS': ('L1', 'L2'),
}


def get_freq_labels_from_sat_id(sat_id: str) -> tuple:
    """Frequency band labels for a satellite id like 'GPS_12'."""
    return CONSTELLATION_FREQ_LABELS.get(sat_id.upper().split('_')[0], ('F1', 'F2'))


def _single_freq_tag(sat_results: Dict[str, Any]) -> str:
    """Title suffix for single-frequency (no ionospheric correction) results."""
    try:
        b = pd.read_csv(sat_results.get('bending_csv'), usecols=['freq_mode'], nrows=1)
        if str(b['freq_mode'].iloc[0]) == 'single':
            return '  [SINGLE-FREQ: no iono correction]'
    except Exception:
        pass
    return ''


# ---------------------------------------------------------------- map helpers
_MERC_R = 6378137.0


def _merc(lat, lon):
    lat = np.clip(np.asarray(lat, float), -85.0, 85.0)
    return _MERC_R * np.radians(lon), _MERC_R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _merc_inv(x, y):
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y, float) / _MERC_R)) - np.pi / 2), np.degrees(np.asarray(x, float) / _MERC_R)


def fetch_basemap(x0, x1, y0, y1, tiles_across: int = 4):
    """
    Stitch OpenStreetMap tiles covering a Web-Mercator box.
    Returns (image array, (left, right, bottom, top)) or (None, reason).
    Tiles are cached on disk (MAP_TILE_CACHE) and fetched with an identifying
    User-Agent, as the OSM tile usage policy requires; after one failure no
    further requests are made in this process (offline use stays fast).
    """
    if not MAP_TILE_URL:
        return None, 'basemap disabled'
    try:
        from PIL import Image
        import io
        import urllib.request
    except ImportError:
        return None, 'Pillow not installed'
    world = 2 * np.pi * _MERC_R
    z = int(np.clip(np.round(np.log2(world * tiles_across / max(x1 - x0, 1.0))), 2, 14))
    n = 2 ** z
    tx = lambda x: (x + world / 2) / world * n
    ty = lambda y: (world / 2 - y) / world * n
    ix0, ix1 = int(np.floor(tx(x0))), int(np.floor(tx(x1)))
    iy0, iy1 = int(np.floor(ty(y1))), int(np.floor(ty(y0)))
    if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 64:
        return None, 'too many tiles'
    canvas = None
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            xx, yy = ix % n, iy
            if not (0 <= yy < n):
                continue
            path = os.path.join(MAP_TILE_CACHE, str(z), str(xx), f'{yy}.png')
            data = None
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = f.read()
            elif not _TILES_OFFLINE['flag']:
                try:
                    req = urllib.request.Request(MAP_TILE_URL.format(z=z, x=xx, y=yy, s='a'),
                                                 headers={'User-Agent': MAP_USER_AGENT})
                    with urllib.request.urlopen(req, timeout=6) as r:
                        data = r.read()
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'wb') as f:
                        f.write(data)
                except Exception as e:
                    _TILES_OFFLINE['flag'] = True
                    return None, f'tiles unavailable ({type(e).__name__})'
            if data is None:
                return None, 'tiles unavailable (offline)'
            tile = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            if canvas is None:
                ts = tile.shape[0]
                canvas = np.zeros(((iy1 - iy0 + 1) * ts, (ix1 - ix0 + 1) * ts, 3), dtype=np.uint8)
            canvas[(iy - iy0) * ts:(iy - iy0 + 1) * ts, (ix - ix0) * ts:(ix - ix0 + 1) * ts] = tile[:ts, :ts, :3]
    if canvas is None:
        return None, 'no tiles'
    left = ix0 / n * world - world / 2
    right = (ix1 + 1) / n * world - world / 2
    top = world / 2 - iy0 / n * world
    bottom = world / 2 - (iy1 + 1) / n * world
    return canvas, (left, right, bottom, top)


def draw_tangent_point_map(ax, st_lat: float, st_lon: float, tp: pd.DataFrame, fig=None,
                           levels: Optional[pd.DataFrame] = None, h_range: Optional[tuple] = None):
    """Station-centred map: basemap, range rings, tangent points coloured by height.
    levels  : the profile levels (filled dots); other tp epochs are drawn as small rings.
    h_range : (low, high) km - the colour scale, shared with the y-axes of plots (b)-(d)."""
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    tp = tp.dropna(subset=['tp_lat', 'tp_lon']) if tp is not None and not tp.empty else pd.DataFrame()
    if levels is not None and not levels.empty and {'tp_lat', 'tp_lon', 'tp_height_km'} <= set(levels.columns):
        _lv = levels.dropna(subset=['tp_lat', 'tp_lon'])
        if not _lv.empty:                      # 3.5.1: map extent from the profile levels only
            tp = _lv.assign(azimuth_deg=_lv['tp_azimuth_deg']) if 'tp_azimuth_deg' in _lv.columns else _lv
    if tp.empty:
        ax.text(0.5, 0.5, 'No tangent-point positions\n(re-run with v4.4 to compute them)', ha='center',
                va='center', transform=ax.transAxes, fontsize=12, color=PLOT_INK2)
        ax.set_axis_off()
        return
    dmax = float(np.nanmax(tp['tp_distance_km'])) if 'tp_distance_km' in tp else 100.0
    steps = [5, 10, 20, 25, 50, 100, 200, 250, 500]
    step = next((s for s in steps if dmax / s <= 4), 500)
    n_rings = max(1, int(np.ceil(dmax / step)))
    r_view = (n_rings + 0.25) * step
    cx, cy = _merc(st_lat, st_lon)
    half = r_view * 1000.0 / np.cos(np.radians(st_lat))
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    img, ext = fetch_basemap(x0, x1, y0, y1)
    if img is not None:
        ax.imshow(img, extent=ext, origin='upper', interpolation='bilinear', alpha=0.85, zorder=0)
        note = '© OpenStreetMap contributors'
    else:
        ax.set_facecolor('#eef0ec')
        note = f'Basemap: {ext}'
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal')
    # range rings, labelled on the side away from the tangent points
    az_lbl = (float(np.nanmedian(tp['azimuth_deg'] if 'azimuth_deg' in tp else tp.get('tp_azimuth_deg', pd.Series([0.0]))))
              + 180.0 + 35.0) % 360.0
    az = np.linspace(0, 360, 361)
    for k in range(1, n_rings + 1):
        la, lo = destination_point(st_lat, st_lon, az, k * step * 1000.0, 6371000.0)
        rx, ry = _merc(la, lo)
        ax.plot(rx, ry, color='#3b3b38', linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.75, zorder=2)
        la_t, lo_t = destination_point(st_lat, st_lon, az_lbl, k * step * 1000.0, 6371000.0)
        lx, ly = _merc(la_t, lo_t)
        ax.text(lx, ly, f'{k * step:g} km', fontsize=8.5, color=PLOT_INK, ha='center', va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
    # tangent points (deeper = darker)
    cmap = mcolors.LinearSegmentedColormap.from_list('tp_blue', PLOT_SEQ_BLUE[::-1])
    px, py = _merc(tp['tp_lat'].values, tp['tp_lon'].values)
    norm = mcolors.Normalize(*h_range) if h_range else mcolors.Normalize(np.nanmin(tp['tp_height_km']),
                                                                          np.nanmax(tp['tp_height_km']))
    lv = levels.dropna(subset=['tp_lat', 'tp_lon']) if levels is not None and not levels.empty else pd.DataFrame()
    if lv.empty:
        sc = ax.scatter(px, py, c=tp['tp_height_km'], cmap=cmap, norm=norm, s=30, edgecolors='white',
                        linewidths=0.6, zorder=5)
    else:
        # 3.5.1: only the profile levels (the points of plots b-d)
        lx_, ly_ = _merc(lv['tp_lat'].values, lv['tp_lon'].values)
        sc = ax.scatter(lx_, ly_, c=lv['tp_height_km'], cmap=cmap, norm=norm, s=38, edgecolors='white',
                        linewidths=0.7, zorder=5)
    ax.scatter([cx], [cy], marker='^', s=120, color=PLOT_INK, edgecolors='white', linewidths=1.2, zorder=6)
    ax.annotate('Station', (cx, cy), xytext=(6, -12), textcoords='offset points', fontsize=9, color=PLOT_INK,
                zorder=6, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
    cb = (fig or ax.figure).colorbar(sc, ax=ax, fraction=0.04, pad=0.1, location='left')
    cb.set_label('Tangent-point height (km)', color=PLOT_INK2)
    cb.outline.set_visible(False)
    # ticks at round degrees
    la_lo, lo_lo = _merc_inv(x0, y0)
    la_hi, lo_hi = _merc_inv(x1, y1)
    def nice(span):
        return next((s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5) if span / s <= 4.5), 10)
    st_lo, st_la = nice(lo_hi - lo_lo), nice(la_hi - la_lo)
    lons = np.arange(np.ceil(lo_lo / st_lo) * st_lo, lo_hi, st_lo)
    lats = np.arange(np.ceil(la_lo / st_la) * st_la, la_hi, st_la)
    ax.set_xticks(_merc(np.zeros_like(lons), lons)[0])
    ax.set_xticklabels([f"{v:.{0 if st_lo >= 1 else 2 if st_lo < 0.1 else 1}f}°E" for v in lons])
    ax.set_yticks(_merc(lats, np.zeros_like(lats))[1])
    ax.set_yticklabels([f"{v:.{0 if st_la >= 1 else 2 if st_la < 0.1 else 1}f}°N" for v in lats])
    ax.tick_params(labelsize=8.5, colors=PLOT_INK2)
    ax.text(0.99, 0.01, note, transform=ax.transAxes, fontsize=7.5, color=PLOT_INK2, ha='right', va='bottom',
            zorder=7, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_title('(a) Occultation Map', fontsize=13, fontweight='bold', color=PLOT_INK, loc='left')


STEP5_CHECK_NAMES = ['Epochs in this occultation', 'Ray geometry solved', 'Bending > 0 (no Doppler bias)',
                     'Crosses the apparent horizon', 'Below/above rays overlap', 'Profile levels']
CHECK_SHORT = {'Atmospheric Doppler (reference satellite found)': 'Atmos. Doppler',
               'Below the horizon (0°)': 'Below 0°', 'Epochs in this occultation': 'Event epochs',
               'Ray geometry solved': 'Geometry', 'Bending > 0 (no Doppler bias)': 'Bending > 0',
               'Crosses the apparent horizon': 'Crosses horizon', 'Below/above rays overlap': 'Overlap',
               'Profile levels': 'Levels', "Partial bending physical (model alpha_P)": "Physical α′ (model α_P)"}
CHECK_COLOR = {True: '#1a8a4a', False: '#c62828', None: '#e07b00', 'not_reached': '#a8a7a2'}


def load_ro_checks(bending_dir: str, item_id: str) -> Optional[Dict[str, Any]]:
    """The RO tests written by step 5 (bending/ro_checks.json) for one list item."""
    try:
        with open(os.path.join(bending_dir, 'ro_checks.json')) as f:
            data = json.load(f)
    except Exception:
        return None
    return data.get(item_id)


def _short_check_name(name: str) -> str:
    if name.startswith('Tracked below'):
        return 'Below ' + name.split('below ')[-1].replace(' elevation', '')
    if name.startswith('Dual frequency'):
        return 'Dual freq.'
    return CHECK_SHORT.get(name, name)


def _checks_strip(fig, checks: List[Dict[str, Any]], y_top: float, x0: float = 0.015, width: float = 0.97,
                  fontsize: float = 9.0) -> float:
    """One wrapped line of '● test: value' items. Returns the y below the strip."""
    renderer = fig.canvas.get_renderer()
    fw = fig.get_window_extent(renderer=renderer).width
    x, y, line_h = x0, y_top, 0.022
    for c in checks:
        col = CHECK_COLOR.get(c.get('passed'), CHECK_COLOR[None])
        txt = f"{_short_check_name(c['name'])}: {c['value']}"
        if len(txt) > 70:
            txt = txt[:67] + '…'
        t = fig.text(x + 0.012, y, txt, fontsize=fontsize, color=PLOT_INK2, va='center', ha='left')
        w = t.get_window_extent(renderer=renderer).width / fw
        if x + 0.012 + w > x0 + width and x > x0:            # wrap
            t.remove()
            x, y = x0, y - line_h
            t = fig.text(x + 0.012, y, txt, fontsize=fontsize, color=PLOT_INK2, va='center', ha='left')
            w = t.get_window_extent(renderer=renderer).width / fw
        fig.text(x, y, '●', fontsize=fontsize + 1, color=col, va='center', ha='left')
        x += 0.012 + w + 0.018
    return y - line_h


def generate_ro_checklist_plot(item_id: str, info: Optional[Dict[str, Any]], output_path: str,
                               reason: str = '', dpi: int = 150) -> bool:
    """
    Radio Occultation panel for an item WITHOUT a profile: every RO test in
    pipeline order with its value, the requirement and pass / fail; tests after
    the one that stopped the item are shown as not reached.
    """
    try:
        import matplotlib.pyplot as plt
        from plot_style import apply_plot_fonts
    except ImportError:
        return False
    apply_plot_fonts()
    checks = list((info or {}).get('checks', []))
    names = {c['name'] for c in checks}
    if info is not None and info.get('candidate'):
        stopped = next((c['name'] for c in checks if c.get('passed') is False), None)
    else:
        stopped = next((c['name'] for c in checks if c.get('passed') is False), 'RO candidate test')
    for nm in STEP5_CHECK_NAMES:
        if nm not in names:
            checks.append({'name': nm, 'passed': 'not_reached', 'value': 'not reached', 'need': ''})
    fig = plt.figure(figsize=(13, 11))
    fig.text(0.015, 0.975, f'Radio Occultation: {item_id}', fontsize=17, fontweight='bold', color=PLOT_INK, va='top')
    head = ('RO candidate, but no profile' if (info or {}).get('candidate') else 'Not a radio occultation')
    fig.text(0.015, 0.935, head + (f' - stopped at: {stopped}' if stopped else ''), fontsize=12, color='#8a3b12', va='top')
    if reason:
        fig.text(0.015, 0.905, f'Reason: {reason}', fontsize=10.5, color=PLOT_INK2, va='top')
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.80])
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(checks)
    dy = min(0.075, 0.92 / max(n, 1))
    ax.text(0.04, 0.98, 'Test', fontsize=11, fontweight='bold', color=PLOT_INK, va='center')
    ax.text(0.40, 0.98, 'Value', fontsize=11, fontweight='bold', color=PLOT_INK, va='center')
    ax.text(0.84, 0.98, 'Required', fontsize=11, fontweight='bold', color=PLOT_INK, va='center')
    stage_break = sum(1 for c in checks if c['name'] not in STEP5_CHECK_NAMES
                      and c['name'] != "Partial bending physical (model alpha_P)")
    for i, c in enumerate(checks):
        y = 0.98 - (i + 1) * dy - (0.02 if i >= stage_break else 0.0)
        p = c.get('passed')
        col = CHECK_COLOR.get(p, CHECK_COLOR[None])
        ax.scatter([0.015], [y], s=150, color=col, edgecolors='white', linewidths=1.0)
        mark = {True: 'pass', False: 'FAIL', None: 'note', 'not_reached': ''}.get(p, '')
        ax.text(0.03, y, mark, fontsize=8.5, color=col, va='center', ha='left', fontweight='bold')
        ink = PLOT_INK if p != 'not_reached' else '#a8a7a2'
        ax.text(0.08, y, c['name'], fontsize=11, color=ink, va='center')
        val = c['value'] if len(c['value']) <= 62 else c['value'][:59] + '…'
        ax.text(0.40, y, val, fontsize=10.5, color=PLOT_INK2 if p != 'not_reached' else '#a8a7a2', va='center')
        ax.text(0.84, y, c.get('need', ''), fontsize=10.5, color=PLOT_INK2, va='center')
        ax.plot([0.0, 1.0], [y - dy / 2, y - dy / 2], color=PLOT_GRID, linewidth=0.6)
    yb = 0.98 - (stage_break + 0.5) * dy - 0.01
    ax.text(1.0, yb, 'retrieval (step 5) ↓', fontsize=9, color=PLOT_INK2, ha='right', va='center')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return True


def _station_level_msl_km(sat_results: Dict[str, Any], fallback_m: Optional[float]) -> Optional[float]:
    try:
        b = pd.read_csv(sat_results.get('bending_csv'), nrows=1)
        return float(b['station_height_km'].iloc[0]) - float(b.get('geoid_sep_m', pd.Series([0.0])).iloc[0]) / 1000.0
    except Exception:
        return fallback_m / 1000.0 if fallback_m is not None else None


def _station_line(ax, h_km: Optional[float]):
    if h_km is None or not np.isfinite(h_km):
        return
    ax.axhline(h_km, color=PLOT_INK2, linewidth=1.0, linestyle='--', zorder=1)
    ax.annotate(f'station {h_km:.2f} km', xy=(1.0, h_km), xycoords=('axes fraction', 'data'), xytext=(-4, 3),
                textcoords='offset points', ha='right', va='bottom', fontsize=9, color=PLOT_INK2)


MK_RO = dict(marker='o', s=34, color=PLOT_BLUE, edgecolors='white', linewidths=0.6)
MK_ERA = dict(marker='s', s=34, facecolors='none', edgecolors=PLOT_ORANGE, linewidths=1.4)
MK_3 = dict(marker='^', s=38, color=PLOT_AQUA, edgecolors='#0b5a3e', linewidths=0.7)
H_LABEL = 'Tangent-point height above sea level (km)'


def generate_derived_plots(sat_results: Dict[str, Any], sat_id: str, output_path: str, dpi: int = 150,
                           station_altitude: Optional[float] = None, site: Optional[Dict[str, Any]] = None) -> bool:
    """
    Panel 2 - Radio Occultation (v4.4):
      (a) map: station, range rings, tangent point of every epoch coloured by height
      (b) bending angles vs tangent-point height
      (c) refractivity RO vs ERA5          (d) refractivity error vs ERA5 (%)
    Height = tangent-point height above mean sea level (Abel r = a/n), the same
    reference as ERA5. Impact height (a - R) is NOT used: it sits ~1.5 km high.
    """
    try:
        import matplotlib.pyplot as plt
        from plot_style import apply_plot_fonts
    except ImportError:
        return False
    apply_plot_fonts()
    tag = _single_freq_tag(sat_results)
    h_sta = _station_level_msl_km(sat_results, station_altitude)
    bend = pd.read_csv(sat_results['bending_csv']) if sat_results.get('bending_csv') and os.path.exists(sat_results['bending_csv']) else pd.DataFrame()
    tp_csv = sat_results.get('tp_csv') or (sat_results.get('bending_csv') or '').replace('_bending.csv', '_tangent_points.csv')
    tp = pd.read_csv(tp_csv) if tp_csv and os.path.exists(tp_csv) else pd.DataFrame()
    if tp.empty and sat_results.get('refrac_csv') and os.path.exists(sat_results['refrac_csv']):
        rf = pd.read_csv(sat_results['refrac_csv'])
        if 'tp_lat' in rf.columns:
            tp = rf.rename(columns={'height_km': 'tp_height_km'})

    # one height scale for (a) colours and the (b)-(d) y-axes
    hs = [h_sta] if h_sta is not None and np.isfinite(h_sta) else []
    for d_, c_ in ((tp, 'tp_height_km'), (bend, 'tp_height_km')):
        if not d_.empty and c_ in d_.columns:
            hs += list(pd.to_numeric(d_[c_], errors='coerce').dropna())
    if sat_results.get('refrac_csv') and os.path.exists(sat_results['refrac_csv']):
        hs += list(pd.read_csv(sat_results['refrac_csv'], usecols=['height_km'])['height_km'].dropna())
    h_range = (float(min(hs)), float(max(hs))) if len(hs) >= 2 and max(hs) > min(hs) else None
    h_pad = 0.06 * (h_range[1] - h_range[0]) if h_range else 0.0
    y_lim = (h_range[0] - h_pad, h_range[1] + h_pad) if h_range else None

    fig = plt.figure(figsize=(13, 12))
    st_lat = float(bend['station_lat'].iloc[0]) if 'station_lat' in bend.columns else (site or {}).get('lat')
    st_lon = float(bend['station_lon'].iloc[0]) if 'station_lon' in bend.columns else (site or {}).get('lon')
    sub = ''
    _sd = bend if (not bend.empty and 'tp_distance_km' in bend.columns) else tp
    if not tp.empty and 'tp_distance_km' in tp.columns:
        sub = (f"tangent points {np.nanmin(_sd['tp_distance_km']):.0f}–{np.nanmax(_sd['tp_distance_km']):.0f} km from the station"
               + (f", azimuth {np.nanmedian(tp['azimuth_deg']):.0f}°" if 'azimuth_deg' in tp.columns else ''))
    fig.suptitle(f'Radio Occultation: {sat_id}', fontsize=17, fontweight='bold', x=0.015, ha='left', y=0.988,
                 color=PLOT_INK)
    if sub:
        fig.text(0.015, 0.953, sub, fontsize=10.5, color=PLOT_INK2, ha='left')
    y_next = 0.928
    if tag:                                            # v4.6: single-frequency alert banner
        sig1 = str(bend['sig1'].iloc[0]) if 'sig1' in bend.columns else get_freq_labels_from_sat_id(sat_id)[0]
        fig.text(0.015, y_next, f"  ⚠  SINGLE-FREQUENCY occultation ({sig1} only): the second frequency was not "
                                f"available, so there is NO ionospheric correction (about 1–2 % in N).  ",
                 fontsize=10.5, color='#7a2e00', fontweight='bold', va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.45', facecolor='#ffe3cc', edgecolor='#e07b00', linewidth=1.2))
        y_next -= 0.035
    # v4.7: per-level flags - L1-only (L2 gap) and model alpha_P (no above-horizon ray)
    flag_l1 = (~bend['iono_corrected'].astype(bool)).values if (not tag and 'iono_corrected' in bend.columns) \
        else np.zeros(len(bend), bool)
    flag_mod = (bend['alpha_P_src'] == 'model').values if 'alpha_P_src' in bend.columns else np.zeros(len(bend), bool)
    flagged = flag_l1 | flag_mod
    # Flagged levels (3.5.1): hollow marker = L1-only level (L2 gap, no ionospheric
    # correction). The marker explains itself; the legend names it once.
    from matplotlib.lines import Line2D
    FLAG_TXT_L1 = 'L1-only (L2 gap)'

    def _flag_on(mask, h):
        """per-level flag for rows of another table (refractivity/comparison), matched by height."""
        if not mask.any() or 'tp_height_km' not in bend.columns:
            return np.zeros(len(h), bool)
        hb = bend['tp_height_km'].values
        return np.array([bool(mask[np.argmin(np.abs(hb - x))]) and np.min(np.abs(hb - x)) < 0.002
                         for x in np.asarray(h, float)])

    def _sc(ax_, x, y, mk, fl_l1=None, fl_mod=None, label=None):
        """normal markers; flagged levels hollow (L1-only) or half-filled (model alpha_P)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        f1 = np.zeros(len(x), bool) if fl_l1 is None else np.asarray(fl_l1, bool)
        fm = np.zeros(len(x), bool) if fl_mod is None else np.asarray(fl_mod, bool)
        f1 = f1 & ~fm
        ax_.scatter(x[~(f1 | fm)], y[~(f1 | fm)], **mk, label=label)
        col = mk.get('color', mk.get('edgecolors'))
        ms = float(np.sqrt(mk.get('s', 34)))
        style = dict(linestyle='none', marker=mk.get('marker', 'o'), markersize=ms, markeredgecolor=col,
                     markeredgewidth=1.3, zorder=3)
        if f1.any():
            ax_.plot(x[f1], y[f1], markerfacecolor='white', **style)
        if fm.any():                                    # model alpha_P levels (feature off in 3.5.1)
            ax_.plot(x[fm], y[fm], markerfacecolor='white', **style)
        ax_._flags = getattr(ax_, '_flags', set()) | ({'l1'} if f1.any() else set()) | ({'mod'} if fm.any() else set())

    def _legend(ax_, **kw):
        """legend + one neutral entry per flag type present in this plot."""
        h, l = ax_.get_legend_handles_labels()
        fl = getattr(ax_, '_flags', set())
        base = dict(linestyle='none', marker='o', markersize=6.5, markeredgecolor=PLOT_INK2, markeredgewidth=1.3)
        if fl & {'l1', 'mod'}:
            h.append(Line2D([], [], markerfacecolor='white', **base)); l.append(FLAG_TXT_L1)
        ax_.legend(h, l, **kw)
    info = load_ro_checks(os.path.dirname(sat_results.get('bending_csv') or ''), sat_id)
    if info and info.get('checks'):
        fig.text(0.015, y_next, 'RO tests', fontsize=9, fontweight='bold', color=PLOT_INK, va='center')
        y_next = _checks_strip(fig, info['checks'], y_next, x0=0.075, width=0.91, fontsize=8.6)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=y_next - 0.025, bottom=0.055, hspace=0.3, wspace=0.24)

    # (a) map
    ax = fig.add_subplot(gs[0, 0])
    if st_lat is not None:
        draw_tangent_point_map(ax, st_lat, st_lon, tp, fig,
                               levels=bend if 'tp_lat' in bend.columns else None, h_range=y_lim)
    else:
        ax.text(0.5, 0.5, 'Station position unknown', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    # (b) bending - per channel and ionosphere-corrected, as in Hajj et al. (2002) Fig. 9.
    # Mountain receiver: every curve is the PARTIAL bending (below-horizon ray minus
    # the above-horizon ray at the same impact parameter), i.e. the bending
    # accumulated below the station only.
    ax = fig.add_subplot(gs[0, 1])
    if not bend.empty:
        ycol = 'tp_height_km' if 'tp_height_km' in bend.columns else 'impact_height_km'
        ylab = H_LABEL if ycol == 'tp_height_km' else 'Impact height a − R (km) — not a real height'
        f1l, f2l = get_freq_labels_from_sat_id(sat_id)
        if 'sig1' in bend.columns and str(bend['sig1'].iloc[0]) not in ('', 'nan'):
            f1l = str(bend['sig1'].iloc[0])                 # the actual signal, e.g. L1OF, L1C/A
        if 'sig2' in bend.columns and str(bend['sig2'].iloc[0]) not in ('', 'nan'):
            f2l = str(bend['sig2'].iloc[0])
        single = 'freq_mode' in bend.columns and str(bend['freq_mode'].iloc[0]) == 'single'
        if 'partial_L1_rad' in bend.columns:
            _sc(ax, np.degrees(bend['partial_L1_rad']), bend[ycol], MK_RO, None, flag_mod, label=f'{f1l}')
            if not single and 'partial_L2_rad' in bend.columns and bend['partial_L2_rad'].notna().any():
                ax.scatter(np.degrees(bend['partial_L2_rad']), bend[ycol], **MK_ERA, label=f'{f2l}')
        _sc(ax, np.degrees(bend['bending_angle_rad']), bend[ycol], MK_3, flag_l1, flag_mod,
            label=f'{f1l} (no iono correction)' if single else 'ionosphere-corrected')
        _legend(ax, fontsize=9, frameon=True, framealpha=0.9, edgecolor='none', loc='best')
        ax.set_ylabel(ylab)
        _station_line(ax, h_sta)
    _style_axes(ax, '(b) Bending angle'); ax.set_xlabel('Bending angle (°)')
    if y_lim: ax.set_ylim(*y_lim)

    # (c) refractivity
    ax = fig.add_subplot(gs[1, 0])
    comp = pd.read_csv(sat_results['comp_csv']) if sat_results.get('comp_csv') and os.path.exists(sat_results['comp_csv']) else pd.DataFrame()
    refr = pd.read_csv(sat_results['refrac_csv']) if sat_results.get('refrac_csv') and os.path.exists(sat_results['refrac_csv']) else pd.DataFrame()
    def _split(d):
        if 'height_above_station_km' in d.columns:
            top = d['height_above_station_km'].abs() < 0.005
        else:
            top = d['height_km'] >= d['height_km'].max() - 1e-6
        return d[~top], d[top]
    if not refr.empty:
        r_lv, r_top = _split(refr)
        _sc(ax, r_lv['refractivity_N'], r_lv['height_km'], MK_RO, _flag_on(flag_l1, r_lv['height_km']),
            _flag_on(flag_mod, r_lv['height_km']), label='RO')
        if not comp.empty:
            c_lv, _ = _split(comp)
            ax.scatter(c_lv['N_ERA5'], c_lv['height_km'], **MK_ERA, label='.nc')
        if not r_top.empty:
            ax.scatter(r_top['refractivity_N'], r_top['height_km'], marker='D', s=46, facecolors='none',
                       edgecolors=PLOT_INK, linewidths=1.3, zorder=4, label='station (boundary)')
        _legend(ax, fontsize=9, frameon=True, framealpha=0.9, edgecolor='none', loc='best')
        _station_line(ax, h_sta)
    _style_axes(ax, '(c) Refractivity'); ax.set_xlabel('Refractivity N (N-units)'); ax.set_ylabel(H_LABEL)
    if y_lim: ax.set_ylim(*y_lim)

    # (d) refractivity error
    ax = fig.add_subplot(gs[1, 1])
    if not comp.empty:
        c_lv, _ = _split(comp)
        pct = (c_lv['N_RO'] - c_lv['N_ERA5']) / c_lv['N_ERA5'] * 100.0
        ax.axvline(0.0, color=PLOT_INK2, linewidth=1.0)
        _sc(ax, pct, c_lv['height_km'], MK_RO, _flag_on(flag_l1, c_lv['height_km']),
            _flag_on(flag_mod, c_lv['height_km']), label='RO − .nc')
        _legend(ax, fontsize=9, frameon=True, framealpha=0.9, edgecolor='none', loc='best')
        err_stats = (f"\nmean {pct.mean():+.1f} %  ·  RMS {np.sqrt((pct ** 2).mean()):.1f} %  ·  "
                     f"{len(pct)} levels below the station") if len(pct) else ''
        _station_line(ax, h_sta)
    else:
        ax.text(0.5, 0.5, 'Needs a .nc file for the comparison', ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color=PLOT_INK2)
    _style_axes(ax, '(d) Refractivity error vs .nc')
    if y_lim: ax.set_ylim(*y_lim)
    ax.set_xlabel('(N_RO − N_nc) / N_nc (%)' + (err_stats if not comp.empty else '')); ax.set_ylabel(H_LABEL)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return True


def generate_atmospheric_plots(sat_results: Dict[str, Any], sat_id: str, output_path: str, dpi: int = 150,
                               station_altitude: Optional[float] = None) -> bool:
    """
    Panel 3 - Atmospheric profiles (v4.4): pressure, water-vapour pressure,
    relative humidity, temperature (ERA5, the retrieval's constraint), against
    tangent-point height above MSL. RO = filled blue circles, ERA5 = open
    orange squares.
    """
    try:
        import matplotlib.pyplot as plt
        from plot_style import apply_plot_fonts
    except ImportError:
        return False
    apply_plot_fonts()
    tag = _single_freq_tag(sat_results)
    h_sta = _station_level_msl_km(sat_results, station_altitude)
    atm = pd.read_csv(sat_results['atm_csv']) if sat_results.get('atm_csv') and os.path.exists(sat_results['atm_csv']) else pd.DataFrame()
    fig = plt.figure(figsize=(13, 10.6))
    fig.suptitle(f'Atmospheric Profiles: {sat_id}{tag}', fontsize=17, fontweight='bold', x=0.015, ha='left', y=0.985,
                 color='#B23A0B' if tag else PLOT_INK)
    fig.text(0.015, 0.948, 'Temperature (from the .nc file) is the constraint; P, Pw and RH are retrieved from the RO refractivity.',
             fontsize=10.5, color=PLOT_INK2, ha='left')
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.905, bottom=0.065, hspace=0.32, wspace=0.22)
    panels = [('(a) Pressure', 'Pressure (hPa)', 'pressure_hPa', 'P_era5'),
              ('(b) Water-vapour pressure', 'Water-vapour pressure (hPa)', 'water_vapor_hPa', 'Pw_era5'),
              ('(c) Relative humidity', 'Relative humidity (%)', 'RH', 'RH_era5'),
              ('(d) Temperature', 'Temperature (°C)', None, 'T_C')]
    if not atm.empty and 'T_era5' in atm.columns:
        T_c = atm['T_era5'] - 273.15
        es = 6.1094 * np.exp(17.625 * T_c / (T_c + 243.04))
        if 'water_vapor_hPa' in atm.columns:
            atm['RH'] = (atm['water_vapor_hPa'] / es * 100).clip(0, 100)
        if 'Pw_era5' in atm.columns:
            atm['RH_era5'] = (atm['Pw_era5'] / es * 100).clip(0, 100)
        atm['T_C'] = T_c
    for k, (title, xlab, ro_col, era_col) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        if atm.empty:
            ax.text(0.5, 0.5, 'Needs the atmospheric retrieval (.nc file)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color=PLOT_INK2)
        else:
            if ro_col and ro_col in atm.columns:
                ax.scatter(atm[ro_col], atm['height_km'], **MK_RO, label='RO')
            if era_col in atm.columns:
                ax.scatter(atm[era_col], atm['height_km'], **MK_ERA,
                           label='.nc' if ro_col else 'T (from .nc file)')
            ax.legend(fontsize=9.5, frameon=False, loc='best')
            _station_line(ax, h_sta)
        _style_axes(ax, title); ax.set_xlabel(xlab); ax.set_ylabel(H_LABEL)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return True


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

def event_station_from_bending(bending_csv: str, base: 'StationConfig') -> 'StationConfig':
    """Station of one occultation (its receiver fix and met at its time), as written by step 5."""
    try:
        b = pd.read_csv(bending_csv, nrows=1)
    except Exception:
        return base
    g = lambda c, v: float(b[c].iloc[0]) if c in b.columns and pd.notna(b[c].iloc[0]) else v
    return StationConfig(latitude=g('station_lat', base.latitude), longitude=g('station_lon', base.longitude),
                         altitude=g('station_height_km', base.altitude / 1000.0) * 1000.0, name=base.name,
                         surface_pressure_hPa=g('station_P_hPa', base.surface_pressure_hPa),
                         surface_temp_K=g('station_T_K', base.surface_temp_K),
                         surface_humidity_hPa=g('station_e_hPa', base.surface_humidity_hPa),
                         geoid_sep_m=g('geoid_sep_m', base.geoid_sep_m))


class GNSSROPipeline:
    def __init__(self, station: StationConfig, config: PipelineConfig = PipelineConfig()):
        self.station = station
        self.config = config
        self.results: Dict[str, ProcessingResult] = {}

    def run_full_pipeline(self, ubx_dir: str, sp3_file: str,
                          era5_file: Optional[str] = None,
                          output_dir: str = "./output",
                          progress_callback: Optional[Callable] = None) -> Dict[str, ProcessingResult]:
        """
        Run the complete mountain-top GNSS-RO pipeline.
        Supports UBX or RINEX input (auto-detected).
        """
        os.makedirs(output_dir, exist_ok=True)
        st = self.station

        # Step 1: Parse observations (UBX or RNX)
        if progress_callback:
            progress_callback("Parsing observation files...", 0.0)
        self.results['step1'] = parse_gnss_directory(
            ubx_dir, f"{output_dir}/step1_observations.csv", progress_callback)
        if not self.results['step1'].success:
            return self.results
        source = self.results['step1'].metadata.get('source', 'unknown')
        if progress_callback:
            progress_callback(f"Using {source} data source", 0.05)

        # v4.1/4.5: station position from the receiver, met from ERA5 if not supplied
        md = self.results['step1'].metadata
        rec = md.get('ubx_station') or md.get('rinex_station') or extract_station_info(ubx_dir)
        if rec is not None and 'source' not in rec:
            rec = {**rec, 'source': 'UBX NAV-PVT' if md.get('ubx_station') else 'RINEX header'}
        self.station, pos_src = resolve_station(self.station, rec)
        st = self.station
        obs_csv = f"{output_dir}/step1_observations.csv"
        df1 = pd.read_csv(obs_csv)
        if pos_src == 'cra' or 'sta_lat' not in df1.columns:
            # .cra wins (forced, or no receiver position): every observation uses it
            for c, v in (('sta_lat', st.latitude), ('sta_lon', st.longitude), ('sta_h', st.altitude),
                         ('sta_geoid', st.geoid_sep_m)):
                df1[c] = v
            df1['sta_file'] = 'metadata.cra'
        else:
            # each observation keeps its own file's 3D fix; gaps get the session position
            for c, v in (('sta_lat', st.latitude), ('sta_lon', st.longitude), ('sta_h', st.altitude),
                         ('sta_geoid', st.geoid_sep_m)):
                df1[c] = pd.to_numeric(df1[c], errors='coerce').fillna(v)
        df1.to_csv(obs_csv, index=False)
        spread = (rec or {}).get('max_file_spread_m', 0.0) or 0.0
        if pos_src != 'cra' and spread > STATION_SPLIT_WARN_M:
            pos_src += (f"; files from {len(rec.get('files', []))} positions up to {spread:.0f} m apart - "
                        f"each file uses its own 3D fix")
        met_src = 'cra'
        self.met_fn = None
        if st.surface_N is None and st.surface_pressure_hPa is None and era5_file:
            # v4.5: station met is taken per occultation, at its own time and place
            self.met_fn = lambda ev_st, when, _f=era5_file: era5_station_met(
                _f, ev_st.latitude, ev_st.longitude, ev_st.altitude - ev_st.geoid_sep_m, when)
            utc = pd.to_datetime(df1['utc'], format='mixed', errors='coerce').dropna()
            when = str(utc.iloc[len(utc) // 2]) if not utc.empty else None
            met = era5_station_met(era5_file, st.latitude, st.longitude, st.altitude - st.geoid_sep_m, when)
            if met:
                st.surface_pressure_hPa, st.surface_temp_K, st.surface_humidity_hPa = met['P'], met['T'], met['e']
                met_src = '.nc (per occultation, at its time)'
        if st.surface_pressure_hPa is None and st.surface_N is None:
            met_src = 'standard atmosphere (inaccurate)'
        self.results['station'] = ProcessingResult(
            True, None,
            f"Station {st.latitude:.6f}, {st.longitude:.6f}, GPS height {st.altitude - st.geoid_sep_m:.1f} m "
            f"from {pos_src}; n_r from {met_src}: N = {st.get_surface_refractivity():.1f}",
            {'position_source': pos_src, 'met_source': met_src})
        if progress_callback:
            progress_callback(self.results['station'].message, 0.06)

        # Step 2: SP3 matching
        if progress_callback:
            progress_callback("Matching with SP3 ephemeris...", 0.1)
        self.results['step2'] = match_observations_with_sp3(
            f"{output_dir}/step1_observations.csv", sp3_file,
            f"{output_dir}/step2_matched.csv", progress_callback)
        if not self.results['step2'].success:
            return self.results

        # Step 3a: Elevation (geodetic normal)
        if progress_callback:
            progress_callback("Calculating elevations...", 0.35)
        self.results['step3a'] = calculate_accurate_elevations(
            f"{output_dir}/step2_matched.csv", st, f"{output_dir}/step3a_elevations.csv")
        if not self.results['step3a'].success:
            return self.results

        # Step 3b: Geometric Doppler
        if progress_callback:
            progress_callback("Calculating geometric Doppler...", 0.45)
        self.results['step3b'] = calculate_geometric_doppler(
            f"{output_dir}/step3a_elevations.csv", st, f"{output_dir}/step3b_doppler.csv")
        if not self.results['step3b'].success:
            return self.results

        # Step 4: Single differencing + Fresnel-adaptive smoothing
        if progress_callback:
            progress_callback("Applying single differencing...", 0.55)
        self.results['step4'] = apply_single_differencing(
            f"{output_dir}/step3b_doppler.csv", self.config,
            f"{output_dir}/step4_differenced.csv", station_alt_m=st.altitude)

        # Step 5: Bending angles (alpha_N, alpha_P, partial bending)
        if progress_callback:
            progress_callback("Retrieving bending angles...", 0.65)
        self.results['step5'] = retrieve_bending_angles(
            f"{output_dir}/step4_differenced.csv", st, self.config, f"{output_dir}/bending",
            met_fn=getattr(self, 'met_fn', None),
            nprof_fn=((lambda ev_st, when, _f=era5_file: era5_column_above_station(_f, ev_st, when))
                      if era5_file else None))

        # Steps 6-7: per occultation
        if self.results['step5'].success and self.results['step5'].data is not None:
            for idx, row in self.results['step5'].data.iterrows():
                sat_id = row['sat_id']
                when = row.get('utc_mid', None) or None
                bending_csv = f"{output_dir}/bending/{sat_id}_bending.csv"
                if not os.path.exists(bending_csv):
                    continue
                if progress_callback:
                    progress_callback(f"Abel inversion: {sat_id}...", min(0.7 + idx * 0.03, 0.98))
                refrac_csv = f"{output_dir}/refractivity/{sat_id}_refractivity.csv"
                os.makedirs(os.path.dirname(refrac_csv), exist_ok=True)
                result = retrieve_refractivity(bending_csv, refrac_csv)
                self.results[f'step6_{sat_id}'] = result

                if era5_file and result.success:
                    comp_csv = f"{output_dir}/comparison/{sat_id}_comparison.csv"
                    atm_csv = f"{output_dir}/atmospheric/{sat_id}_atmospheric.csv"
                    os.makedirs(os.path.dirname(comp_csv), exist_ok=True)
                    os.makedirs(os.path.dirname(atm_csv), exist_ok=True)
                    ev_st = event_station_from_bending(bending_csv, st)      # v4.5: this event's fix + met
                    self.results[f'step6b_{sat_id}'] = compare_with_era5(
                        refrac_csv, era5_file, ev_st.latitude, ev_st.longitude, comp_csv, when=when)
                    self.results[f'step7_{sat_id}'] = retrieve_atmospheric_profile(
                        refrac_csv, era5_file, ev_st.latitude, ev_st.longitude, atm_csv,
                        when=when, station=ev_st)

        if progress_callback:
            progress_callback("Pipeline complete", 1.0)
        return self.results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mountain-top GNSS-RO Processing Pipeline")
    parser.add_argument("--ubx-dir", required=True)
    parser.add_argument("--sp3-file", required=True)
    parser.add_argument("--era5-file")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--alt", type=float, required=True, help="GPS height above sea level (m), as the receiver shows it")
    parser.add_argument("--name", default="Station")
    parser.add_argument("--surface-p", type=float, help="station pressure (hPa)")
    parser.add_argument("--surface-t", type=float, help="station temperature (K)")
    parser.add_argument("--surface-e", type=float, help="station water vapour pressure (hPa)")
    parser.add_argument("--obs-time-is-gps", action="store_true",
                        help="observation times are GPS time (no leap-second shift)")
    args = parser.parse_args()

    if args.obs_time_is_gps:
        apply_processing_config({**PROCESSING_DEFAULTS, 'OBS_TIME_IS_GPS': True})
    station = StationConfig(latitude=args.lat, longitude=args.lon, altitude=args.alt, height_ref='msl',
                            name=args.name, surface_pressure_hPa=args.surface_p,
                            surface_temp_K=args.surface_t, surface_humidity_hPa=args.surface_e)
    if station.surface_pressure_hPa is None:
        print("WARNING: no station met data; n_r falls back to a standard atmosphere. "
              "For mountain RO, n_r is the Abel top boundary: pass --surface-p/-t/-e.")
    pipeline = GNSSROPipeline(station)
    results = pipeline.run_full_pipeline(args.ubx_dir, args.sp3_file, args.era5_file, args.output_dir)
    print("\n" + "=" * 60 + "\nPIPELINE SUMMARY\n" + "=" * 60)
    for step, result in results.items():
        print(f"{'✓' if result.success else '✗'} {step}: {result.message}")
