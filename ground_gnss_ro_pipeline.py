"""
GNSS Radio Occultation Processing Pipeline v3.4.4.2
================================================

Pipeline Steps:
    1. UBX Parsing: Raw GNSS observations from u-blox receiver
    2. SP3 Matching: Interpolate precise satellite orbits to observation times
    3a. Elevation Calculation: Accurate elevation from ECEF coordinates
    3b. Geometric Doppler: Expected Doppler from satellite-receiver geometry
    4. Single Differencing: Remove receiver clock drift using reference satellite
    5. Bending Angle: Compute bending angle and impact parameter with iono correction
    6. Abel Inversion: Retrieve refractivity profile from bending angles
    7. Atmospheric Retrieval: Derive P, Pw, q using ERA5 temperature constraint

v1.2 Changes:
    - Added plotting functions (generate_raw_plots, generate_derived_plots)
    - Fixed T_era5 column in atmospheric output
    - Added progress callback support for SP3 matching
    - Added RO status evaluation function
    - Loosened RO threshold to ±2.5 Hz
"""

from __future__ import annotations
import os
import struct
import glob
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import fsolve

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

RO_ELEVATION_THRESHOLD = 5.0
RO_DOPPLER_THRESHOLD = 1
RO_MIN_EPOCHS = 25  # v3.4.4: raised from 10

POLYNOMIAL_WINDOW = 150 #(seconds, for a 50Hz sampling be 150/50 = 3)

# Polyfit segmentation: gap (sec) above which the rolling polynomial fit restarts.
POLYFIT_GAP_THRESHOLD = 5.0

# ----------------------------------------------------------------------------
# v3.4.4 — Configurable processing constants exposed via the .cra "PROCESSING" key.
# Anything the user puts under "PROCESSING" overrides the corresponding default
# below. Missing keys fall back to the defaults — i.e. the .cra is additive.
# ----------------------------------------------------------------------------
PROCESSING_DEFAULTS = {
    # Smoothing
    'POLY_SMOOTH_WINDOW': 150,        # Polynomial smoothing window (sec). 50Hz→3s effective.
    'POLYFIT_GAP_THRESHOLD': 5.0,     # Restart polyfit when gap >= this many seconds.

    # RO detection
    'RO_ELEVATION_THRESHOLD': 5.0,    # deg
    'RO_DOPPLER_THRESHOLD': 1.0,      # Hz
    'RO_MIN_EPOCHS': 25,              # Minimum RO epochs for a valid event.

    # Reference satellite selection (kept for forward compatibility w/ ref-rework)
    'REF_SAT_ELEVATION_THRESHOLD': 50.0,
    'REF_SAT_MIN_EPOCHS': 100,
    'REF_SAT_JUMP_THRESHOLD': 2.0,

    # Smith-Weintraub refractivity coefficients
    'N_COEFF_A1': 77.6,
    'N_COEFF_A2': 3.73e5,

    # Pipeline behaviour
    'KEEP_INTERMEDIATE_CSVS': False,   # If true, keep step1/step2/step3 CSVs after run.
    'FORCE_CRA_STATION_COORDS': False, # If true, use .cra station coords even if RINEX has APPROX POSITION XYZ.
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
    global RO_ELEVATION_THRESHOLD, RO_DOPPLER_THRESHOLD, RO_MIN_EPOCHS
    global POLYNOMIAL_WINDOW, POLYFIT_GAP_THRESHOLD
    global N_COEFF_A1, N_COEFF_A2

    if not cfg:
        return

    RO_ELEVATION_THRESHOLD = float(cfg.get('RO_ELEVATION_THRESHOLD', RO_ELEVATION_THRESHOLD))
    RO_DOPPLER_THRESHOLD   = float(cfg.get('RO_DOPPLER_THRESHOLD', RO_DOPPLER_THRESHOLD))
    RO_MIN_EPOCHS          = int(cfg.get('RO_MIN_EPOCHS', RO_MIN_EPOCHS))
    POLYNOMIAL_WINDOW      = float(cfg.get('POLY_SMOOTH_WINDOW', POLYNOMIAL_WINDOW))
    POLYFIT_GAP_THRESHOLD  = float(cfg.get('POLYFIT_GAP_THRESHOLD', POLYFIT_GAP_THRESHOLD))
    N_COEFF_A1             = float(cfg.get('N_COEFF_A1', N_COEFF_A1))
    N_COEFF_A2             = float(cfg.get('N_COEFF_A2', N_COEFF_A2))


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


def calculate_elevation_angle(sat_xyz: np.ndarray, station_xyz: np.ndarray) -> float:
    d = sat_xyz - station_xyz
    d_mag = np.linalg.norm(d)
    r_e_mag = np.linalg.norm(station_xyz)
    cos_zenith = np.dot(d, station_xyz) / (d_mag * r_e_mag)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = math.acos(cos_zenith)
    return math.degrees(math.pi / 2 - zenith_rad)


def compute_gravity(h_m: float, lat_deg: Optional[float] = None) -> float:
    g0 = 9.80665
    if lat_deg is None:
        return g0 * (R_EARTH / (R_EARTH + h_m)) ** 2
    lat_rad = np.radians(lat_deg)
    sin2 = np.sin(lat_rad) ** 2
    sin22 = np.sin(2 * lat_rad) ** 2
    g_surf = 9.780327 * (1 + 0.0053024 * sin2 - 0.0000058 * sin22)
    return g_surf - 3.086e-6 * h_m


def evaluate_ro_status(
    sat_data: pd.DataFrame,
    elevation_threshold: Optional[float] = None,
    doppler_threshold: Optional[float] = None,
    min_epochs: Optional[int] = None,
    min_dual_freq_epochs: Optional[int] = None
) -> Dict[str, bool]:
    """Evaluate RO status for each satellite.

    v3.4.4: thresholds default to live module-level values so .cra-overridden
    PROCESSING values take effect downstream.
    """
    # Resolve live so .cra overrides apply at call time.
    if elevation_threshold is None:
        elevation_threshold = RO_ELEVATION_THRESHOLD
    if doppler_threshold is None:
        doppler_threshold = RO_DOPPLER_THRESHOLD
    if min_epochs is None:
        min_epochs = RO_MIN_EPOCHS
    if min_dual_freq_epochs is None:
        min_dual_freq_epochs = RO_MIN_EPOCHS

    ro_status = {}
    
    if sat_data.empty:
        return ro_status
    
    df = sat_data.copy()
    if 'sat_id' not in df.columns:
        if 'gnssId' in df.columns and 'svId' in df.columns:
            df['sat_id'] = df['gnssId'].astype(str) + '_' + df['svId'].astype(str)
        else:
            return ro_status
    
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    
    for sat_id, group in df.groupby('sat_id'):
        if 'atmos_dopp_poli' not in group.columns or elev_col not in group.columns:
            ro_status[sat_id] = False
            continue
        
        # Check 1: atmos_doppler polinomial threshold at low elevation
        ro_mask = (
            (group[elev_col] < elevation_threshold) & 
            (group['atmos_dopp_poli'].abs() > doppler_threshold)
        )
        ro_count = ro_mask.sum()
        
        if ro_count < min_epochs:
            ro_status[sat_id] = False
            continue
        
        # Check 2: dual-frequency availability at low elevation
        gnss_id = group['gnssId'].iloc[0] if 'gnssId' in group.columns else None
        if gnss_id not in FREQ_PAIRS:
            ro_status[sat_id] = False
            continue
        
        sig1, sig2 = FREQ_PAIRS[gnss_id]
        low_elev = group[group[elev_col] < elevation_threshold]
        
        if 'timestamp' in low_elev.columns:
            timestamps_sig1 = set(low_elev[low_elev['sigID'] == sig1]['timestamp'])
            timestamps_sig2 = set(low_elev[low_elev['sigID'] == sig2]['timestamp'])
        elif 'utc' in low_elev.columns:
            timestamps_sig1 = set(low_elev[low_elev['sigID'] == sig1]['utc'])
            timestamps_sig2 = set(low_elev[low_elev['sigID'] == sig2]['utc'])
        else:
            ro_status[sat_id] = False
            continue
        
        dual_freq_count = len(timestamps_sig1 & timestamps_sig2)
        ro_status[sat_id] = dual_freq_count >= min_dual_freq_epochs
    
    return ro_status

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
                        key = (sat['iTOW'], sat['gnss'], sat['svId'])
                        self.measx_data[key] = sat
                elif (msg_class, msg_id) == (0x02, 0x15):
                    for sat in self._parse_rxm_rawx(payload):
                        key = (sat['rcvTow'], sat['gnss'], sat['svId'], sat['sigID'])
                        self.rawx_data[key] = sat
                elif (msg_class, msg_id) == (0x01, 0x35):
                    for sat in self._parse_nav_sat(payload):
                        key = (sat['iTOW'], sat['gnss'], sat['svId'])
                        self.navsat_data[key] = sat
                elif (msg_class, msg_id) == (0x01, 0x07):
                    navpvt = self._parse_nav_pvt(payload)
                    self.navpvt_data[navpvt['iTOW']] = navpvt
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
            gnssId, svId, sigID = struct.unpack_from('<BBB', payload, offset + 20)
            cno = struct.unpack_from('<B', payload, offset + 26)[0]
            sats.append({
                'rcvTow': int(rcvTow), 'gnss': self.GNSS_ID_MAP.get(gnssId, '?'),
                'svId': svId, 'prMes': prMes, 'cpMes': cpMes, 'doppler': doMes, 'cno': cno,
                'sigID': self.SIGNAL_MAP.get((gnssId, sigID), f"unknown({sigID})")
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
        return {
            'iTOW': iTOW * 1e-3, 'lat': lat * 1e-7, 'lon': lon * 1e-7,
            'height': height * 1e-3, 'fixType': fixType,
            'utc': f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{int(nano / 1000):06d}'
        }

    def _merge_data(self) -> List[Dict]:
        rows = []
        for key in sorted(self.rawx_data):
            rcvTow, gnss, svId, sigID = key
            rawx = self.rawx_data[key]
            iTOW_key = (rcvTow, gnss, svId)
            measx = self.measx_data.get(iTOW_key, {})
            navsat = self.navsat_data.get(iTOW_key, {})
            pvt = self.navpvt_data.get(rcvTow, {})

            if sigID in ['L1C/A', 'L1OF', 'L1C', 'E1C', 'B1I', 'B1C']:
                codePhase = measx.get('codePhase', '')
            else:
                codePhase = ''

            rows.append({
                'timestamp': rcvTow, 'utc': pvt.get('utc', ''), 'gnssId': gnss,
                'svId': svId, 'sigID': sigID, 'elevation': navsat.get('elev', ''),
                'azimuth': navsat.get('azim', ''), 'carrierPhase': rawx.get('cpMes', ''),
                'pseudorange': rawx.get('prMes', ''), 'doppler': rawx.get('doppler', ''),
                'codePhase': codePhase, 'cno': rawx.get('cno', '')
            })
        return rows


def parse_ubx_directory(
    input_dir: str, output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> ProcessingResult:
    parser = UBXParser()
    result = parser.parse_directory(input_dir, progress_callback)
    if result.success and output_csv and result.data is not None:
        result.data.to_csv(output_csv, index=False)
    return result


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
                     'azimuth', 'carrierPhase', 'pseudorange', 'doppler', 'codePhase', 'cno']
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


def derive_doppler_from_carrier_phase(
    df: pd.DataFrame,
    window_size: int = POLYNOMIAL_WINDOW,
    poly_order: int = 2,
    progress_callback: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Derive Doppler shift from carrier phase measurements.
    
    Method:
        1. Group by satellite and signal
        2. Sort by timestamp
        3. Fit polynomial to carrier phase within sliding window
        4. Doppler = derivative of polynomial at center point
    
    Args:
        df: DataFrame with columns [timestamp, gnssId, svId, sigID, carrierPhase]
        window_size: Number of points for polynomial fitting
        poly_order: Polynomial order (2 recommended)
        progress_callback: Optional progress callback
    
    Returns:
        DataFrame with derived 'doppler' column
    """
    df = df.copy()
    
    # Ensure carrierPhase is numeric
    df['carrierPhase'] = pd.to_numeric(df['carrierPhase'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # Initialize doppler column
    df['doppler'] = np.nan
    df['doppler_derived'] = True  # Flag to indicate derived values
    
    # Group by satellite and signal
    if 'gnssId' in df.columns and 'svId' in df.columns and 'sigID' in df.columns:
        group_cols = ['gnssId', 'svId', 'sigID']
    elif 'sat_id' in df.columns and 'sigID' in df.columns:
        group_cols = ['sat_id', 'sigID']
    else:
        group_cols = ['gnssId', 'svId'] if 'gnssId' in df.columns else []
    
    if not group_cols:
        if progress_callback:
            progress_callback("Warning: Cannot group data for doppler derivation", None)
        return df
    
    groups = df.groupby(group_cols, sort=False)
    total_groups = len(groups)
    
    for i, (group_key, group_df) in enumerate(groups):
        if progress_callback and i % 50 == 0:
            progress_callback(f"Deriving Doppler: {i}/{total_groups} groups", i / total_groups * 0.1)
        
        # Sort by timestamp
        group_df = group_df.sort_values('timestamp')
        indices = group_df.index.tolist()
        
        if len(indices) < 3:
            continue
        
        timestamps = group_df['timestamp'].values  # milliseconds
        phases = group_df['carrierPhase'].values   # cycles
        
        # Convert timestamps to seconds
        t_sec = timestamps / 1000.0
        
        # Skip if carrier phase data is mostly invalid
        valid_mask = ~np.isnan(phases)
        if valid_mask.sum() < 3:
            continue
        
        # Interpolate missing carrier phase values for continuity
        if not valid_mask.all():
            phases_interp = np.interp(
                t_sec,
                t_sec[valid_mask],
                phases[valid_mask]
            )
            phases = phases_interp
        
        # Compute Doppler using polynomial derivative
        doppler_values = _compute_doppler_polynomial(
            t_sec, phases, window_size, poly_order
        )
        
        # Assign back to dataframe
        df.loc[indices, 'doppler'] = doppler_values
    
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
    
    # Derive Doppler
    df_processed = derive_doppler_from_carrier_phase(df, progress_callback=progress_callback)
    
    # Count derived values
    derived_count = df_processed['doppler'].notna().sum()
    
    if progress_callback:
        progress_callback(f"Derived {derived_count} Doppler values from carrier phase", 0.1)
    
    result_status = {
        'doppler_source': 'derived',
        'original_valid_ratio': 1.0 - status['missing_ratio'],
        'derived_count': int(derived_count)
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
    
    # === DOPPLER FALLBACK: Ensure Doppler data exists ===
    if result.data is not None:
        df, doppler_status = ensure_doppler_data(result.data, progress_callback)
        result.data = df
        result.metadata['doppler_source'] = doppler_status['doppler_source']
        result.metadata['doppler_derived_count'] = doppler_status.get('derived_count', 0)
        
        if doppler_status['doppler_source'] == 'derived':
            result.message += f" | Doppler derived from carrier phase ({doppler_status['derived_count']} values)"
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


@dataclass
class ProcessingResult:
    success: bool
    data: Optional[pd.DataFrame]
    message: str
    metadata: Dict


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
        target_time_gps = target_time_utc + timedelta(seconds=GPS_LEAP_SECONDS)

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


def match_observations_with_sp3(
    obs_csv: str,
    sp3_file: str,
    output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    batch_size: int = 2000
) -> ProcessingResult:
    """
    Match observations from CSV with SP3 satellite positions
    
    Args:
        obs_csv: Path to observations CSV file (UTC timestamps)
        sp3_file: Path to SP3 file (GPS time)
        output_csv: Optional path for matched output
        progress_callback: Optional callback(message, fraction)
        batch_size: Progress update interval
    
    Returns:
        ProcessingResult with matched data
    """
    # Load observations
    df = pd.read_csv(obs_csv)
    

    # Parse UTC timestamps
    df['parsed_utc'] = pd.to_datetime(df['utc'], format='mixed', errors='coerce')
    df['parsed_utc'] = df['parsed_utc'].dt.tz_localize(None)


    # === TEMP DEBUG: Write next to the input CSV ===
    import os
    debug_path = obs_csv.replace('.csv', '_DEBUG.txt')
    with open(debug_path, 'w') as f:
        f.write(f"Rows before dropna: {len(df)}\n")
        f.write(f"Rows with valid parsed_utc: {df['parsed_utc'].notna().sum()}\n")
    # === END DEBUG ===


    
    df = df.dropna(subset=['parsed_utc'])
    

    # Create satellite identifier
    df['sat_identifier'] = df['gnssId'].astype(str) + ' ' + df['svId'].astype(str)

    # Initialize SP3 parser
    sp3 = SP3Parser(sp3_file)

    # Add SP3 columns
    sp3_cols = [
        'gps_time_used',
        'interp_x', 'interp_y', 'interp_z',
        'interp_vel_x', 'interp_vel_y', 'interp_vel_z',
        'interp_speed', 'interp_clk', 'interp_clk_rate',
        'sp3_match_status'
    ]
    for col in sp3_cols:
        df[col] = np.nan
    df['sp3_match_status'] = 'no_match'

    matched_count = 0
    total_count = len(df)

    # Match each observation
    for idx, row in df.iterrows():
        sp3_data = sp3.interpolate(row['sat_identifier'], row['parsed_utc'])
        
        if sp3_data:
            for key, value in sp3_data.items():
                if key in df.columns:
                    df.at[idx, key] = value
            df.at[idx, 'sp3_match_status'] = 'matched'
            matched_count += 1

        if progress_callback and (idx + 1) % batch_size == 0:
            frac = 0.1 + 0.25 * ((idx + 1) / total_count)
            progress_callback(f"SP3 matching: {idx + 1:,}/{total_count:,}", frac)


    # Extract matched rows
    df_matched = df[df['sp3_match_status'] == 'matched'].copy()
    
    # Save if requested
    if output_csv and not df_matched.empty:
        df_matched.to_csv(output_csv, index=False)

    return ProcessingResult(
        success=True,
        data=df_matched,
        message=f"Matched {matched_count}/{total_count} observations",
        metadata={'total': total_count, 'matched': matched_count}
    )

# ============================================================================
# STEP 3A & 3B: ELEVATION AND DOPPLER
# ============================================================================


def calculate_accurate_elevations(
    input_csv: str, 
    station: StationConfig, 
    output_csv: Optional[str] = None
) -> ProcessingResult:
    """Calculate accurate elevation angles from SP3 satellite positions"""
    
    df = pd.read_csv(input_csv)
    
    # Convert elevation to numeric, coercing empty/invalid to NaN
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    
    # Only filter if elevation data exists; otherwise keep all rows
    # (elevation will be computed from SP3 positions)
    if df['elevation'].notna().any():
        df = df[df['elevation'] >= -5]
    
    if df.empty:
        return ProcessingResult(
            success=False,
            data=None,
            message="No observations after elevation filter",
            metadata={'filtered_count': 0}  # FIXED: Added metadata
        )
    
    station_xyz = station.to_ecef()
    
    elevations = []
    failed_count = 0
    
    for _, r in df.iterrows():
        try:
            sat_xyz = np.array([r['interp_x'], r['interp_y'], r['interp_z']])
            elev = calculate_elevation_angle(sat_xyz, station_xyz)
            elevations.append(elev)
        except (KeyError, ValueError):
            elevations.append(np.nan)
            failed_count += 1
    
    df['accurate_elevation'] = elevations
    
    # Count before filtering
    before_filter = len(df)
    
    # Now filter on computed accurate_elevation
    df = df[df['accurate_elevation'] >= -5]
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    return ProcessingResult(
        success=True,
        data=df,
        message=f"Calculated elevations for {len(df)} observations",
        metadata={  # FIXED: Added metadata
            'total_processed': before_filter,
            'passed_filter': len(df),
            'failed_calculations': failed_count
        }
    )


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
    
    station_ecef = station.to_ecef()
    
    # Time delay: use pseudorange when available; fall back to geometric range.
    # RINEX carrier-only rows have NaN pseudorange; using geometric range
    # introduces < 0.01 Hz error in geometric Doppler — well within thresholds.
    station_ecef_tmp = station.to_ecef()
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
        sat_vel_ecef[:, 0] * station_ecef[1] - sat_vel_ecef[:, 1] * station_ecef[0]
    ) / SPEED_OF_LIGHT
    
    # ROBUST FREQUENCY LOOKUP with fallback
    gnss_col = 'gnssId' if 'gnssId' in df.columns else None
    
    carrier_freq = np.zeros(len(df))
    freq_missing_count = 0
    
    for i, row in df.iterrows():
        sig_id = row['sigID']
        gnss_id = row[gnss_col] if gnss_col else None
        
        freq = get_signal_frequency(sig_id, gnss_id)
        
        if np.isnan(freq):
            freq_missing_count += 1
            # Last resort: use L1 frequency
            freq = 1575.420e6
        
        idx = df.index.get_loc(i)
        carrier_freq[idx] = freq
    
    # Clock drift contribution
    clk_rate = df['interp_clk_rate'].values if 'interp_clk_rate' in df.columns else 0
    clock_doppler = carrier_freq * clk_rate
    
    # Calculate geometric Doppler
    df['geometric_doppler'] = -carrier_freq * (range_rate + sagnac_rate) / SPEED_OF_LIGHT + clock_doppler
    df['carrier_freq_hz'] = carrier_freq  # Store for debugging
    
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
    fresnel_window_sec: float = POLYNOMIAL_WINDOW,
    reference_elevation_threshold: float = 50.0,
    min_reference_epochs: int = 100
) -> ProcessingResult:
    """
    Apply single differencing with robust reference satellite selection.
    
    Strategy:
    1. Identify candidate reference satellites (high elevation throughout)
    2. Score candidates by stability and coverage
    3. Use best candidate as primary reference
    4. Fall back to elevation-weighted average when primary unavailable
    """
    df = pd.read_csv(input_csv)
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    
    df['excess_doppler'] = df['doppler'] - df['geometric_doppler']
    df['sat_id'] = df['gnssId'] + '_' + df['svId'].astype(str)
    
    # Process each constellation+signal combination separately
    df['atmos_doppler'] = np.nan
    df['reference_sat'] = ''
    df['reference_type'] = ''  # 'primary' or 'weighted_avg'
    
    for (gnss_id, sig_id), sig_group in df.groupby(['gnssId', 'sigID']):
        sig_indices = sig_group.index
        
        # Find primary reference satellite for this constellation+signal
        primary_ref = _select_primary_reference(
            sig_group, 
            elev_col, 
            reference_elevation_threshold,
            min_reference_epochs
        )
        
        # Get all epochs for this signal
        epochs = sig_group['utc'].unique()
        
        for epoch in epochs:
            epoch_mask = (df.index.isin(sig_indices)) & (df['utc'] == epoch)
            epoch_data = df.loc[epoch_mask]
            
            if epoch_data.empty:
                continue
            
            ref_doppler = None
            ref_sat = None
            ref_type = None
            
            # Try primary reference first
            if primary_ref is not None:
                primary_data = epoch_data[epoch_data['sat_id'] == primary_ref]
                if not primary_data.empty and primary_data[elev_col].iloc[0] >= reference_elevation_threshold:
                    ref_doppler = primary_data['excess_doppler'].iloc[0]
                    ref_sat = primary_ref
                    ref_type = 'primary'
            
            # Fallback: elevation-weighted average
            if ref_doppler is None:
                ref_doppler, ref_sat = _compute_weighted_reference(
                    epoch_data, 
                    elev_col, 
                    reference_elevation_threshold
                )
                if ref_doppler is not None:
                    ref_type = 'weighted_avg'
            
            # Last resort: highest elevation satellite
            if ref_doppler is None:
                highest_idx = epoch_data[elev_col].idxmax()
                ref_doppler = epoch_data.loc[highest_idx, 'excess_doppler']
                ref_sat = epoch_data.loc[highest_idx, 'sat_id']
                ref_type = 'highest_elev'
            
            # Apply differencing
            df.loc[epoch_mask, 'atmos_doppler'] = epoch_data['excess_doppler'] - ref_doppler
            df.loc[epoch_mask, 'reference_sat'] = ref_sat
            df.loc[epoch_mask, 'reference_type'] = ref_type
    
    # Mask high elevation satellites (they are reference, not targets)
    df.loc[df[elev_col] >= config.elevation_mask_high, 'atmos_doppler'] = np.nan
    
    # Apply polynomial smoothing
    df = apply_fresnel_polynomial_smoothing(df, fresnel_window_sec)
    
    # Summary statistics
    ref_stats = df.groupby('reference_type').size().to_dict()
    primary_refs = df[df['reference_type'] == 'primary']['reference_sat'].unique()
    
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
        n_jumps = (excess_diff > 2.0).sum()  # >2 Hz jump = suspect
        
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
    
def apply_fresnel_polynomial_smoothing(df: pd.DataFrame, fresnel_window_sec: float = None) -> pd.DataFrame:
    """
    Apply 2nd-order polynomial fit to atmos_doppler over Fresnel time window.
    Takes value at center of window. Respects gaps >= POLYFIT_GAP_THRESHOLD seconds.

    v3.4.4: window and gap threshold are read live from module-level constants
    so .cra overrides take effect. Pass fresnel_window_sec to override per-call.
    """
    df = df.copy()
    df['atmos_dopp_poli'] = np.nan
    
    if 'atmos_doppler' not in df.columns or 'timestamp' not in df.columns:
        return df
    
    # Read live so .cra-overridden values are respected.
    if fresnel_window_sec is None:
        fresnel_window_sec = POLYNOMIAL_WINDOW
    gap_threshold = POLYFIT_GAP_THRESHOLD
    half_window = fresnel_window_sec / 2.0
    
    for (sat_id, sig_id), group in df.groupby(['sat_id', 'sigID'] if 'sat_id' in df.columns else ['gnssId', 'svId', 'sigID']):
        group = group.sort_values('timestamp').copy()
        timestamps = group['timestamp'].values
        atmos_vals = group['atmos_doppler'].values
        indices = group.index.values
        
        # Detect gap boundaries (>=5 sec)
        time_diffs = np.diff(timestamps)
        gap_indices = np.where(time_diffs >= gap_threshold)[0]
        
        # Create segments between gaps
        segment_starts = np.concatenate([[0], gap_indices + 1])
        segment_ends = np.concatenate([gap_indices + 1, [len(timestamps)]])
        
        for seg_start, seg_end in zip(segment_starts, segment_ends):
            seg_timestamps = timestamps[seg_start:seg_end]
            seg_atmos = atmos_vals[seg_start:seg_end]
            seg_indices = indices[seg_start:seg_end]
            
            for i, (t_center, idx) in enumerate(zip(seg_timestamps, seg_indices)):
                if np.isnan(seg_atmos[i]):
                    continue
                
                # Find points within Fresnel window
                mask = (seg_timestamps >= t_center - half_window) & (seg_timestamps <= t_center + half_window)
                t_window = seg_timestamps[mask]
                v_window = seg_atmos[mask]
                
                # Remove NaN
                valid = ~np.isnan(v_window)
                t_valid = t_window[valid]
                v_valid = v_window[valid]
                
                if len(v_valid) < 2:
                    df.loc[idx, 'atmos_dopp_poli'] = seg_atmos[i]
                    continue
                
                try:
                    t_rel = t_valid - t_center
                    coeffs = np.polyfit(t_rel, v_valid, 1)
                    df.loc[idx, 'atmos_dopp_poli'] = np.polyval(coeffs, 0)
                except:
                    df.loc[idx, 'atmos_dopp_poli'] = seg_atmos[i]
    
    return df

# ============================================================================
# STEP 5: BENDING ANGLE RETRIEVAL
# ============================================================================


class BendingAngleRetriever:
    """
    Retrieve bending angles from atmospheric Doppler.
    All calculations in ECEF frame. Receiver is stationary.
    """
    
    def __init__(self, station: StationConfig):
        self.station = station
        self.r_rec_ecef = station.to_ecef()
        self.R_local = station.get_gaussian_radius()
        self.c = SPEED_OF_LIGHT

    def _get_height_range_max(self, config: 'PipelineConfig') -> float:
        """
        Compute the maximum physically meaningful impact height (km).
        
        From Bouguer's law, the maximum impact parameter for a ground-based 
        receiver is a_max = n_r * r_r (when the ray is horizontal). The 
        corresponding impact height is:
        
            h_max = a_max - R_c = (n_r - 1)*R_c + n_r*h_station
        
        For a sea-level station this is ~2 km; higher stations get slightly
        less excess because n_r decreases with altitude.
        
        A 1 km margin is added for measurement noise and numerical tolerance.
        """
        if config.height_range_max > 0:
            return config.height_range_max  # user override
        
        h_station = self.station.altitude  # meters
        R_c = self.R_local                 # meters
        
        # Refractive index at station
        N_station = self.station.get_surface_refractivity()
        n_r = 1.0 + N_station * 1e-6
        
        # Bouguer upper bound + 1 km margin
        r_r = R_c + h_station
        h_max_m = n_r * r_r - R_c + 1000.0  # 1 km margin
        return h_max_m / 1000.0  # convert to km

    @staticmethod
    def _ecef_to_geodetic_lat(x, y, z):
        """Fast ECEF to geodetic latitude only (degrees). Bowring method."""
        a = WGS84_A
        b = a * (1 - 1 / 298.257223563)
        e2 = WGS84_E2
        ep2 = (a / b) ** 2 - 1
        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * a, p * b)
        lat = np.arctan2(
            z + ep2 * b * np.sin(theta)**3,
            p - e2 * a * np.cos(theta)**3
        )
        return np.degrees(lat)

    @staticmethod
    def _gaussian_radius_at_lat(lat_deg):
        """
        Compute Gaussian mean radius of curvature at a given geodetic latitude.
        R = sqrt(M * N) where M = meridional, N = normal radius of curvature.
        """
        lat_r = np.radians(lat_deg)
        sin_lat = np.sin(lat_r)
        denom = 1 - WGS84_E2 * sin_lat**2
        M = WGS84_A * (1 - WGS84_E2) / denom**1.5
        N = WGS84_A / np.sqrt(denom)
        return np.sqrt(M * N)

    @staticmethod
    def _impact_param_to_tangent_height(a_m, R_c_m, N_surface=315.0, H_scale=7000.0,
                                         tol=0.01, max_iter=20):
        """
        Solve for geometric tangent point height from impact parameter.
        
        At the tangent point, Bouguer's law gives (Hajj Eq. 14b, sinφ=1):
            a = n(r_tp) · r_tp
        
        Given a and R_c, solve for t where r_tp = R_c + t and
        n(r) = 1 + N_surface·10⁻⁶ · exp(-t / H_scale).
        
        Uses fixed-point iteration: t_{k+1} = a / n(R_c + t_k) - R_c
        Converges in 3-5 iterations to sub-cm accuracy.
        
        Args:
            a_m: Impact parameter (m), scalar or array
            R_c_m: Local radius of curvature (m), scalar or array
            N_surface: Surface refractivity in N-units (default 315)
            H_scale: Atmospheric scale height in meters (default 7000)
            tol: Convergence tolerance in meters
            max_iter: Maximum iterations
            
        Returns:
            Geometric tangent point height in meters
        """
        a_m = np.asarray(a_m, dtype=float)
        R_c_m = np.asarray(R_c_m, dtype=float)
        
        # Initial guess: vacuum approximation (n=1)
        t = a_m - R_c_m
        
        for _ in range(max_iter):
            # Clamp t for exp stability (below surface is fine physically)
            t_safe = np.clip(t, -50000.0, 200000.0)
            n = 1.0 + N_surface * 1e-6 * np.exp(-t_safe / H_scale)
            t_new = a_m / n - R_c_m
            if np.all(np.abs(t_new - t) < tol):
                break
            t = t_new
        
        return t_new

    def _estimate_tangent_point_radius(self, r_sat_ecef):
        """
        Estimate the local Earth radius of curvature at the ray's tangent point.
        
        For ground-based RO, the tangent point (where the ray is closest to Earth)
        is always near the station — unlike space-based RO where it can be thousands
        of km from both endpoints. The ray bends down from the satellite and arrives
        at the ground station; the lowest point of the ray is between the station 
        and the satellite but displaced by only a few tens of km from the station.
        
        At the station's horizon distance (~25 km for h=50m), the latitude change 
        is only ~0.2°, giving negligible R change. Therefore, for ground-based RO,
        the station's local radius of curvature is the appropriate R_c.
        
        For better accuracy, we compute R at a latitude slightly displaced toward 
        the satellite's sub-point, weighted by the ratio of station altitude to 
        satellite distance (which determines how far the tangent point is from 
        the station).
        
        Reference: Hajj et al. 2002, Section 4.2 (adapted for ground-based geometry)
        """
        r_r = self.r_rec_ecef  # receiver ECEF
        r_t = r_sat_ecef       # transmitter ECEF
        
        r_r_mag = np.linalg.norm(r_r)
        r_t_mag = np.linalg.norm(r_t)
        
        if r_r_mag < 1e-6 or r_t_mag < 1e-6:
            return self.R_local  # fallback
        
        # For ground-based RO, the tangent point is displaced from the station
        # toward the satellite's sub-point by a small amount. The displacement
        # depends on the elevation angle: at horizon (~0°), it's at the geometric
        # horizon distance; at higher elevations, it's essentially at the station.
        #
        # A first-order estimate: interpolate latitude toward satellite sub-point
        # with a small weight proportional to (horizon_distance / sat_distance).
        # For h_station=50m: horizon ≈ 25km, sat_distance ≈ 25000km → weight ≈ 0.001
        # This makes the correction negligible in practice.
        
        r_rec_dir = r_r / r_r_mag
        r_sat_dir = r_t / r_t_mag
        
        # Compute elevation angle to determine how close tangent point is to station
        los = r_t - r_r
        los_mag = np.linalg.norm(los)
        if los_mag < 1e-6:
            return self.R_local
        los_unit = los / los_mag
        up = r_rec_dir
        sin_elev = np.dot(los_unit, up)
        
        # For low elevation (RO regime), tangent point is slightly displaced
        # toward horizon. Weight based on geometric horizon fraction.
        horizon_dist = np.sqrt(2 * r_r_mag * self.station.altitude) if self.station.altitude > 0 else 0
        weight = min(horizon_dist / los_mag, 0.01)  # tiny correction
        
        # Interpolate direction toward satellite sub-point
        tp_dir = (1 - weight) * r_rec_dir + weight * r_sat_dir
        tp_norm = np.linalg.norm(tp_dir)
        if tp_norm < 1e-10:
            return self.R_local
        tp_dir = tp_dir / tp_norm
        
        # Get geodetic latitude at tangent point
        tp_lat = self._ecef_to_geodetic_lat(
            tp_dir[0] * WGS84_A, tp_dir[1] * WGS84_A, tp_dir[2] * WGS84_A
        )
        
        return self._gaussian_radius_at_lat(tp_lat)

    def solve_single_freq(
        self, 
        doppler: float, 
        v_sat_ecef: np.ndarray, 
        r_sat_ecef: np.ndarray, 
        wavelength: float
    ) -> Tuple[float, float]:
        """
        Solve for bending angle and impact parameter from atmospheric Doppler.
        
        In ECEF frame:
        - Receiver is stationary (v_rec = 0)
        - Atmospheric Doppler arises from ray bending changing the effective
          direction of signal propagation at the satellite
        
        For ground-based RO, the receiver is inside the atmosphere, so 
        Bouguer's law requires n_r at the receiver:
            a = r_t * n_t * sin(θ_t + δ_t) = r_r * n_r * sin(θ_r + δ_r)
        where n_t ≈ 1 at GPS altitude but n_r ≈ 1.0003 at the ground.
        (Hajj et al. 2002, Eq. 14b)
        
        Args:
            doppler: Atmospheric Doppler (Hz) - geometric already removed
            v_sat_ecef: Satellite velocity in ECEF (m/s)
            r_sat_ecef: Satellite position in ECEF (m)
            wavelength: Signal wavelength (m)
        
        Returns:
            (bending_angle, impact_parameter) or (nan, nan) on failure
        """
        r_t = r_sat_ecef
        v_t = v_sat_ecef
        r_r = self.r_rec_ecef
        
        # Refractive index at receiver (ground station, inside atmosphere)
        # Uses actual surface met data if available, otherwise standard atmosphere
        N_station = self.station.get_surface_refractivity()  # N-units at station level
        n_receiver = 1.0 + N_station * 1e-6
        
        # At GPS altitude (~20,200 km), n_transmitter ≈ 1 (vacuum)
        n_transmitter = 1.0
        
        # Straight-line direction (no bending)
        L_vec = r_r - r_t
        dist = np.linalg.norm(L_vec)
        k0 = L_vec / dist
        
        # Occultation plane normal
        normal = np.cross(r_t, r_r)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-6:
            return np.nan, np.nan
        axis = normal / norm_mag

        def rotate_vec(vec, k, theta):
            """Rodrigues rotation formula"""
            return (vec * np.cos(theta) + 
                    np.cross(k, vec) * np.sin(theta) + 
                    k * np.dot(k, vec) * (1 - np.cos(theta)))

        def equations(vars):
            dt, dr = vars  # bending at transmitter, bending at receiver
            
            # Bent ray directions
            k_t = rotate_vec(k0, axis, -dt)  # ray leaving satellite
            k_r = rotate_vec(k0, axis, dr)   # ray arriving at receiver
            
            # Doppler equation (ECEF, receiver stationary so v_r term = 0)
            # Atmospheric Doppler = (1/λ) * v_t · (k_t - k0)
            doppler_model = np.dot(v_t, k_t - k0) / wavelength
            eq1 = doppler - doppler_model
            
            # Bouguer's law: a = n*r*sin(ψ) = n * |r × k̂|
            # At transmitter: n_t ≈ 1, so a_t = |r_t × k̂_t|
            # At receiver: n_r ≈ 1.0003, so a_r = n_r * |r_r × k̂_r|
            a_t = n_transmitter * np.linalg.norm(np.cross(r_t, k_t))
            a_r = n_receiver * np.linalg.norm(np.cross(r_r, k_r))
            eq2 = a_t - a_r
            
            return [eq1, eq2]

        try:
            result = fsolve(equations, [1e-5, 1e-4], full_output=True)
            dt_sol, dr_sol = result[0]
            info_dict = result[1]   # dict with 'fvec', 'fjac', etc.
            ier = result[2]          # integer flag: 1 = solution found
            mesg = result[3]         # string message
            
            # Check convergence: ier==1 means solution found
            if ier != 1:
                return np.nan, np.nan
            
            # Check residual magnitude
            residual = np.linalg.norm(info_dict['fvec'])
            if residual > 1e-3:
                return np.nan, np.nan
            
            # Compute total bending angle.
            # For ground-based RO, the rotation convention means individual 
            # components dt_sol and dr_sol can have either sign. The total 
            # deflection angle is the sum of absolute values.
            # Physical sanity: atmospheric bending at GPS frequencies is
            # ~0.02 rad at surface, up to ~0.05 rad in extreme humidity.
            bending_angle = abs(dt_sol) + abs(dr_sol)
            
            if bending_angle > 0.1 or bending_angle < 1e-10:
                return np.nan, np.nan  # unphysical
            
            # Impact parameter: use RECEIVER side for ground-based RO.
            # The transmitter side (r_t ~ 26,500 km) amplifies tiny angular
            # errors in k_t into large impact parameter errors. The receiver
            # side (r_r ~ 6,371 km) is 4× less sensitive to angular errors.
            k_r_sol = rotate_vec(k0, axis, dr_sol)
            impact_parameter = n_receiver * np.linalg.norm(np.cross(r_r, k_r_sol))
            
            # Bouguer upper bound: a ≤ n_r * r_r (equality when ray is horizontal).
            # Values exceeding this are unphysical for ground-based geometry.
            a_max_physical = n_receiver * np.linalg.norm(r_r)
            if impact_parameter > a_max_physical * 1.001:  # 0.1% tolerance
                return np.nan, np.nan
            
            return bending_angle, impact_parameter
        except Exception:
            return np.nan, np.nan

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
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Filter to RO satellites only
        ro_status = evaluate_ro_status(df)
        ro_sats = [sat for sat, is_ro in ro_status.items() if is_ro]
        
        if not ro_sats:
            return ProcessingResult(success=True, data=pd.DataFrame(), 
                                   message="No RO satellites found")
        
        df = df[df['sat_id'].isin(ro_sats)]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        generated_files, summary_stats = [], []
        sat_groups = list(df.groupby('sat_id'))
        total_sats = len(sat_groups)
        
        for idx, (sat_id, sat_data) in enumerate(sat_groups):
            if progress_callback:
                progress_callback(f"Bending angles: {sat_id} ({idx+1}/{total_sats})", 
                                0.55 + 0.25 * ((idx + 1) / total_sats))
            
            gnss_id = sat_data['gnssId'].iloc[0]
            if gnss_id not in FREQ_PAIRS:
                continue
            s1, s2 = FREQ_PAIRS[gnss_id]
            df1 = sat_data[sat_data['sigID'] == s1].set_index('timestamp').sort_index()
            df2 = sat_data[sat_data['sigID'] == s2].set_index('timestamp').sort_index()
            merged = df1.join(df2, lsuffix='_f1', rsuffix='_f2', how='inner')
            if len(merged) < config.min_epochs_for_bending:
                continue

            f1, f2 = SIGNAL_FREQUENCIES[s1], SIGNAL_FREQUENCIES[s2]
            lam1, lam2 = self.c / f1, self.c / f2
            res_f1, res_f2 = {'utc': [], 'a': [], 'alpha': [], 'r_sat_x': [], 'r_sat_y': [], 'r_sat_z': []}, {'utc': [], 'a': [], 'alpha': []}

            for t, row in merged.iterrows():
                # Satellite state (ECEF)
                r_sat = np.array([row['interp_x_f1'], row['interp_y_f1'], row['interp_z_f1']])
                v_sat = np.array([row['interp_vel_x_f1'], row['interp_vel_y_f1'], row['interp_vel_z_f1']])
                
                # Use polynomial-fitted atmos_doppler if available
                dopp_f1 = row.get('atmos_dopp_poli_f1', np.nan)
                if np.isnan(dopp_f1):
                    dopp_f1 = row['atmos_doppler_f1']
                dopp_f2 = row.get('atmos_dopp_poli_f2', np.nan)
                if np.isnan(dopp_f2):
                    dopp_f2 = row['atmos_doppler_f2']
                
                # Solve for each frequency
                alpha1, a1 = self.solve_single_freq(dopp_f1, v_sat, r_sat, lam1)
                alpha2, a2 = self.solve_single_freq(dopp_f2, v_sat, r_sat, lam2)

                if not np.isnan(a1) and alpha1 > 0:
                    res_f1['utc'].append(row['utc_f1'])
                    res_f1['a'].append(a1)
                    res_f1['alpha'].append(alpha1)
                    res_f1['r_sat_x'].append(r_sat[0])
                    res_f1['r_sat_y'].append(r_sat[1])
                    res_f1['r_sat_z'].append(r_sat[2])
                if not np.isnan(a2) and alpha2 > 0:
                    res_f2['utc'].append(row['utc_f2'])
                    res_f2['a'].append(a2)
                    res_f2['alpha'].append(alpha2)
                    
                if progress_callback and hasattr(progress_callback, '__self__'):
                    if getattr(progress_callback.__self__, '_stopped', False):
                        return ProcessingResult(False, message="Cancelled")

            d1, d2 = pd.DataFrame(res_f1).sort_values('a'), pd.DataFrame(res_f2).sort_values('a')
            if d1.empty or d2.empty:
                continue

            try:
                # Ionospheric correction via dual-frequency combination
                # Use smooth-cubic (piecewise Hermite) interpolation as described
                # in Hajj et al. 2002, §4.3: "This 'smooth-cubic' interpolation 
                # scheme avoids introducing sharp variations between the points 
                # when the data is noisy."
                #
                # Conditions: f(t_j) = f_j and f'(t_j) = (f_{j+1}-f_{j-1})/(t_{j+1}-t_{j-1})
                # This is equivalent to a Catmull-Rom / cardinal spline.
                
                # Restrict interpolation to the overlap range (no extrapolation)
                a1_vals = d1['a'].values
                a2_vals = d2['a'].values
                a2_alpha = d2['alpha'].values
                
                a_min_overlap = max(a1_vals.min(), a2_vals.min())
                a_max_overlap = min(a1_vals.max(), a2_vals.max())
                
                # Only process points within overlap range
                overlap_mask = (a1_vals >= a_min_overlap) & (a1_vals <= a_max_overlap)
                if overlap_mask.sum() < config.min_epochs_for_bending:
                    continue
                
                # Compute smooth-cubic (Hermite) interpolation
                # Estimate derivatives at each knot as central differences
                n_knots = len(a2_vals)
                slopes = np.zeros(n_knots)
                if n_knots >= 2:
                    # Central differences for interior points
                    for ki in range(1, n_knots - 1):
                        slopes[ki] = (a2_alpha[ki+1] - a2_alpha[ki-1]) / (a2_vals[ki+1] - a2_vals[ki-1])
                    # Forward/backward differences at endpoints
                    slopes[0] = (a2_alpha[1] - a2_alpha[0]) / (a2_vals[1] - a2_vals[0]) if n_knots > 1 else 0
                    slopes[-1] = (a2_alpha[-1] - a2_alpha[-2]) / (a2_vals[-1] - a2_vals[-2]) if n_knots > 1 else 0
                
                # Build piecewise cubic Hermite interpolation (PCHIP-like)
                # scipy's CubicHermiteSpline matches Hajj's description exactly
                from scipy.interpolate import CubicHermiteSpline
                hermite_interp = CubicHermiteSpline(a2_vals, a2_alpha, slopes)
                
                # Clamp to overlap range (no extrapolation)
                a1_clamped = np.clip(a1_vals, a_min_overlap, a_max_overlap)
                alpha2_interp = hermite_interp(a1_clamped)
                
                # Apply overlap mask: set out-of-range points to NaN
                alpha2_full = np.full_like(a1_vals, np.nan)
                alpha2_full[overlap_mask] = alpha2_interp[overlap_mask]


                coeff_1 =  f1**2 / (f1**2 - f2**2)
                coeff_2 =  f2**2 / (f1**2 - f2**2)
                alpha_neut =  coeff_1 * d1['alpha'].values - coeff_2 * alpha2_full
                
                # Filter out points where interpolation was not available
                valid_iono = ~np.isnan(alpha_neut)
                if valid_iono.sum() < config.min_epochs_for_bending:
                    continue
                
                d1_valid = d1[valid_iono].copy()
                alpha_neut_valid = alpha_neut[valid_iono]
                
                # Compute local Earth radius at each tangent point (not station!)
                # The tangent point can be hundreds of km from the station at a 
                # different latitude where R differs by up to ~43 km (Hajj et al. 2002 §4.2)
                R_tp = d1_valid.apply(
                    lambda row: self._estimate_tangent_point_radius(
                        np.array([row['r_sat_x'], row['r_sat_y'], row['r_sat_z']])
                    ), axis=1
                )
                tangent_height = (d1_valid['a'].values - R_tp.values) / 1000.0

                # Compute true geometric tangent point height by inverting
                # Bouguer's law: a = n(r_tp) · r_tp → t = a/n(t) - R_c
                # This differs from impact height by ~2 km near the surface
                # (see Hajj et al. 2002, Eq. 14b)
                # 
                # Use station's actual surface refractivity for the n(t) model.
                # The exponential model is n(t) = 1 + N0·10⁻⁶·exp(-t/H) where
                # N0 is the sea-level value. Given N at station altitude h:
                #   N0 = N_station · exp(h/H)
                N_station = self.station.get_surface_refractivity()
                H_scale = 7000.0
                N_sea_level = N_station * np.exp(self.station.altitude / H_scale)
                
                geom_tangent_m = self._impact_param_to_tangent_height(
                    d1_valid['a'].values, R_tp.values,
                    N_surface=N_sea_level, H_scale=H_scale
                )
                geom_tangent_km = geom_tangent_m / 1000.0

                out_df = pd.DataFrame({
                    'utc': d1_valid['utc'].values, 
                    'impact_parameter_m': d1_valid['a'].values, 
                    'impact_height_km': tangent_height,
                    'tangent_height_km': geom_tangent_km,
                    'local_radius_km': R_tp.values / 1000.0,
                    'bending_angle_rad': alpha_neut_valid, 
                    'bending_angle_deg': np.degrees(alpha_neut_valid),
                    'bending_L1': d1_valid['alpha'].values, 
                    'bending_L2': alpha2_full[valid_iono],
                })
                out_df = out_df[
                    (out_df['impact_height_km'] > config.height_range_min) & 
                    (out_df['impact_height_km'] < self._get_height_range_max(config))
                ]

                if not out_df.empty and output_dir:
                    fname = f"{output_dir}/{sat_id}_bending.csv"
                    out_df.to_csv(fname, index=False)
                    generated_files.append(fname)
                    summary_stats.append({
                        'sat_id': sat_id, 
                        'gnss_system': gnss_id, 
                        'valid_epochs': len(out_df),
                        'min_height_km': out_df['tangent_height_km'].min(), 
                        'max_height_km': out_df['tangent_height_km'].max(),
                        'min_impact_height_km': out_df['impact_height_km'].min(), 
                        'max_impact_height_km': out_df['impact_height_km'].max(),
                        'max_bending_rad': out_df['bending_angle_rad'].max()
                    })
            except Exception:
                continue

        summary_df = pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()
        if output_dir and not summary_df.empty:
            summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
        return ProcessingResult(
            success=True, 
            data=summary_df, 
            message=f"Generated bending angles for {len(summary_stats)} satellites", 
            metadata={'files': generated_files}
        )


def retrieve_bending_angles(
    input_csv: str, 
    station: StationConfig, 
    config: PipelineConfig = PipelineConfig(), 
    output_dir: Optional[str] = None, 
    progress_callback: Optional[Callable] = None
) -> ProcessingResult:
    return BendingAngleRetriever(station).process(input_csv, config, output_dir, progress_callback)

# ============================================================================
# STEP 6: ABEL INVERSION
# ============================================================================


class AbelInversion:
    """
    Abel integral inversion for retrieving atmospheric refractivity from bending angles.
    Implements statistical optimization blending observed and climatological bending angles.
    
    Uses per-level local radius of curvature from the bending angle step (Hajj et al. 2002 §4.2)
    rather than a global mean Earth radius, which avoids latitude-dependent height biases 
    of up to ~14 km.
    """
    
    def __init__(self, climatology_blend_km: float = 50.0):
        """
        Initialize Abel inversion processor.
        
        Args:
            climatology_blend_km: Height above which to blend with climatology (km)
        """
        self.R_earth_mean = R_EARTH  # fallback only
        self.climatology_blend_km = climatology_blend_km
    
    def run(self, bending_csv: str, output_csv: Optional[str] = None) -> ProcessingResult:
        """
        Perform Abel inversion on bending angle profile.
        
        Args:
            bending_csv: Input CSV with bending angles
            output_csv: Optional output path for refractivity profile
            
        Returns:
            ProcessingResult with refractivity data
        """
        # Load and sort by impact parameter
        df = pd.read_csv(bending_csv).sort_values('impact_parameter_m').reset_index(drop=True)
        
        if df.empty:
            return ProcessingResult(
                success=False,
                data=None,
                message="Empty bending angle file",
                metadata={'input_rows': 0}
            )
        
        # Use local radius from bending angle step if available;
        # otherwise fall back to global mean (with warning).
        if 'local_radius_km' in df.columns:
            R_local = df['local_radius_km'].values * 1000.0  # per-level, in meters
            R_reference = np.median(R_local)  # representative value for height estimation
        else:
            R_local = np.full(len(df), self.R_earth_mean)
            R_reference = self.R_earth_mean
        
        # Calculate approximate heights using local radius
        df['approx_height_km'] = (df['impact_parameter_m'].values - R_local) / 1000
        
        # Fit exponential to lower atmosphere for climatological reference
        fit_data = df[df['approx_height_km'] <= 40.0]
        
        if len(fit_data) < 2:
            return ProcessingResult(
                success=False,
                data=None,
                message="Insufficient data for climatology fit (<2 points below 40km)",
                metadata={
                    'input_rows': len(df),
                    'fit_points': len(fit_data),
                    'R_reference_km': R_reference / 1000.0
                }
            )
        
        # Fit log-linear model to get scale height
        coeffs = np.polyfit(
            fit_data['impact_parameter_m'].values,
            np.log(fit_data['bending_angle_rad'].values + 1e-10),
            1
        )
        scale_height = -1.0 / coeffs[0] if coeffs[0] < 0 else 7000.0
        
        # Reference point for climatology
        mid_idx = len(fit_data) // 2
        a_ref = fit_data['impact_parameter_m'].iloc[mid_idx]
        alpha_ref = fit_data['bending_angle_rad'].iloc[mid_idx]
        
        # Get arrays for processing
        a_values = df['impact_parameter_m'].values
        alpha_meas = df['bending_angle_rad'].values
        
        # Build climatological profile
        alpha_clim = alpha_ref * np.exp(-(a_values - a_ref) / scale_height)
        
        # Statistical optimization weights (Hajj Eq. 21)
        a_upper_mask = df['approx_height_km'] <= self.climatology_blend_km
        if a_upper_mask.any():
            a_upper = df.loc[a_upper_mask, 'impact_parameter_m'].max()
        else:
            a_upper = a_values.min()
        
        # Measurement uncertainty (small, trust measurements)
        sigma_meas = np.full_like(alpha_meas, 1e-7)
        
        # Climatology uncertainty (large below blend height, smaller above)
        sigma_clim = np.full_like(alpha_clim, 1e10)
        sigma_clim[a_values > a_upper] = 0.05 * alpha_clim[a_values > a_upper]
        
        # Compute weights
        w_meas = 1.0 / sigma_meas**2
        w_clim = 1.0 / sigma_clim**2
        
        # Optimally blended bending angle profile
        df['bending_optimized'] = (alpha_meas * w_meas + alpha_clim * w_clim) / (w_meas + w_clim)
        
        # Prepare for Abel integral
        a = df['impact_parameter_m'].values
        alpha = df['bending_optimized'].values
        n_levels = len(a)
        ln_n = np.zeros(n_levels)
        
        # Abel integral computation (Hajj Eqs. 22-23)
        for i in range(n_levels):
            a_i = a[i]
            
            # Integration intermediate limit (a few levels above current)
            a_int_idx = min(i + 3, n_levels - 1)
            a_int = a[a_int_idx]
            
            # Analytical part (integration by parts boundary term, Eq. 23)
            analytical = (
                alpha[a_int_idx] * np.log(a_int + np.sqrt(max(a_int**2 - a_i**2, 1e-20))) -
                alpha[i] * np.log(a_i + 1e-10)
            )
            
            # Parts integral (integration by parts main term)
            parts_integral = sum(
                -np.log((a[j] + a[j+1])/2 + np.sqrt(max(((a[j] + a[j+1])/2)**2 - a_i**2, 1e-20))) * 
                (alpha[j+1] - alpha[j])
                for j in range(i, min(a_int_idx, n_levels-1))
                if (a[j] + a[j+1])/2 > a_i
            )
            
            # Regular integral (above a_int)
            regular_integral = sum(
                ((alpha[j] + alpha[j+1])/2) / 
                np.sqrt(max(((a[j] + a[j+1])/2)**2 - a_i**2, 1e-20)) * 
                (a[j+1] - a[j])
                for j in range(a_int_idx, n_levels-1)
                if ((a[j] + a[j+1])/2)**2 - a_i**2 > 0
            )
            
            # Combined Abel integral (Eq. 22)
            ln_n[i] = (1.0 / np.pi) * (analytical + parts_integral + regular_integral)
        
        # Convert to refractive index
        df['refractive_index'] = np.exp(ln_n)
        
        # Calculate geometric height using per-level local radius
        # r_tangent = a / n, then height = r_tangent - R_local
        df['height_km'] = (df['impact_parameter_m'].values / df['refractive_index'].values - R_local) / 1000.0
        
        # Convert to refractivity N-units — standard formula, no fudge factors
        # N = (n - 1) × 10^6  (Hajj et al. 2002, Eq. 24)
        df['refractivity_N'] = (df['refractive_index'] - 1.0) * 1e6
        
        # Prepare output
        result_df = df[[
            'height_km',
            'refractivity_N',
            'impact_parameter_m',
            'bending_optimized'
        ]].copy()
        
        # Save if requested
        if output_csv:
            result_df.to_csv(output_csv, index=False)
        
        return ProcessingResult(
            success=True,
            data=result_df,
            message=f"Retrieved refractivity: {result_df['height_km'].min():.2f}-{result_df['height_km'].max():.2f} km",
            metadata={
                'input_levels': len(df),
                'output_levels': len(result_df),
                'height_range_km': (result_df['height_km'].min(), result_df['height_km'].max()),
                'scale_height_m': scale_height,
                'climatology_blend_km': self.climatology_blend_km,
                'R_reference_km': R_reference / 1000.0,
                'uses_local_radius': 'local_radius_km' in pd.read_csv(bending_csv).columns,
            }
        )


def retrieve_refractivity(
    bending_csv: str,
    output_csv: Optional[str] = None,
    climatology_blend_km: float = 50.0
) -> ProcessingResult:
    """
    Convenience wrapper for Abel inversion.
    
    Args:
        bending_csv: Input CSV with bending angles
        output_csv: Optional output path
        climatology_blend_km: Height for climatology blending (km)
        
    Returns:
        ProcessingResult with refractivity profile
    """
    return AbelInversion(climatology_blend_km).run(bending_csv, output_csv)



# ============================================================================
# STEP 6B: ERA5 COMPARISON
# ============================================================================

def _era5_extract(ds, lat, lon):
    """
    Extract ERA5 T, q, z profiles with automatic fallback.

    Priority:
      1. Spatial interpolation to (lat, lon) — used when the point falls
         inside the ERA5 grid.
      2. Nearest-grid-point selection — used when interp() returns all-NaN
         (point outside or on the edge of the grid).
      3. Full spatial mean — last resort when neither above yields data.

    ERA5 files from Copernicus sometimes cover only a small region; an RO
    session whose station coords fall just outside that region will cause
    xarray.interp() to silently return NaN for every pressure level.  That
    propagates through refractivity calculation and produces an empty CSV
    — the symptom the user sees as "ERA5 not plotted".

    The coordinate mismatch is non-fatal: the ERA5 file is still useful as
    a climatological reference even if it was downloaded for a slightly
    different area.  The log message records which fallback was used.
    """
    fallback_used = 'interp'

    def _time_mean(da):
        """Average over valid_time / time dimension, whichever exists."""
        for dim in ('valid_time', 'time'):
            if dim in da.dims:
                return da.mean(dim=dim)
        return da

    if lat is not None and lon is not None:
        try:
            T = _time_mean(ds['t'].interp(latitude=lat, longitude=lon))
            q = _time_mean(ds['q'].interp(latitude=lat, longitude=lon))
            z = _time_mean(ds['z'].interp(latitude=lat, longitude=lon))
            # Check whether interp produced valid data
            if not (np.isfinite(T.values).any() and
                    np.isfinite(z.values).any()):
                raise ValueError("interp returned all-NaN (point outside grid)")
        except Exception:
            # Fallback 1: nearest grid point
            fallback_used = 'nearest'
            try:
                T = _time_mean(ds['t'].sel(latitude=lat, longitude=lon, method='nearest'))
                q = _time_mean(ds['q'].sel(latitude=lat, longitude=lon, method='nearest'))
                z = _time_mean(ds['z'].sel(latitude=lat, longitude=lon, method='nearest'))
                if not np.isfinite(T.values).any():
                    raise ValueError("nearest also returned all-NaN")
            except Exception:
                # Fallback 2: full spatial mean
                fallback_used = 'spatial_mean'
                T = _time_mean(ds['t'].mean(dim=['latitude', 'longitude']))
                q = _time_mean(ds['q'].mean(dim=['latitude', 'longitude']))
                z = _time_mean(ds['z'].mean(dim=['latitude', 'longitude']))
    else:
        fallback_used = 'spatial_mean'
        T = _time_mean(ds['t'].mean(dim=['latitude', 'longitude']))
        q = _time_mean(ds['q'].mean(dim=['latitude', 'longitude']))
        z = _time_mean(ds['z'].mean(dim=['latitude', 'longitude']))

    return T, q, z, fallback_used


def compare_with_era5(refractivity_csv: str, era5_file: str,
                      lat: Optional[float] = None, lon: Optional[float] = None,
                      output_csv: Optional[str] = None) -> ProcessingResult:
    """
    Compare retrieved refractivity against ERA5.

    v3.4.5 fixes:
    - Automatic fallback when station coords are outside the ERA5 grid.
    - height_km array is explicitly sorted before interp1d.
    - Height filter relaxed from > 0 to > -1 km to retain near-surface data.
    - Robust time-dimension handling (valid_time OR time).
    """
    try:
        import xarray as xr
    except ImportError:
        return ProcessingResult(False, None, "xarray required", {})

    ro_df = pd.read_csv(refractivity_csv)
    if ro_df.empty or 'height_km' not in ro_df.columns:
        return ProcessingResult(False, None, "Empty or invalid refractivity CSV", {})

    h_ro = ro_df['height_km'].values
    N_ro = ro_df['refractivity_N'].values
    # Relaxed filter: keep near-surface data (> -1 km)
    valid = np.isfinite(h_ro) & np.isfinite(N_ro) & (h_ro > -1.0)
    h_ro, N_ro = h_ro[valid], N_ro[valid]

    if len(h_ro) == 0:
        return ProcessingResult(False, None,
                                "No valid RO refractivity points (all NaN or below -1 km)", {})

    try:
        ds = xr.open_dataset(era5_file)
    except Exception as e:
        return ProcessingResult(False, None, f"Cannot open ERA5 file: {e}", {})

    try:
        T, q, z, fallback = _era5_extract(ds, lat, lon)
        P = ds['pressure_level']

        height_km = (z.values / 9.80665) / 1000.0
        e = (q * P) / (0.622 + (1.0 - 0.622) * q)
        N_era5 = 77.6 * (P / T) + 3.73e5 * (e / T ** 2)
    except Exception as e:
        ds.close()
        return ProcessingResult(False, None, f"ERA5 profile extraction failed: {e}", {})
    finally:
        ds.close()

    # Sort by height before interpolation (ERA5 pressure levels may be in
    # any order depending on file origin)
    sort_idx = np.argsort(height_km)
    height_km_s = height_km[sort_idx]
    N_era5_s    = N_era5.values[sort_idx]

    # Remove duplicate heights (can occur at boundaries)
    _, uniq = np.unique(height_km_s, return_index=True)
    height_km_s = height_km_s[uniq]
    N_era5_s    = N_era5_s[uniq]

    if not np.isfinite(N_era5_s).any():
        return ProcessingResult(False, None,
                                f"ERA5 refractivity all-NaN (fallback={fallback})", {})

    N_era5_interp = interp1d(
        height_km_s, N_era5_s,
        kind='linear', bounds_error=False, fill_value=np.nan
    )(h_ro)

    valid2 = np.isfinite(N_era5_interp)
    if not valid2.any():
        return ProcessingResult(False, None,
                                "No overlap between RO height range and ERA5 grid "
                                f"(RO: {h_ro.min():.1f}–{h_ro.max():.1f} km, "
                                f"ERA5: {height_km_s.min():.1f}–{height_km_s.max():.1f} km, "
                                f"fallback={fallback})", {})

    h_c = h_ro[valid2]
    N_ro_c = N_ro[valid2]
    N_era5_c = N_era5_interp[valid2]
    error = N_ro_c - N_era5_c

    comparison_df = pd.DataFrame({
        'height_km': h_c, 'N_RO': N_ro_c,
        'N_ERA5': N_era5_c, 'error': error
    })
    if output_csv:
        comparison_df.to_csv(output_csv, index=False)

    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    corr = float(np.corrcoef(N_ro_c, N_era5_c)[0, 1]) if len(N_ro_c) > 1 else np.nan
    return ProcessingResult(
        success=True, data=comparison_df,
        message=f"ERA5 comparison OK (fallback={fallback}): RMSE={rmse:.4f}, bias={bias:+.4f}",
        metadata={'rmse': rmse, 'bias': bias, 'correlation': corr, 'era5_fallback': fallback}
    )


# ============================================================================
# STEP 7: ATMOSPHERIC RETRIEVAL
# ============================================================================

def retrieve_atmospheric_profile(refractivity_csv: str, era5_file: str,
                                  lat: Optional[float] = None, lon: Optional[float] = None,
                                  output_csv: Optional[str] = None) -> ProcessingResult:
    """
    Retrieve P, Pw, q profiles using ERA5 temperature as a constraint.

    v3.4.5 fixes:
    - Uses _era5_extract() for robust fallback when coords outside ERA5 grid.
    - Explicit sort + dedup of ERA5 height array before interp1d.
    - Robust time-dimension handling.
    - Returns ProcessingResult(False) with descriptive message on failure
      instead of crashing or returning an empty profile.
    """
    try:
        import xarray as xr
    except ImportError:
        return ProcessingResult(False, None, "xarray required", {})

    ro_df = pd.read_csv(refractivity_csv)
    if ro_df.empty or 'height_km' not in ro_df.columns:
        return ProcessingResult(False, None, "Empty or invalid refractivity CSV", {})

    # Keep only finite, non-negative heights for the hydrostatic integration
    valid_mask = np.isfinite(ro_df['height_km'].values) & np.isfinite(ro_df['refractivity_N'].values)
    ro_df = ro_df[valid_mask].copy()
    if ro_df.empty:
        return ProcessingResult(False, None, "No finite refractivity rows", {})

    h_m = ro_df['height_km'].values * 1000.0
    N   = ro_df['refractivity_N'].values

    try:
        ds = xr.open_dataset(era5_file)
    except Exception as e:
        return ProcessingResult(False, None, f"Cannot open ERA5 file: {e}", {})

    try:
        T_era5_da, q_era5_da, z_era5_da, fallback = _era5_extract(ds, lat, lon)
        P_levels = ds['pressure_level'].values
    except Exception as e:
        ds.close()
        return ProcessingResult(False, None, f"ERA5 extraction failed: {e}", {})
    finally:
        ds.close()

    h_era5 = (z_era5_da.values / 9.80665)           # metres
    T_era5_arr = T_era5_da.values
    q_era5_arr = q_era5_da.values

    # Sort + dedup by height
    sort_idx = np.argsort(h_era5)
    h_era5     = h_era5[sort_idx]
    T_era5_arr = T_era5_arr[sort_idx]
    P_era5_arr = P_levels[sort_idx]
    q_era5_arr = q_era5_arr[sort_idx]
    _, uniq = np.unique(h_era5, return_index=True)
    h_era5     = h_era5[uniq]
    T_era5_arr = T_era5_arr[uniq]
    P_era5_arr = P_era5_arr[uniq]
    q_era5_arr = q_era5_arr[uniq]

    if not np.isfinite(T_era5_arr).any():
        return ProcessingResult(False, None,
                                f"ERA5 temperature all-NaN (fallback={fallback})", {})

    T     = interp1d(h_era5, T_era5_arr, bounds_error=False, fill_value='extrapolate')(h_m)
    P_era5 = interp1d(h_era5, P_era5_arr, bounds_error=False, fill_value='extrapolate')(h_m)
    q_era5 = interp1d(h_era5, q_era5_arr, bounds_error=False, fill_value='extrapolate')(h_m)

    n, i_top = len(h_m), int(np.argmax(h_m))
    Pw, P = np.zeros(n), np.zeros(n)
    P[i_top] = P_era5[i_top]

    R, m_dry, m_water = 8.314462, 28.97e-3, 18.015e-3
    for _ in range(15):
        Pw_old = Pw.copy()
        for i in range(i_top - 1, -1, -1):
            dh   = h_m[i] - h_m[i + 1]
            h_mid = 0.5 * (h_m[i] + h_m[i + 1])
            T_mid = 0.5 * (T[i] + T[i + 1])
            g = compute_gravity(h_mid, lat)
            rho = (m_dry * P[i + 1] * 100 +
                   (m_water - m_dry) * 0.5 * (Pw[i] + Pw[i + 1]) * 100) / (R * T_mid)
            P[i] = P[i + 1] - rho * g * dh / 100
        for i in range(n):
            Pw[i] = max(0.0,
                        (T[i] ** 2 / N_COEFF_A2) * (N[i] - N_COEFF_A1 * P[i] / T[i]))
            T_c = T[i] - 273.15
            Pw[i] = min(Pw[i],
                        6.1094 * np.exp(17.625 * T_c / (T_c + 243.04)))
        if np.max(np.abs(Pw - Pw_old)) < 0.01:
            break

    epsilon = m_water / m_dry
    q      = epsilon * Pw / (P - Pw + epsilon * Pw) * 1000
    Pw_era5_out = (q_era5 * P_era5) / (epsilon + (1.0 - epsilon) * q_era5)

    profile = pd.DataFrame({
        'height_km':             h_m / 1000,
        'temperature_K':         T,
        'pressure_hPa':          P,
        'water_vapor_hPa':       Pw,
        'specific_humidity_g_kg': q,
        'refractivity_N':        N,
        'T_era5':                T,
        'P_era5':                P_era5,
        'Pw_era5':               Pw_era5_out,
        'q_era5':                q_era5 * 1000,
    })
    if output_csv:
        profile.to_csv(output_csv, index=False)

    P_rmse = float(np.sqrt(np.mean((P - P_era5) ** 2)))
    return ProcessingResult(
        success=True, data=profile,
        message=f"Atmospheric retrieval OK (ERA5 fallback={fallback})",
        metadata={'P_rmse': P_rmse, 'era5_fallback': fallback}
    )


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def generate_raw_plots(sat_data: pd.DataFrame, sat_id: str, output_path: str, dpi: int = 150) -> bool:
    """Generate 2x2 raw GNSS observation plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from plot_style import apply_plot_fonts
    except ImportError:
        return False

    apply_plot_fonts()   # v3.4.7 — larger tick / axis / legend text

    if sat_data.empty:
        return False

    df = sat_data.copy()
    if 'utc' in df.columns:
        df['utc_parsed'] = pd.to_datetime(df['utc'], errors='coerce')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'GNSS Raw Observations: {sat_id}', fontsize=18, fontweight='bold')
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'

    ax = axes[0, 0]
    if 'atmos_doppler' in df.columns and 'utc_parsed' in df.columns and 'sigID' in df.columns:
        valid = df.dropna(subset=['atmos_doppler', 'utc_parsed'])
        if not valid.empty:
            # Scatter dot colors per frequency (lighter shades)
            dot_colors = {
                'L1C/A': '#EF5350', 'L2CL': '#42A5F5',
                'B1I D1': '#EF5350', 'B2I D1': '#42A5F5',
                'E1C': '#EF5350', 'E5bQ': '#42A5F5',
                'L1OF': '#EF5350', 'L2OF': '#42A5F5',
            }
            # Darker line colors for polynomial interpolation
            line_colors = {
                'L1C/A': '#B71C1C', 'L2CL': '#0D47A1',
                'B1I D1': '#B71C1C', 'B2I D1': '#0D47A1',
                'E1C': '#B71C1C', 'E5bQ': '#0D47A1',
                'L1OF': '#B71C1C', 'L2OF': '#0D47A1',
            }
            for sig in valid['sigID'].unique():
                subset = valid[valid['sigID'] == sig]
                c = dot_colors.get(sig, '#EF5350')
                ax.scatter(subset['utc_parsed'], subset['atmos_doppler'], s=3, alpha=0.6, c=c, label=f'{sig}')
            
            # Overlay polynomial fit (frequency-dependent darker color, respecting gaps)
            if 'atmos_dopp_poli' in df.columns and 'timestamp' in df.columns:
                gap_threshold = 5.0
                for sig in valid['sigID'].unique():
                    sig_data = df[df['sigID'] == sig].dropna(subset=['atmos_dopp_poli', 'utc_parsed']).sort_values('timestamp')
                    if sig_data.empty:
                        continue
                    
                    lc = line_colors.get(sig, '#B71C1C')
                    timestamps = sig_data['timestamp'].values
                    time_diffs = np.diff(timestamps)
                    gap_indices = np.where(time_diffs >= gap_threshold)[0]
                    
                    segment_starts = np.concatenate([[0], gap_indices + 1])
                    segment_ends = np.concatenate([gap_indices + 1, [len(timestamps)]])
                    
                    first_seg = True
                    for seg_start, seg_end in zip(segment_starts, segment_ends):
                        seg_data = sig_data.iloc[seg_start:seg_end]
                        if len(seg_data) < 2:
                            continue
                        lbl = f'{sig} fit' if first_seg else None
                        ax.plot(seg_data['utc_parsed'], seg_data['atmos_dopp_poli'], 
                                color=lc, linewidth=2, alpha=0.85, label=lbl)
                        first_seg = False
            
            ax.axhline(y=RO_DOPPLER_THRESHOLD, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=-RO_DOPPLER_THRESHOLD, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.legend(fontsize=9, markerscale=2, loc='best')
    ax.set_xlabel('UTC Time'); ax.set_ylabel('Atmospheric Doppler (Hz)')
    ax.set_title('(a) Atmospheric Doppler', fontsize=15, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if 'cno' in df.columns and 'accurate_elevation' in df.columns and 'sigID' in df.columns:
        valid = df.dropna(subset=['cno', 'accurate_elevation', 'sigID'])
        if not valid.empty:
            colors = ['#1976D2', '#D32F2F', '#388E3C']
            for i, sig in enumerate(valid['sigID'].unique()):
                subset = valid[valid['sigID'] == sig]
                ax.scatter(subset['accurate_elevation'], subset['cno'], s=3, alpha=0.6, c=colors[i % 3], label=sig)
            ax.legend(fontsize=9, markerscale=2, loc='lower right')
    ax.set_xlabel('Accurate Elevation (°)'); ax.set_ylabel('C/N₀ (dB-Hz)')
    ax.set_title('(b) Signal Strength vs Elevation', fontsize=15, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if elev_col in df.columns and 'utc_parsed' in df.columns:
        valid = df.dropna(subset=[elev_col, 'utc_parsed'])
        if not valid.empty:
            ax.scatter(valid['utc_parsed'], valid[elev_col], s=3, alpha=0.6, c='#F57C00')
            ax.axhline(y=RO_ELEVATION_THRESHOLD, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel('UTC Time'); ax.set_ylabel('Elevation (°)')
    ax.set_title('(c) Satellite Elevation', fontsize=15, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if 'doppler' in df.columns and 'utc_parsed' in df.columns and 'sigID' in df.columns:
        valid = df.dropna(subset=['doppler', 'utc_parsed', 'sigID'])
        if not valid.empty:
            colors = ['#1976D2', '#D32F2F', '#388E3C']
            for i, sig in enumerate(valid['sigID'].unique()):
                subset = valid[valid['sigID'] == sig]
                ax.scatter(subset['utc_parsed'], subset['doppler'], s=3, alpha=0.6, c=colors[i % 3], label=sig)
            ax.legend(fontsize=9, markerscale=2, loc='lower right')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel('UTC Time'); ax.set_ylabel('Measured Doppler (Hz)')
    ax.set_title('(d) Raw Doppler', fontsize=15, fontweight='bold'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True

# Frequency band display names by constellation
CONSTELLATION_FREQ_LABELS = {
    'GPS': ('L1', 'L2'),
    'GAL': ('E1', 'E5b'),
    'BDS': ('B1', 'B2'),
    'GLO': ('G1', 'G2'),
    'SBAS': ('L1', 'L5'),
    'QZSS': ('L1', 'L2'),
}


def get_freq_labels_from_sat_id(sat_id: str) -> tuple:
    """
    Extract frequency band labels from satellite ID.
    
    Args:
        sat_id: Satellite identifier like 'GPS_12', 'GAL_5', 'BDS_21', etc.
        
    Returns:
        Tuple of (freq1_label, freq2_label) for the constellation
    """
    sat_id_upper = sat_id.upper()
    
    if sat_id_upper.startswith('GPS') or sat_id_upper.startswith('G_'):
        return ('L1', 'L2')
    elif sat_id_upper.startswith('GAL') or sat_id_upper.startswith('E_'):
        return ('E1', 'E5b')
    elif sat_id_upper.startswith('BDS') or sat_id_upper.startswith('BEI') or sat_id_upper.startswith('C_'):
        return ('B1', 'B2')
    elif sat_id_upper.startswith('GLO') or sat_id_upper.startswith('R_'):
        return ('G1', 'G2')
    elif sat_id_upper.startswith('SBAS') or sat_id_upper.startswith('S_'):
        return ('L1', 'L5')
    elif sat_id_upper.startswith('QZSS') or sat_id_upper.startswith('J_'):
        return ('L1', 'L2')
    else:
        # Default fallback
        return ('F1', 'F2')


def generate_derived_plots(sat_results: Dict[str, Any], sat_id: str, output_path: str, dpi: int = 150, station_altitude: Optional[float] = None) -> bool:
    """
    Generate 2x2 derived profile plots (Panel 2).
    
    Subplots: Bending Angle, Refractivity, Refractivity % Error, Specific Humidity
    
    Changes from original:
    - Dynamic frequency labels based on constellation
    - Bending angle in degrees (not mrad)
    - Added Refractivity % Error and Specific Humidity panels
    - Station height shown as upper bound on bending angle plot
    """
    try:
        import matplotlib.pyplot as plt
        from plot_style import apply_plot_fonts
    except ImportError:
        return False

    apply_plot_fonts()   # v3.4.7 — larger tick / axis / legend text

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Derived Profiles: {sat_id}', fontsize=18, fontweight='bold')

    # v3.4.4: scatter (not line) for retrieved profiles — line plots in height
    # space connect outliers and visually corrupt the panel. Markers leave
    # outliers isolated and easy to spot.
    sct_kw = dict(s=8, alpha=0.7, edgecolors='none')

    # Get constellation-specific frequency labels
    freq1_label, freq2_label = get_freq_labels_from_sat_id(sat_id)

    # (a) Bending Angle Profile - IN DEGREES
    ax = axes[0, 0]
    bending_csv = sat_results.get('bending_csv')
    if bending_csv and os.path.exists(bending_csv):
        df = pd.read_csv(bending_csv)
        # Use geometric tangent height if available, fall back to impact height
        y_col = 'tangent_height_km' if 'tangent_height_km' in df.columns else 'impact_height_km'
        y_label = 'Geometric Height, a/n − Rₑ (km)' if y_col == 'tangent_height_km' else 'Impact Height, a − Rₑ (km)'
        if 'bending_L1' in df.columns:
            ax.scatter(np.degrees(df['bending_L1']), df[y_col],
                       c='b', label=freq1_label, **sct_kw)
        if 'bending_L2' in df.columns:
            ax.scatter(np.degrees(df['bending_L2']), df[y_col],
                       c='r', marker='x', label=freq2_label, s=10, alpha=0.7)
        if 'bending_angle_rad' in df.columns:
            ax.scatter(np.degrees(df['bending_angle_rad']), df[y_col],
                       c='g', label='Iono-free', **sct_kw)
        ax.legend(fontsize=10, loc='upper right')
    else:
        y_label = 'Geometric Height, a/n − Rₑ (km)'
    # Station height as upper bound for retrieved profile
    if station_altitude is not None:
        station_alt_km = station_altitude / 1000.0
        ax.axhline(y=station_alt_km, color='#4CAF50', linestyle=':', linewidth=1.5, alpha=0.8,
                   label=f'Station height ({station_alt_km:.2f} km)')
        ax.legend(fontsize=10, loc='upper right')
    ax.set_xlabel('Bending Angle (°)')
    ax.set_ylabel(y_label)
    ax.set_title('(a) Bending Angle', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (b) Refractivity Profile
    ax = axes[0, 1]
    comp_csv, refrac_csv = sat_results.get('comp_csv'), sat_results.get('refrac_csv')
    if comp_csv and os.path.exists(comp_csv):
        df = pd.read_csv(comp_csv)
        ax.scatter(df['N_RO'], df['height_km'], c='r', label='RO Retrieved', **sct_kw)
        ax.scatter(df['N_ERA5'], df['height_km'], c='b', marker='x', s=10, alpha=0.7, label='ERA5')
        ax.legend(fontsize=11, loc='upper right')
    elif refrac_csv and os.path.exists(refrac_csv):
        df = pd.read_csv(refrac_csv)
        ax.scatter(df['refractivity_N'], df['height_km'], c='r', label='RO Retrieved', **sct_kw)
        ax.legend(fontsize=11, loc='upper right')
    ax.set_xlabel('Refractivity (N-units)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(b) Refractivity', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (c) Refractivity % Error
    ax = axes[1, 0]
    if comp_csv and os.path.exists(comp_csv):
        df = pd.read_csv(comp_csv)
        if 'N_RO' in df.columns and 'N_ERA5' in df.columns:
            pct_error = ((df['N_RO'] - df['N_ERA5']) / df['N_ERA5']) * 100
            ax.scatter(pct_error, df['height_km'], c='m', label='% Error', **sct_kw)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.legend(fontsize=11, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Requires ERA5\ncomparison data',
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
    ax.set_xlabel('Refractivity Error (%)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(c) Refractivity % Error', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (d) Specific Humidity
    ax = axes[1, 1]
    atm_csv = sat_results.get('atm_csv')
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'specific_humidity_g_kg' in df.columns:
            ax.scatter(df['specific_humidity_g_kg'], df['height_km'], c='c', label='q (RO)', **sct_kw)
        if 'q_era5' in df.columns:
            ax.scatter(df['q_era5'], df['height_km'], c='c', marker='x', s=10, alpha=0.6, label='q (ERA5)')
        ax.legend(fontsize=11, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Requires atmospheric\nretrieval data',
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
    ax.set_xlabel('Specific Humidity (g/kg)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(d) Specific Humidity', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


def generate_atmospheric_plots(sat_results: Dict[str, Any], sat_id: str, output_path: str, dpi: int = 150) -> bool:
    """
    Generate 2x2 atmospheric retrieval plots (Panel 3).
    
    Subplots: Pressure, Water Vapor Pressure, Relative Humidity, Temperature (ERA5)
    """
    try:
        import matplotlib.pyplot as plt
        from plot_style import apply_plot_fonts
    except ImportError:
        return False

    apply_plot_fonts()   # v3.4.7 — larger tick / axis / legend text

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Atmospheric Profiles: {sat_id}', fontsize=18, fontweight='bold')

    # v3.4.4: scatter (not line) for retrieved profiles.
    sct_kw = dict(s=8, alpha=0.7, edgecolors='none')

    atm_csv = sat_results.get('atm_csv')

    # (a) Pressure Profile
    ax = axes[0, 0]
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'pressure_hPa' in df.columns:
            ax.scatter(df['pressure_hPa'], df['height_km'], c='r', label='P (RO)', **sct_kw)
        if 'P_era5' in df.columns:
            ax.scatter(df['P_era5'], df['height_km'], c='r', marker='x', s=10, alpha=0.6, label='P (ERA5)')
        ax.legend(fontsize=11, loc='upper right')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(a) Pressure', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) Water Vapor Pressure Profile
    ax = axes[0, 1]
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'water_vapor_hPa' in df.columns:
            ax.scatter(df['water_vapor_hPa'], df['height_km'], c='b', label='Pw (RO)', **sct_kw)
        if 'Pw_era5' in df.columns:
            ax.scatter(df['Pw_era5'], df['height_km'], c='b', marker='x', s=10, alpha=0.6, label='Pw (ERA5)')
        ax.legend(fontsize=11, loc='upper right')
    ax.set_xlabel('Water Vapor Pressure (hPa)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(b) Water Vapor Pressure', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (c) Relative Humidity (computed from Pw and T)
    ax = axes[1, 0]
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'water_vapor_hPa' in df.columns and 'T_era5' in df.columns:
            T_c = df['T_era5'] - 273.15
            Ps = 6.1094 * np.exp(17.625 * T_c / (T_c + 243.04))
            RH_ro = (df['water_vapor_hPa'] / Ps) * 100
            RH_ro = RH_ro.clip(0, 100)
            ax.scatter(RH_ro, df['height_km'], c='#8E24AA', label='RH (RO)', **sct_kw)
            if 'Pw_era5' in df.columns:
                RH_era5 = (df['Pw_era5'] / Ps) * 100
                RH_era5 = RH_era5.clip(0, 100)
                ax.scatter(RH_era5, df['height_km'], c='#8E24AA', marker='x',
                           s=10, alpha=0.6, label='RH (ERA5)')
            ax.legend(fontsize=11, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Requires Pw and T\ndata for RH', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
    else:
        ax.text(0.5, 0.5, 'Requires atmospheric\nretrieval data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(c) Relative Humidity', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (d) Temperature Profile - IN CELSIUS
    ax = axes[1, 1]
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'T_era5' in df.columns:
            temp_celsius = df['T_era5'] - 273.15
            ax.scatter(temp_celsius, df['height_km'], c='g', label='T (ERA5)', **sct_kw)
            ax.legend(fontsize=11, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Temperature data\nnot available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
    else:
        ax.text(0.5, 0.5, 'Requires ERA5 data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Geometric Height, a/n − Rₑ (km)')
    ax.set_title('(d) Temperature (ERA5)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True




# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

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
        Run complete GNSS-RO processing pipeline.
        Supports UBX or RINEX input (auto-detected).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Parse observations (UBX or RNX)
        if progress_callback: 
            progress_callback("Parsing observation files...", 0.0)
        self.results['step1'] = parse_gnss_directory(
            ubx_dir, 
            f"{output_dir}/step1_observations.csv", 
            progress_callback
        )
        if not self.results['step1'].success: 
            return self.results
        
        # Log data source
        source = self.results['step1'].metadata.get('source', 'unknown')
        if progress_callback:
            progress_callback(f"Using {source} data source", 0.05)

        # Step 2: SP3 matching
        if progress_callback: 
            progress_callback("Matching with SP3 ephemeris...", 0.1)
        self.results['step2'] = match_observations_with_sp3(
            f"{output_dir}/step1_observations.csv", 
            sp3_file, 
            f"{output_dir}/step2_matched.csv", 
            progress_callback
        )
        if not self.results['step2'].success: 
            return self.results

        # Step 3a: Elevation calculation
        if progress_callback: 
            progress_callback("Calculating elevations...", 0.35)
        self.results['step3a'] = calculate_accurate_elevations(
            f"{output_dir}/step2_matched.csv", 
            self.station, 
            f"{output_dir}/step3a_elevations.csv"
        )

        # Step 3b: Geometric Doppler
        if progress_callback: 
            progress_callback("Calculating geometric Doppler...", 0.45)
        self.results['step3b'] = calculate_geometric_doppler(
            f"{output_dir}/step3a_elevations.csv", 
            self.station, 
            f"{output_dir}/step3b_doppler.csv"
        )

        # Step 4: Single differencing
        if progress_callback: 
            progress_callback("Applying single differencing...", 0.55)
        self.results['step4'] = apply_single_differencing(
            f"{output_dir}/step3b_doppler.csv", 
            self.config, 
            f"{output_dir}/step4_differenced.csv"
        )

        # Step 5: Bending angles
        if progress_callback: 
            progress_callback("Retrieving bending angles...", 0.65)
        self.results['step5'] = retrieve_bending_angles(
            f"{output_dir}/step4_differenced.csv", 
            self.station, 
            self.config, 
            f"{output_dir}/bending"
        )

        # Steps 6-7: Per-satellite processing
        if self.results['step5'].success and self.results['step5'].data is not None:
            for idx, row in self.results['step5'].data.iterrows():
                sat_id = row['sat_id']
                bending_csv = f"{output_dir}/bending/{sat_id}_bending.csv"
                if os.path.exists(bending_csv):
                    if progress_callback: 
                        progress_callback(f"Abel inversion: {sat_id}...", 0.7 + idx * 0.03)
                    refrac_csv = f"{output_dir}/refractivity/{sat_id}_refractivity.csv"
                    os.makedirs(os.path.dirname(refrac_csv), exist_ok=True)
                    result = retrieve_refractivity(bending_csv, refrac_csv)
                    self.results[f'step6_{sat_id}'] = result

                    if era5_file and result.success:
                        comp_csv = f"{output_dir}/comparison/{sat_id}_comparison.csv"
                        atm_csv = f"{output_dir}/atmospheric/{sat_id}_atmospheric.csv"
                        os.makedirs(os.path.dirname(comp_csv), exist_ok=True)
                        os.makedirs(os.path.dirname(atm_csv), exist_ok=True)
                        self.results[f'step6b_{sat_id}'] = compare_with_era5(
                            refrac_csv, era5_file, 
                            self.station.latitude, self.station.longitude, 
                            comp_csv
                        )
                        self.results[f'step7_{sat_id}'] = retrieve_atmospheric_profile(
                            refrac_csv, era5_file, 
                            self.station.latitude, self.station.longitude, 
                            atm_csv
                        )

        if progress_callback: 
            progress_callback("Pipeline complete", 1.0)
        return self.results



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNSS-RO Processing Pipeline")
    parser.add_argument("--ubx-dir", required=True); parser.add_argument("--sp3-file", required=True)
    parser.add_argument("--era5-file"); parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--lat", type=float, required=True); parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--alt", type=float, required=True); parser.add_argument("--name", default="Station")
    args = parser.parse_args()

    station = StationConfig(latitude=args.lat, longitude=args.lon, altitude=args.alt, name=args.name)
    pipeline = GNSSROPipeline(station)
    results = pipeline.run_full_pipeline(args.ubx_dir, args.sp3_file, args.era5_file, args.output_dir)
    print("\n" + "=" * 60 + "\nPIPELINE SUMMARY\n" + "=" * 60)
    for step, result in results.items():
        print(f"{'✓' if result.success else '✗'} {step}: {result.message}")
