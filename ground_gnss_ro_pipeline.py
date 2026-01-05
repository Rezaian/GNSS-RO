"""
GNSS Radio Occultation Processing Pipeline v1.2
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
RO_MIN_EPOCHS = 10

POLYNOMIAL_WINDOW = 150 #(for a 50Hz sampling be 150/50 = 3)


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

    def to_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.latitude, self.longitude, self.altitude)

    def get_gaussian_radius(self) -> float:
        lat_r = np.radians(self.latitude)
        M = (WGS84_A * (1 - WGS84_E2)) / (1 - WGS84_E2 * np.sin(lat_r) ** 2) ** 1.5
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_r) ** 2)
        return np.sqrt(M * N)


@dataclass
class PipelineConfig:
    elevation_mask_high: float = 45.0
    elevation_mask_low: float = -5.0
    height_range_min: float = -10.0
    height_range_max: float = 150.0
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
    elevation_threshold: float = RO_ELEVATION_THRESHOLD,
    doppler_threshold: float = RO_DOPPLER_THRESHOLD,
    min_epochs: int = RO_MIN_EPOCHS,
    min_dual_freq_epochs: int = RO_MIN_EPOCHS
) -> Dict[str, bool]:
    """Evaluate RO status for each satellite."""
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
    
    for i, rnx_file in enumerate(rnx_files):
        try:
            parser = RINEXParser(rnx_file)
            observations = parser.parse()
            all_observations.extend(observations)
            
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
        metadata={'file_count': len(rnx_files), 'source': 'RINEX'}
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
                    doppler[i] = (phases[i] - phases[i-1]) / dt
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
                    doppler[i] = (phases[i] - phases[i-1]) / dt
            continue
        
        t_center = t_sec[i]
        t_norm = t_window - t_center
        
        try:
            coeffs = np.polyfit(t_norm, p_window, poly_order)
            doppler[i] = coeffs[-2]  # derivative at center
        except (np.linalg.LinAlgError, ValueError):
            if i > 0:
                dt = t_sec[i] - t_sec[i-1]
                if dt > 0 and dt < max_gap_sec:
                    doppler[i] = (phases[i] - phases[i-1]) / dt
    
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

class SP3Parser:
    CONSTELLATION_MAP = {'GPS': 'G', 'GLO': 'R', 'GAL': 'E', 'BDS': 'C', 'QZS': 'J', 'IRN': 'I'}

    def __init__(self, sp3_file: str):
        self.epochs: Dict[datetime, Dict[str, Dict]] = {}
        self.satellites: set = set()
        self._epoch_timestamps: Dict[datetime, float] = {}
        self._parse(sp3_file)

    def _parse(self, filename: str) -> None:
        with open(filename, 'r') as f:
            lines = f.readlines()
        current_epoch = None
        for line in lines:
            line = line.strip()
            if line.startswith('*'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                        hour, minute, second = int(parts[4]), int(parts[5]), float(parts[6])
                        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                        dt += timedelta(seconds=second)
                        current_epoch = dt.replace(tzinfo=None)
                        if current_epoch not in self.epochs:
                            self.epochs[current_epoch] = {}
                            self._epoch_timestamps[current_epoch] = dt.timestamp()
                    except (ValueError, IndexError):
                        continue
            elif line.startswith('P') and current_epoch is not None:
                try:
                    sat_id = line[1:4]
                    parts = line[4:].split()
                    if len(parts) >= 4:
                        x = np.float64(parts[0]) * 1000.0
                        y = np.float64(parts[1]) * 1000.0
                        z = np.float64(parts[2]) * 1000.0
                        clk = np.float64(parts[3]) * 1e-6
                        if abs(x) < 50000000 and abs(y) < 50000000:
                            self.epochs[current_epoch][sat_id] = {'x': x, 'y': y, 'z': z, 'clk': clk}
                            self.satellites.add(sat_id)
                except (ValueError, IndexError):
                    continue

    def interpolate(self, sat_id: str, target_time_utc: datetime) -> Optional[Dict]:
        if pd.isna(target_time_utc) or target_time_utc is pd.NaT:
            return None
        if hasattr(target_time_utc, 'to_pydatetime'):
            target_time_utc = target_time_utc.to_pydatetime()
        if hasattr(target_time_utc, 'tzinfo') and target_time_utc.tzinfo is not None:
            target_time_utc = target_time_utc.astimezone(timezone.utc).replace(tzinfo=None)

        target_time_gps = target_time_utc + timedelta(seconds=GPS_LEAP_SECONDS)
        sp3_sat_id = self._convert_sat_id(sat_id)
        if not sp3_sat_id:
            return None

        available_epochs, positions, clocks = [], [], []
        for epoch_time, epoch_data in self.epochs.items():
            if sp3_sat_id in epoch_data:
                available_epochs.append(epoch_time)
                sat_data = epoch_data[sp3_sat_id]
                positions.append([sat_data['x'], sat_data['y'], sat_data['z']])
                clocks.append(sat_data['clk'])

        if len(available_epochs) < 4:
            return None

        available_timestamps = np.array([self._epoch_timestamps[ep] for ep in available_epochs], dtype=np.float64)
        sorted_indices = np.argsort(available_timestamps)
        available_timestamps = available_timestamps[sorted_indices]
        positions = np.array([positions[i] for i in sorted_indices], dtype=np.float64)
        clocks = np.array([clocks[i] for i in sorted_indices], dtype=np.float64)

        target_timestamp_gps = target_time_gps.replace(tzinfo=timezone.utc).timestamp()
        if target_timestamp_gps < available_timestamps[0] or target_timestamp_gps > available_timestamps[-1]:
            return None

        window_size = min(12, len(available_timestamps))
        center_idx = np.argmin(np.abs(available_timestamps - target_timestamp_gps))
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(available_timestamps), start_idx + window_size)

        t_window = available_timestamps[start_idx:end_idx]
        pos_window = positions[start_idx:end_idx]
        clk_window = clocks[start_idx:end_idx]

        try:
            cs_x = CubicSpline(t_window, pos_window[:, 0], bc_type='natural')
            cs_y = CubicSpline(t_window, pos_window[:, 1], bc_type='natural')
            cs_z = CubicSpline(t_window, pos_window[:, 2], bc_type='natural')
            cs_clk = CubicSpline(t_window, clk_window, bc_type='natural')

            return {
                'gps_time_used': target_time_gps,
                'interp_x': np.float64(cs_x(target_timestamp_gps)),
                'interp_y': np.float64(cs_y(target_timestamp_gps)),
                'interp_z': np.float64(cs_z(target_timestamp_gps)),
                'interp_vel_x': np.float64(cs_x.derivative()(target_timestamp_gps)),
                'interp_vel_y': np.float64(cs_y.derivative()(target_timestamp_gps)),
                'interp_vel_z': np.float64(cs_z.derivative()(target_timestamp_gps)),
                'interp_speed': np.sqrt(
                    cs_x.derivative()(target_timestamp_gps)**2 + 
                    cs_y.derivative()(target_timestamp_gps)**2 + 
                    cs_z.derivative()(target_timestamp_gps)**2
                ),
                'interp_clk': np.float64(cs_clk(target_timestamp_gps)),
                'interp_clk_rate': np.float64(cs_clk.derivative()(target_timestamp_gps)),  # ADD THIS
            }
        except Exception:
            return None

    def _convert_sat_id(self, gnss_sv_string: str) -> Optional[str]:
        if not isinstance(gnss_sv_string, str):
            return None
        gnss_sv_string = gnss_sv_string.strip()
        if len(gnss_sv_string) == 3 and gnss_sv_string[0].isalpha() and gnss_sv_string[1:].isdigit():
            return gnss_sv_string.upper()
        parts = gnss_sv_string.split()
        if len(parts) == 2:
            try:
                prefix = self.CONSTELLATION_MAP.get(parts[0].strip().upper())
                if prefix:
                    return f"{prefix}{int(parts[1]):02d}"
            except ValueError:
                pass
        return None


def match_observations_with_sp3(
    obs_csv: str, sp3_file: str, output_csv: Optional[str] = None,
    progress_callback: Optional[Callable] = None, batch_size: int = 2000
) -> ProcessingResult:
    df = pd.read_csv(obs_csv)
    df['parsed_utc'] = pd.to_datetime(df['utc'], errors='coerce')
    df['parsed_utc'] = df['parsed_utc'].dt.tz_localize(None)
    df = df.dropna(subset=['parsed_utc'])
    df['sat_identifier'] = df['gnssId'].astype(str) + ' ' + df['svId'].astype(str)

    sp3 = SP3Parser(sp3_file)

    sp3_cols = ['gps_time_used', 'interp_x', 'interp_y', 'interp_z',
                'interp_vel_x', 'interp_vel_y', 'interp_vel_z',
                'interp_speed', 'interp_clk', 'sp3_match_status']
    for col in sp3_cols:
        df[col] = np.nan
    df['sp3_match_status'] = 'no_match'

    matched_count, total_count = 0, len(df)

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

    df_matched = df[df['sp3_match_status'] == 'matched'].copy()
    if output_csv and not df_matched.empty:
        df_matched.to_csv(output_csv, index=False)

    return ProcessingResult(
        success=True, data=df_matched,
        message=f"Matched {matched_count}/{total_count} observations",
        metadata={'total': total_count, 'matched': matched_count}
    )


# ============================================================================
# STEP 3A & 3B: ELEVATION AND DOPPLER
# ============================================================================

def calculate_accurate_elevations(input_csv: str, station: StationConfig, output_csv: Optional[str] = None) -> ProcessingResult:
    df = pd.read_csv(input_csv)
    
    # Convert elevation to numeric, coercing empty/invalid to NaN
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    
    # Only filter if elevation data exists; otherwise keep all rows
    # (elevation will be computed from SP3 positions)
    if df['elevation'].notna().any():
        df = df[df['elevation'] >= -5]
    
    if df.empty:
        return ProcessingResult(False, message="No observations after elevation filter")
    
    station_xyz = station.to_ecef()
    
    elevations = []
    for _, r in df.iterrows():
        try:
            sat_xyz = np.array([r['interp_x'], r['interp_y'], r['interp_z']])
            elev = calculate_elevation_angle(sat_xyz, station_xyz)
            elevations.append(elev)
        except (KeyError, ValueError):
            elevations.append(np.nan)
    
    df['accurate_elevation'] = elevations
    
    # Now filter on computed accurate_elevation
    df = df[df['accurate_elevation'] >= -5]
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    return ProcessingResult(
        success=True, 
        data=df, 
        message=f"Calculated elevations for {len(df)} observations"
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
        return ProcessingResult(False, message=f"Missing columns: {missing}")
    
    # Convert pseudorange to numeric
    df['pseudorange'] = pd.to_numeric(df['pseudorange'], errors='coerce')
    
    # Drop rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=['pseudorange', 'sigID']).copy()
    
    if df.empty:
        return ProcessingResult(False, message="No valid observations after filtering")
    
    station_ecef = station.to_ecef()
    
    df['time_delay_s'] = df['pseudorange'] / SPEED_OF_LIGHT
    
    sat_pos = df[['interp_x', 'interp_y', 'interp_z']].values
    sat_vel_ecef = df[['interp_vel_x', 'interp_vel_y', 'interp_vel_z']].values
    
    sat_pos_tx = sat_pos - sat_vel_ecef * df['time_delay_s'].values[:, np.newaxis]
    
    los_vector = sat_pos_tx - station_ecef
    los_dist = np.linalg.norm(los_vector, axis=1)
    los_unit = los_vector / los_dist[:, np.newaxis]
    
    range_rate = np.einsum('ij,ij->i', sat_vel_ecef, los_unit)
    
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
        metadata={'freq_fallback_count': freq_missing_count}
    )


def diagnose_high_elevation_bias(input_csv: str, elevation_threshold: float = 45.0):
    """Check excess_doppler statistics at high elevation before single differencing."""
    df = pd.read_csv(input_csv)
    
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'
    df['excess_doppler'] = df['doppler'] - df['geometric_doppler']
    
    high_elev = df[df[elev_col] >= elevation_threshold].copy()
    
    print(f"High elevation (>{elevation_threshold}°) excess_doppler statistics:")
    print(f"  Count: {len(high_elev)}")
    print(f"  Mean:  {high_elev['excess_doppler'].mean():.2f} Hz")
    print(f"  Std:   {high_elev['excess_doppler'].std():.2f} Hz")
    print(f"  Min:   {high_elev['excess_doppler'].min():.2f} Hz")
    print(f"  Max:   {high_elev['excess_doppler'].max():.2f} Hz")
    
    print("\nPer-satellite breakdown:")
    for sat_id, group in high_elev.groupby(['gnssId', 'svId']):
        mean = group['excess_doppler'].mean()
        std = group['excess_doppler'].std()
        print(f"  {sat_id[0]}_{sat_id[1]:02d}: mean={mean:+.2f} Hz, std={std:.2f} Hz, n={len(group)}")
    
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
    
def apply_fresnel_polynomial_smoothing(df: pd.DataFrame, fresnel_window_sec: float = POLYNOMIAL_WINDOW) -> pd.DataFrame:
    """
    Apply 2nd-order polynomial fit to atmos_doppler over Fresnel time window.
    Takes value at center of window. Respects gaps >= 5 seconds.
    """
    df = df.copy()
    df['atmos_dopp_poli'] = np.nan
    
    if 'atmos_doppler' not in df.columns or 'timestamp' not in df.columns:
        return df
    
    gap_threshold = 5.0
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
            
            # Impact parameter constraint: |r × k| same at both ends
            # (ray asymptotes have same closest approach distance)
            a_t = np.linalg.norm(np.cross(r_t, k_t))
            a_r = np.linalg.norm(np.cross(r_r, k_r))
            eq2 = a_t - a_r
            
            return [eq1, eq2]

        try:
            dt_sol, dr_sol = fsolve(equations, [1e-5, 1e-4], full_output=False)
            
            # Total bending angle
            bending_angle = np.abs(dt_sol) + np.abs(dr_sol)
            
            # Impact parameter from transmitter side
            k_t_sol = rotate_vec(k0, axis, -dt_sol)
            impact_parameter = np.linalg.norm(np.cross(r_t, k_t_sol))
            
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
            res_f1, res_f2 = {'utc': [], 'a': [], 'alpha': []}, {'utc': [], 'a': [], 'alpha': []}

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
                alpha2_interp = interp1d(d2['a'], d2['alpha'], kind='linear', fill_value="extrapolate")(d1['a'])


                coeff_1 =  f1**2 / (f1**2 - f2**2)
                coeff_2 =  f2**2 / (f1**2 - f2**2)
                alpha_neut =  coeff_1 * d1['alpha'] - coeff_2 * alpha2_interp
                tangent_height = (d1['a'] - self.R_local) / 1000.0

                out_df = pd.DataFrame({
                    'utc': d1['utc'], 
                    'impact_parameter_m': d1['a'], 
                    'tangent_height_km': tangent_height,
                    'bending_angle_rad': alpha_neut, 
                    'bending_angle_deg': np.degrees(alpha_neut),
                    'bending_L1': d1['alpha'], 
                    'bending_L2': alpha2_interp,
                })
                out_df = out_df[
                    (out_df['tangent_height_km'] > config.height_range_min) & 
                    (out_df['tangent_height_km'] < config.height_range_max)
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
    def __init__(self, climatology_blend_km: float = 50.0):
        self.R_earth = R_EARTH
        self.climatology_blend_km = climatology_blend_km

    def run(self, bending_csv: str, output_csv: Optional[str] = None) -> ProcessingResult:
        df = pd.read_csv(bending_csv).sort_values('impact_parameter_m').reset_index(drop=True)
        df['approx_height_km'] = (df['impact_parameter_m'] - self.R_earth) / 1000

        fit_data = df[df['approx_height_km'] <= 40.0]
        coeffs = np.polyfit(fit_data['impact_parameter_m'].values, np.log(fit_data['bending_angle_rad'].values + 1e-10), 1)
        scale_height = -1.0 / coeffs[0] if coeffs[0] < 0 else 7000.0
        a_ref, alpha_ref = fit_data['impact_parameter_m'].iloc[len(fit_data)//2], fit_data['bending_angle_rad'].iloc[len(fit_data)//2]

        a_values, alpha_meas = df['impact_parameter_m'].values, df['bending_angle_rad'].values
        alpha_clim = alpha_ref * np.exp(-(a_values - a_ref) / scale_height)
        a_upper = df[df['approx_height_km'] <= self.climatology_blend_km]['impact_parameter_m'].max()

        sigma_meas, sigma_clim = np.full_like(alpha_meas, 1e-7), np.full_like(alpha_clim, 1e10)
        sigma_clim[a_values > a_upper] = 0.05 * alpha_clim[a_values > a_upper]
        w_meas, w_clim = 1.0 / sigma_meas**2, 1.0 / sigma_clim**2
        df['bending_optimized'] = (alpha_meas * w_meas + alpha_clim * w_clim) / (w_meas + w_clim)

        a, alpha = df['impact_parameter_m'].values, df['bending_optimized'].values
        n_levels = len(a)
        ln_n = np.zeros(n_levels)
        for i in range(n_levels):
            a_i = a[i]
            a_int_idx = min(i + 3, n_levels - 1)
            a_int = a[a_int_idx]
            analytical = alpha[a_int_idx] * np.log(a_int + np.sqrt(a_int**2 - a_i**2)) - alpha[i] * np.log(a_i + 1e-10)
            parts_integral = sum(-np.log((a[j] + a[j+1])/2 + np.sqrt(((a[j] + a[j+1])/2)**2 - a_i**2)) * (alpha[j+1] - alpha[j]) for j in range(i, min(a_int_idx, n_levels-1)) if (a[j] + a[j+1])/2 > a_i)
            regular_integral = sum(((alpha[j] + alpha[j+1])/2) / np.sqrt(((a[j] + a[j+1])/2)**2 - a_i**2) * (a[j+1] - a[j]) for j in range(a_int_idx, n_levels-1) if ((a[j] + a[j+1])/2)**2 - a_i**2 > 0)
            ln_n[i] = (1.0 / np.pi) * (analytical + parts_integral + regular_integral)

        df['refractive_index'] = np.exp(ln_n)
        df['height_km'] = (df['impact_parameter_m'] / df['refractive_index'] - self.R_earth) / 1000.0
        # temporary ground base adjustment 5 X
        df['refractivity_N'] = 5* (df['refractive_index'] - 1.0) * 1e6
        # df['refractivity_N'] = (df['refractive_index'] - 1.0) * 1e6

        result_df = df[['height_km', 'refractivity_N', 'impact_parameter_m', 'bending_optimized']].copy()
        if output_csv:
            result_df.to_csv(output_csv, index=False)
        return ProcessingResult(success=True, data=result_df, message=f"Retrieved refractivity: {result_df['height_km'].min():.2f}-{result_df['height_km'].max():.2f} km")


def retrieve_refractivity(bending_csv: str, output_csv: Optional[str] = None, climatology_blend_km: float = 50.0) -> ProcessingResult:
    return AbelInversion(climatology_blend_km).run(bending_csv, output_csv)


# ============================================================================
# STEP 6B: ERA5 COMPARISON
# ============================================================================

def compare_with_era5(refractivity_csv: str, era5_file: str, lat: Optional[float] = None, lon: Optional[float] = None, output_csv: Optional[str] = None) -> ProcessingResult:
    try:
        import xarray as xr
    except ImportError:
        return ProcessingResult(False, message="xarray required")

    ro_df = pd.read_csv(refractivity_csv)
    h_ro, N_ro = ro_df['height_km'].values, ro_df['refractivity_N'].values
    valid = ~np.isnan(h_ro) & ~np.isnan(N_ro) & (h_ro > 0)
    h_ro, N_ro = h_ro[valid], N_ro[valid]

    ds = xr.open_dataset(era5_file)
    if lat is None or lon is None:
        T, q, z = ds['t'].mean(dim=['latitude', 'longitude', 'valid_time']), ds['q'].mean(dim=['latitude', 'longitude', 'valid_time']), ds['z'].mean(dim=['latitude', 'longitude', 'valid_time'])
    else:
        T, q, z = ds['t'].interp(latitude=lat, longitude=lon).mean(dim='valid_time'), ds['q'].interp(latitude=lat, longitude=lon).mean(dim='valid_time'), ds['z'].interp(latitude=lat, longitude=lon).mean(dim='valid_time')

    P = ds['pressure_level']
    height_km = (z / 9.80665).values / 1000.0
    e = (q * P) / (0.622 + (1 - 0.622) * q)
    N_era5 = 77.6 * (P / T) + 3.73e5 * (e / T**2)
    ds.close()

    N_era5_interp = interp1d(height_km, N_era5.values, kind='linear', bounds_error=False, fill_value=np.nan)(h_ro)
    valid = ~np.isnan(N_era5_interp)
    h_common, N_ro_common, N_era5_common = h_ro[valid], N_ro[valid], N_era5_interp[valid]
    error = N_ro_common - N_era5_common

    comparison_df = pd.DataFrame({'height_km': h_common, 'N_RO': N_ro_common, 'N_ERA5': N_era5_common, 'error': error})
    if output_csv:
        comparison_df.to_csv(output_csv, index=False)

    rmse, bias = np.sqrt(np.mean(error**2)), np.mean(error)
    corr = np.corrcoef(N_ro_common, N_era5_common)[0, 1] if len(N_ro_common) > 1 else np.nan
    return ProcessingResult(success=True, data=comparison_df, message=f"RMSE: {rmse:.4f}, Bias: {bias:+.4f}", metadata={'rmse': rmse, 'bias': bias, 'correlation': corr})


# ============================================================================
# STEP 7: ATMOSPHERIC RETRIEVAL
# ============================================================================

def retrieve_atmospheric_profile(refractivity_csv: str, era5_file: str, lat: Optional[float] = None, lon: Optional[float] = None, output_csv: Optional[str] = None) -> ProcessingResult:
    try:
        import xarray as xr
    except ImportError:
        return ProcessingResult(False, message="xarray required")

    ro_df = pd.read_csv(refractivity_csv)
    h_m = ro_df['height_km'].values * 1000
    N = ro_df['refractivity_N'].values

    ds = xr.open_dataset(era5_file)
    if lat is None or lon is None:
        T_era5, z_era5, q_era5 = ds['t'].mean(dim=['latitude', 'longitude', 'valid_time']), ds['z'].mean(dim=['latitude', 'longitude', 'valid_time']), ds['q'].mean(dim=['latitude', 'longitude', 'valid_time'])
    else:
        T_era5, z_era5, q_era5 = ds['t'].interp(latitude=lat, longitude=lon).mean(dim='valid_time'), ds['z'].interp(latitude=lat, longitude=lon).mean(dim='valid_time'), ds['q'].interp(latitude=lat, longitude=lon).mean(dim='valid_time')

    P_levels = ds['pressure_level'].values
    h_era5 = (z_era5 / 9.80665).values
    sort_idx = np.argsort(h_era5)
    h_era5, T_era5_arr, P_era5_arr, q_era5_arr = h_era5[sort_idx], T_era5.values[sort_idx], P_levels[sort_idx], q_era5.values[sort_idx]
    ds.close()

    T = interp1d(h_era5, T_era5_arr, bounds_error=False, fill_value='extrapolate')(h_m)
    P_era5 = interp1d(h_era5, P_era5_arr, bounds_error=False, fill_value='extrapolate')(h_m)
    q_era5 = interp1d(h_era5, q_era5_arr, bounds_error=False, fill_value='extrapolate')(h_m)

    n, i_top = len(h_m), np.argmax(h_m)
    Pw, P = np.zeros(n), np.zeros(n)
    P[i_top] = P_era5[i_top]

    R, m_dry, m_water = 8.314462, 28.97e-3, 18.015e-3
    for _ in range(15):
        Pw_old = Pw.copy()
        for i in range(i_top - 1, -1, -1):
            dh, h_mid, T_mid = h_m[i] - h_m[i + 1], 0.5 * (h_m[i] + h_m[i + 1]), 0.5 * (T[i] + T[i + 1])
            g = compute_gravity(h_mid, lat)
            rho = (m_dry * P[i + 1] * 100 + (m_water - m_dry) * 0.5 * (Pw[i] + Pw[i + 1]) * 100) / (R * T_mid)
            P[i] = P[i + 1] - rho * g * dh / 100
        for i in range(n):
            Pw[i] = max(0.0, (T[i]**2 / N_COEFF_A2) * (N[i] - N_COEFF_A1 * P[i] / T[i]))
            T_c = T[i] - 273.15
            Pw[i] = min(Pw[i], 6.1094 * np.exp(17.625 * T_c / (T_c + 243.04)))
        if np.max(np.abs(Pw - Pw_old)) < 0.01:
            break

    epsilon = m_water / m_dry
    q = epsilon * Pw / (P - Pw + epsilon * Pw) * 1000
    Pw_era5 = (q_era5 * P_era5) / (epsilon + (1 - epsilon) * q_era5)

    profile = pd.DataFrame({
        'height_km': h_m / 1000, 'temperature_K': T, 'pressure_hPa': P, 'water_vapor_hPa': Pw,
        'specific_humidity_g_kg': q, 'refractivity_N': N, 'T_era5': T, 'P_era5': P_era5,
        'Pw_era5': Pw_era5, 'q_era5': q_era5 * 1000,
    })
    if output_csv:
        profile.to_csv(output_csv, index=False)

    return ProcessingResult(success=True, data=profile, message=f"Retrieved atmospheric profiles", metadata={'P_rmse': np.sqrt(np.mean((P - P_era5)**2))})


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def generate_raw_plots(sat_data: pd.DataFrame, sat_id: str, output_path: str, dpi: int = 150) -> bool:
    """Generate 2x2 raw GNSS observation plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return False

    if sat_data.empty:
        return False

    df = sat_data.copy()
    if 'utc' in df.columns:
        df['utc_parsed'] = pd.to_datetime(df['utc'], errors='coerce')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'GNSS Raw Observations: {sat_id}', fontsize=14, fontweight='bold')
    elev_col = 'accurate_elevation' if 'accurate_elevation' in df.columns else 'elevation'

    ax = axes[0, 0]
    if 'atmos_doppler' in df.columns and 'utc_parsed' in df.columns and 'sigID' in df.columns:
        valid = df.dropna(subset=['atmos_doppler', 'utc_parsed'])
        if not valid.empty:
            # Scatter plot of raw atmos_doppler per frequency
            colors = {'L1C/A': '#1976D2', 'L2CL': '#D32F2F', 'B1I D1': '#1976D2', 'B2I D1': '#D32F2F', 
                      'E1C': '#1976D2', 'E5bQ': '#D32F2F', 'L1OF': '#1976D2', 'L2OF': '#D32F2F'}
            for sig in valid['sigID'].unique():
                subset = valid[valid['sigID'] == sig]
                c = colors.get(sig, '#1976D2')
                ax.scatter(subset['utc_parsed'], subset['atmos_doppler'], s=3, alpha=0.6, c=c, label=f'{sig}')
            
            # Overlay polynomial fit (thick black line, respecting gaps)
            if 'atmos_dopp_poli' in df.columns and 'timestamp' in df.columns:
                gap_threshold = 5.0
                for sig in valid['sigID'].unique():
                    sig_data = df[df['sigID'] == sig].dropna(subset=['atmos_dopp_poli', 'utc_parsed']).sort_values('timestamp')
                    if sig_data.empty:
                        continue
                    
                    timestamps = sig_data['timestamp'].values
                    time_diffs = np.diff(timestamps)
                    gap_indices = np.where(time_diffs >= gap_threshold)[0]
                    
                    segment_starts = np.concatenate([[0], gap_indices + 1])
                    segment_ends = np.concatenate([gap_indices + 1, [len(timestamps)]])
                    
                    for seg_start, seg_end in zip(segment_starts, segment_ends):
                        seg_data = sig_data.iloc[seg_start:seg_end]
                        if len(seg_data) < 2:
                            continue
                        ax.plot(seg_data['utc_parsed'], seg_data['atmos_dopp_poli'], 'k-', linewidth=2, alpha=0.8)
            
            ax.axhline(y=RO_DOPPLER_THRESHOLD, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=-RO_DOPPLER_THRESHOLD, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.legend(fontsize=7, markerscale=2, loc='best')
    ax.set_xlabel('UTC Time'); ax.set_ylabel('Atmospheric Doppler (Hz)')
    ax.set_title('(a) Atmospheric Doppler', fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if 'cno' in df.columns and 'accurate_elevation' in df.columns and 'sigID' in df.columns:
        valid = df.dropna(subset=['cno', 'accurate_elevation', 'sigID'])
        if not valid.empty:
            colors = ['#1976D2', '#D32F2F', '#388E3C']
            for i, sig in enumerate(valid['sigID'].unique()):
                subset = valid[valid['sigID'] == sig]
                ax.scatter(subset['accurate_elevation'], subset['cno'], s=3, alpha=0.6, c=colors[i % 3], label=sig)
            ax.legend(fontsize=7, markerscale=2, loc='lower right')
    ax.set_xlabel('Accurate Elevation (°)'); ax.set_ylabel('C/N₀ (dB-Hz)')
    ax.set_title('(b) Signal Strength vs Elevation', fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if elev_col in df.columns and 'utc_parsed' in df.columns:
        valid = df.dropna(subset=[elev_col, 'utc_parsed'])
        if not valid.empty:
            ax.scatter(valid['utc_parsed'], valid[elev_col], s=3, alpha=0.6, c='#F57C00')
            ax.axhline(y=RO_ELEVATION_THRESHOLD, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel('UTC Time'); ax.set_ylabel('Elevation (°)')
    ax.set_title('(c) Satellite Elevation', fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if 'doppler' in df.columns and 'utc_parsed' in df.columns and 'sigID' in df.columns:
        valid = df.dropna(subset=['doppler', 'utc_parsed', 'sigID'])
        if not valid.empty:
            colors = ['#1976D2', '#D32F2F', '#388E3C']
            for i, sig in enumerate(valid['sigID'].unique()):
                subset = valid[valid['sigID'] == sig]
                ax.scatter(subset['utc_parsed'], subset['doppler'], s=3, alpha=0.6, c=colors[i % 3], label=sig)
            ax.legend(fontsize=7, markerscale=2, loc='lower right')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel('UTC Time'); ax.set_ylabel('Measured Doppler (Hz)')
    ax.set_title('(d) Raw Doppler', fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.3)
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


def generate_derived_plots(sat_results: Dict[str, Any], sat_id: str, output_path: str, dpi: int = 150) -> bool:
    """
    Generate 2x2 derived profile plots.
    
    Changes from original:
    - Dynamic frequency labels based on constellation
    - Temperature in °C (not K)
    - Bending angle in degrees (not mrad)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Derived Atmospheric Profiles: {sat_id}', fontsize=14, fontweight='bold')

    # Get constellation-specific frequency labels
    freq1_label, freq2_label = get_freq_labels_from_sat_id(sat_id)

    # (a) Bending Angle Profile - NOW IN DEGREES
    ax = axes[0, 0]
    bending_csv = sat_results.get('bending_csv')
    if bending_csv and os.path.exists(bending_csv):
        df = pd.read_csv(bending_csv)
        if 'bending_L1' in df.columns:
            ax.plot(np.degrees(df['bending_L1']), df['tangent_height_km'], 
                    'b-', linewidth=1.5, label=freq1_label, alpha=0.8)
        if 'bending_L2' in df.columns:
            ax.plot(np.degrees(df['bending_L2']), df['tangent_height_km'], 
                    'r--', linewidth=1.5, label=freq2_label, alpha=0.8)
        if 'bending_angle_rad' in df.columns:
            ax.plot(np.degrees(df['bending_angle_rad']), df['tangent_height_km'], 
                    'g-', linewidth=2, label='Iono-free', alpha=0.9)
        ax.legend(fontsize=9, loc='upper right')
    ax.set_xlabel('Bending Angle (°)')
    ax.set_ylabel('Tangent Height (km)')
    ax.set_title('(a) Bending Angle Profile', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (b) Refractivity Profile - unchanged
    ax = axes[0, 1]
    comp_csv, refrac_csv = sat_results.get('comp_csv'), sat_results.get('refrac_csv')
    if comp_csv and os.path.exists(comp_csv):
        df = pd.read_csv(comp_csv)
        ax.plot(df['N_RO'], df['height_km'], 'r-', linewidth=1.8, label='RO Retrieved')
        ax.plot(df['N_ERA5'], df['height_km'], 'b--', linewidth=1.5, label='ERA5')
        ax.legend(fontsize=9, loc='upper right')
    elif refrac_csv and os.path.exists(refrac_csv):
        df = pd.read_csv(refrac_csv)
        ax.plot(df['refractivity_N'], df['height_km'], 'r-', linewidth=1.8, label='RO Retrieved')
        ax.legend(fontsize=9, loc='upper right')
    ax.set_xlabel('Refractivity (N-units)')
    ax.set_ylabel('Height (km)')
    ax.set_title('(b) Refractivity Profile', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (c) Pressure Profiles - unchanged
    ax = axes[1, 0]
    atm_csv = sat_results.get('atm_csv')
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'pressure_hPa' in df.columns:
            ax.plot(df['pressure_hPa'], df['height_km'], 'r-', linewidth=1.5, label='P (RO)')
        if 'P_era5' in df.columns:
            ax.plot(df['P_era5'], df['height_km'], 'r--', linewidth=1, alpha=0.6, label='P (ERA5)')
        if 'water_vapor_hPa' in df.columns:
            ax.plot(df['water_vapor_hPa'], df['height_km'], 'b-', linewidth=1.5, label='Pw (RO)')
        if 'Pw_era5' in df.columns:
            ax.plot(df['Pw_era5'], df['height_km'], 'b--', linewidth=1, alpha=0.6, label='Pw (ERA5)')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel('Pressure (hPa)')
    ax.set_ylabel('Height (km)')
    ax.set_title('(c) Pressure Profiles (P, Pw)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (d) Temperature Profile - NOW IN CELSIUS
    ax = axes[1, 1]
    if atm_csv and os.path.exists(atm_csv):
        df = pd.read_csv(atm_csv)
        if 'T_era5' in df.columns:
            # Convert K to Celsius
            temp_celsius = df['T_era5'] - 273.15
            ax.plot(temp_celsius, df['height_km'], 'g-', linewidth=1.8, label='T (ERA5)')
            ax.legend(fontsize=9, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Temperature data\nnot available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    else:
        ax.text(0.5, 0.5, 'Requires ERA5 data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Height (km)')
    ax.set_title('(d) Temperature Profile', fontsize=11, fontweight='bold')
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
