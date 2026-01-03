"""
LEO GNSS Radio Occultation Processing Pipeline
===============================================

Processes COSMIC-2 Level 1b (conPhs) data to derive atmospheric profiles.
Compares results against Level 2 products (atmPrf, wetPf2).

Key differences from ground-based RO:
    - Both receiver (LEO) and transmitter (GNSS) are moving
    - Excess phase already extracted in conPhs
    - No single-differencing needed (clock pre-calibrated)
    - Tangent point latitude varies along profile

Pipeline Steps:
    1. Load conPhs: Read excess phase and orbit data
    2. Orbit Interpolation: Interpolate low-rate orbits to high-rate observations
    3. Geometry Computation: Impact parameter, tangent point, Doppler
    4. Bending Angle Retrieval: Doppler inversion with dual-frequency iono correction
    5. Abel Inversion: Bending angle → refractivity
    6. Validation: Compare against atmPrf/wetPf2

Data source: https://data.cosmic.ucar.edu/gnss-ro/cosmic2/

Author: LEO RO Pipeline
"""

from __future__ import annotations
import os
import glob
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq, fsolve
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

SPEED_OF_LIGHT = 299792458.0  # m/s
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s

# WGS84 Ellipsoid
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_B = 6356752.3142  # Semi-minor axis (m)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # First eccentricity squared
R_EARTH = 6371000.0  # Mean Earth radius (m)

# GPS time epoch
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

# GNSS Signal Frequencies (Hz)
GNSS_FREQUENCIES = {
    # GPS
    'G': {'L1': 1575.42e6, 'L2': 1227.60e6, 'L5': 1176.45e6},
    # GLONASS (center frequencies, actual varies by channel)
    'R': {'L1': 1602.0e6, 'L2': 1246.0e6},
    # Galileo
    'E': {'L1': 1575.42e6, 'L5a': 1176.45e6, 'L5b': 1207.14e6},
    # BeiDou
    'C': {'B1': 1561.098e6, 'B2': 1207.14e6, 'B3': 1268.52e6},
}

# Refractivity coefficients (Smith-Weintraub)
K1 = 77.6  # K/hPa (dry term)
K2 = 3.73e5  # K²/hPa (wet term)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OccultationEvent:
    """Container for a single occultation event."""
    event_id: str
    constellation: str  # G, R, E, C
    prn: int
    leo_id: str
    start_time: datetime
    stop_time: datetime
    
    # High-rate data (typically 50-100 Hz)
    time_hr: np.ndarray  # GPS seconds
    excess_L1: np.ndarray  # meters
    excess_L2: np.ndarray  # meters
    snr_L1: np.ndarray
    snr_L2: np.ndarray
    occ_height: np.ndarray  # km (from conPhs)
    
    # Interpolated orbit data (at high-rate times)
    leo_pos: np.ndarray  # (N, 3) meters ECEF
    leo_vel: np.ndarray  # (N, 3) m/s ECEF
    gnss_pos: np.ndarray  # (N, 3) meters ECEF
    gnss_vel: np.ndarray  # (N, 3) m/s ECEF
    
    # Metadata
    leap_seconds: float = 18.0
    
    def __post_init__(self):
        self.n_obs = len(self.time_hr)


@dataclass
class BendingProfile:
    """Bending angle profile output."""
    impact_param: np.ndarray  # meters
    tangent_height: np.ndarray  # km
    bending_L1: np.ndarray  # rad
    bending_L2: np.ndarray  # rad
    bending_neutral: np.ndarray  # rad (iono-corrected)
    latitude: np.ndarray  # deg
    longitude: np.ndarray  # deg
    time: np.ndarray  # GPS seconds


@dataclass
class RefractivityProfile:
    """Refractivity profile output."""
    height: np.ndarray  # km
    refractivity: np.ndarray  # N-units
    impact_param: np.ndarray  # meters
    bending: np.ndarray  # rad
    latitude: np.ndarray  # deg
    longitude: np.ndarray  # deg


@dataclass 
class ValidationResult:
    """Comparison against Level 2 products."""
    height_common: np.ndarray
    retrieved: np.ndarray
    reference: np.ndarray
    difference: np.ndarray
    rmse: float
    bias: float
    correlation: float
    n_points: int


@dataclass
class PipelineConfig:
    """Processing configuration."""
    # Quality filters
    min_snr: float = 50.0  # minimum SNR threshold
    min_height_km: float = -2.0  # minimum tangent height
    max_height_km: float = 60.0  # maximum tangent height for retrieval
    top_height_km: float = 80.0  # top of bending angle profile
    
    # Bending angle processing
    smoothing_window: int = 11  # samples for excess phase smoothing
    min_bending_rad: float = 1e-7  # minimum valid bending angle
    max_bending_rad: float = 0.1  # maximum valid bending angle
    
    # Abel inversion
    climatology_blend_km: float = 40.0  # blend with climatology above this
    abel_top_km: float = 150.0  # upper integration limit
    
    # Output
    output_resolution_m: float = 100.0  # vertical resolution for output


# ============================================================================
# COORDINATE UTILITIES
# ============================================================================

def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF (x, y, z) to geodetic (lat, lon, height).
    Uses iterative method for accuracy.
    
    Returns: (latitude_deg, longitude_deg, height_m)
    """
    lon = np.arctan2(y, x)
    
    # Iterative latitude computation
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    
    for _ in range(10):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    
    return np.degrees(lat), np.degrees(lon), h


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    """Convert geodetic to ECEF coordinates."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
    
    return np.array([x, y, z])


def gaussian_radius(lat_deg: float) -> float:
    """
    Compute Gaussian mean radius of curvature at given latitude.
    R_g = sqrt(M * N) where M = meridional, N = prime vertical radius
    """
    lat = np.radians(lat_deg)
    sin2 = np.sin(lat)**2
    
    M = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sin2)**1.5
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin2)
    
    return np.sqrt(M * N)


def gps_seconds_to_datetime(gps_sec: float, leap_sec: float = 18.0) -> datetime:
    """Convert GPS seconds since 1980-01-06 to UTC datetime."""
    gps_time = GPS_EPOCH + timedelta(seconds=float(gps_sec))
    utc_time = gps_time - timedelta(seconds=leap_sec)
    return utc_time


# ============================================================================
# STEP 1: LOAD CONPHS DATA
# ============================================================================

class LeoOrbReader:
    """
    Read COSMIC-2 Level 1b leoOrb (precise LEO orbit) NetCDF files.
    
    leoOrb provides post-processed precise orbits, higher quality than
    the real-time orbits embedded in conPhs.
    
    File structure:
        - time: epoch times (GPS seconds)
        - x, y, z: ECEF position (km)
        - xdot, ydot, zdot: ECEF velocity (km/s)
        
    Naming: leoOrb_C2E1.2025.001_0001.0001_nc
    """
    
    def __init__(self):
        self.orbits: Dict[str, Dict] = {}  # leo_id -> orbit data
    
    def load_file(self, filepath: str) -> Optional[Dict]:
        """Load a single leoOrb file."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required")
        
        ds = xr.open_dataset(filepath)
        
        # Extract LEO ID from filename
        fname = Path(filepath).stem
        leo_id = self._extract_leo_id(fname)
        
        # Time (GPS seconds)
        time = ds['time'].values.astype(np.float64)
        
        # Position (km → m)
        x = ds['x'].values.astype(np.float64) * 1000.0
        y = ds['y'].values.astype(np.float64) * 1000.0
        z = ds['z'].values.astype(np.float64) * 1000.0
        
        # Velocity (km/s → m/s)
        if 'xdot' in ds:
            vx = ds['xdot'].values.astype(np.float64) * 1000.0
            vy = ds['ydot'].values.astype(np.float64) * 1000.0
            vz = ds['zdot'].values.astype(np.float64) * 1000.0
        else:
            # Compute velocity from position derivative
            dt = np.gradient(time)
            vx = np.gradient(x) / dt
            vy = np.gradient(y) / dt
            vz = np.gradient(z) / dt
        
        ds.close()
        
        orbit_data = {
            'leo_id': leo_id,
            'time': time,
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz
        }
        
        self.orbits[leo_id] = orbit_data
        return orbit_data
    
    def load_directory(self, dirpath: str, pattern: str = "leoOrb_*.nc") -> Dict[str, Dict]:
        """Load all leoOrb files from directory."""
        files = glob.glob(os.path.join(dirpath, "**", pattern), recursive=True)
        
        for f in files:
            try:
                self.load_file(f)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        
        return self.orbits
    
    def interpolate_to_times(
        self,
        leo_id: str,
        target_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate orbit to target times.
        
        Returns:
            position (N, 3), velocity (N, 3) in meters and m/s
        """
        if leo_id not in self.orbits:
            raise ValueError(f"No orbit data for {leo_id}")
        
        orb = self.orbits[leo_id]
        
        # Build splines
        cs_x = CubicSpline(orb['time'], orb['x'], bc_type='natural')
        cs_y = CubicSpline(orb['time'], orb['y'], bc_type='natural')
        cs_z = CubicSpline(orb['time'], orb['z'], bc_type='natural')
        
        # Position
        pos = np.column_stack([
            cs_x(target_times),
            cs_y(target_times),
            cs_z(target_times)
        ])
        
        # Velocity from spline derivative
        vel = np.column_stack([
            cs_x.derivative()(target_times),
            cs_y.derivative()(target_times),
            cs_z.derivative()(target_times)
        ])
        
        return pos, vel
    
    def _extract_leo_id(self, filename: str) -> str:
        """Extract LEO satellite ID from filename."""
        import re
        match = re.search(r'(C2[A-Z]\d)', filename)
        if match:
            return match.group(1)
        return "C2XX"


class ConPhsReader:
    """
    Read COSMIC-2 Level 1b conPhs (excess phase) NetCDF files.
    
    File structure:
        - time: high-rate observation times (GPS seconds)
        - exL1, exL2: excess phase at L1/L2 frequencies (meters)
        - txmitLR: low-rate transmission times (GPS seconds)
        - xLeoLR, yLeoLR, zLeoLR: LEO position (km)
        - xGnssLR, yGnssLR, zGnssLR: GNSS position (km)
        - occheight: tangent point height (km)
        - SNR fields: caL1Snr, pL1Snr, pL2Snr
    
    Can optionally use external leoOrb data for higher precision LEO orbits.
    """
    
    CONSTELLATION_MAP = {
        'G': 'GPS', 'R': 'GLO', 'E': 'GAL', 'C': 'BDS',
        'J': 'QZS', 'I': 'IRN'
    }
    
    def __init__(self, leoorb_reader: Optional[LeoOrbReader] = None):
        self.events: List[OccultationEvent] = []
        self.leoorb_reader = leoorb_reader

    def load_file(
            self, 
            filepath: str,
            use_leoorb: bool = True
        ) -> Optional[OccultationEvent]:
            """Load a single conPhs NetCDF file with time unit correction."""
            try:
                import xarray as xr
            except ImportError:
                raise ImportError("xarray required: pip install xarray netCDF4")
            
            # --- FIX START: Open without decoding to prevent DateParseError ---
            ds = xr.open_dataset(filepath, decode_times=False)
            
            # Standardize the GPS time units for all variables that use them
            bad_units = 'seconds since 0Z Jan 6, 1980 (GPS seconds)'
            good_units = 'seconds since 1980-01-06 00:00:00'
            
            for var_name in ds.variables:
                if ds[var_name].attrs.get('units') == bad_units:
                    ds[var_name].attrs['units'] = good_units
            
            # Now decode the variables into proper timestamps/objects
            ds = xr.decode_cf(ds)
            # --- FIX END ---
            
            # Extract metadata from filename and attributes
            fname = Path(filepath).stem
            event_id = ds.attrs.get('fileStamp', fname)
            con_id = ds.attrs.get('conId', 'G')
            leap_sec = float(ds.attrs.get('leapsec', 18.0))
            
            # Parse PRN and LEO ID
            prn = self._extract_prn(fname, con_id)
            leo_id = self._extract_leo_id(fname)
            
            # High-rate data (Convert to float64 for precision)
            # Note: If 'time' was decoded to datetime64, we convert back to 
            # GPS seconds for the internal math of the pipeline
            if np.issubdtype(ds['time'].dtype, np.datetime64):
                # Extract raw values or convert back to seconds from 1980
                # A common way is to access the 'units' originally provided
                time_hr = ds['time'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
                # Offset to GPS epoch if needed, but usually ds['time'] raw is what we want
            else:
                time_hr = ds['time'].values.astype(np.float64)
                
            # Excess phase (meters)
            ex_L1 = ds['exL1'].values.astype(np.float64)
            ex_L2 = ds['exL2'].values.astype(np.float64)
            
            # SNR
            snr_L1 = ds['caL1Snr'].values.astype(np.float32) # conPhs typically uses caL1Snr
            snr_L2 = ds['pL2Snr'].values.astype(np.float32)
            
            # Occultation height (km)
            occ_height = ds['occheight'].values.astype(np.float32)
            
            # Low-rate orbit data
            # Handle txmitLR being decoded to datetime
            if np.issubdtype(ds['txmitLR'].dtype, np.datetime64):
                t_lr = ds['txmitLR'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
            else:
                t_lr = ds['txmitLR'].values.astype(np.float64)
            
            # LEO and GNSS position (km -> m)
            leo_pos_lr = np.column_stack([ds['xLeoLR'], ds['yLeoLR'], ds['zLeoLR']]) * 1000.0
            gnss_pos_lr = np.column_stack([ds['xGnssLR'], ds['yGnssLR'], ds['zGnssLR']]) * 1000.0
            
            # Start/stop times
            start_gps = float(ds.attrs.get('startTime', t_lr[0]))
            stop_gps = float(ds.attrs.get('stopTime', t_lr[-1]))
            
            ds.close()
            
            # Interpolate LEO orbits
            if use_leoorb and self.leoorb_reader and leo_id in self.leoorb_reader.orbits:
                leo_pos, leo_vel = self.leoorb_reader.interpolate_to_times(leo_id, time_hr)
            else:
                leo_pos, leo_vel = self._interpolate_orbit(
                    t_lr, leo_pos_lr[:,0], leo_pos_lr[:,1], leo_pos_lr[:,2], time_hr
                )
            
            # GNSS orbits
            gnss_pos, gnss_vel = self._interpolate_orbit(
                t_lr, gnss_pos_lr[:,0], gnss_pos_lr[:,1], gnss_pos_lr[:,2], time_hr
            )
            
            return OccultationEvent(
                event_id=event_id, constellation=con_id, prn=prn, leo_id=leo_id,
                start_time=gps_seconds_to_datetime(start_gps, leap_sec),
                stop_time=gps_seconds_to_datetime(stop_gps, leap_sec),
                time_hr=time_hr, excess_L1=ex_L1, excess_L2=ex_L2,
                snr_L1=snr_L1, snr_L2=snr_L2, occ_height=occ_height,
                leo_pos=leo_pos, leo_vel=leo_vel,
                gnss_pos=gnss_pos, gnss_vel=gnss_vel,
                leap_seconds=leap_sec
            )
    
    def load_directory(self, dirpath: str, pattern: str = "conPhs_*.nc") -> List[OccultationEvent]:
        """Load all conPhs files from directory."""
        files = sorted(glob.glob(os.path.join(dirpath, pattern)))
        
        events = []
        for f in files:
            try:
                event = self.load_file(f)
                if event is not None:
                    events.append(event)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        
        self.events = events
        return events
    
    def _interpolate_orbit(
        self,
        t_lr: np.ndarray,
        x_lr: np.ndarray,
        y_lr: np.ndarray,
        z_lr: np.ndarray,
        t_hr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate low-rate orbit to high-rate times.
        Returns position (N,3) and velocity (N,3) arrays.
        """
        # Remove any NaN/invalid values
        valid = ~(np.isnan(x_lr) | np.isnan(y_lr) | np.isnan(z_lr) | 
                  (np.abs(x_lr) < 1e3) | (np.abs(y_lr) < 1e3))
        
        if np.sum(valid) < 4:
            # Not enough valid points, return NaN
            n = len(t_hr)
            return np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        
        t_v = t_lr[valid]
        x_v = x_lr[valid]
        y_v = y_lr[valid]
        z_v = z_lr[valid]
        
        # Cubic spline interpolation
        try:
            cs_x = CubicSpline(t_v, x_v, bc_type='natural')
            cs_y = CubicSpline(t_v, y_v, bc_type='natural')
            cs_z = CubicSpline(t_v, z_v, bc_type='natural')
            
            # Position
            pos = np.column_stack([
                cs_x(t_hr),
                cs_y(t_hr),
                cs_z(t_hr)
            ])
            
            # Velocity (derivative of spline)
            vel = np.column_stack([
                cs_x.derivative()(t_hr),
                cs_y.derivative()(t_hr),
                cs_z.derivative()(t_hr)
            ])
            
            return pos, vel
            
        except Exception:
            n = len(t_hr)
            return np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    
    def _extract_prn(self, filename: str, con_id: str) -> int:
        """Extract PRN number from filename."""
        # Pattern: ...R08... or ...G12...
        import re
        pattern = rf'{con_id}(\d{{2}})'
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return 0
    
    def _extract_leo_id(self, filename: str) -> str:
        """Extract LEO satellite ID from filename."""
        # Pattern: C2E1 or C2E2 etc.
        import re
        match = re.search(r'(C2[A-Z]\d)', filename)
        if match:
            return match.group(1)
        return "C2XX"


# ============================================================================
# STEP 2: OCCULTATION GEOMETRY
# ============================================================================

class OccultationGeometry:
    """
    Compute geometric quantities for LEO-GNSS occultation.
    
    Key quantities:
        - Impact parameter: perpendicular distance from Earth center to ray
        - Tangent point: closest point on ray to Earth center
        - Straight-line (geometric) path delay and Doppler
    """
    
    def __init__(self, event: OccultationEvent):
        self.event = event
        self.c = SPEED_OF_LIGHT
    
    def compute_impact_parameter(self) -> np.ndarray:
        """
        Compute impact parameter for each observation.
        
        Impact parameter a = |r_leo × L_hat| where L = r_gnss - r_leo
        This is the perpendicular distance from Earth center to the ray.
        """
        r_leo = self.event.leo_pos  # (N, 3)
        r_gnss = self.event.gnss_pos  # (N, 3)
        
        # Ray vector
        L = r_gnss - r_leo
        L_norm = np.linalg.norm(L, axis=1, keepdims=True)
        L_hat = L / (L_norm + 1e-10)
        
        # Impact parameter = |r_leo × L_hat|
        cross = np.cross(r_leo, L_hat)
        impact = np.linalg.norm(cross, axis=1)
        
        return impact
    
    def compute_tangent_point(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute tangent point coordinates for each observation.
        
        Tangent point is closest point on LEO-GNSS ray to Earth center.
        
        Returns:
            latitude (deg), longitude (deg), height (m)
        """
        r_leo = self.event.leo_pos
        r_gnss = self.event.gnss_pos
        
        # Ray direction
        L = r_gnss - r_leo
        L_norm = np.linalg.norm(L, axis=1, keepdims=True)
        L_hat = L / (L_norm + 1e-10)
        
        # Parameter t for closest point: t = -dot(r_leo, L_hat)
        t = -np.sum(r_leo * L_hat, axis=1, keepdims=True)
        
        # Tangent point in ECEF
        r_tp = r_leo + t * L_hat
        
        # Convert to geodetic
        lat = np.zeros(len(r_tp))
        lon = np.zeros(len(r_tp))
        height = np.zeros(len(r_tp))
        
        for i, r in enumerate(r_tp):
            lat[i], lon[i], height[i] = ecef_to_geodetic(r[0], r[1], r[2])
        
        return lat, lon, height
    
    def compute_geometric_doppler(self, frequency: float) -> np.ndarray:
        """
        Compute geometric (vacuum) Doppler shift.
        
        This is the Doppler shift expected from pure geometry,
        without any atmospheric effects.
        
        Doppler_geo = -f/c * d(range)/dt
        """
        r_leo = self.event.leo_pos
        v_leo = self.event.leo_vel
        r_gnss = self.event.gnss_pos
        v_gnss = self.event.gnss_vel
        
        # Range vector
        L = r_gnss - r_leo
        L_norm = np.linalg.norm(L, axis=1)
        L_hat = L / (L_norm[:, np.newaxis] + 1e-10)
        
        # Range rate = projection of relative velocity onto LOS
        v_rel = v_gnss - v_leo
        range_rate = np.sum(v_rel * L_hat, axis=1)
        
        # Geometric Doppler
        doppler_geo = -frequency * range_rate / self.c
        
        return doppler_geo
    
    def compute_straight_line_range(self) -> np.ndarray:
        """Compute straight-line range between LEO and GNSS."""
        L = self.event.gnss_pos - self.event.leo_pos
        return np.linalg.norm(L, axis=1)


# ============================================================================
# STEP 3: BENDING ANGLE RETRIEVAL
# ============================================================================

class BendingAngleRetriever:
    """
    Retrieve bending angle from excess phase using Doppler method.
    
    Theory:
        Excess Doppler = d(excess_phase)/dt / wavelength
        Atmospheric Doppler = Excess Doppler - Geometric Doppler
        Bending angle solved from atmospheric Doppler using ray geometry
    
    Ionospheric correction:
        α_neutral = (f1²·α1 - f2²·α2) / (f1² - f2²)
    """
    def __init__(self, event: OccultationEvent, config: PipelineConfig):
        self.event = event
        self.config = config
        self.c = SPEED_OF_LIGHT
        
        # Get frequencies
        con = event.constellation
        freqs = GNSS_FREQUENCIES.get(con, GNSS_FREQUENCIES['G'])
        self.f1 = freqs.get('L1', freqs.get('B1', 1575.42e6))
        self.f2 = freqs.get('L2', freqs.get('B2', 1227.60e6))

    
    def compute_excess_doppler(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute excess Doppler from excess phase derivative.
        
        Doppler = (1/λ) * d(phase)/dt [Hz]
        
        Uses smoothing to reduce noise in derivative.
        """
        time = self.event.time_hr
        ex_L1 = self.event.excess_L1
        ex_L2 = self.event.excess_L2
        
        # Smooth excess phase before differentiation
        if self.config.smoothing_window > 1:
            ex_L1_smooth = uniform_filter1d(ex_L1, self.config.smoothing_window)
            ex_L2_smooth = uniform_filter1d(ex_L2, self.config.smoothing_window)
        else:
            ex_L1_smooth = ex_L1
            ex_L2_smooth = ex_L2
        
        # Time derivative (m/s)
        dt = np.gradient(time)
        dex_L1_dt = np.gradient(ex_L1_smooth) / dt
        dex_L2_dt = np.gradient(ex_L2_smooth) / dt
        
        # Convert to Doppler (Hz)
        # Doppler = (1/λ) * d(phase)/dt but phase is in meters, so
        # Doppler = d(phase)/dt / λ = dex_dt * f / c
        doppler_L1 = dex_L1_dt * self.f1 / self.c
        doppler_L2 = dex_L2_dt * self.f2 / self.c
        
        return doppler_L1, doppler_L2
    
    def compute_atmospheric_doppler(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute atmospheric Doppler by subtracting geometric component.
        
        Atmospheric = Excess - Geometric
        """
        geom = OccultationGeometry(self.event)
        
        # Excess Doppler from phase
        excess_L1, excess_L2 = self.compute_excess_doppler()
        
        # Geometric Doppler
        geo_L1 = geom.compute_geometric_doppler(self.f1)
        geo_L2 = geom.compute_geometric_doppler(self.f2)
        
        # Atmospheric component
        atm_L1 = excess_L1 - geo_L1
        atm_L2 = excess_L2 - geo_L2
        
        return atm_L1, atm_L2
    
    def doppler_to_bending(
        self,
        atm_doppler: np.ndarray,
        frequency: float
    ) -> np.ndarray:
        """
        Convert atmospheric Doppler to bending angle.
        
        Uses simplified relationship for LEO geometry:
            α ≈ -λ * f_atm / v_perp
        
        where v_perp is the cross-track velocity component.
        
        More accurate: solve full ray-tracing equations.
        """
        r_leo = self.event.leo_pos
        v_leo = self.event.leo_vel
        r_gnss = self.event.gnss_pos
        v_gnss = self.event.gnss_vel
        
        wavelength = self.c / frequency
        
        # Occultation plane normal
        L = r_gnss - r_leo
        occ_normal = np.cross(r_leo, L)
        occ_normal_mag = np.linalg.norm(occ_normal, axis=1, keepdims=True)
        occ_normal_hat = occ_normal / (occ_normal_mag + 1e-10)
        
        # Perpendicular velocity component
        v_rel = v_gnss - v_leo
        
        # Project out the component along occ plane normal
        L_hat = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-10)
        
        # Effective velocity for bending conversion
        # This is the velocity component perpendicular to the ray
        v_perp = np.sqrt(
            np.sum(v_rel**2, axis=1) - np.sum(v_rel * L_hat, axis=1)**2
        )
        
        # Bending angle (simplified formula)
        # More rigorous: solve ray equations as in ground-based code
        bending = -wavelength * atm_doppler / (v_perp + 1e-10)
        
        return np.abs(bending)
    
    def doppler_to_bending_rigorous(
        self,
        atm_doppler: np.ndarray,
        frequency: float
    ) -> np.ndarray:
        """
        Rigorous bending angle computation using ray-tracing.
        
        Solves for transmitter and receiver deflection angles
        that satisfy both Doppler and impact parameter constraints.
        """
        r_leo = self.event.leo_pos
        v_leo = self.event.leo_vel
        r_gnss = self.event.gnss_pos
        v_gnss = self.event.gnss_vel
        
        wavelength = self.c / frequency
        n_obs = len(atm_doppler)
        bending = np.zeros(n_obs)
        
        for i in range(n_obs):
            if np.isnan(atm_doppler[i]):
                bending[i] = np.nan
                continue
            
            r_t = r_gnss[i]
            v_t = v_gnss[i]
            r_r = r_leo[i]
            v_r = v_leo[i]
            
            # Straight-line direction
            L = r_r - r_t
            dist = np.linalg.norm(L)
            k0 = L / dist
            
            # Occultation plane
            normal = np.cross(r_t, r_r)
            norm_mag = np.linalg.norm(normal)
            if norm_mag < 1e-6:
                bending[i] = np.nan
                continue
            axis = normal / norm_mag
            
            def rotate_vec(vec, k, theta):
                """Rodrigues rotation."""
                return (vec * np.cos(theta) +
                        np.cross(k, vec) * np.sin(theta) +
                        k * np.dot(k, vec) * (1 - np.cos(theta)))
            
            def equations(vars):
                dt, dr = vars
                
                # Rotated ray directions
                k_t = rotate_vec(k0, axis, -dt)
                k_r = rotate_vec(k0, axis, dr)
                
                # Doppler equation
                lhs = wavelength * atm_doppler[i]
                rhs = np.dot(v_t, (k_t - k0)) - np.dot(v_r, (k_r - k0))
                eq1 = lhs - rhs
                
                # Impact parameter equality
                a_t = np.linalg.norm(np.cross(r_t, k_t))
                a_r = np.linalg.norm(np.cross(r_r, k_r))
                eq2 = a_t - a_r
                
                return [eq1, eq2]
            
            try:
                dt_sol, dr_sol = fsolve(equations, [1e-5, 1e-4], full_output=False)
                bending[i] = np.abs(dt_sol) + np.abs(dr_sol)
            except Exception:
                bending[i] = np.nan
        
        return bending
    
    def apply_ionospheric_correction(
        self,
        bending_L1: np.ndarray,
        bending_L2: np.ndarray,
        impact_L1: np.ndarray,
        impact_L2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dual-frequency ionospheric correction.
        
        Linear combination:
            α_neutral = (f1²·α1 - f2²·α2) / (f1² - f2²)
        
        Requires interpolation to common impact parameter grid.
        """
        # Coefficients
        c1 = self.f1**2 / (self.f1**2 - self.f2**2)
        c2 = self.f2**2 / (self.f1**2 - self.f2**2)
        
        # Sort by impact parameter
        idx1 = np.argsort(impact_L1)
        idx2 = np.argsort(impact_L2)
        
        a1_sorted = impact_L1[idx1]
        a2_sorted = impact_L2[idx2]
        b1_sorted = bending_L1[idx1]
        b2_sorted = bending_L2[idx2]
        
        # Remove NaN
        valid1 = ~np.isnan(b1_sorted) & ~np.isnan(a1_sorted)
        valid2 = ~np.isnan(b2_sorted) & ~np.isnan(a2_sorted)
        
        a1_v = a1_sorted[valid1]
        b1_v = b1_sorted[valid1]
        a2_v = a2_sorted[valid2]
        b2_v = b2_sorted[valid2]
        
        if len(a1_v) < 10 or len(a2_v) < 10:
            return impact_L1, bending_L1  # Return uncorrected
        
        # Interpolate L2 onto L1 grid
        try:
            interp_L2 = interp1d(
                a2_v, b2_v, kind='linear',
                bounds_error=False, fill_value=np.nan
            )
            b2_on_L1 = interp_L2(a1_v)
            
            # Ionospheric correction
            bending_neutral = c1 * b1_v - c2 * b2_on_L1
            
            # Filter out negative and extreme values
            valid = (bending_neutral > 0) & (bending_neutral < 0.1)
            
            return a1_v[valid], bending_neutral[valid]
            
        except Exception:
            return a1_v, b1_v
    
    def retrieve(self, rigorous: bool = False) -> Optional[BendingProfile]:
        """
        Execute full bending angle retrieval with consistent array shapes.
        """
        # 1. Vertical Crop: Focus only on the relevant atmosphere (-20km to 120km)
        # This resolves the broadcast error by ensuring all arrays start at the same length
        valid_height_mask = (self.event.occ_height > -20.0) & (self.event.occ_height < 120.0)
        
        if np.sum(valid_height_mask) < 500:
            print(f"Error: Insufficient data in height window. Found: {np.sum(valid_height_mask)}")
            return None

        # 2. Apply the mask to ALL input data immediately
        time_c = self.event.time_hr[valid_height_mask]
        exL1_c = self.event.excess_L1[valid_height_mask]
        exL2_c = self.event.excess_L2[valid_height_mask]
        snr1_c = self.event.snr_L1[valid_height_mask]
        snr2_c = self.event.snr_L2[valid_height_mask]
        
        # 3. Apply the mask to Orbit/Geometry data
        # This was the likely source of the (9772,) vs (30900,) mismatch
        leo_pos_c = self.event.leo_pos[valid_height_mask]
        leo_vel_c = self.event.leo_vel[valid_height_mask]
        gnss_pos_c = self.event.gnss_pos[valid_height_mask]
        gnss_vel_c = self.event.gnss_vel[valid_height_mask]

        # 4. Compute Doppler on cropped segment
        dt = np.gradient(time_c)
        # Smooth with a window of 21 for 50/100Hz data
        doppler_L1 = (np.gradient(uniform_filter1d(exL1_c, 21)) / dt) * (self.f1 / self.c)
        doppler_L2 = (np.gradient(uniform_filter1d(exL2_c, 21)) / dt) * (self.f2 / self.c)

        # 5. Compute geometric quantities for the CROPPED data
        # Temporarily swap event positions to use the OccultationGeometry logic on cropped arrays
        # Or more simply, compute impact parameter directly:
        L = gnss_pos_c - leo_pos_c
        L_hat = L / np.linalg.norm(L, axis=1, keepdims=True)
        impact_c = np.linalg.norm(np.cross(leo_pos_c, L_hat), axis=1)

        # 6. Convert Doppler to Bending
        # We manually compute bending for the cropped segment to keep shapes aligned
        wavelength1 = self.c / self.f1
        v_rel = gnss_vel_c - leo_vel_c
        v_perp = np.sqrt(np.sum(v_rel**2, axis=1) - np.sum(v_rel * L_hat, axis=1)**2)
        
        bending_L1 = np.abs(-wavelength1 * doppler_L1 / (v_perp + 1e-10))
        
        # 7. Final Quality Filter (Positive bending and SNR)
        final_mask = (snr1_c > self.config.min_snr) & (bending_L1 > 0) & (bending_L1 < 0.05)
        
        if np.sum(final_mask) < 100:
            print(f"Failed: Only {np.sum(final_mask)} points passed final SNR/Physicality filters")
            return None

        # 8. Ionospheric Correction (L1/L2)
        # For simplicity in this step, we'll use the same impact grid
        c1 = self.f1**2 / (self.f1**2 - self.f2**2)
        c2 = self.f2**2 / (self.f1**2 - self.f2**2)
        
        # Compute L2 bending for the final mask
        wavelength2 = self.c / self.f2
        bending_L2 = np.abs(-wavelength2 * doppler_L2 / (v_perp + 1e-10))
        
        alpha_neutral = c1 * bending_L1[final_mask] - c2 * bending_L2[final_mask]
        a_neutral = impact_c[final_mask]

        # 9. Extract Tangent Point Coordinates
        geom = OccultationGeometry(self.event)
        lat, lon, _ = geom.compute_tangent_point()
        lat_final = lat[valid_height_mask][final_mask]
        lon_final = lon[valid_height_mask][final_mask]

        R_g = gaussian_radius(np.mean(lat_final))
        
        return BendingProfile(
            impact_param=a_neutral,
            tangent_height=(a_neutral - R_g) / 1000.0,
            bending_L1=bending_L1[final_mask],
            bending_L2=bending_L2[final_mask],
            bending_neutral=alpha_neutral,
            latitude=lat_final,
            longitude=lon_final,
            time=time_c[final_mask]
        )


# ============================================================================
# STEP 4: ABEL INVERSION
# ============================================================================

class AbelInversion:
    """
    Inverse Abel transform: bending angle → refractivity.
    
    Theory:
        ln(n(a)) = (1/π) ∫[a→∞] α(a') / √(a'² - a²) da'
    
    Where:
        n = refractive index
        a = impact parameter
        α = bending angle
    
    Implementation uses statistical optimization to blend measurements
    with exponential climatology at high altitudes.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.R_earth = R_EARTH
    
    def statistical_optimization(
        self,
        impact: np.ndarray,
        bending: np.ndarray,
        height_km: np.ndarray
    ) -> np.ndarray:
        """
        Blend measurements with exponential climatology above threshold.
        
        Uses weighted average based on measurement uncertainty and
        climatology confidence.
        """
        # Fit exponential to mid-atmosphere (20-40 km)
        fit_mask = (height_km >= 20.0) & (height_km <= 40.0) & (bending > 0)
        
        if np.sum(fit_mask) < 10:
            # Not enough data for fit, return original
            return bending.copy()
        
        a_fit = impact[fit_mask]
        alpha_fit = bending[fit_mask]
        
        # Linear fit to log(bending) vs impact
        try:
            log_alpha = np.log(alpha_fit + 1e-12)
            coeffs = np.polyfit(a_fit, log_alpha, 1)
            
            # Scale height from slope
            if coeffs[0] < 0:
                H_scale = -1.0 / coeffs[0]
            else:
                H_scale = 7000.0  # Default ~7 km scale height
            
            # Reference values
            a_ref = np.median(a_fit)
            alpha_ref = np.exp(np.polyval(coeffs, a_ref))
            
        except Exception:
            H_scale = 7000.0
            a_ref = np.median(impact)
            alpha_ref = np.median(bending[bending > 0])
        
        # Climatology model
        alpha_clim = alpha_ref * np.exp(-(impact - a_ref) / H_scale)
        
        # Blending weights
        sigma_meas = np.full_like(bending, 1e-7)  # Measurement uncertainty
        sigma_clim = np.full_like(bending, 1e10)  # Very uncertain below blend height
        
        # Above blend height, trust climatology more
        above_blend = height_km > self.config.climatology_blend_km
        sigma_clim[above_blend] = 0.05 * alpha_clim[above_blend]
        
        # Weighted average
        w_meas = 1.0 / sigma_meas**2
        w_clim = 1.0 / sigma_clim**2
        
        alpha_opt = (bending * w_meas + alpha_clim * w_clim) / (w_meas + w_clim)
        
        return alpha_opt
    
    def inverse_abel_transform(
        self,
        impact: np.ndarray,
        bending: np.ndarray
    ) -> np.ndarray:
        """
        Compute inverse Abel transform.
        
        ln(n(a)) = (1/π) ∫[a→∞] α(a') / √(a'² - a²) da'
        
        Uses numerical integration with singularity handling.
        """
        n_levels = len(impact)
        ln_n = np.zeros(n_levels)
        
        for i in range(n_levels):
            a_i = impact[i]
            
            # Skip lowest few levels for numerical stability
            start_idx = min(i + 3, n_levels - 1)
            a_start = impact[start_idx]
            
            # Analytical term near singularity
            if start_idx < n_levels:
                term1 = bending[start_idx] * np.log(
                    a_start + np.sqrt(a_start**2 - a_i**2 + 1e-10)
                )
                term2 = bending[i] * np.log(a_i + 1e-10)
                analytical = term1 - term2
            else:
                analytical = 0.0
            
            # Integration by parts term
            parts_integral = 0.0
            for j in range(i, min(start_idx, n_levels - 1)):
                a_mid = 0.5 * (impact[j] + impact[j + 1])
                if a_mid > a_i:
                    d_alpha = bending[j + 1] - bending[j]
                    sqrt_term = np.sqrt(a_mid**2 - a_i**2)
                    parts_integral -= np.log(a_mid + sqrt_term) * d_alpha
            
            # Regular numerical integration
            regular_integral = 0.0
            for j in range(start_idx, n_levels - 1):
                a_mid = 0.5 * (impact[j] + impact[j + 1])
                alpha_mid = 0.5 * (bending[j] + bending[j + 1])
                da = impact[j + 1] - impact[j]
                
                denom_sq = a_mid**2 - a_i**2
                if denom_sq > 0:
                    regular_integral += alpha_mid / np.sqrt(denom_sq) * da
            
            ln_n[i] = (1.0 / np.pi) * (analytical + parts_integral + regular_integral)
        
        return ln_n
    
    def compute_refractivity(
        self,
        impact: np.ndarray,
        ln_n: np.ndarray,
        latitude: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert refractive index to refractivity and geometric height.
        
        N = (n - 1) × 10^6
        height = a/n - R_local(lat)
        """
        n = np.exp(ln_n)
        
        # Height with latitude-varying radius
        height = np.zeros_like(impact)
        for i in range(len(impact)):
            R_local = gaussian_radius(latitude[i])
            r_tangent = impact[i] / n[i]
            height[i] = (r_tangent - R_local) / 1000.0  # km
        
        # Refractivity
        N = (n - 1.0) * 1e6
        
        return height, N
    
    def run(self, profile: BendingProfile) -> RefractivityProfile:
        """Execute Abel inversion on bending angle profile."""
        impact = profile.impact_param
        bending = profile.bending_neutral
        height_km = profile.tangent_height
        latitude = profile.latitude
        longitude = profile.longitude
        
        # CRITICAL: Sort by ascending impact parameter for Abel integral
        sort_idx = np.argsort(impact)
        impact = impact[sort_idx]
        bending = bending[sort_idx]
        height_km = height_km[sort_idx]
        latitude = latitude[sort_idx]
        longitude = longitude[sort_idx]
        
        # Filter to valid atmospheric range before Abel inversion
        atm_mask = (height_km >= self.config.min_height_km) & \
                   (height_km <= self.config.abel_top_km) & \
                   (bending > self.config.min_bending_rad) & \
                   (bending < self.config.max_bending_rad)
        
        if np.sum(atm_mask) < 50:
            raise ValueError(f"Insufficient valid points for Abel inversion: {np.sum(atm_mask)}")
        
        impact = impact[atm_mask]
        bending = bending[atm_mask]
        height_km = height_km[atm_mask]
        latitude = latitude[atm_mask]
        longitude = longitude[atm_mask]
        
        # Statistical optimization
        bending_opt = self.statistical_optimization(impact, bending, height_km)
        
        # Inverse Abel transform
        ln_n = self.inverse_abel_transform(impact, bending_opt)
        
        # Convert to refractivity
        height_out, N_out = self.compute_refractivity(impact, ln_n, latitude)
        
        # Filter valid output range
        valid = (height_out >= self.config.min_height_km) & \
                (height_out <= self.config.max_height_km) & \
                (N_out > 0) & (N_out < 500) & \
                ~np.isnan(N_out) & ~np.isinf(N_out)
        
        if np.sum(valid) == 0:
            raise ValueError("No valid refractivity points after Abel inversion")
        
        return RefractivityProfile(
            height=height_out[valid],
            refractivity=N_out[valid],
            impact_param=impact[valid],
            bending=bending_opt[valid],
            latitude=latitude[valid],
            longitude=longitude[valid]
        )


# ============================================================================
# STEP 5: VALIDATION AGAINST LEVEL 2 PRODUCTS
# ============================================================================

class Level2Validator:
    """
    Compare retrieved profiles against COSMIC-2 Level 2 products.
    
    Products:
        - atmPrf: Bending angle and refractivity (Abel inversion)
        - wetPf2: 1D-Var retrieval (T, P, q)
    """
    
    def __init__(self):
        pass
    
    def load_atmPrf(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load atmPrf (bending/refractivity) validation data."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required")
        
        ds = xr.open_dataset(filepath)
        
        data = {
            'impact_height': ds['Impact_height'].values,  # km
            'bending': ds['Bend_ang'].values,  # rad
            'bending_L1': ds['Bend_ang1'].values if 'Bend_ang1' in ds else None,
            'bending_L2': ds['Bend_ang2'].values if 'Bend_ang2' in ds else None,
            'refractivity': ds['Ref'].values,  # N-units
            'latitude': ds['Lat'].values,
            'longitude': ds['Lon'].values,
            'pressure': ds['Pres'].values if 'Pres' in ds else None,
            'temperature': ds['Temp'].values if 'Temp' in ds else None,
        }
        
        ds.close()
        return data
    
    def load_wetPf2(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load wetPf2 (1D-Var) validation data."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required")
        
        ds = xr.open_dataset(filepath)
        
        data = {
            'height': ds['gph'].values,  # km (geopotential height)
            'temperature': ds['Temp'].values,  # Celsius
            'pressure': ds['Pres'].values,  # hPa
            'vapor_pressure': ds['Vp'].values,  # hPa
            'refractivity': ds['ref'].values,  # N-units
            'latitude': ds['lat'].values,
            'longitude': ds['lon'].values,
        }
        
        ds.close()
        return data
    
    def compare_bending(
        self,
        retrieved: BendingProfile,
        reference: Dict[str, np.ndarray]
    ) -> ValidationResult:
        """Compare retrieved bending angles against atmPrf."""
        # Reference data
        h_ref = reference['impact_height']  # km
        b_ref = reference['bending']  # rad
        
        # Remove NaN
        valid_ref = ~np.isnan(b_ref) & ~np.isnan(h_ref) & (b_ref > 0)
        h_ref = h_ref[valid_ref]
        b_ref = b_ref[valid_ref]
        
        # Retrieved data
        h_ret = retrieved.tangent_height
        b_ret = retrieved.bending_neutral
        
        valid_ret = ~np.isnan(b_ret) & (b_ret > 0)
        h_ret = h_ret[valid_ret]
        b_ret = b_ret[valid_ret]
        
        if len(h_ref) < 10 or len(h_ret) < 10:
            return ValidationResult(
                height_common=np.array([]),
                retrieved=np.array([]),
                reference=np.array([]),
                difference=np.array([]),
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=0
            )
        
        # Interpolate to common grid
        h_min = max(h_ref.min(), h_ret.min())
        h_max = min(h_ref.max(), h_ret.max())
        h_common = np.linspace(h_min, h_max, 200)
        
        try:
            interp_ref = interp1d(h_ref, b_ref, bounds_error=False, fill_value=np.nan)
            interp_ret = interp1d(h_ret, b_ret, bounds_error=False, fill_value=np.nan)
            
            b_ref_interp = interp_ref(h_common)
            b_ret_interp = interp_ret(h_common)
            
            valid = ~np.isnan(b_ref_interp) & ~np.isnan(b_ret_interp)
            h_c = h_common[valid]
            b_ref_c = b_ref_interp[valid]
            b_ret_c = b_ret_interp[valid]
            
        except Exception:
            return ValidationResult(
                height_common=np.array([]),
                retrieved=np.array([]),
                reference=np.array([]),
                difference=np.array([]),
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=0
            )
        
        if len(h_c) < 10:
            return ValidationResult(
                height_common=h_c, retrieved=b_ret_c, reference=b_ref_c,
                difference=b_ret_c - b_ref_c,
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=len(h_c)
            )
        
        diff = b_ret_c - b_ref_c
        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)
        corr = np.corrcoef(b_ret_c, b_ref_c)[0, 1]
        
        return ValidationResult(
            height_common=h_c,
            retrieved=b_ret_c,
            reference=b_ref_c,
            difference=diff,
            rmse=rmse,
            bias=bias,
            correlation=corr,
            n_points=len(h_c)
        )
    
    def compare_refractivity(
        self,
        retrieved: RefractivityProfile,
        reference: Dict[str, np.ndarray],
        source: str = 'atmPrf'
    ) -> ValidationResult:
        """Compare retrieved refractivity against atmPrf or wetPf2."""
        # Reference data
        if source == 'atmPrf':
            h_ref = reference['impact_height']  # Use impact height as proxy
        else:
            h_ref = reference['height']
        
        N_ref = reference['refractivity']
        
        valid_ref = ~np.isnan(N_ref) & ~np.isnan(h_ref) & (N_ref > 0)
        h_ref = h_ref[valid_ref]
        N_ref = N_ref[valid_ref]
        
        # Retrieved
        h_ret = retrieved.height
        N_ret = retrieved.refractivity
        
        valid_ret = ~np.isnan(N_ret) & (N_ret > 0)
        h_ret = h_ret[valid_ret]
        N_ret = N_ret[valid_ret]
        
        if len(h_ref) < 10 or len(h_ret) < 10:
            return ValidationResult(
                height_common=np.array([]),
                retrieved=np.array([]),
                reference=np.array([]),
                difference=np.array([]),
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=0
            )
        
        # Common height grid
        h_min = max(h_ref.min(), h_ret.min())
        h_max = min(h_ref.max(), h_ret.max())
        h_common = np.linspace(h_min, h_max, 200)
        
        try:
            interp_ref = interp1d(h_ref, N_ref, bounds_error=False, fill_value=np.nan)
            interp_ret = interp1d(h_ret, N_ret, bounds_error=False, fill_value=np.nan)
            
            N_ref_interp = interp_ref(h_common)
            N_ret_interp = interp_ret(h_common)
            
            valid = ~np.isnan(N_ref_interp) & ~np.isnan(N_ret_interp)
            h_c = h_common[valid]
            N_ref_c = N_ref_interp[valid]
            N_ret_c = N_ret_interp[valid]
            
        except Exception:
            return ValidationResult(
                height_common=np.array([]),
                retrieved=np.array([]),
                reference=np.array([]),
                difference=np.array([]),
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=0
            )
        
        if len(h_c) < 10:
            return ValidationResult(
                height_common=h_c, retrieved=N_ret_c, reference=N_ref_c,
                difference=N_ret_c - N_ref_c,
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=len(h_c)
            )
        
        diff = N_ret_c - N_ref_c
        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)
        corr = np.corrcoef(N_ret_c, N_ref_c)[0, 1]
        
        # Fractional RMSE (percentage)
        frac_rmse = 100.0 * rmse / np.mean(N_ref_c)
        
        return ValidationResult(
            height_common=h_c,
            retrieved=N_ret_c,
            reference=N_ref_c,
            difference=diff,
            rmse=rmse,
            bias=bias,
            correlation=corr,
            n_points=len(h_c)
        )


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class LEOROPipeline:
    """
    Complete LEO GNSS Radio Occultation Processing Pipeline.
    
    Usage:
        pipeline = LEOROPipeline()
        
        # Optional: load precise orbits
        pipeline.load_leoorb('./leoOrb_2025.001/')
        
        results = pipeline.process_event(
            conphs_file='conPhs_C2E1.2025.001.00.08.R08_0001.0001_nc',
            atmprf_file='atmPrf_C2E1.2025.001.00.08.R08_0001.0001_nc',  # optional
            wetpf2_file='wetPf2_C2E1.2025.001.00.08.R08_0001.0001_nc'   # optional
        )
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.leoorb_reader: Optional[LeoOrbReader] = None
        self.reader: Optional[ConPhsReader] = None
        self.validator = Level2Validator()
    
    def load_leoorb(self, leoorb_dir: str) -> int:
        """
        Load precise LEO orbit data from leoOrb directory.
        
        Parameters:
            leoorb_dir: Directory containing leoOrb_*.nc files
        
        Returns:
            Number of satellites loaded
        """
        self.leoorb_reader = LeoOrbReader()
        self.leoorb_reader.load_directory(leoorb_dir)
        self.reader = ConPhsReader(self.leoorb_reader)
        return len(self.leoorb_reader.orbits)
    
    def _get_reader(self) -> ConPhsReader:
        """Get or create ConPhsReader."""
        if self.reader is None:
            self.reader = ConPhsReader(self.leoorb_reader)
        return self.reader
    
    def process_event(
        self,
        conphs_file: str,
        atmprf_file: Optional[str] = None,
        wetpf2_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        rigorous_bending: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single occultation event.
        
        Parameters:
            conphs_file: Path to conPhs NetCDF file
            atmprf_file: Path to atmPrf NetCDF file (for validation)
            wetpf2_file: Path to wetPf2 NetCDF file (for validation)
            output_dir: Directory to save output CSV files
            rigorous_bending: Use rigorous ray-tracing (slower)
        
        Returns:
            Dictionary with processing results and validation metrics
        """
        results = {
            'event_id': None,
            'success': False,
            'bending_profile': None,
            'refractivity_profile': None,
            'validation_bending': None,
            'validation_refractivity_atmPrf': None,
            'validation_refractivity_wetPf2': None,
            'messages': []
        }
        
        # Step 1: Load conPhs
        try:
            reader = self._get_reader()
            event = reader.load_file(conphs_file)
            if event is None:
                results['messages'].append("Failed to load conPhs file")
                return results
            results['event_id'] = event.event_id
            
            # Note orbit source
            orbit_source = "leoOrb" if (self.leoorb_reader and 
                event.leo_id in self.leoorb_reader.orbits) else "conPhs"
            results['messages'].append(
                f"Loaded event {event.event_id}: {event.n_obs} obs, LEO orbit from {orbit_source}"
            )
        except Exception as e:
            results['messages'].append(f"Error loading conPhs: {e}")
            return results
        
        # Step 2: Retrieve bending angles
        try:
            retriever = BendingAngleRetriever(event, self.config)
            bending_profile = retriever.retrieve(rigorous=rigorous_bending)
            
            if bending_profile is None:
                results['messages'].append("Bending angle retrieval failed: insufficient valid data")
                return results
            
            results['bending_profile'] = bending_profile
            results['messages'].append(
                f"Retrieved bending: {len(bending_profile.impact_param)} points, "
                f"{bending_profile.tangent_height.min():.1f}-{bending_profile.tangent_height.max():.1f} km"
            )
        except Exception as e:
            results['messages'].append(f"Error in bending retrieval: {e}")
            return results
        
        # Step 3: Abel inversion
        try:
            abel = AbelInversion(self.config)
            refrac_profile = abel.run(bending_profile)
            
            results['refractivity_profile'] = refrac_profile
            results['messages'].append(
                f"Retrieved refractivity: {len(refrac_profile.height)} points, "
                f"{refrac_profile.height.min():.1f}-{refrac_profile.height.max():.1f} km"
            )
        except Exception as e:
            results['messages'].append(f"Error in Abel inversion: {e}")
            return results
        
        results['success'] = True
        
        # Step 4: Validation against atmPrf
        if atmprf_file and os.path.exists(atmprf_file):
            try:
                atmPrf_data = self.validator.load_atmPrf(atmprf_file)
                
                # Bending comparison
                val_bending = self.validator.compare_bending(bending_profile, atmPrf_data)
                results['validation_bending'] = val_bending
                results['messages'].append(
                    f"Bending validation: RMSE={val_bending.rmse:.2e} rad, "
                    f"r={val_bending.correlation:.3f}, N={val_bending.n_points}"
                )
                
                # Refractivity comparison
                val_refrac = self.validator.compare_refractivity(
                    refrac_profile, atmPrf_data, source='atmPrf'
                )
                results['validation_refractivity_atmPrf'] = val_refrac
                results['messages'].append(
                    f"Refractivity validation (atmPrf): RMSE={val_refrac.rmse:.2f} N, "
                    f"r={val_refrac.correlation:.3f}"
                )
            except Exception as e:
                results['messages'].append(f"Warning: atmPrf validation failed: {e}")
        
        # Step 5: Validation against wetPf2
        if wetpf2_file and os.path.exists(wetpf2_file):
            try:
                wetPf2_data = self.validator.load_wetPf2(wetpf2_file)
                
                val_refrac = self.validator.compare_refractivity(
                    refrac_profile, wetPf2_data, source='wetPf2'
                )
                results['validation_refractivity_wetPf2'] = val_refrac
                results['messages'].append(
                    f"Refractivity validation (wetPf2): RMSE={val_refrac.rmse:.2f} N, "
                    f"r={val_refrac.correlation:.3f}"
                )
            except Exception as e:
                results['messages'].append(f"Warning: wetPf2 validation failed: {e}")
        
        # Save outputs
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def process_directory(
        self,
        conphs_dir: str,
        level2_dir: Optional[str] = None,
        leoorb_dir: Optional[str] = None,
        output_dir: str = "./output",
        max_events: Optional[int] = None,
        generate_plots: bool = False
    ) -> pd.DataFrame:
        """
        Process all events in a directory.
        
        Parameters:
            conphs_dir: Directory containing conPhs files
            level2_dir: Directory containing atmPrf/wetPf2 files (or None to search in conphs_dir)
            leoorb_dir: Directory containing leoOrb files (optional, for precise orbits)
            output_dir: Output directory
            max_events: Maximum number of events to process
            generate_plots: Whether to generate plots for each event
        
        Returns:
            Summary DataFrame with validation metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load precise orbits if provided
        if leoorb_dir and self.leoorb_reader is None:
            self.load_leoorb(leoorb_dir)
        
        # Find conPhs files (support both .nc and _nc patterns)
        conphs_files = sorted(
            glob.glob(os.path.join(conphs_dir, "conPhs_*.nc")) +
            glob.glob(os.path.join(conphs_dir, "conPhs_*_nc"))
        )
        
        if max_events:
            conphs_files = conphs_files[:max_events]
        
        if not conphs_files:
            print(f"No conPhs files found in {conphs_dir}")
            return pd.DataFrame()
        
        # Determine search directory for Level 2 files
        search_dir = level2_dir if level2_dir else conphs_dir
        
        summary = []
        
        for i, conphs_file in enumerate(conphs_files):
            fname = os.path.basename(conphs_file)
            print(f"\n[{i+1}/{len(conphs_files)}] Processing: {fname}")
            
            # Extract event identifier
            # Pattern: conPhs_{event_id}.nc or conPhs_{event_id}_nc
            event_id = self._extract_event_id(fname)
            
            # Find matching Level 2 files
            atmprf_file = self._find_matching_file(search_dir, 'atmPrf', event_id)
            wetpf2_file = self._find_matching_file(search_dir, 'wetPf2', event_id)
            
            if atmprf_file:
                print(f"  Found atmPrf: {os.path.basename(atmprf_file)}")
            if wetpf2_file:
                print(f"  Found wetPf2: {os.path.basename(wetpf2_file)}")
            
            # Process event - output goes to subdirectory named by event_id
            event_output = os.path.join(output_dir, event_id)
            results = self.process_event(
                conphs_file=conphs_file,
                atmprf_file=atmprf_file,
                wetpf2_file=wetpf2_file,
                output_dir=event_output
            )
            
            # Generate plots if requested and processing succeeded
            if generate_plots and results['success']:
                self._generate_event_plots(
                    conphs_file=conphs_file,
                    results=results,
                    atmprf_file=atmprf_file,
                    wetpf2_file=wetpf2_file,
                    output_dir=event_output,
                    event_id=event_id
                )
            
            # Collect summary stats
            row = {
                'event_id': event_id,
                'success': results['success'],
                'conphs_file': fname,
                'atmprf_found': atmprf_file is not None,
                'wetpf2_found': wetpf2_file is not None
            }
            
            if results['bending_profile'] is not None:
                bp = results['bending_profile']
                row['n_bending_pts'] = len(bp.impact_param)
                row['height_min_km'] = bp.tangent_height.min()
                row['height_max_km'] = bp.tangent_height.max()
            
            if results['validation_bending'] is not None:
                vb = results['validation_bending']
                row['bending_rmse_urad'] = vb.rmse * 1e6
                row['bending_corr'] = vb.correlation
            
            if results['validation_refractivity_atmPrf'] is not None:
                vr = results['validation_refractivity_atmPrf']
                row['refrac_rmse_N'] = vr.rmse
                row['refrac_corr'] = vr.correlation
            
            summary.append(row)
            
            # Print status
            status = "OK" if results['success'] else "FAILED"
            print(f"  Status: {status}")
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'processing_summary.csv'), index=False)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total events: {len(summary_df)}")
        print(f"Successful: {summary_df['success'].sum()}")
        if 'refrac_rmse_N' in summary_df.columns:
            valid = summary_df['refrac_rmse_N'].notna()
            if valid.any():
                print(f"Mean refractivity RMSE: {summary_df.loc[valid, 'refrac_rmse_N'].mean():.2f} N")
        
        return summary_df
    
    def _extract_event_id(self, filename: str) -> str:
        """
        Extract event identifier from filename.
        
        Examples:
            conPhs_C2E2.2025.001.17.18.R01_0001.0001_nc -> C2E2.2025.001.17.18.R01_0001.0001
            conPhs_C2E1.2025.001.00.08.R08_0001.0001.nc -> C2E1.2025.001.00.08.R08_0001.0001
        """
        # Remove prefix
        event_id = filename.replace('conPhs_', '')
        event_id = event_id.replace('atmPrf_', '')
        event_id = event_id.replace('wetPf2_', '')
        
        # Remove suffix
        event_id = event_id.replace('.nc', '')
        event_id = event_id.replace('_nc', '')
        
        return event_id
    
    def _find_matching_file(self, search_dir: str, product: str, event_id: str) -> Optional[str]:
        """
        Find matching Level 2 file for given product type and event ID.
        
        Args:
            search_dir: Directory to search
            product: Product type ('atmPrf' or 'wetPf2')
            event_id: Event identifier
        
        Returns:
            Path to matching file or None
        """
        # Try both naming conventions
        patterns = [
            os.path.join(search_dir, f"{product}_{event_id}.nc"),
            os.path.join(search_dir, f"{product}_{event_id}_nc"),
        ]
        
        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern
        
        return None
    
    def _generate_event_plots(
        self,
        conphs_file: str,
        results: Dict,
        atmprf_file: Optional[str],
        wetpf2_file: Optional[str],
        output_dir: str,
        event_id: str
    ) -> List[str]:
        """Generate plots for a single event."""
        # Reload event for raw data
        reader = self._get_reader()
        event = reader.load_file(conphs_file)
        
        if event is None:
            return []
        
        # Load reference data
        atmPrf_data = None
        wetPf2_data = None
        
        if atmprf_file:
            try:
                atmPrf_data = self.validator.load_atmPrf(atmprf_file)
            except Exception:
                pass
        
        if wetpf2_file:
            try:
                wetPf2_data = self.validator.load_wetPf2(wetpf2_file)
            except Exception:
                pass
        
        # Create plotter and generate plots
        plot_dir = os.path.join(output_dir, 'plots')
        plotter = PipelinePlotter(output_dir=plot_dir)
        
        plot_files = plotter.plot_all(
            event=event,
            bending_profile=results['bending_profile'],
            refrac_profile=results['refractivity_profile'],
            validation_bending=results['validation_bending'],
            validation_refrac_atmPrf=results['validation_refractivity_atmPrf'],
            validation_refrac_wetPf2=results['validation_refractivity_wetPf2'],
            atmPrf_data=atmPrf_data,
            wetPf2_data=wetPf2_data,
            event_id=event_id
        )
        
        print(f"  Generated {len(plot_files)} plots")
        return plot_files
    
    def _save_results(self, results: Dict, output_dir: str) -> None:
        """Save processing results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Bending profile
        if results['bending_profile'] is not None:
            bp = results['bending_profile']
            df = pd.DataFrame({
                'impact_param_m': bp.impact_param,
                'tangent_height_km': bp.tangent_height,
                'bending_L1_rad': bp.bending_L1,
                'bending_L2_rad': bp.bending_L2,
                'bending_neutral_rad': bp.bending_neutral,
                'latitude_deg': bp.latitude,
                'longitude_deg': bp.longitude,
                'time_gps': bp.time
            })
            df.to_csv(os.path.join(output_dir, 'bending_profile.csv'), index=False)
        
        # Refractivity profile
        if results['refractivity_profile'] is not None:
            rp = results['refractivity_profile']
            df = pd.DataFrame({
                'height_km': rp.height,
                'refractivity_N': rp.refractivity,
                'impact_param_m': rp.impact_param,
                'bending_rad': rp.bending,
                'latitude_deg': rp.latitude,
                'longitude_deg': rp.longitude
            })
            df.to_csv(os.path.join(output_dir, 'refractivity_profile.csv'), index=False)
        
        # Validation results
        if results['validation_bending'] is not None:
            vb = results['validation_bending']
            if len(vb.height_common) > 0:
                df = pd.DataFrame({
                    'height_km': vb.height_common,
                    'retrieved_rad': vb.retrieved,
                    'reference_rad': vb.reference,
                    'difference_rad': vb.difference
                })
                df.to_csv(os.path.join(output_dir, 'validation_bending.csv'), index=False)
        
        if results['validation_refractivity_atmPrf'] is not None:
            vr = results['validation_refractivity_atmPrf']
            if len(vr.height_common) > 0:
                df = pd.DataFrame({
                    'height_km': vr.height_common,
                    'retrieved_N': vr.retrieved,
                    'reference_N': vr.reference,
                    'difference_N': vr.difference
                })
                df.to_csv(os.path.join(output_dir, 'validation_refractivity.csv'), index=False)


# ============================================================================
# PLOTTING MODULE
# ============================================================================

class PipelinePlotter:
    """
    Two-panel visualization system for LEO GNSS-RO pipeline.
    
    Panel 1: Raw GNSS Observations (2x2)
        (a) Excess Phase L1/L2 vs Time
        (b) SNR L1/L2 vs Time  
        (c) Excess Phase vs Occultation Height
        (d) Occultation Height vs Time
    
    Panel 2: Derived Atmospheric Profiles (2x2)
        (a) Bending Angle vs Tangent Height (with atmPrf reference)
        (b) Refractivity vs Height (with atmPrf/wetPf2 reference)
        (c) Pressure Profile (with wetPf2 reference)
        (d) Temperature Profile (with wetPf2 reference)
    """
    
    def __init__(self, output_dir: str = "./plots", dpi: int = 150):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dpi = dpi
        
        # Color scheme
        self.colors = {
            'L1': '#1976D2',
            'L2': '#D32F2F',
            'neutral': '#388E3C',
            'retrieved': '#1976D2',
            'atmPrf': '#D32F2F',
            'wetPf2': '#F57C00',
        }
    
    def plot_all(
        self,
        event: OccultationEvent,
        bending_profile: BendingProfile,
        refrac_profile: RefractivityProfile,
        validation_bending: Optional[ValidationResult] = None,
        validation_refrac_atmPrf: Optional[ValidationResult] = None,
        validation_refrac_wetPf2: Optional[ValidationResult] = None,
        atmPrf_data: Optional[Dict[str, np.ndarray]] = None,
        wetPf2_data: Optional[Dict[str, np.ndarray]] = None,
        event_id: str = "event"
    ) -> List[str]:
        """Generate both panel plots."""
        import matplotlib
        matplotlib.use('Agg')
        
        saved_files = []
        
        # Panel 1: Raw GNSS Observations
        path = self._generate_raw_panel(event, event_id)
        saved_files.append(path)
        
        # Panel 2: Derived Profiles
        path = self._generate_derived_panel(
            bending_profile, refrac_profile,
            validation_bending, validation_refrac_atmPrf, validation_refrac_wetPf2,
            atmPrf_data, wetPf2_data, event_id
        )
        saved_files.append(path)
        
        return saved_files
    
    def _generate_raw_panel(self, event: OccultationEvent, event_id: str) -> str:
        """
        Generate Panel 1: Raw GNSS Observations (2x2).
        
        (a) Excess Phase L1/L2 vs Time
        (b) SNR L1/L2 vs Time
        (c) Excess Phase vs Occultation Height
        (d) Occultation Height vs Time
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import timedelta
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Raw GNSS Observations: {event_id}', fontsize=14, fontweight='bold')
        
        # Time array (relative seconds from start)
        t_rel = event.time_hr - event.time_hr[0]
        
        # Convert to datetime for better formatting
        base_time = event.start_time
        t_datetime = [base_time + timedelta(seconds=float(s)) for s in t_rel]
        
        # (a) Excess Phase vs Time
        ax = axes[0, 0]
        ax.plot(t_datetime, event.excess_L1, color=self.colors['L1'], lw=0.8, label='L1', alpha=0.8)
        ax.plot(t_datetime, event.excess_L2, color=self.colors['L2'], lw=0.8, label='L2', alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel('UTC Time')
        ax.set_ylabel('Excess Phase (m)')
        ax.set_title('(a) Excess Phase vs Time', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # (b) SNR vs Time
        ax = axes[0, 1]
        ax.plot(t_datetime, event.snr_L1, color=self.colors['L1'], lw=0.8, label='L1', alpha=0.8)
        ax.plot(t_datetime, event.snr_L2, color=self.colors['L2'], lw=0.8, label='L2', alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel('UTC Time')
        ax.set_ylabel('SNR (V/V)')
        ax.set_title('(b) Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # (c) Excess Phase vs Occultation Height
        ax = axes[1, 0]
        ax.plot(event.excess_L1, event.occ_height, color=self.colors['L1'], lw=0.8, label='L1', alpha=0.8)
        ax.plot(event.excess_L2, event.occ_height, color=self.colors['L2'], lw=0.8, label='L2', alpha=0.8)
        ax.set_xlabel('Excess Phase (m)')
        ax.set_ylabel('Occultation Height (km)')
        ax.set_title('(c) Excess Phase vs Height', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # (d) Occultation Height vs Time
        ax = axes[1, 1]
        ax.scatter(t_datetime, event.occ_height, s=3, alpha=0.6, c='#F57C00')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel('UTC Time')
        ax.set_ylabel('Occultation Height (km)')
        ax.set_title('(d) Occultation Height vs Time', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'{event_id}_panel1_raw.png')
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return filepath
    
    def _generate_derived_panel(
        self,
        bending: BendingProfile,
        refrac: RefractivityProfile,
        val_bending: Optional[ValidationResult],
        val_refrac_atm: Optional[ValidationResult],
        val_refrac_wet: Optional[ValidationResult],
        atmPrf_data: Optional[Dict[str, np.ndarray]],
        wetPf2_data: Optional[Dict[str, np.ndarray]],
        event_id: str
    ) -> str:
        """
        Generate Panel 2: Derived Atmospheric Profiles (2x2).
        
        (a) Bending Angle vs Tangent Height
        (b) Refractivity vs Height
        (c) Pressure Profile
        (d) Temperature Profile
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Derived Atmospheric Profiles: {event_id}', fontsize=14, fontweight='bold')
        
        # (a) Bending Angle Profile
        ax = axes[0, 0]
        # Plot L1, L2, Iono-corrected
        ax.plot(bending.bending_L1 * 1e3, bending.tangent_height, 
                color=self.colors['L1'], lw=1, label='L1', alpha=0.7)
        ax.plot(bending.bending_L2 * 1e3, bending.tangent_height, 
                color=self.colors['L2'], lw=1, ls='--', label='L2', alpha=0.7)
        ax.plot(bending.bending_neutral * 1e3, bending.tangent_height, 
                color=self.colors['neutral'], lw=2, label='Iono-free', alpha=0.9)
        
        # Overlay atmPrf reference if available
        if atmPrf_data is not None and 'bending' in atmPrf_data:
            h_ref = atmPrf_data.get('impact_height', np.array([]))
            b_ref = atmPrf_data.get('bending', np.array([]))
            valid = ~np.isnan(b_ref) & ~np.isnan(h_ref) & (b_ref > 0)
            if np.any(valid):
                ax.plot(b_ref[valid] * 1e3, h_ref[valid], 
                        color=self.colors['atmPrf'], lw=1.5, ls=':', label='atmPrf', alpha=0.8)
        
        ax.set_xlabel('Bending Angle (mrad)')
        ax.set_ylabel('Tangent Height (km)')
        ax.set_title('(a) Bending Angle Profile', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 60])
        
        # Add validation stats if available
        if val_bending and val_bending.n_points > 0:
            stats = f'RMSE: {val_bending.rmse*1e6:.1f} µrad\nr: {val_bending.correlation:.3f}'
            ax.text(0.95, 0.05, stats, transform=ax.transAxes, fontsize=8,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # (b) Refractivity Profile
        ax = axes[0, 1]
        valid_r = (refrac.refractivity > 0) & ~np.isnan(refrac.refractivity)
        ax.plot(refrac.refractivity[valid_r], refrac.height[valid_r], 
                color=self.colors['retrieved'], lw=2, label='Retrieved')
        
        # Overlay atmPrf refractivity
        if atmPrf_data is not None and 'refractivity' in atmPrf_data:
            h_ref = atmPrf_data.get('impact_height', np.array([]))
            n_ref = atmPrf_data.get('refractivity', np.array([]))
            valid = ~np.isnan(n_ref) & ~np.isnan(h_ref) & (n_ref > 0)
            if np.any(valid):
                ax.plot(n_ref[valid], h_ref[valid], 
                        color=self.colors['atmPrf'], lw=1.5, ls='--', label='atmPrf', alpha=0.8)
        
        # Overlay wetPf2 refractivity
        if wetPf2_data is not None and 'refractivity' in wetPf2_data:
            h_ref = wetPf2_data.get('height', np.array([]))
            n_ref = wetPf2_data.get('refractivity', np.array([]))
            valid = ~np.isnan(n_ref) & ~np.isnan(h_ref) & (n_ref > 0)
            if np.any(valid):
                ax.plot(n_ref[valid], h_ref[valid], 
                        color=self.colors['wetPf2'], lw=1.5, ls=':', label='wetPf2', alpha=0.8)
        
        ax.set_xlabel('Refractivity (N-units)')
        ax.set_ylabel('Height (km)')
        ax.set_title('(b) Refractivity Profile', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim([0, 60])
        
        # Add validation stats
        if val_refrac_atm and val_refrac_atm.n_points > 0:
            stats = f'vs atmPrf: RMSE={val_refrac_atm.rmse:.1f}N, r={val_refrac_atm.correlation:.3f}'
            ax.text(0.95, 0.05, stats, transform=ax.transAxes, fontsize=7,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # (c) Pressure Profile
        ax = axes[1, 0]
        has_pressure = False
        
        # Plot wetPf2 pressure and vapor pressure if available
        if wetPf2_data is not None:
            h_ref = wetPf2_data.get('height', np.array([]))
            
            if 'pressure' in wetPf2_data:
                p_ref = wetPf2_data['pressure']
                valid = ~np.isnan(p_ref) & ~np.isnan(h_ref)
                if np.any(valid):
                    ax.plot(p_ref[valid], h_ref[valid], 
                            color=self.colors['wetPf2'], lw=2, label='P (wetPf2)')
                    has_pressure = True
            
            if 'vapor_pressure' in wetPf2_data:
                vp_ref = wetPf2_data['vapor_pressure']
                valid = ~np.isnan(vp_ref) & ~np.isnan(h_ref)
                if np.any(valid):
                    ax.plot(vp_ref[valid], h_ref[valid], 
                            color=self.colors['L1'], lw=2, ls='--', label='Pw (wetPf2)')
                    has_pressure = True
        
        # Plot atmPrf pressure if available
        if atmPrf_data is not None and 'pressure' in atmPrf_data:
            h_ref = atmPrf_data.get('impact_height', np.array([]))
            p_ref = atmPrf_data['pressure']
            if p_ref is not None:
                valid = ~np.isnan(p_ref) & ~np.isnan(h_ref)
                if np.any(valid):
                    ax.plot(p_ref[valid], h_ref[valid], 
                            color=self.colors['atmPrf'], lw=1.5, ls=':', label='P (atmPrf)', alpha=0.7)
                    has_pressure = True
        
        if has_pressure:
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Pressure data\nnot available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_xlabel('Pressure (hPa)')
        ax.set_ylabel('Height (km)')
        ax.set_title('(c) Pressure Profiles (P, Pw)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 40])
        
        # (d) Temperature Profile
        ax = axes[1, 1]
        has_temp = False
        
        # Plot wetPf2 temperature if available
        if wetPf2_data is not None and 'temperature' in wetPf2_data:
            h_ref = wetPf2_data.get('height', np.array([]))
            t_ref = wetPf2_data['temperature']
            valid = ~np.isnan(t_ref) & ~np.isnan(h_ref)
            if np.any(valid):
                # Convert Celsius to Kelvin if needed
                t_plot = t_ref[valid]
                if np.nanmean(t_plot) < 100:  # Likely Celsius
                    t_plot = t_plot + 273.15
                ax.plot(t_plot, h_ref[valid], 
                        color=self.colors['wetPf2'], lw=2, label='T (wetPf2)')
                has_temp = True
        
        # Plot atmPrf temperature if available
        if atmPrf_data is not None and 'temperature' in atmPrf_data:
            h_ref = atmPrf_data.get('impact_height', np.array([]))
            t_ref = atmPrf_data['temperature']
            if t_ref is not None:
                valid = ~np.isnan(t_ref) & ~np.isnan(h_ref)
                if np.any(valid):
                    t_plot = t_ref[valid]
                    if np.nanmean(t_plot) < 100:  # Likely Celsius
                        t_plot = t_plot + 273.15
                    ax.plot(t_plot, h_ref[valid], 
                            color=self.colors['atmPrf'], lw=1.5, ls='--', label='T (atmPrf)', alpha=0.7)
                    has_temp = True
        
        if has_temp:
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Temperature data\nnot available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Height (km)')
        ax.set_title('(d) Temperature Profile', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 40])
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'{event_id}_panel2_derived.png')
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return filepath


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface for LEO RO pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LEO GNSS Radio Occultation Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single event
  python leo_gnss_ro_pipeline.py --conphs conPhs_C2E1.2025.001.00.08.R08_0001.0001_nc \\
                                  --atmprf atmPrf_C2E1.2025.001.00.08.R08_0001.0001_nc \\
                                  --output ./output --plot

  # Process directory with all files together (conPhs, atmPrf, wetPf2)
  python leo_gnss_ro_pipeline.py --data-dir ./ro_data --output ./results --plot

  # Process with separate Level 2 directory
  python leo_gnss_ro_pipeline.py --conphs-dir ./conphs_data \\
                                  --level2-dir ./level2_data \\
                                  --output ./output --plot
        """
    )
    
    # Input options
    parser.add_argument('--conphs', type=str, help='Single conPhs file')
    parser.add_argument('--conphs-dir', type=str, help='Directory of conPhs files')
    parser.add_argument('--data-dir', type=str, 
                        help='Directory containing all files (conPhs, atmPrf, wetPf2)')
    parser.add_argument('--leoorb-dir', type=str, help='Directory of leoOrb files (precise orbits)')
    parser.add_argument('--atmprf', type=str, help='Single atmPrf file (validation)')
    parser.add_argument('--wetpf2', type=str, help='Single wetPf2 file (validation)')
    parser.add_argument('--level2-dir', type=str, help='Directory of Level 2 files')
    
    # Output
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    
    # Processing options
    parser.add_argument('--rigorous', action='store_true', help='Use rigorous ray-tracing')
    parser.add_argument('--max-events', type=int, help='Maximum events to process')
    parser.add_argument('--plot', action='store_true', help='Generate validation plots')
    
    # Config overrides
    parser.add_argument('--min-snr', type=float, default=50.0, help='Minimum SNR')
    parser.add_argument('--min-height', type=float, default=-2.0, help='Min height (km)')
    parser.add_argument('--max-height', type=float, default=60.0, help='Max height (km)')
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig(
        min_snr=args.min_snr,
        min_height_km=args.min_height,
        max_height_km=args.max_height
    )
    
    pipeline = LEOROPipeline(config)
    
    # Load precise orbits if provided
    if args.leoorb_dir:
        n_sats = pipeline.load_leoorb(args.leoorb_dir)
        print(f"Loaded leoOrb data for {n_sats} satellites")
    
    if args.conphs:
        # Single event mode
        results = pipeline.process_event(
            conphs_file=args.conphs,
            atmprf_file=args.atmprf,
            wetpf2_file=args.wetpf2,
            output_dir=args.output,
            rigorous_bending=args.rigorous
        )
        
        print("\n" + "=" * 60)
        print("PROCESSING RESULTS")
        print("=" * 60)
        for msg in results['messages']:
            print(f"  {msg}")
        print(f"\nSuccess: {results['success']}")
        
        # Generate plots if requested
        if args.plot and results['success']:
            print("\n" + "=" * 60)
            print("GENERATING PLOTS")
            print("=" * 60)
            
            # Need to reload event for plotting (contains raw data)
            reader = pipeline._get_reader()
            event = reader.load_file(args.conphs)
            
            # Load reference data for plotting
            atmPrf_data = None
            wetPf2_data = None
            
            if args.atmprf and os.path.exists(args.atmprf):
                try:
                    atmPrf_data = pipeline.validator.load_atmPrf(args.atmprf)
                except Exception as e:
                    print(f"  Warning: Could not load atmPrf for plotting: {e}")
            
            if args.wetpf2 and os.path.exists(args.wetpf2):
                try:
                    wetPf2_data = pipeline.validator.load_wetPf2(args.wetpf2)
                except Exception as e:
                    print(f"  Warning: Could not load wetPf2 for plotting: {e}")
            
            plot_dir = os.path.join(args.output, 'plots')
            plotter = PipelinePlotter(output_dir=plot_dir)
            
            plot_files = plotter.plot_all(
                event=event,
                bending_profile=results['bending_profile'],
                refrac_profile=results['refractivity_profile'],
                validation_bending=results['validation_bending'],
                validation_refrac_atmPrf=results['validation_refractivity_atmPrf'],
                validation_refrac_wetPf2=results['validation_refractivity_wetPf2'],
                atmPrf_data=atmPrf_data,
                wetPf2_data=wetPf2_data,
                event_id=results['event_id']
            )
            
            print(f"  Generated {len(plot_files)} plots:")
            for pf in plot_files:
                print(f"    - {os.path.basename(pf)}")
        
    elif args.data_dir or args.conphs_dir:
        # Directory/batch mode
        # --data-dir: all files in same directory
        # --conphs-dir + --level2-dir: separate directories
        
        if args.data_dir:
            conphs_dir = args.data_dir
            level2_dir = None  # Will search in same directory
        else:
            conphs_dir = args.conphs_dir
            level2_dir = args.level2_dir
        
        summary = pipeline.process_directory(
            conphs_dir=conphs_dir,
            level2_dir=level2_dir,
            leoorb_dir=args.leoorb_dir,
            output_dir=args.output,
            max_events=args.max_events,
            generate_plots=args.plot
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
