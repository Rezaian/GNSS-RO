"""
LEO GNSS Radio Occultation Processing Pipeline (CORRECTED)
===========================================================

Processes COSMIC-2 Level 1b (conPhs) data to derive atmospheric profiles.
Compares results against Level 2 products (atmPrf, wetPf2).

CORRECTIONS APPLIED (2026-01-03):
    - Fixed Doppler-to-bending conversion using Hajj et al. 2002 method
    - Added local center of curvature correction (Syndergaard 1998)
    - Impact parameter computed from bent ray geometry
    - Ionospheric correction uses smoothed L1-L2 difference (Hajj Eq. 18)
    - Adaptive Fresnel-scale smoothing

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

References:
    Hajj et al. 2002, J. Atmos. Sol.-Terr. Phys. 64, 451-469
    Ho et al. 2022, Remote Sens. 14, 5588
    Syndergaard 1998, J. Atmos. Sol.-Terr. Phys. 60, 171-180
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
class LocalCurvatureFrame:
    """
    Local coordinate frame for occultation processing.
    
    Per Hajj et al. 2002 and Syndergaard 1998, the center of spherical
    symmetry should be the center of a circle tangent to the ellipsoid
    at the tangent point, not Earth's center.
    """
    center: np.ndarray      # Local center of curvature (ECEF, m)
    radius: float           # Radius of curvature in occultation plane (m)
    normal: np.ndarray      # Occultation plane normal (unit vector)
    tangent_lat: float      # Tangent point latitude (deg)
    tangent_lon: float      # Tangent point longitude (deg)


@dataclass
class PipelineConfig:
    """Processing configuration."""
    # Quality filters
    min_snr: float = 50.0  # minimum SNR threshold
    min_height_km: float = -2.0  # minimum tangent height
    max_height_km: float = 60.0  # maximum tangent height for retrieval
    top_height_km: float = 80.0  # top of bending angle profile
    
    # Bending angle processing
    smoothing_window: int = 51  # samples for excess phase smoothing (CORRECTED: was 11)
    min_bending_rad: float = 1e-7  # minimum valid bending angle
    max_bending_rad: float = 0.1  # maximum valid bending angle
    
    # Abel inversion
    climatology_blend_km: float = 40.0  # blend with climatology above this
    abel_top_km: float = 150.0  # upper integration limit
    
    # Output
    output_resolution_m: float = 100.0  # vertical resolution for output
    
    # NEW: Rigorous bending computation
    use_rigorous_bending: bool = True  # Use Hajj et al. 2002 method
    use_local_curvature: bool = True   # Use local center of curvature


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
# NEW: LOCAL CENTER OF CURVATURE (Hajj et al. 2002, Syndergaard 1998)
# ============================================================================

def compute_local_curvature_frame(
    r_gnss: np.ndarray,
    r_leo: np.ndarray,
    tangent_lat: float,
    tangent_lon: float
) -> LocalCurvatureFrame:
    """
    Compute local center of curvature for spherical symmetry assumption.
    
    Per Hajj et al. 2002 p. 459-460:
    "The center of symmetry is taken to be the center of a circle in the
    occultation plane which is tangent to the ellipse at the ray path
    tangent point with a radius equal to the ellipse's radius of curvature
    at the same tangent point."
    
    This can be up to 40 km from Earth's true center at equatorial latitudes.
    
    Parameters
    ----------
    r_gnss : array (3,)
        GNSS satellite ECEF position (m)
    r_leo : array (3,)
        LEO satellite ECEF position (m)
    tangent_lat : float
        Tangent point geodetic latitude (degrees)
    tangent_lon : float
        Tangent point geodetic longitude (degrees)
        
    Returns
    -------
    LocalCurvatureFrame
        Contains center, radius, and plane normal
    """
    lat_rad = np.radians(tangent_lat)
    lon_rad = np.radians(tangent_lon)
    
    # Meridional radius of curvature (M)
    # M = a(1-e²) / (1 - e²sin²φ)^(3/2)
    sin2_lat = np.sin(lat_rad)**2
    M = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sin2_lat)**1.5
    
    # Prime vertical radius of curvature (N)
    # N = a / √(1 - e²sin²φ)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin2_lat)
    
    # Occultation plane normal
    occ_normal = np.cross(r_leo, r_gnss)
    occ_norm_mag = np.linalg.norm(occ_normal)
    if occ_norm_mag > 1e-6:
        occ_normal = occ_normal / occ_norm_mag
    else:
        occ_normal = np.array([0, 0, 1])
    
    # Unit vectors at tangent point
    e_n = np.array([  # North
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad)
    ])
    e_e = np.array([  # East
        -np.sin(lon_rad),
        np.cos(lon_rad),
        0.0
    ])
    e_u = np.array([  # Up (local vertical)
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])
    
    # Compute azimuth of occultation plane at tangent point
    ray_dir = r_gnss - r_leo
    ray_norm = np.linalg.norm(ray_dir)
    if ray_norm > 1e-6:
        ray_dir = ray_dir / ray_norm
    
    # Project ray onto local horizontal plane
    ray_horiz = ray_dir - np.dot(ray_dir, e_u) * e_u
    ray_horiz_norm = np.linalg.norm(ray_horiz)
    if ray_horiz_norm > 1e-6:
        ray_horiz = ray_horiz / ray_horiz_norm
    
    cos_az = np.dot(ray_horiz, e_n)
    sin_az = np.dot(ray_horiz, e_e)
    azimuth = np.arctan2(sin_az, cos_az)
    
    # Radius of curvature in occultation plane
    # R = MN / (M·sin²α + N·cos²α) where α is azimuth
    cos2_az = np.cos(azimuth)**2
    sin2_az = np.sin(azimuth)**2
    R_occ = M * N / (M * sin2_az + N * cos2_az)
    
    # Surface point at tangent location
    N_surf = WGS84_A / np.sqrt(1 - WGS84_E2 * sin2_lat)
    r_surface = np.array([
        N_surf * np.cos(lat_rad) * np.cos(lon_rad),
        N_surf * np.cos(lat_rad) * np.sin(lon_rad),
        N_surf * (1 - WGS84_E2) * np.sin(lat_rad)
    ])
    
    # Local center is along the local vertical, at distance R_occ below surface
    center = r_surface - R_occ * e_u
    
    return LocalCurvatureFrame(
        center=center,
        radius=R_occ,
        normal=occ_normal,
        tangent_lat=tangent_lat,
        tangent_lon=tangent_lon
    )


# ============================================================================
# STEP 1: LOAD CONPHS DATA
# ============================================================================

class LeoOrbReader:
    """
    Read COSMIC-2 Level 1b leoOrb (precise LEO orbit) NetCDF files.
    """
    
    def __init__(self):
        self.orbits: Dict[str, Dict] = {}
    
    def load_file(self, filepath: str) -> Optional[Dict]:
        """Load a single leoOrb file."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required")
        
        ds = xr.open_dataset(filepath)
        
        fname = Path(filepath).stem
        leo_id = self._extract_leo_id(fname)
        
        time = ds['time'].values.astype(np.float64)
        x = ds['x'].values.astype(np.float64) * 1000.0
        y = ds['y'].values.astype(np.float64) * 1000.0
        z = ds['z'].values.astype(np.float64) * 1000.0
        
        if 'xdot' in ds:
            vx = ds['xdot'].values.astype(np.float64) * 1000.0
            vy = ds['ydot'].values.astype(np.float64) * 1000.0
            vz = ds['zdot'].values.astype(np.float64) * 1000.0
        else:
            dt = np.gradient(time)
            vx = np.gradient(x) / dt
            vy = np.gradient(y) / dt
            vz = np.gradient(z) / dt
        
        ds.close()
        
        orbit_data = {
            'leo_id': leo_id, 'time': time,
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
    
    def interpolate_to_times(self, leo_id: str, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate orbit to target times."""
        if leo_id not in self.orbits:
            raise ValueError(f"No orbit data for {leo_id}")
        
        orb = self.orbits[leo_id]
        cs_x = CubicSpline(orb['time'], orb['x'], bc_type='natural')
        cs_y = CubicSpline(orb['time'], orb['y'], bc_type='natural')
        cs_z = CubicSpline(orb['time'], orb['z'], bc_type='natural')
        
        pos = np.column_stack([cs_x(target_times), cs_y(target_times), cs_z(target_times)])
        vel = np.column_stack([
            cs_x.derivative()(target_times),
            cs_y.derivative()(target_times),
            cs_z.derivative()(target_times)
        ])
        
        return pos, vel
    
    def _extract_leo_id(self, filename: str) -> str:
        import re
        match = re.search(r'(C2[A-Z]\d)', filename)
        return match.group(1) if match else "C2XX"


class ConPhsReader:
    """
    Read COSMIC-2 Level 1b conPhs (excess phase) NetCDF files.
    """
    
    CONSTELLATION_MAP = {
        'G': 'GPS', 'R': 'GLO', 'E': 'GAL', 'C': 'BDS',
        'J': 'QZS', 'I': 'IRN'
    }
    
    def __init__(self, leoorb_reader: Optional[LeoOrbReader] = None):
        self.events: List[OccultationEvent] = []
        self.leoorb_reader = leoorb_reader

    def load_file(self, filepath: str, use_leoorb: bool = True) -> Optional[OccultationEvent]:
        """Load a single conPhs NetCDF file."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required: pip install xarray netCDF4")
        
        ds = xr.open_dataset(filepath, decode_times=False)
        
        bad_units = 'seconds since 0Z Jan 6, 1980 (GPS seconds)'
        good_units = 'seconds since 1980-01-06 00:00:00'
        
        for var_name in ds.variables:
            if ds[var_name].attrs.get('units') == bad_units:
                ds[var_name].attrs['units'] = good_units
        
        ds = xr.decode_cf(ds)
        
        fname = Path(filepath).stem
        event_id = ds.attrs.get('fileStamp', fname)
        con_id = ds.attrs.get('conId', 'G')
        leap_sec = float(ds.attrs.get('leapsec', 18.0))
        
        prn = self._extract_prn(fname, con_id)
        leo_id = self._extract_leo_id(fname)
        
        if np.issubdtype(ds['time'].dtype, np.datetime64):
            time_hr = ds['time'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        else:
            time_hr = ds['time'].values.astype(np.float64)
        
        ex_L1 = ds['exL1'].values.astype(np.float64)
        ex_L2 = ds['exL2'].values.astype(np.float64)
        snr_L1 = ds['caL1Snr'].values.astype(np.float32)
        snr_L2 = ds['pL2Snr'].values.astype(np.float32)
        occ_height = ds['occheight'].values.astype(np.float32)
        
        if np.issubdtype(ds['txmitLR'].dtype, np.datetime64):
            t_lr = ds['txmitLR'].values.astype('datetime64[ns]').astype(np.float64) / 1e9
        else:
            t_lr = ds['txmitLR'].values.astype(np.float64)
        
        leo_pos_lr = np.column_stack([ds['xLeoLR'], ds['yLeoLR'], ds['zLeoLR']]) * 1000.0
        gnss_pos_lr = np.column_stack([ds['xGnssLR'], ds['yGnssLR'], ds['zGnssLR']]) * 1000.0
        
        start_gps = float(ds.attrs.get('startTime', t_lr[0]))
        stop_gps = float(ds.attrs.get('stopTime', t_lr[-1]))
        
        ds.close()
        
        if use_leoorb and self.leoorb_reader and leo_id in self.leoorb_reader.orbits:
            leo_pos, leo_vel = self.leoorb_reader.interpolate_to_times(leo_id, time_hr)
        else:
            leo_pos, leo_vel = self._interpolate_orbit(
                t_lr, leo_pos_lr[:,0], leo_pos_lr[:,1], leo_pos_lr[:,2], time_hr
            )
        
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
    
    def _interpolate_orbit(self, t_lr, x_lr, y_lr, z_lr, t_hr) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate low-rate orbit to high-rate times."""
        valid = ~(np.isnan(x_lr) | np.isnan(y_lr) | np.isnan(z_lr) | 
                  (np.abs(x_lr) < 1e3) | (np.abs(y_lr) < 1e3))
        
        if np.sum(valid) < 4:
            n = len(t_hr)
            return np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        
        t_v, x_v, y_v, z_v = t_lr[valid], x_lr[valid], y_lr[valid], z_lr[valid]
        
        try:
            cs_x = CubicSpline(t_v, x_v, bc_type='natural')
            cs_y = CubicSpline(t_v, y_v, bc_type='natural')
            cs_z = CubicSpline(t_v, z_v, bc_type='natural')
            
            pos = np.column_stack([cs_x(t_hr), cs_y(t_hr), cs_z(t_hr)])
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
        import re
        pattern = rf'{con_id}(\d{{2}})'
        match = re.search(pattern, filename)
        return int(match.group(1)) if match else 0
    
    def _extract_leo_id(self, filename: str) -> str:
        import re
        match = re.search(r'(C2[A-Z]\d)', filename)
        return match.group(1) if match else "C2XX"


# ============================================================================
# STEP 2: OCCULTATION GEOMETRY
# ============================================================================

class OccultationGeometry:
    """Compute geometric quantities for LEO-GNSS occultation."""
    
    def __init__(self, event: OccultationEvent):
        self.event = event
        self.c = SPEED_OF_LIGHT
    
    def compute_impact_parameter(self) -> np.ndarray:
        """Compute straight-line impact parameter (for initial guess only)."""
        r_leo = self.event.leo_pos
        r_gnss = self.event.gnss_pos
        
        L = r_gnss - r_leo
        L_norm = np.linalg.norm(L, axis=1, keepdims=True)
        L_hat = L / (L_norm + 1e-10)
        
        cross = np.cross(r_leo, L_hat)
        impact = np.linalg.norm(cross, axis=1)
        
        return impact
    
    def compute_tangent_point(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute tangent point coordinates for each observation."""
        r_leo = self.event.leo_pos
        r_gnss = self.event.gnss_pos
        
        L = r_gnss - r_leo
        L_norm = np.linalg.norm(L, axis=1, keepdims=True)
        L_hat = L / (L_norm + 1e-10)
        
        t = -np.sum(r_leo * L_hat, axis=1, keepdims=True)
        r_tp = r_leo + t * L_hat
        
        lat = np.zeros(len(r_tp))
        lon = np.zeros(len(r_tp))
        height = np.zeros(len(r_tp))
        
        for i, r in enumerate(r_tp):
            lat[i], lon[i], height[i] = ecef_to_geodetic(r[0], r[1], r[2])
        
        return lat, lon, height
    
    def compute_geometric_doppler(self, frequency: float) -> np.ndarray:
        """Compute geometric (vacuum) Doppler shift."""
        r_leo = self.event.leo_pos
        v_leo = self.event.leo_vel
        r_gnss = self.event.gnss_pos
        v_gnss = self.event.gnss_vel
        
        L = r_gnss - r_leo
        L_norm = np.linalg.norm(L, axis=1)
        L_hat = L / (L_norm[:, np.newaxis] + 1e-10)
        
        v_rel = v_gnss - v_leo
        range_rate = np.sum(v_rel * L_hat, axis=1)
        
        doppler_geo = -frequency * range_rate / self.c
        
        return doppler_geo
    
    def compute_straight_line_range(self) -> np.ndarray:
        """Compute straight-line range between LEO and GNSS."""
        L = self.event.gnss_pos - self.event.leo_pos
        return np.linalg.norm(L, axis=1)


# ============================================================================
# STEP 3: BENDING ANGLE RETRIEVAL (CORRECTED)
# ============================================================================

class BendingAngleRetriever:
    """
    Retrieve bending angle from excess phase using Doppler method.
    
    CORRECTED: Uses Hajj et al. 2002 rigorous method instead of 
    simplified formula. Implements proper local curvature correction.
    """
    
    def __init__(self, event: OccultationEvent, config: PipelineConfig):
        self.event = event
        self.config = config
        self.c = SPEED_OF_LIGHT
        
        con = event.constellation
        freqs = GNSS_FREQUENCIES.get(con, GNSS_FREQUENCIES['G'])
        self.f1 = freqs.get('L1', freqs.get('B1', 1575.42e6))
        self.f2 = freqs.get('L2', freqs.get('B2', 1227.60e6))

    def compute_fresnel_smoothing_window(self, tangent_height_km: np.ndarray) -> int:
        """
        Compute adaptive Fresnel-scale smoothing window.
        
        Per Hajj et al. 2002: T = 2F/V where F is Fresnel diameter, V is tangent velocity.
        """
        # Typical values
        D_leo = 700e3   # LEO altitude ~700 km
        D_gnss = 20200e3  # GNSS altitude ~20,200 km
        wavelength = self.c / self.f1
        
        # Fresnel diameter: F = sqrt(λ * D_t * D_r / (D_t + D_r))
        F = np.sqrt(wavelength * D_leo * D_gnss / (D_leo + D_gnss))  # ~1-2 km
        
        # Tangent point velocity ~2-3 km/s
        V_tangent = 2500.0  # m/s
        
        # Smoothing time
        T_smooth = 2.0 * F / V_tangent  # seconds
        
        # Convert to samples (assuming 50 Hz data)
        sample_rate = 50.0
        window = int(T_smooth * sample_rate)
        
        # Ensure odd window size
        window = max(21, window)
        if window % 2 == 0:
            window += 1
        
        return window

    def compute_atmospheric_doppler(
            self,
            time: np.ndarray,
            excess_L1: np.ndarray,
            excess_L2: np.ndarray,
            leo_pos: np.ndarray,
            leo_vel: np.ndarray,
            gnss_pos: np.ndarray,
            gnss_vel: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute atmospheric Doppler from excess phase.
            
            NOTE: conPhs excess phase (exL1, exL2) is ATMOSPHERIC excess phase -
            the geometric component has already been removed by CDAAC processing.
            Therefore we do NOT subtract geometric Doppler here.
            
            Atmospheric Doppler = d(excess_phase)/dt * f/c
            """
            # Compute sampling rate
            dt = np.gradient(time)
            sample_rate = 1.0 / np.median(dt)
            
            # Adaptive smoothing window (~0.5 second)
            window = max(21, int(sample_rate * 0.5))
            if window % 2 == 0:
                window += 1
            
            # Smooth excess phase before differentiation
            ex_L1_smooth = uniform_filter1d(excess_L1, window)
            ex_L2_smooth = uniform_filter1d(excess_L2, window)
            
            # Compute excess phase rate (m/s)
            dex_L1_dt = np.gradient(ex_L1_smooth) / dt
            dex_L2_dt = np.gradient(ex_L2_smooth) / dt
            
            # Convert to Doppler (Hz)
            # Doppler = phase_rate * f / c
            atm_doppler_L1 = dex_L1_dt * self.f1 / self.c
            atm_doppler_L2 = dex_L2_dt * self.f2 / self.c
            
            return atm_doppler_L1, atm_doppler_L2

    def doppler_to_bending_hajj(
        self,
        atm_doppler: np.ndarray,
        r_gnss: np.ndarray,
        v_gnss: np.ndarray,
        r_leo: np.ndarray,
        v_leo: np.ndarray,
        frequency: float,
        tangent_lat: np.ndarray,
        tangent_lon: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert atmospheric Doppler to bending angle using Hajj et al. 2002 method.
        
        Solves the system:
            Eq. 14a (Doppler): λΔf = velocity changes due to bending
            Eq. 14b (Bouguer): Impact parameter equality at both ends
        
        Returns:
            bending: Total bending angle (radians)
            impact_param: Impact parameter from bent ray (meters)
        """
        wavelength = self.c / frequency
        n_obs = len(atm_doppler)
        
        bending = np.full(n_obs, np.nan)
        impact_param = np.full(n_obs, np.nan)

        n_converged = 0
        n_failed = 0
        fail_reasons = {'nan_doppler': 0, 'small_occ_norm': 0, 'fsolve_fail': 0, 'sanity_fail': 0}
        
        for i in range(n_obs):
            if np.isnan(atm_doppler[i]):
                continue
            
            r_t = r_gnss[i]
            v_t = v_gnss[i]
            r_r = r_leo[i]
            v_r = v_leo[i]
            
            # Compute local curvature frame
            if self.config.use_local_curvature:
                frame = compute_local_curvature_frame(
                    r_t, r_r, tangent_lat[i], tangent_lon[i]
                )
                r_t_local = r_t - frame.center
                r_r_local = r_r - frame.center
                occ_axis = frame.normal
            else:
                r_t_local = r_t
                r_r_local = r_r
                occ_axis = np.cross(r_t, r_r)
                occ_norm = np.linalg.norm(occ_axis)
                if occ_norm > 1e-6:
                    occ_axis = occ_axis / occ_norm
                else:
                    continue
            
            # Straight-line ray direction
            L = r_r_local - r_t_local
            dist = np.linalg.norm(L)
            if dist < 1e-6:
                continue
            k_hat = L / dist
            
            # Magnitudes
            r_t_mag = np.linalg.norm(r_t_local)
            r_r_mag = np.linalg.norm(r_r_local)
            v_t_mag = np.linalg.norm(v_t)
            v_r_mag = np.linalg.norm(v_r)
            
            # Angles between position and straight-line direction
            cos_theta_t = np.dot(r_t_local, k_hat) / (r_t_mag + 1e-10)
            cos_theta_r = -np.dot(r_r_local, k_hat) / (r_r_mag + 1e-10)
            theta_t = np.arccos(np.clip(cos_theta_t, -1, 1))
            theta_r = np.arccos(np.clip(cos_theta_r, -1, 1))
            
            # Angles between velocity and position
            cos_phi_t = np.dot(v_t, r_t_local) / (v_t_mag * r_t_mag + 1e-10)
            cos_phi_r = np.dot(v_r, r_r_local) / (v_r_mag * r_r_mag + 1e-10)
            phi_t = np.arccos(np.clip(cos_phi_t, -1, 1))
            phi_r = np.arccos(np.clip(cos_phi_r, -1, 1))
            
            def rotate_vec(vec, axis, theta):
                """Rodrigues rotation formula."""
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                return (vec * cos_t + 
                        np.cross(axis, vec) * sin_t + 
                        axis * np.dot(axis, vec) * (1 - cos_t))
            
            def equations(vars):
                delta_t, delta_r = vars
                
                # Bent ray directions
                k_t = rotate_vec(k_hat, occ_axis, -delta_t)
                k_r = rotate_vec(-k_hat, occ_axis, delta_r)
                
                # Doppler equation (Hajj et al. 2002 Eq. 14a)
                # λΔf = v_t·(k_t - k0) - v_r·(k_r - (-k0))
                lhs = wavelength * atm_doppler[i]
                rhs = np.dot(v_t, k_t - k_hat) - np.dot(v_r, k_r - (-k_hat))
                eq1 = lhs - rhs
                
                # Bouguer/Snell constraint (Eq. 14b)
                # Impact parameters must be equal
                a_t = np.linalg.norm(np.cross(r_t_local, k_t))
                a_r = np.linalg.norm(np.cross(r_r_local, k_r))
                eq2 = a_t - a_r
                
                return [eq1, eq2]
            
            # Initial guess based on simplified formula
            L_straight = r_gnss[i] - r_leo[i]
            L_hat_straight = L_straight / np.linalg.norm(L_straight)
            v_rel = v_gnss[i] - v_leo[i]
            v_perp = np.sqrt(np.sum(v_rel**2) - np.dot(v_rel, L_hat_straight)**2)
            alpha_init = abs(wavelength * atm_doppler[i] / (v_perp + 1e-10))
            
            try:
                # Solve the system
                sol = fsolve(equations, [alpha_init * 0.3, alpha_init * 0.7], 
                            full_output=True)
                delta_t_sol, delta_r_sol = sol[0]
                info = sol[1]
                
                # Check convergence
                if np.max(np.abs(info['fvec'])) < 1e-8:
                    total_bending = abs(delta_t_sol) + abs(delta_r_sol)
                    
                    # Compute impact parameter from bent ray
                    k_t_final = rotate_vec(k_hat, occ_axis, -delta_t_sol)
                    a_final = np.linalg.norm(np.cross(r_t_local, k_t_final))
                    
                    # Sanity check
                    if 0 < total_bending < 0.1 and a_final > 6.3e6:
                        bending[i] = total_bending
                        impact_param[i] = a_final
                        n_converged += 1
                    else:
                        fail_reasons['sanity_fail'] += 1
                else:
                    fail_reasons['fsolve_fail'] += 1

            except Exception:
                pass
                
        print(f"  [DEBUG] Hajj solver: {n_converged}/{n_obs} converged. Failures: {fail_reasons}")
        return bending, impact_param

    def doppler_to_bending_simple(
        self,
        atm_doppler: np.ndarray,
        r_gnss: np.ndarray,
        v_gnss: np.ndarray,
        r_leo: np.ndarray,
        v_leo: np.ndarray,
        frequency: float
    ) -> np.ndarray:
        """
        Simple (INCORRECT) Doppler-to-bending conversion.
        Kept for comparison purposes only.
        """
        wavelength = self.c / frequency
        
        L = r_gnss - r_leo
        L_hat = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-10)
        
        v_rel = v_gnss - v_leo
        v_perp = np.sqrt(
            np.sum(v_rel**2, axis=1) - np.sum(v_rel * L_hat, axis=1)**2
        )
        
        bending = np.abs(-wavelength * atm_doppler / (v_perp + 1e-10))
        
        return bending

    def apply_ionospheric_correction_hajj(
        self,
        bending_L1: np.ndarray,
        bending_L2: np.ndarray,
        impact_L1: np.ndarray,
        impact_L2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ionospheric correction per Hajj et al. 2002 Eq. 18.
        
        α_neut(a) = α_L1(a) + c_iono × (α̃_L1(a) - α̃_L2(a))
        
        Uses smoothed L1-L2 difference to preserve L1 resolution while
        removing ionospheric effects.
        """
        # Ionospheric coefficient
        c_iono = self.f2**2 / (self.f1**2 - self.f2**2)  # ~1.5457 for GPS
        
        # Sort by impact parameter
        idx1 = np.argsort(impact_L1)
        idx2 = np.argsort(impact_L2)
        
        a1 = impact_L1[idx1]
        b1 = bending_L1[idx1]
        a2 = impact_L2[idx2]
        b2 = bending_L2[idx2]
        
        # Remove invalid values
        valid1 = np.isfinite(b1) & np.isfinite(a1) & (b1 > 0)
        valid2 = np.isfinite(b2) & np.isfinite(a2) & (b2 > 0)
        
        a1, b1 = a1[valid1], b1[valid1]
        a2, b2 = a2[valid2], b2[valid2]
        
        if len(a1) < 20 or len(a2) < 20:
            return a1, b1
        
        # Smoothing window for ionospheric term (~2x Fresnel scale)
        smooth_win = max(51, int(len(b1) * 0.03))
        if smooth_win % 2 == 0:
            smooth_win += 1
        
        # Smooth bending angles
        b1_smooth = uniform_filter1d(b1, smooth_win)
        b2_smooth = uniform_filter1d(b2, smooth_win)
        
        # Interpolate smoothed L2 onto L1 grid
        try:
            interp_b2_smooth = interp1d(
                a2, b2_smooth, kind='linear',
                bounds_error=False, fill_value=np.nan
            )
            b2_on_L1 = interp_b2_smooth(a1)
            
            # Hajj et al. Eq. 18:
            # α_neut = α_L1 + c_iono × (α̃_L1 - α̃_L2)
            iono_correction = c_iono * (b1_smooth - b2_on_L1)
            bending_neutral = b1 + iono_correction
            
            # Quality filter
            valid = (
                np.isfinite(bending_neutral) & 
                (bending_neutral > 0) & 
                (bending_neutral < 0.1)
            )
            
            return a1[valid], bending_neutral[valid]
            
        except Exception:
            return a1, b1

    def retrieve(self, rigorous: bool = True) -> Optional[BendingProfile]:
        """
        Execute full bending angle retrieval.
        
        CORRECTED: Uses rigorous Hajj et al. 2002 method by default.
        """
        self._diag = []  # Diagnostic messages

        # 1. Height crop
        valid_height_mask = (self.event.occ_height > -20.0) & (self.event.occ_height < 120.0)
        n_valid = np.sum(valid_height_mask)
        self._diag.append(f"Height crop: {n_valid}/{len(self.event.occ_height)} in range")      
        if n_valid < 500:
            self._diag.append(f"FAIL: Only {n_valid} points, need 500")
            return None

        n_valid = np.sum(valid_height_mask)
        print(f"  [DEBUG] Height crop: {n_valid}/{len(self.event.occ_height)} points in -20 to 120 km")
        
        if n_valid < 500:
            print(f"  [ERROR] Insufficient data in height window. Found: {n_valid}, need 500")
            return None

        # 2. Apply mask to all data
        time_c = self.event.time_hr[valid_height_mask]
        exL1_c = self.event.excess_L1[valid_height_mask]
        exL2_c = self.event.excess_L2[valid_height_mask]
        snr1_c = self.event.snr_L1[valid_height_mask]
        snr2_c = self.event.snr_L2[valid_height_mask]
        leo_pos_c = self.event.leo_pos[valid_height_mask]
        leo_vel_c = self.event.leo_vel[valid_height_mask]
        gnss_pos_c = self.event.gnss_pos[valid_height_mask]
        gnss_vel_c = self.event.gnss_vel[valid_height_mask]
        time_c = self.event.time_hr[valid_height_mask]

        # 3. Compute tangent point coordinates
        geom = OccultationGeometry(self.event)
        lat_full, lon_full, _ = geom.compute_tangent_point()
        lat_c = lat_full[valid_height_mask]
        lon_c = lon_full[valid_height_mask]

        # 4. Compute atmospheric Doppler
        atm_doppler_L1, atm_doppler_L2 = self.compute_atmospheric_doppler(
            time_c, exL1_c, exL2_c,
            leo_pos_c, leo_vel_c, gnss_pos_c, gnss_vel_c
        )

        self._diag.append(f"Atm Doppler L1: median={np.nanmedian(atm_doppler_L1):.1f} Hz, valid={np.sum(np.isfinite(atm_doppler_L1))}")
        print(f"  [DEBUG] Atm Doppler L1: min={np.nanmin(atm_doppler_L1):.1f}, max={np.nanmax(atm_doppler_L1):.1f}, median={np.nanmedian(atm_doppler_L1):.1f} Hz")
        print(f"  [DEBUG] Atm Doppler L2: min={np.nanmin(atm_doppler_L2):.1f}, max={np.nanmax(atm_doppler_L2):.1f}, median={np.nanmedian(atm_doppler_L2):.1f} Hz")       

        # 5. Convert Doppler to bending angle
        if rigorous or self.config.use_rigorous_bending:
            # Use Hajj et al. 2002 rigorous method
            bending_L1, impact_L1 = self.doppler_to_bending_hajj(
                atm_doppler_L1, gnss_pos_c, gnss_vel_c,
                leo_pos_c, leo_vel_c, self.f1, lat_c, lon_c
            )
            bending_L2, impact_L2 = self.doppler_to_bending_hajj(
                atm_doppler_L2, gnss_pos_c, gnss_vel_c,
                leo_pos_c, leo_vel_c, self.f2, lat_c, lon_c
            )

        n_valid_b = np.sum(np.isfinite(bending_L1))
        self._diag.append(f"Hajj L1: {n_valid_b}/{len(bending_L1)} valid bending values")
        
        n_valid_b1 = np.sum(np.isfinite(bending_L1))
        n_valid_b2 = np.sum(np.isfinite(bending_L2))
        print(f"  [DEBUG] Hajj bending: L1={n_valid_b1} valid, L2={n_valid_b2} valid out of {len(bending_L1)}")
        if n_valid_b1 > 0:
            print(f"  [DEBUG] Bending L1: min={np.nanmin(bending_L1)*1e3:.3f}, max={np.nanmax(bending_L1)*1e3:.3f} mrad")

        else:
            # Simple method (for comparison)
            bending_L1 = self.doppler_to_bending_simple(
                atm_doppler_L1, gnss_pos_c, gnss_vel_c,
                leo_pos_c, leo_vel_c, self.f1
            )
            bending_L2 = self.doppler_to_bending_simple(
                atm_doppler_L2, gnss_pos_c, gnss_vel_c,
                leo_pos_c, leo_vel_c, self.f2
            )
            # Compute impact parameter from straight line
            L = gnss_pos_c - leo_pos_c
            L_hat = L / np.linalg.norm(L, axis=1, keepdims=True)
            impact_L1 = np.linalg.norm(np.cross(leo_pos_c, L_hat), axis=1)
            impact_L2 = impact_L1.copy()

        # 6. Quality filter
        valid_L1 = (
            np.isfinite(bending_L1) & 
            np.isfinite(impact_L1) &
            (bending_L1 > 0) & 
            (bending_L1 < 0.1) &
            (snr1_c > self.config.min_snr)
        )
        self._diag.append(f"Quality filter: {np.sum(valid_L1)} passed")
        
        if np.sum(valid_L1) < 100:
            print(f"Failed: Only {np.sum(valid_L1)} points passed quality filters")
            self._diag.append(f"FAIL: Only {np.sum(valid_L1)} points passed quality filter")
            return None

        # 7. Apply ionospheric correction
        a_neutral, b_neutral = self.apply_ionospheric_correction_hajj(
            bending_L1[valid_L1], bending_L2[valid_L1],
            impact_L1[valid_L1], impact_L2[valid_L1]
        )
        self._diag.append(f"Iono correction: {len(a_neutral)} points")
        
        if len(a_neutral) < 50:
            print(f"Failed: Only {len(a_neutral)} points after iono correction")
            self._diag.append(f"FAIL: Only {len(a_neutral)} after iono correction")
            return None

        # 8. Compute tangent height
        lat_valid = lat_c[valid_L1]
        lon_valid = lon_c[valid_L1]
        time_valid = time_c[valid_L1]
        
        # Need to map neutral grid back to lat/lon
        # Use interpolation from original impact parameter
        idx_sort = np.argsort(impact_L1[valid_L1])
        a_sorted = impact_L1[valid_L1][idx_sort]
        lat_sorted = lat_valid[idx_sort]
        lon_sorted = lon_valid[idx_sort]
        time_sorted = time_valid[idx_sort]
        
        try:
            lat_interp = interp1d(a_sorted, lat_sorted, bounds_error=False, fill_value='extrapolate')
            lon_interp = interp1d(a_sorted, lon_sorted, bounds_error=False, fill_value='extrapolate')
            time_interp = interp1d(a_sorted, time_sorted, bounds_error=False, fill_value='extrapolate')
            
            lat_neutral = lat_interp(a_neutral)
            lon_neutral = lon_interp(a_neutral)
            time_neutral = time_interp(a_neutral)
        except:
            lat_neutral = np.full_like(a_neutral, np.nanmean(lat_valid))
            lon_neutral = np.full_like(a_neutral, np.nanmean(lon_valid))
            time_neutral = np.linspace(time_valid.min(), time_valid.max(), len(a_neutral))
        
        # Compute tangent height using local radius
        R_local = np.array([gaussian_radius(lat) for lat in lat_neutral])
        h_tangent = (a_neutral - R_local) / 1000.0  # km
        
        # Also get L1/L2 bending on the same grid
        try:
            b1_interp = interp1d(impact_L1[valid_L1], bending_L1[valid_L1], 
                                bounds_error=False, fill_value=np.nan)
            b2_interp = interp1d(impact_L2[valid_L1], bending_L2[valid_L1],
                                bounds_error=False, fill_value=np.nan)
            b1_neutral = b1_interp(a_neutral)
            b2_neutral = b2_interp(a_neutral)
        except:
            b1_neutral = b_neutral.copy()
            b2_neutral = b_neutral.copy()
        
        return BendingProfile(
            impact_param=a_neutral,
            tangent_height=h_tangent,
            bending_L1=b1_neutral,
            bending_L2=b2_neutral,
            bending_neutral=b_neutral,
            latitude=lat_neutral,
            longitude=lon_neutral,
            time=time_neutral
        )


# ============================================================================
# STEP 4: ABEL INVERSION
# ============================================================================

class AbelInversion:
    """
    Inverse Abel transform: bending angle → refractivity.
    
    Theory:
        ln(n(a)) = (1/π) ∫[a→∞] α(a') / √(a'² - a²) da'
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
        """Blend measurements with exponential climatology above threshold."""
        fit_mask = (height_km >= 20.0) & (height_km <= 40.0) & (bending > 0)
        
        if np.sum(fit_mask) < 10:
            return bending.copy()
        
        a_fit = impact[fit_mask]
        alpha_fit = bending[fit_mask]
        
        try:
            log_alpha = np.log(alpha_fit + 1e-12)
            coeffs = np.polyfit(a_fit, log_alpha, 1)
            
            if coeffs[0] < 0:
                H_scale = -1.0 / coeffs[0]
            else:
                H_scale = 7000.0
            
            a_ref = np.median(a_fit)
            alpha_ref = np.exp(np.polyval(coeffs, a_ref))
            
        except Exception:
            H_scale = 7000.0
            a_ref = np.median(impact)
            alpha_ref = np.median(bending[bending > 0])
        
        alpha_clim = alpha_ref * np.exp(-(impact - a_ref) / H_scale)
        
        sigma_meas = np.full_like(bending, 1e-7)
        sigma_clim = np.full_like(bending, 1e10)
        
        above_blend = height_km > self.config.climatology_blend_km
        sigma_clim[above_blend] = 0.05 * alpha_clim[above_blend]
        
        w_meas = 1.0 / sigma_meas**2
        w_clim = 1.0 / sigma_clim**2
        
        alpha_opt = (bending * w_meas + alpha_clim * w_clim) / (w_meas + w_clim)
        
        return alpha_opt
    
    def inverse_abel_transform(
        self,
        impact: np.ndarray,
        bending: np.ndarray
    ) -> np.ndarray:
        """Compute inverse Abel transform."""
        n_levels = len(impact)
        ln_n = np.zeros(n_levels)
        
        for i in range(n_levels):
            a_i = impact[i]
            
            start_idx = min(i + 3, n_levels - 1)
            a_start = impact[start_idx]
            
            if start_idx < n_levels:
                term1 = bending[start_idx] * np.log(
                    a_start + np.sqrt(a_start**2 - a_i**2 + 1e-10)
                )
                term2 = bending[i] * np.log(a_i + 1e-10)
                analytical = term1 - term2
            else:
                analytical = 0.0
            
            parts_integral = 0.0
            for j in range(i, min(start_idx, n_levels - 1)):
                a_mid = 0.5 * (impact[j] + impact[j + 1])
                if a_mid > a_i:
                    d_alpha = bending[j + 1] - bending[j]
                    sqrt_term = np.sqrt(a_mid**2 - a_i**2)
                    parts_integral -= np.log(a_mid + sqrt_term) * d_alpha
            
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
        """Convert refractive index to refractivity and geometric height."""
        n = np.exp(ln_n)
        
        height = np.zeros_like(impact)
        for i in range(len(impact)):
            R_local = gaussian_radius(latitude[i])
            r_tangent = impact[i] / n[i]
            height[i] = (r_tangent - R_local) / 1000.0
        
        N = (n - 1.0) * 1e6
        
        return height, N
    
    def run(self, profile: BendingProfile) -> RefractivityProfile:
        """Execute Abel inversion on bending angle profile."""
        impact = profile.impact_param
        bending = profile.bending_neutral
        height_km = profile.tangent_height
        latitude = profile.latitude
        longitude = profile.longitude
        
        # Sort by ascending impact parameter
        sort_idx = np.argsort(impact)
        impact = impact[sort_idx]
        bending = bending[sort_idx]
        height_km = height_km[sort_idx]
        latitude = latitude[sort_idx]
        longitude = longitude[sort_idx]
        
        # Filter to valid atmospheric range
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
        
        # Filter valid output
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
    """Compare retrieved profiles against COSMIC-2 Level 2 products."""
    
    def __init__(self):
        pass
    
    def load_atmPrf(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load atmPrf validation data."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required")
        
        ds = xr.open_dataset(filepath)
        
        data = {
            'impact_height': ds['Impact_height'].values,
            'bending': ds['Bend_ang'].values,
            'bending_L1': ds['Bend_ang1'].values if 'Bend_ang1' in ds else None,
            'bending_L2': ds['Bend_ang2'].values if 'Bend_ang2' in ds else None,
            'refractivity': ds['Ref'].values,
            'latitude': ds['Lat'].values,
            'longitude': ds['Lon'].values,
            'pressure': ds['Pres'].values if 'Pres' in ds else None,
            'temperature': ds['Temp'].values if 'Temp' in ds else None,
        }
        
        ds.close()
        return data
    
    def load_wetPf2(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load wetPf2 validation data."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required")
        
        ds = xr.open_dataset(filepath)
        
        data = {
            'height': ds['MSL_alt'].values if 'MSL_alt' in ds else ds['Alt'].values,
            'refractivity': ds['Ref'].values if 'Ref' in ds else None,
            'pressure': ds['Pres'].values if 'Pres' in ds else None,
            'temperature': ds['Temp'].values if 'Temp' in ds else None,
            'vapor_pressure': ds['Vp'].values if 'Vp' in ds else None,
            'latitude': ds['Lat'].values,
            'longitude': ds['Lon'].values,
        }
        
        ds.close()
        return data
    
    def compare_bending(
        self,
        retrieved: BendingProfile,
        reference: Dict[str, np.ndarray]
    ) -> ValidationResult:
        """Compare retrieved bending angles against atmPrf."""
        h_ref = reference['impact_height']
        b_ref = reference['bending']
        
        valid_ref = ~np.isnan(b_ref) & ~np.isnan(h_ref) & (b_ref > 0)
        h_ref = h_ref[valid_ref]
        b_ref = b_ref[valid_ref]
        
        h_ret = retrieved.tangent_height
        b_ret = retrieved.bending_neutral
        
        valid_ret = ~np.isnan(b_ret) & (b_ret > 0)
        h_ret = h_ret[valid_ret]
        b_ret = b_ret[valid_ret]
        
        if len(h_ref) < 10 or len(h_ret) < 10:
            return ValidationResult(
                height_common=np.array([]), retrieved=np.array([]),
                reference=np.array([]), difference=np.array([]),
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=0
            )
        
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
                height_common=np.array([]), retrieved=np.array([]),
                reference=np.array([]), difference=np.array([]),
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
            height_common=h_c, retrieved=b_ret_c, reference=b_ref_c,
            difference=diff, rmse=rmse, bias=bias, correlation=corr, n_points=len(h_c)
        )
    
    def compare_refractivity(
        self,
        retrieved: RefractivityProfile,
        reference: Dict[str, np.ndarray],
        source: str = 'atmPrf'
    ) -> ValidationResult:
        """Compare retrieved refractivity against atmPrf or wetPf2."""
        if source == 'atmPrf':
            h_ref = reference['impact_height']
        else:
            h_ref = reference['height']
        
        N_ref = reference['refractivity']
        
        valid_ref = ~np.isnan(N_ref) & ~np.isnan(h_ref) & (N_ref > 0)
        h_ref = h_ref[valid_ref]
        N_ref = N_ref[valid_ref]
        
        h_ret = retrieved.height
        N_ret = retrieved.refractivity
        
        valid_ret = ~np.isnan(N_ret) & (N_ret > 0)
        h_ret = h_ret[valid_ret]
        N_ret = N_ret[valid_ret]
        
        if len(h_ref) < 10 or len(h_ret) < 10:
            return ValidationResult(
                height_common=np.array([]), retrieved=np.array([]),
                reference=np.array([]), difference=np.array([]),
                rmse=np.nan, bias=np.nan, correlation=np.nan, n_points=0
            )
        
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
                height_common=np.array([]), retrieved=np.array([]),
                reference=np.array([]), difference=np.array([]),
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
        
        return ValidationResult(
            height_common=h_c, retrieved=N_ret_c, reference=N_ref_c,
            difference=diff, rmse=rmse, bias=bias, correlation=corr, n_points=len(h_c)
        )


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class LEOROPipeline:
    """
    Complete LEO GNSS Radio Occultation Processing Pipeline.
    
    CORRECTED VERSION with:
    - Rigorous Doppler-to-bending conversion (Hajj et al. 2002)
    - Local center of curvature correction
    - Proper ionospheric correction with smoothed L1-L2 difference
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.leoorb_reader: Optional[LeoOrbReader] = None
        self.reader: Optional[ConPhsReader] = None
        self.validator = Level2Validator()
    
    def load_leoorb(self, leoorb_dir: str) -> int:
        """Load precise LEO orbit data from leoOrb directory."""
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
        rigorous_bending: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single occultation event.
        
        Parameters:
            conphs_file: Path to conPhs NetCDF file
            atmprf_file: Path to atmPrf NetCDF file (for validation)
            wetpf2_file: Path to wetPf2 NetCDF file (for validation)
            output_dir: Directory to save output CSV files
            rigorous_bending: Use rigorous ray-tracing (default: True)
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
            
            orbit_source = "leoOrb" if (self.leoorb_reader and 
                event.leo_id in self.leoorb_reader.orbits) else "conPhs"
            results['messages'].append(
                f"Loaded event {event.event_id}: {event.n_obs} obs, LEO orbit from {orbit_source}"
            )
        except Exception as e:
            results['messages'].append(f"Error loading conPhs: {e}")
            return results
        
        # Step 2: Retrieve bending angles (CORRECTED)
        try:
            retriever = BendingAngleRetriever(event, self.config)
            print(f"  [DEBUG] Starting bending retrieval (rigorous={rigorous_bending})")
            bending_profile = retriever.retrieve(rigorous=rigorous_bending)

            # Log diagnostics
            if hasattr(retriever, '_diag'):
                for msg in retriever._diag:
                    results['messages'].append(f"[DIAG] {msg}")

            
            if bending_profile is None:
                results['messages'].append("Bending angle retrieval failed: insufficient valid data")
                print(f"  [ERROR] Bending retrieval returned None")
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
                
                val_bending = self.validator.compare_bending(bending_profile, atmPrf_data)
                results['validation_bending'] = val_bending
                results['messages'].append(
                    f"Bending validation: RMSE={val_bending.rmse:.2e} rad, "
                    f"r={val_bending.correlation:.3f}, N={val_bending.n_points}"
                )
                
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
        """Process all events in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        if leoorb_dir and self.leoorb_reader is None:
            self.load_leoorb(leoorb_dir)
        
        conphs_files = sorted(
            glob.glob(os.path.join(conphs_dir, "conPhs_*.nc")) +
            glob.glob(os.path.join(conphs_dir, "conPhs_*_nc"))
        )
        
        if max_events:
            conphs_files = conphs_files[:max_events]
        
        if not conphs_files:
            print(f"No conPhs files found in {conphs_dir}")
            return pd.DataFrame()
        
        search_dir = level2_dir if level2_dir else conphs_dir
        
        summary = []
        
        for i, conphs_file in enumerate(conphs_files):
            fname = os.path.basename(conphs_file)
            print(f"\n[{i+1}/{len(conphs_files)}] Processing: {fname}")
            
            event_id = self._extract_event_id(fname)
            
            atmprf_file = self._find_matching_file(search_dir, 'atmPrf', event_id)
            wetpf2_file = self._find_matching_file(search_dir, 'wetPf2', event_id)
            
            if atmprf_file:
                print(f"  Found atmPrf: {os.path.basename(atmprf_file)}")
            if wetpf2_file:
                print(f"  Found wetPf2: {os.path.basename(wetpf2_file)}")
            
            event_output = os.path.join(output_dir, event_id)
            results = self.process_event(
                conphs_file=conphs_file,
                atmprf_file=atmprf_file,
                wetpf2_file=wetpf2_file,
                output_dir=event_output
            )
            
            if generate_plots and results['success']:
                self._generate_event_plots(
                    conphs_file=conphs_file,
                    results=results,
                    atmprf_file=atmprf_file,
                    wetpf2_file=wetpf2_file,
                    output_dir=event_output,
                    event_id=event_id
                )
            
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
            
            status = "OK" if results['success'] else "FAILED"
            print(f"  Status: {status}")
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'processing_summary.csv'), index=False)
        
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
        """Extract event identifier from filename."""
        event_id = filename.replace('conPhs_', '')
        event_id = event_id.replace('atmPrf_', '')
        event_id = event_id.replace('wetPf2_', '')
        event_id = event_id.replace('.nc', '')
        event_id = event_id.replace('_nc', '')
        return event_id
    
    def _find_matching_file(self, search_dir: str, product: str, event_id: str) -> Optional[str]:
        """Find matching Level 2 file."""
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
        reader = self._get_reader()
        event = reader.load_file(conphs_file)
        
        if event is None:
            return []
        
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
    """Two-panel visualization system for LEO GNSS-RO pipeline."""
    
    def __init__(self, output_dir: str = "./plots", dpi: int = 150):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dpi = dpi
        
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
        
        path = self._generate_raw_panel(event, event_id)
        saved_files.append(path)
        
        path = self._generate_derived_panel(
            bending_profile, refrac_profile,
            validation_bending, validation_refrac_atmPrf, validation_refrac_wetPf2,
            atmPrf_data, wetPf2_data, event_id
        )
        saved_files.append(path)
        
        return saved_files
    
    def _generate_raw_panel(self, event: OccultationEvent, event_id: str) -> str:
        """Generate Panel 1: Raw GNSS Observations (2x2)."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import timedelta
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Raw GNSS Observations: {event_id}', fontsize=14, fontweight='bold')
        
        t_rel = event.time_hr - event.time_hr[0]
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
        
        # (c) Excess Phase vs Height
        ax = axes[1, 0]
        ax.plot(event.excess_L1, event.occ_height, color=self.colors['L1'], lw=0.8, label='L1', alpha=0.8)
        ax.plot(event.excess_L2, event.occ_height, color=self.colors['L2'], lw=0.8, label='L2', alpha=0.8)
        ax.set_xlabel('Excess Phase (m)')
        ax.set_ylabel('Occultation Height (km)')
        ax.set_title('(c) Excess Phase vs Height', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # (d) Height vs Time
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
        """Generate Panel 2: Derived Atmospheric Profiles (2x2)."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Derived Atmospheric Profiles: {event_id}', fontsize=14, fontweight='bold')
        
        # (a) Bending Angle Profile
        ax = axes[0, 0]
        ax.plot(bending.bending_L1 * 1e3, bending.tangent_height, 
                color=self.colors['L1'], lw=1, label='L1', alpha=0.7)
        ax.plot(bending.bending_L2 * 1e3, bending.tangent_height, 
                color=self.colors['L2'], lw=1, ls='--', label='L2', alpha=0.7)
        ax.plot(bending.bending_neutral * 1e3, bending.tangent_height, 
                color=self.colors['neutral'], lw=2, label='Iono-free', alpha=0.9)
        
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
        
        if val_bending and val_bending.n_points > 0:
            stats = f'RMSE: {val_bending.rmse*1e6:.1f} µrad\nr: {val_bending.correlation:.3f}'
            ax.text(0.95, 0.05, stats, transform=ax.transAxes, fontsize=8,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # (b) Refractivity Profile
        ax = axes[0, 1]
        valid_r = (refrac.refractivity > 0) & ~np.isnan(refrac.refractivity)
        ax.plot(refrac.refractivity[valid_r], refrac.height[valid_r], 
                color=self.colors['retrieved'], lw=2, label='Retrieved')
        
        if atmPrf_data is not None and 'refractivity' in atmPrf_data:
            h_ref = atmPrf_data.get('impact_height', np.array([]))
            n_ref = atmPrf_data.get('refractivity', np.array([]))
            valid = ~np.isnan(n_ref) & ~np.isnan(h_ref) & (n_ref > 0)
            if np.any(valid):
                ax.plot(n_ref[valid], h_ref[valid], 
                        color=self.colors['atmPrf'], lw=1.5, ls='--', label='atmPrf', alpha=0.8)
        
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
        
        if val_refrac_atm and val_refrac_atm.n_points > 0:
            stats = f'vs atmPrf: RMSE={val_refrac_atm.rmse:.1f}N, r={val_refrac_atm.correlation:.3f}'
            ax.text(0.95, 0.05, stats, transform=ax.transAxes, fontsize=7,
                    ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # (c) Pressure Profile
        ax = axes[1, 0]
        has_pressure = False
        
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
        
        if wetPf2_data is not None and 'temperature' in wetPf2_data:
            h_ref = wetPf2_data.get('height', np.array([]))
            t_ref = wetPf2_data['temperature']
            valid = ~np.isnan(t_ref) & ~np.isnan(h_ref)
            if np.any(valid):
                t_plot = t_ref[valid]
                if np.nanmean(t_plot) < 100:
                    t_plot = t_plot + 273.15
                ax.plot(t_plot, h_ref[valid], 
                        color=self.colors['wetPf2'], lw=2, label='T (wetPf2)')
                has_temp = True
        
        if atmPrf_data is not None and 'temperature' in atmPrf_data:
            h_ref = atmPrf_data.get('impact_height', np.array([]))
            t_ref = atmPrf_data['temperature']
            if t_ref is not None:
                valid = ~np.isnan(t_ref) & ~np.isnan(h_ref)
                if np.any(valid):
                    t_plot = t_ref[valid]
                    if np.nanmean(t_plot) < 100:
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
        description="LEO GNSS Radio Occultation Processing Pipeline (CORRECTED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single event
  python sat_gnss_ro_pipeline.py --conphs conPhs_C2E1.2025.001.00.08.R08_0001.0001_nc \\
                                  --atmprf atmPrf_C2E1.2025.001.00.08.R08_0001.0001_nc \\
                                  --output ./output --plot

  # Process directory
  python sat_gnss_ro_pipeline.py --data-dir ./ro_data --output ./results --plot
        """
    )
    
    parser.add_argument('--conphs', type=str, help='Single conPhs file')
    parser.add_argument('--conphs-dir', type=str, help='Directory of conPhs files')
    parser.add_argument('--data-dir', type=str, help='Directory containing all files')
    parser.add_argument('--leoorb-dir', type=str, help='Directory of leoOrb files')
    parser.add_argument('--atmprf', type=str, help='Single atmPrf file')
    parser.add_argument('--wetpf2', type=str, help='Single wetPf2 file')
    parser.add_argument('--level2-dir', type=str, help='Directory of Level 2 files')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--rigorous', action='store_true', default=True, help='Use rigorous ray-tracing')
    parser.add_argument('--max-events', type=int, help='Maximum events to process')
    parser.add_argument('--plot', action='store_true', help='Generate validation plots')
    parser.add_argument('--min-snr', type=float, default=50.0, help='Minimum SNR')
    parser.add_argument('--min-height', type=float, default=-2.0, help='Min height (km)')
    parser.add_argument('--max-height', type=float, default=60.0, help='Max height (km)')
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        min_snr=args.min_snr,
        min_height_km=args.min_height,
        max_height_km=args.max_height,
        use_rigorous_bending=args.rigorous
    )
    
    pipeline = LEOROPipeline(config)
    
    if args.leoorb_dir:
        n_sats = pipeline.load_leoorb(args.leoorb_dir)
        print(f"Loaded leoOrb data for {n_sats} satellites")
    
    if args.conphs:
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
        
        if args.plot and results['success']:
            print("\n" + "=" * 60)
            print("GENERATING PLOTS")
            print("=" * 60)
            
            reader = pipeline._get_reader()
            event = reader.load_file(args.conphs)
            
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
        if args.data_dir:
            conphs_dir = args.data_dir
            level2_dir = None
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
