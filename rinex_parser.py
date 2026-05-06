#!/usr/bin/env python3
"""
RINEX 3.x Observation File Parser
Parses .rnx files and outputs to CSV format.
"""

import re
import sys
import csv
from datetime import datetime, timedelta
from pathlib import Path

# GNSS System mappings
GNSS_SYSTEMS = {
    'G': 'GPS',
    'R': 'GLONASS',
    'E': 'GAL',
    'C': 'BDS',
    'J': 'QZSS',
    'I': 'IRNSS',
    'S': 'SBAS'
}

# Signal ID mappings (observation code to signal name)
SIGNAL_NAMES = {
    # GPS
    'G': {
        'C1C': 'L1 C/A', 'L1C': 'L1 C/A', 'D1C': 'L1 C/A', 'S1C': 'L1 C/A',
        'C1S': 'L1C(D)', 'L1S': 'L1C(D)', 'D1S': 'L1C(D)', 'S1S': 'L1C(D)',
        'C1L': 'L1C(P)', 'L1L': 'L1C(P)', 'D1L': 'L1C(P)', 'S1L': 'L1C(P)',
        'C1X': 'L1C(D+P)', 'L1X': 'L1C(D+P)', 'D1X': 'L1C(D+P)', 'S1X': 'L1C(D+P)',
        'C1P': 'L1 P', 'L1P': 'L1 P', 'D1P': 'L1 P', 'S1P': 'L1 P',
        'C1W': 'L1 P(Y)', 'L1W': 'L1 P(Y)', 'D1W': 'L1 P(Y)', 'S1W': 'L1 P(Y)',
        'C2C': 'L2 C/A', 'L2C': 'L2 C/A', 'D2C': 'L2 C/A', 'S2C': 'L2 C/A',
        'C2D': 'L2 semi-codeless', 'L2D': 'L2 semi-codeless', 'D2D': 'L2 semi-codeless', 'S2D': 'L2 semi-codeless',
        'C2P': 'L2 P', 'L2P': 'L2 P', 'D2P': 'L2 P', 'S2P': 'L2 P',
        'C2W': 'L2 P(Y)', 'L2W': 'L2 P(Y)', 'D2W': 'L2 P(Y)', 'S2W': 'L2 P(Y)',
        'C2L': 'L2C(L)', 'L2L': 'L2C(L)', 'D2L': 'L2C(L)', 'S2L': 'L2C(L)',
        'C2S': 'L2C(M)', 'L2S': 'L2C(M)', 'D2S': 'L2C(M)', 'S2S': 'L2C(M)',
        'C2X': 'L2C(M+L)', 'L2X': 'L2C(M+L)', 'D2X': 'L2C(M+L)', 'S2X': 'L2C(M+L)',
        'C5I': 'L5 I', 'L5I': 'L5 I', 'D5I': 'L5 I', 'S5I': 'L5 I',
        'C5Q': 'L5 Q', 'L5Q': 'L5 Q', 'D5Q': 'L5 Q', 'S5Q': 'L5 Q',
        'C5X': 'L5 I+Q', 'L5X': 'L5 I+Q', 'D5X': 'L5 I+Q', 'S5X': 'L5 I+Q',
    },
    # GLONASS
    'R': {
        'C1C': 'G1 C/A', 'L1C': 'G1 C/A', 'D1C': 'G1 C/A', 'S1C': 'G1 C/A',
        'C1P': 'G1 P', 'L1P': 'G1 P', 'D1P': 'G1 P', 'S1P': 'G1 P',
        'C2C': 'G2 C/A', 'L2C': 'G2 C/A', 'D2C': 'G2 C/A', 'S2C': 'G2 C/A',
        'C2P': 'G2 P', 'L2P': 'G2 P', 'D2P': 'G2 P', 'S2P': 'G2 P',
        'C3I': 'G3 I', 'L3I': 'G3 I', 'D3I': 'G3 I', 'S3I': 'G3 I',
        'C3Q': 'G3 Q', 'L3Q': 'G3 Q', 'D3Q': 'G3 Q', 'S3Q': 'G3 Q',
        'C3X': 'G3 I+Q', 'L3X': 'G3 I+Q', 'D3X': 'G3 I+Q', 'S3X': 'G3 I+Q',
    },
    # Galileo
    'E': {
        'C1A': 'E1 PRS', 'L1A': 'E1 PRS', 'D1A': 'E1 PRS', 'S1A': 'E1 PRS',
        'C1B': 'E1B', 'L1B': 'E1B', 'D1B': 'E1B', 'S1B': 'E1B',
        'C1C': 'E1C', 'L1C': 'E1C', 'D1C': 'E1C', 'S1C': 'E1C',
        'C1X': 'E1B+C', 'L1X': 'E1B+C', 'D1X': 'E1B+C', 'S1X': 'E1B+C',
        'C1Z': 'E1A+B+C', 'L1Z': 'E1A+B+C', 'D1Z': 'E1A+B+C', 'S1Z': 'E1A+B+C',
        'C5I': 'E5aI', 'L5I': 'E5aI', 'D5I': 'E5aI', 'S5I': 'E5aI',
        'C5Q': 'E5aQ', 'L5Q': 'E5aQ', 'D5Q': 'E5aQ', 'S5Q': 'E5aQ',
        'C5X': 'E5a I+Q', 'L5X': 'E5a I+Q', 'D5X': 'E5a I+Q', 'S5X': 'E5a I+Q',
        'C7I': 'E5bI', 'L7I': 'E5bI', 'D7I': 'E5bI', 'S7I': 'E5bI',
        'C7Q': 'E5bQ', 'L7Q': 'E5bQ', 'D7Q': 'E5bQ', 'S7Q': 'E5bQ',
        'C7X': 'E5b I+Q', 'L7X': 'E5b I+Q', 'D7X': 'E5b I+Q', 'S7X': 'E5b I+Q',
        'C8I': 'E5(a+b)I', 'L8I': 'E5(a+b)I', 'D8I': 'E5(a+b)I', 'S8I': 'E5(a+b)I',
        'C8Q': 'E5(a+b)Q', 'L8Q': 'E5(a+b)Q', 'D8Q': 'E5(a+b)Q', 'S8Q': 'E5(a+b)Q',
        'C8X': 'E5 AltBOC', 'L8X': 'E5 AltBOC', 'D8X': 'E5 AltBOC', 'S8X': 'E5 AltBOC',
        'C6A': 'E6A PRS', 'L6A': 'E6A PRS', 'D6A': 'E6A PRS', 'S6A': 'E6A PRS',
        'C6B': 'E6B', 'L6B': 'E6B', 'D6B': 'E6B', 'S6B': 'E6B',
        'C6C': 'E6C', 'L6C': 'E6C', 'D6C': 'E6C', 'S6C': 'E6C',
        'C6X': 'E6B+C', 'L6X': 'E6B+C', 'D6X': 'E6B+C', 'S6X': 'E6B+C',
    },
    # BeiDou
    'C': {
        'C2I': 'B1I', 'L2I': 'B1I', 'D2I': 'B1I', 'S2I': 'B1I',
        'C2Q': 'B1Q', 'L2Q': 'B1Q', 'D2Q': 'B1Q', 'S2Q': 'B1Q',
        'C2X': 'B1 I+Q', 'L2X': 'B1 I+Q', 'D2X': 'B1 I+Q', 'S2X': 'B1 I+Q',
        'C1D': 'B1C Data', 'L1D': 'B1C Data', 'D1D': 'B1C Data', 'S1D': 'B1C Data',
        'C1P': 'B1C Pilot', 'L1P': 'B1C Pilot', 'D1P': 'B1C Pilot', 'S1P': 'B1C Pilot',
        'C1X': 'B1C D+P', 'L1X': 'B1C D+P', 'D1X': 'B1C D+P', 'S1X': 'B1C D+P',
        'C7I': 'B2I', 'L7I': 'B2I', 'D7I': 'B2I', 'S7I': 'B2I',
        'C7Q': 'B2Q', 'L7Q': 'B2Q', 'D7Q': 'B2Q', 'S7Q': 'B2Q',
        'C7X': 'B2 I+Q', 'L7X': 'B2 I+Q', 'D7X': 'B2 I+Q', 'S7X': 'B2 I+Q',
        'C7D': 'B2a Data', 'L7D': 'B2a Data', 'D7D': 'B2a Data', 'S7D': 'B2a Data',
        'C7P': 'B2a Pilot', 'L7P': 'B2a Pilot', 'D7P': 'B2a Pilot', 'S7P': 'B2a Pilot',
        'C6I': 'B3I', 'L6I': 'B3I', 'D6I': 'B3I', 'S6I': 'B3I',
        'C6Q': 'B3Q', 'L6Q': 'B3Q', 'D6Q': 'B3Q', 'S6Q': 'B3Q',
        'C6X': 'B3 I+Q', 'L6X': 'B3 I+Q', 'D6X': 'B3 I+Q', 'S6X': 'B3 I+Q',
    },
    # QZSS
    'J': {
        'C1C': 'L1 C/A', 'L1C': 'L1 C/A', 'D1C': 'L1 C/A', 'S1C': 'L1 C/A',
        'C1S': 'L1C(D)', 'L1S': 'L1C(D)', 'D1S': 'L1C(D)', 'S1S': 'L1C(D)',
        'C1L': 'L1C(P)', 'L1L': 'L1C(P)', 'D1L': 'L1C(P)', 'S1L': 'L1C(P)',
        'C1X': 'L1C(D+P)', 'L1X': 'L1C(D+P)', 'D1X': 'L1C(D+P)', 'S1X': 'L1C(D+P)',
        'C1Z': 'L1-SAIF', 'L1Z': 'L1-SAIF', 'D1Z': 'L1-SAIF', 'S1Z': 'L1-SAIF',
        'C2S': 'L2C(M)', 'L2S': 'L2C(M)', 'D2S': 'L2C(M)', 'S2S': 'L2C(M)',
        'C2L': 'L2C(L)', 'L2L': 'L2C(L)', 'D2L': 'L2C(L)', 'S2L': 'L2C(L)',
        'C2X': 'L2C(M+L)', 'L2X': 'L2C(M+L)', 'D2X': 'L2C(M+L)', 'S2X': 'L2C(M+L)',
        'C5I': 'L5 I', 'L5I': 'L5 I', 'D5I': 'L5 I', 'S5I': 'L5 I',
        'C5Q': 'L5 Q', 'L5Q': 'L5 Q', 'D5Q': 'L5 Q', 'S5Q': 'L5 Q',
        'C5X': 'L5 I+Q', 'L5X': 'L5 I+Q', 'D5X': 'L5 I+Q', 'S5X': 'L5 I+Q',
        'C6S': 'LEX(S)', 'L6S': 'LEX(S)', 'D6S': 'LEX(S)', 'S6S': 'LEX(S)',
        'C6L': 'LEX(L)', 'L6L': 'LEX(L)', 'D6L': 'LEX(L)', 'S6L': 'LEX(L)',
        'C6X': 'LEX(S+L)', 'L6X': 'LEX(S+L)', 'D6X': 'LEX(S+L)', 'S6X': 'LEX(S+L)',
    },
    # SBAS
    'S': {
        'C1C': 'L1 C/A', 'L1C': 'L1 C/A', 'D1C': 'L1 C/A', 'S1C': 'L1 C/A',
        'C5I': 'L5 I', 'L5I': 'L5 I', 'D5I': 'L5 I', 'S5I': 'L5 I',
        'C5Q': 'L5 Q', 'L5Q': 'L5 Q', 'D5Q': 'L5 Q', 'S5Q': 'L5 Q',
        'C5X': 'L5 I+Q', 'L5X': 'L5 I+Q', 'D5X': 'L5 I+Q', 'S5X': 'L5 I+Q',
    },
    # IRNSS
    'I': {
        'C5A': 'L5 SPS', 'L5A': 'L5 SPS', 'D5A': 'L5 SPS', 'S5A': 'L5 SPS',
        'C5B': 'L5 RS(D)', 'L5B': 'L5 RS(D)', 'D5B': 'L5 RS(D)', 'S5B': 'L5 RS(D)',
        'C5C': 'L5 RS(P)', 'L5C': 'L5 RS(P)', 'D5C': 'L5 RS(P)', 'S5C': 'L5 RS(P)',
        'C5X': 'L5 B+C', 'L5X': 'L5 B+C', 'D5X': 'L5 B+C', 'S5X': 'L5 B+C',
        'C9A': 'S SPS', 'L9A': 'S SPS', 'D9A': 'S SPS', 'S9A': 'S SPS',
        'C9B': 'S RS(D)', 'L9B': 'S RS(D)', 'D9B': 'S RS(D)', 'S9B': 'S RS(D)',
        'C9C': 'S RS(P)', 'L9C': 'S RS(P)', 'D9C': 'S RS(P)', 'S9C': 'S RS(P)',
        'C9X': 'S B+C', 'L9X': 'S B+C', 'D9X': 'S B+C', 'S9X': 'S B+C',
    }
}

# Standard observation type prefixes (C=pseudorange, L=phase, D=doppler, S=SNR)
# X is channel number, skip for observation extraction but keep for column counting
VALID_OBS_PREFIXES = {'C', 'L', 'D', 'S'}


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF (X, Y, Z) in meters to geodetic (lat_deg, lon_deg, alt_m).
    Bowring iterative method on WGS84 ellipsoid.
    """
    import math
    a = 6378137.0          # WGS84 semi-major axis
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)        # semi-minor axis
    e2 = 1 - (b / a) ** 2  # first eccentricity squared
    ep2 = (a / b) ** 2 - 1  # second eccentricity squared
    
    lon = math.atan2(y, x)
    p = math.sqrt(x ** 2 + y ** 2)
    
    # Initial estimate using Bowring's method
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3
    )
    
    # Iterate for convergence
    for _ in range(10):
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        lat_new = math.atan2(z + e2 * N * math.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) - b
    
    return math.degrees(lat), math.degrees(lon), alt


class RINEXParser:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.version = None
        self.file_type = None
        self.sat_system = None
        self.obs_types = {}  # {system: [obs_codes]}
        self.time_system = 'GPS'
        self.interval = None
        self.first_obs_time = None
        self.last_obs_time = None
        self.approx_position = None  # (X, Y, Z) in meters
        self.marker_name = None  # Station marker name
        self.antenna_delta = None  # (H, E, N) in meters
        self.leap_seconds = None
        self.glonass_slots = {}  # {slot: freq_num}
        self.phase_shifts = {}  # {sys: {obs_code: shift}}
        self.header_end_line = 0
        self.observations = []
        
    def parse_header(self, lines):
        """Parse the RINEX header section."""
        i = 0
        while i < len(lines):
            line = lines[i]
            label = line[60:80].strip() if len(line) > 60 else ''
            
            if label == 'RINEX VERSION / TYPE':
                try:
                    self.version = float(line[0:9].strip())
                    self.file_type = line[20:21].strip()
                    self.sat_system = line[40:41].strip() if len(line) > 40 else 'M'
                except (ValueError, IndexError):
                    pass
                    
            elif label == 'APPROX POSITION XYZ':
                try:
                    self.approx_position = (
                        float(line[0:14].strip()),
                        float(line[14:28].strip()),
                        float(line[28:42].strip())
                    )
                except (ValueError, IndexError):
                    pass

            elif label == 'MARKER NAME':
                name = line[0:60].strip()
                if name:
                    self.marker_name = name
                    
            elif label == 'ANTENNA: DELTA H/E/N':
                try:
                    self.antenna_delta = (
                        float(line[0:14].strip()),
                        float(line[14:28].strip()),
                        float(line[28:42].strip())
                    )
                except (ValueError, IndexError):
                    pass
                
            elif label == 'SYS / # / OBS TYPES':
                sys_code = line[0:1].strip()
                if not sys_code:
                    # Continuation line without system code - skip, handled below
                    i += 1
                    continue
                    
                try:
                    num_obs = int(line[3:6].strip())
                except ValueError:
                    i += 1
                    continue
                    
                obs_list = []
                
                # Parse observation types from this line (up to 13 per line)
                obs_str = line[7:60]
                for j in range(13):
                    start = j * 4
                    if start + 3 <= len(obs_str):
                        obs = obs_str[start:start+3].strip()
                        if obs:
                            obs_list.append(obs)
                
                # Check for continuation lines
                cont_idx = i + 1
                while len(obs_list) < num_obs and cont_idx < len(lines):
                    cont_line = lines[cont_idx]
                    cont_label = cont_line[60:80].strip() if len(cont_line) > 60 else ''
                    
                    # Continuation line: same label but NO system code in column 0
                    if cont_label == 'SYS / # / OBS TYPES':
                        cont_sys = cont_line[0:1].strip()
                        if cont_sys:
                            # New system starts - stop reading continuations
                            break
                        # Parse continuation observations
                        obs_str = cont_line[7:60]
                        for j in range(13):
                            start = j * 4
                            if start + 3 <= len(obs_str):
                                obs = obs_str[start:start+3].strip()
                                if obs and len(obs_list) < num_obs:
                                    obs_list.append(obs)
                        cont_idx += 1
                    else:
                        break
                        
                if sys_code:
                    self.obs_types[sys_code] = obs_list
                    
            elif label == 'TIME OF FIRST OBS':
                try:
                    year = int(line[0:6].strip())
                    month = int(line[6:12].strip())
                    day = int(line[12:18].strip())
                    hour = int(line[18:24].strip())
                    minute = int(line[24:30].strip())
                    second = float(line[30:43].strip())
                    microseconds = int((second % 1) * 1e6)
                    self.first_obs_time = datetime(year, month, day, hour, minute, 
                                                   int(second), microseconds)
                    if len(line) > 48:
                        self.time_system = line[48:51].strip()
                except (ValueError, IndexError):
                    pass
                    
            elif label == 'TIME OF LAST OBS':
                try:
                    year = int(line[0:6].strip())
                    month = int(line[6:12].strip())
                    day = int(line[12:18].strip())
                    hour = int(line[18:24].strip())
                    minute = int(line[24:30].strip())
                    second = float(line[30:43].strip())
                    microseconds = int((second % 1) * 1e6)
                    self.last_obs_time = datetime(year, month, day, hour, minute,
                                                  int(second), microseconds)
                except (ValueError, IndexError):
                    pass
                    
            elif label == 'INTERVAL':
                try:
                    self.interval = float(line[0:10].strip())
                except ValueError:
                    pass
                    
            elif label == 'LEAP SECONDS':
                try:
                    self.leap_seconds = int(line[0:6].strip())
                except ValueError:
                    pass
                
            elif label == 'GLONASS SLOT / FRQ #':
                # Parse GLONASS slot/frequency assignments
                # Format: nnn Rss kk Rss kk ... (up to 8 satellites per line)
                # nnn = number of satellites (only on first line, may be blank on continuation)
                # Rss = slot number (R01-R24)
                # kk = frequency number (-7 to +6)
                data = line[4:60]
                for j in range(8):
                    start = j * 7
                    end = start + 7
                    if end <= len(data):
                        entry = data[start:end]
                        slot_str = entry[0:3].strip()
                        freq_str = entry[3:7].strip()
                        if slot_str and slot_str.startswith('R') and freq_str:
                            try:
                                slot = int(slot_str[1:])
                                freq = int(freq_str)
                                self.glonass_slots[slot] = freq
                            except ValueError:
                                pass
                                
            elif label == 'SYS / PHASE SHIFT':
                # Parse phase shift corrections
                try:
                    sys_code = line[0:1].strip()
                    obs_code = line[2:5].strip()
                    shift_str = line[6:14].strip()
                    if sys_code and obs_code:
                        if sys_code not in self.phase_shifts:
                            self.phase_shifts[sys_code] = {}
                        shift = float(shift_str) if shift_str else 0.0
                        self.phase_shifts[sys_code][obs_code] = shift
                except (ValueError, IndexError):
                    pass
                                
            elif label == 'END OF HEADER':
                self.header_end_line = i
                break
                
            i += 1
                
    def get_station_geodetic(self):
        """
        Return station position as geodetic (lat, lon, alt) from APPROX POSITION XYZ.
        Returns dict with keys: latitude, longitude, altitude, ecef_x, ecef_y, ecef_z, marker_name
        or None if APPROX POSITION XYZ is not available or is (0,0,0).
        """
        if self.approx_position is None:
            return None
        x, y, z = self.approx_position
        # Skip if position is (0, 0, 0) — means not set
        if abs(x) < 1.0 and abs(y) < 1.0 and abs(z) < 1.0:
            return None
        lat, lon, alt = ecef_to_geodetic(x, y, z)
        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'ecef_x': x,
            'ecef_y': y,
            'ecef_z': z,
            'marker_name': self.marker_name or 'Unknown'
        }

    def parse_header_only(self):
        """
        Parse only the RINEX header (fast - doesn't read observations).
        Useful for extracting station position without full file parsing.
        """
        with open(self.filepath, 'r', errors='replace') as f:
            lines = []
            for line in f:
                lines.append(line.rstrip('\n'))
                if 'END OF HEADER' in line:
                    break
        self.parse_header(lines)

    def parse_observation_value(self, obs_str):
        """Parse a single observation value (14.3 format with optional LLI and SSI).
        
        RINEX format: 14 chars for value + 1 char LLI + 1 char SSI = 16 chars total
        """
        # Pad to 16 chars if shorter
        if len(obs_str) < 16:
            obs_str = obs_str.ljust(16)
        
        value_str = obs_str[0:14].strip()
        lli_str = obs_str[14:15].strip() if len(obs_str) > 14 else ''
        ssi_str = obs_str[15:16].strip() if len(obs_str) > 15 else ''
        
        # Parse value with error handling
        value = None
        if value_str:
            try:
                value = float(value_str)
            except ValueError:
                pass
        
        # Parse LLI (Loss of Lock Indicator)
        lli = None
        if lli_str:
            try:
                lli = int(lli_str)
            except ValueError:
                pass
        
        # Parse SSI (Signal Strength Indicator)
        ssi = None
        if ssi_str:
            try:
                ssi = int(ssi_str)
            except ValueError:
                pass
        
        return value, lli, ssi
        
    def parse_observations(self, lines):
        """Parse observation data records."""
        i = self.header_end_line + 1
        current_epoch = None
        current_gps_time = None
        current_utc_time = None
        epoch_tow = None  # Time of week in milliseconds
        
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines
            if not line or not line.strip():
                i += 1
                continue
                
            # Check for epoch record (starts with '>')
            if line.startswith('>'):
                # Parse epoch header
                # Format: > YYYY MM DD HH MM SS.SSSSSSS  FLAG NUMSAT
                try:
                    year = int(line[2:6].strip())
                    month = int(line[7:9].strip())
                    day = int(line[10:12].strip())
                    hour = int(line[13:15].strip())
                    minute = int(line[16:18].strip())
                    second = float(line[19:29].strip())
                    
                    # Epoch flag (column 31)
                    epoch_flag = 0
                    if len(line) > 31 and line[31:32].strip():
                        try:
                            epoch_flag = int(line[31:32].strip())
                        except ValueError:
                            pass
                    
                    # Number of satellites (columns 32-35)
                    num_sats = 0
                    if len(line) > 32 and line[32:35].strip():
                        try:
                            num_sats = int(line[32:35].strip())
                        except ValueError:
                            pass
                    
                    # Skip special events (epoch_flag > 1)
                    # 0 = OK, 1 = power failure between epochs
                    # 2+ = special events (moving antenna, new site, etc.)
                    if epoch_flag > 1:
                        # Skip the epoch header and all satellite lines for this event
                        i += 1
                        # For special events, skip following lines that aren't new epochs
                        while i < len(lines) and not lines[i].startswith('>'):
                            i += 1
                        continue
                    
                    # Calculate epoch as datetime
                    microseconds = int((second % 1) * 1e6)
                    current_epoch = datetime(year, month, day, hour, minute, 
                                            int(second), microseconds)
                    
                    # Calculate GPS time of week in milliseconds
                    # Python weekday: Monday=0, Sunday=6
                    # GPS week: Sunday=0, Monday=1, ..., Saturday=6
                    days_since_sunday = (current_epoch.weekday() + 1) % 7
                    epoch_tow = int((days_since_sunday * 86400 + hour * 3600 + 
                                    minute * 60 + second) * 1000)
                    
                    # Store GPS time for reference
                    current_gps_time = current_epoch
                    
                    # Convert to UTC if leap seconds are available and time system is GPS
                    # UTC = GPS - leap_seconds
                    if self.leap_seconds and self.time_system == 'GPS':
                        current_utc_time = current_epoch - timedelta(seconds=self.leap_seconds)
                    else:
                        current_utc_time = current_epoch
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse epoch line {i}: {line.rstrip()}", 
                          file=sys.stderr)
                    current_epoch = None
                    current_gps_time = None
                    current_utc_time = None
                    epoch_tow = None
                    
                i += 1
                continue
                
            # Parse satellite observation record
            if current_gps_time is None:
                i += 1
                continue
            
            # Don't strip the line - we need exact column positions
            line_raw = line.rstrip('\n\r')
                
            # Get satellite ID (columns 0-2)
            sat_id = line_raw[0:3].strip()
            if not sat_id or len(sat_id) < 2:
                i += 1
                continue
                
            sys_code = sat_id[0]
            try:
                prn = int(sat_id[1:3])
            except ValueError:
                i += 1
                continue
                
            # Get observation types for this system
            if sys_code not in self.obs_types:
                i += 1
                continue
                
            obs_codes = self.obs_types[sys_code]
            
            # Parse observations from the line (starting at column 3)
            data_part = line_raw[3:]
            
            # Each observation takes 16 characters (14.3 value + 1 LLI + 1 SSI)
            obs_values = {}
            for j, obs_code in enumerate(obs_codes):
                start = j * 16
                end = start + 16
                
                # Extract observation string (may be shorter at end of line)
                if start < len(data_part):
                    obs_str = data_part[start:end] if end <= len(data_part) else data_part[start:]
                    
                    # Skip non-measurement observation types (X=channel, etc.)
                    # but still count their column position
                    if len(obs_code) < 2 or obs_code[0] not in VALID_OBS_PREFIXES:
                        continue
                        
                    value, lli, ssi = self.parse_observation_value(obs_str)
                    
                    # Only store non-zero values
                    if value is not None and value != 0.0:
                        obs_values[obs_code] = {'value': value, 'lli': lli, 'ssi': ssi}
            
            # Group observations by signal (frequency band)
            # Find matching C (pseudorange), L (phase), D (doppler), S (signal strength)
            signals = {}
            for obs_code, obs_data in obs_values.items():
                obs_type = obs_code[0]  # C, L, D, S
                band = obs_code[1:]     # e.g., 1C, 2W, 7Q
                
                if band not in signals:
                    signals[band] = {}
                signals[band][obs_type] = obs_data
            
            # Create output records for each signal
            for band, signal_obs in signals.items():
                # Build observation code for signal name lookup (prefer C, then L, D, S)
                obs_code_sample = None
                for prefix in ['C', 'L', 'D', 'S']:
                    if prefix in signal_obs:
                        obs_code_sample = prefix + band
                        break
                
                if obs_code_sample is None:
                    continue
                
                # Get signal name from lookup table
                sig_name = obs_code_sample
                if sys_code in SIGNAL_NAMES and obs_code_sample in SIGNAL_NAMES[sys_code]:
                    sig_name = SIGNAL_NAMES[sys_code][obs_code_sample]
                
                # Extract values (use empty string if not present for CSV compatibility)
                carrier = signal_obs.get('L', {}).get('value', '')
                pseudo = signal_obs.get('C', {}).get('value', '')
                doppler = signal_obs.get('D', {}).get('value', '')
                cno = signal_obs.get('S', {}).get('value', '')
                
                # Only create record if we have at least one meaningful observation
                # (pseudorange, carrier phase, or doppler)
                if carrier != '' or pseudo != '' or doppler != '':
                    record = {
                        'timestamp': epoch_tow,
                        'gpsTime': current_gps_time.isoformat(),
                        'utc': current_utc_time.isoformat(),
                        'gnssId': GNSS_SYSTEMS.get(sys_code, sys_code),
                        'svId': prn,
                        'sigId': sig_name,
                        'elevation': '',
                        'azimuth': '',
                        'carrierPhase': carrier,
                        'pseudorange': pseudo,
                        'doppler': doppler,
                        'codePhase': '',
                        'cno': cno,
                    }
                    self.observations.append(record)
            
            i += 1
            
    def parse(self):
        """Parse the entire RINEX file."""
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        self.parse_header(lines)
        self.parse_observations(lines)
        
        return self.observations
    
    def to_csv(self, output_path):
        """Write observations to CSV file."""
        fieldnames = ['timestamp', 'gpsTime', 'utc', 'gnssId', 'svId', 'sigId', 'elevation', 
                      'azimuth', 'carrierPhase', 'pseudorange', 'doppler', 'codePhase', 'cno']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.observations)


def main():
    if len(sys.argv) < 2:
        print("Usage: python rinex_parser.py <input.rnx> [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else Path(input_file).stem + '.csv'
    
    parser = RINEXParser(input_file)
    observations = parser.parse()
    
    print(f"Parsed {len(observations)} observation records")
    print(f"RINEX Version: {parser.version}")
    print(f"Time System: {parser.time_system}")
    print(f"Observation Types: {parser.obs_types}")
    if parser.glonass_slots:
        print(f"GLONASS Slots: {parser.glonass_slots}")
    if parser.leap_seconds:
        print(f"Leap Seconds: {parser.leap_seconds}")
    
    parser.to_csv(output_file)
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    main()
