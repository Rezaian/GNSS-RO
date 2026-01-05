#!/usr/bin/env python3
"""
GNSS Radio Occultation Processing GUI
==========================================

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
import json
import glob
from datetime import datetime
from typing import Dict, Optional, Any, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QTabWidget, QProgressBar,
    QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont

import pandas as pd
import numpy as np

from ground_gnss_ro_pipeline import (
    StationConfig, PipelineConfig as GroundPipelineConfig, ProcessingResult,
    evaluate_ro_status, generate_raw_plots, generate_derived_plots,
    parse_gnss_directory, match_observations_with_sp3,
    calculate_accurate_elevations, calculate_geometric_doppler,
    apply_single_differencing, retrieve_bending_angles,
    retrieve_refractivity, compare_with_era5, retrieve_atmospheric_profile
)

from sat_gnss_ro_pipeline import (
    LEOROPipeline, PipelineConfig as SatPipelineConfig
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import multiprocessing as mp
from multiprocessing import Process, Queue
from login_ui import LoginDialog


# ============================================================================
# DATA TYPE DETECTION
# ============================================================================

class DataType:
    NONE = 0
    GROUND = 1
    SATELLITE = 2
    BOTH = 3

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
    
    if has_obs_files and sp3_files and metadata_files:
        has_ground = True
        result['ubx_dir'] = directory
        result['sp3_file'] = sp3_files[0]
        result['metadata_file'] = metadata_files[0]
        
        # Determine observation source
        if ubx_files:
            result['obs_source'] = 'UBX'
            result['info'].append(f"Ground data: {len(ubx_files)} UBX files")
        else:
            result['obs_source'] = 'RNX'
            result['info'].append(f"Ground data: {len(rnx_files)} RINEX files")
        
        # If both exist, note UBX will be preferred
        if ubx_files and rnx_files:
            result['info'].append(f"Note: {len(rnx_files)} RINEX files also found (UBX preferred)")
        
        if len(sp3_files) > 1:
            result['warnings'].append(f"Multiple SP3 files — using {os.path.basename(sp3_files[0])}")
        
        # ERA5 for ground validation
        nc_files = glob.glob(os.path.join(directory, '*.[nN][cC]'))
        era5_candidates = [f for f in nc_files if 'era5' in os.path.basename(f).lower() 
                          or not any(x in os.path.basename(f).lower() for x in ['atmprf', 'wetpf', 'conphs'])]
        if era5_candidates:
            result['era5_file'] = era5_candidates[0]
            result['info'].append(f"ERA5 validation: {os.path.basename(era5_candidates[0])}")
        else:
            result['warnings'].append("No ERA5 data — ground validation limited")
    
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
        result['errors'].append("Ground requires: (.ubx OR .rnx) + .sp3 + metadata.cra")
        result['errors'].append("Satellite requires: conPhs_* files")
    
    return result

def load_metadata(filepath: str) -> Optional[Dict]:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_metadata(filepath: str, data: Dict) -> bool:
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception:
        return False


# ============================================================================
# GROUND PIPELINE PROCESS
# ============================================================================

def run_ground_pipeline(station_dict: dict, ubx_dir: str, sp3_file: str,
                        era5_file: str, output_dir: str, progress_queue: Queue):
    """Ground-based pipeline execution in separate process."""
    import os
    from datetime import datetime
    
    from ground_gnss_ro_pipeline import (
        StationConfig, PipelineConfig, ProcessingResult, SP3Parser,
        evaluate_ro_status, generate_raw_plots, generate_derived_plots,
        parse_gnss_directory, calculate_accurate_elevations,
        calculate_geometric_doppler, apply_single_differencing,
        retrieve_bending_angles, retrieve_refractivity,
        compare_with_era5, retrieve_atmospheric_profile
    )
    
    log_lines = []
    
    def log(message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines.append(f"[{ts}] {message}")
    
    def progress(message: str, fraction: float):
        log(message)
        progress_queue.put(('progress', 'ground', message, fraction))
    
    def write_log():
        log_path = os.path.join(output_dir, 'log.txt')
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(log_lines))
        except:
            pass
    
    try:
        station = StationConfig(**station_dict)
        config = PipelineConfig()
        results = {}
        
        log("=" * 60)
        log("GROUND-BASED GNSS-RO Pipeline")
        log("=" * 60)
        log(f"Station: {station.name}")
        log(f"Location: {station.latitude:.4f}°N, {station.longitude:.4f}°E, {station.altitude:.1f}m")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Parse UBX
        progress("Parsing Observation files...", 0.02)
        obs_csv = f"{output_dir}/step1_observations.csv"
        results['step1'] = parse_gnss_directory(ubx_dir, obs_csv)
        if not results['step1'].success:
            progress_queue.put(('done', 'ground', False, "Opservation file parsing failed", None))
            write_log()
            return
        
        # Step 2: SP3 Matching
        progress("Matching with SP3 ephemeris...", 0.10)
        matched_csv = f"{output_dir}/step2_matched.csv"
        
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
        
        matched_count = 0
        total_count = len(df)
        batch_size = 2000
        
        for i, (idx, row) in enumerate(df.iterrows()):
            sp3_data = sp3.interpolate(row['sat_identifier'], row['parsed_utc'])
            if sp3_data:
                for key, value in sp3_data.items():
                    if key in df.columns:
                        df.at[idx, key] = value
                df.at[idx, 'sp3_match_status'] = 'matched'
                matched_count += 1
            
            if (i + 1) % batch_size == 0:
                frac = 0.10 + 0.25 * ((i + 1) / total_count)
                progress(f"SP3 matching: {i + 1:,}/{total_count:,}", frac)
        
        progress(f"SP3 matching complete: {matched_count:,}/{total_count:,}", 0.35)
        
        df_matched = df[df['sp3_match_status'] == 'matched'].copy()
        if not df_matched.empty:
            df_matched.to_csv(matched_csv, index=False)
        
        results['step2'] = ProcessingResult(
            success=True, data=df_matched,
            message=f"Matched {matched_count}/{total_count}",
            metadata={'total': total_count, 'matched': matched_count}
        )
        
        # Step 3a: Elevations
        progress("Calculating satellite elevations...", 0.38)
        elev_csv = f"{output_dir}/step3a_elevations.csv"
        results['step3a'] = calculate_accurate_elevations(matched_csv, station, elev_csv)
        
        # Step 3b: Geometric Doppler
        progress("Computing geometric Doppler...", 0.42)
        doppler_csv = f"{output_dir}/step3b_doppler.csv"
        results['step3b'] = calculate_geometric_doppler(elev_csv, station, doppler_csv)
        
        # Step 4: Single Differencing
        progress("Applying single differencing...", 0.48)
        diff_csv = f"{output_dir}/step4_differenced.csv"
        results['step4'] = apply_single_differencing(doppler_csv, config, diff_csv)
        
        intermediate_data = None
        if os.path.exists(diff_csv):
            intermediate_data = pd.read_csv(diff_csv)
        
        # Step 5: Bending Angles
        progress("Retrieving bending angles...", 0.52)
        bending_dir = f"{output_dir}/bending"
        results['step5'] = retrieve_bending_angles(
            diff_csv, station, config, bending_dir,
            lambda msg, frac: progress(msg, frac)
        )
        
        # Steps 6 & 7: Per-satellite processing
        if results['step5'].success and results['step5'].data is not None:
            summary = results['step5'].data
            total_sats = len(summary)
            
            for idx, row in summary.iterrows():
                sat_id = row['sat_id']
                bending_csv = f"{bending_dir}/{sat_id}_bending.csv"
                
                frac = 0.80 + 0.10 * ((idx + 1) / max(total_sats, 1))
                progress(f"Abel inversion: {sat_id} ({idx+1}/{total_sats})", frac)
                
                if os.path.exists(bending_csv):
                    refrac_csv = f"{output_dir}/refractivity/{sat_id}_refractivity.csv"
                    os.makedirs(os.path.dirname(refrac_csv), exist_ok=True)
                    result = retrieve_refractivity(bending_csv, refrac_csv)
                    results[f'step6_{sat_id}'] = result
                    
                    if era5_file and result.success:
                        comp_csv = f"{output_dir}/comparison/{sat_id}_comparison.csv"
                        atm_csv = f"{output_dir}/atmospheric/{sat_id}_atmospheric.csv"
                        os.makedirs(os.path.dirname(comp_csv), exist_ok=True)
                        os.makedirs(os.path.dirname(atm_csv), exist_ok=True)
                        
                        results[f'step6b_{sat_id}'] = compare_with_era5(
                            refrac_csv, era5_file, station.latitude, station.longitude, comp_csv
                        )
                        results[f'step7_{sat_id}'] = retrieve_atmospheric_profile(
                            refrac_csv, era5_file, station.latitude, station.longitude, atm_csv
                        )
        
        # Generate plots
        progress("Generating plots...", 0.92)
        
        if intermediate_data is not None:
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            df_plot = intermediate_data.copy()
            if 'sat_id' not in df_plot.columns:
                if 'gnssId' in df_plot.columns and 'svId' in df_plot.columns:
                    df_plot['sat_id'] = df_plot['gnssId'].astype(str) + '_' + df_plot['svId'].astype(str)
            
            ro_status = evaluate_ro_status(df_plot)
            satellites = list(df_plot['sat_id'].unique())
            
            for sat_id in satellites:
                sat_data = df_plot[df_plot['sat_id'] == sat_id]
                
                raw_path = os.path.join(plots_dir, f'{sat_id}_raw.png')
                generate_raw_plots(sat_data, sat_id, raw_path)
                
                if ro_status.get(sat_id, False):
                    sat_results = {
                        'bending_csv': os.path.join(output_dir, 'bending', f'{sat_id}_bending.csv'),
                        'refrac_csv': os.path.join(output_dir, 'refractivity', f'{sat_id}_refractivity.csv'),
                        'comp_csv': os.path.join(output_dir, 'comparison', f'{sat_id}_comparison.csv'),
                        'atm_csv': os.path.join(output_dir, 'atmospheric', f'{sat_id}_atmospheric.csv'),
                    }
                    derived_path = os.path.join(plots_dir, f'{sat_id}_derived.png')
                    generate_derived_plots(sat_results, sat_id, derived_path)
        
        write_log()
        
        success_count = sum(1 for r in results.values() if r.success)
        progress_queue.put(('done', 'ground', True, f"{success_count}/{len(results)} steps completed", diff_csv))
        
    except Exception as e:
        log(f"CRITICAL ERROR: {str(e)}")
        write_log()
        progress_queue.put(('done', 'ground', False, str(e), None))


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
# PLOT CANVAS
# ============================================================================

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#FAFAFA')
        super().__init__(self.fig)
        self.setParent(parent)
    
    def show_placeholder(self, message: str = "Select an item"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center',
                fontsize=13, color='#757575', transform=ax.transAxes)
        ax.axis('off')
        self.draw()
    
    def load_from_png(self, png_path: str):
        self.fig.clear()
        if os.path.exists(png_path):
            img = matplotlib.image.imread(png_path)
            ax = self.fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Plot not available", ha='center', va='center',
                    fontsize=13, color='#757575', transform=ax.transAxes)
            ax.axis('off')
        self.draw()


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
        self.name_edit.setPlaceholderText("Station identifier")
        row1.addWidget(self.name_edit)
        layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Lat:"))
        self.lat_edit = QLineEdit()
        self.lat_edit.setPlaceholderText("°N")
        self.lat_edit.setMaximumWidth(80)
        row2.addWidget(self.lat_edit)
        row2.addWidget(QLabel("Lon:"))
        self.lon_edit = QLineEdit()
        self.lon_edit.setPlaceholderText("°E")
        self.lon_edit.setMaximumWidth(80)
        row2.addWidget(self.lon_edit)
        row2.addStretch()
        layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Altitude:"))
        self.alt_edit = QLineEdit()
        self.alt_edit.setPlaceholderText("meters")
        self.alt_edit.setMaximumWidth(100)
        row3.addWidget(self.alt_edit)
        row3.addWidget(QLabel("m"))
        row3.addStretch()
        layout.addLayout(row3)
    
    def load_from_metadata(self, metadata: Dict):
        self.name_edit.setText(str(metadata.get('STATION_NAME', '')))
        self.lat_edit.setText(str(metadata.get('STATION_LAT', '')))
        self.lon_edit.setText(str(metadata.get('STATION_LON', '')))
        self.alt_edit.setText(str(metadata.get('STATION_HEIGHT', '')))
    
    def get_station_config(self) -> Optional[StationConfig]:
        try:
            return StationConfig(
                latitude=float(self.lat_edit.text()),
                longitude=float(self.lon_edit.text()),
                altitude=float(self.alt_edit.text()),
                name=self.name_edit.text() or "Station"
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
            }
        except ValueError:
            return {}


# ============================================================================
# RESULT LIST WIDGET
# ============================================================================

class ResultListWidget(QListWidget):
    """Unified list for both ground satellites and satellite events."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QListWidget {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
            QListWidget::item { padding: 4px 8px; }
            QListWidget::item:selected {
                background-color: #1976D2;
                color: white;
            }
        """)
        self.current_mode = None  # 'ground' or 'satellite'
    
    def populate_ground(self, ro_status: Dict[str, bool]):
        """Populate with ground-based satellite results."""
        self.clear()
        self.current_mode = 'ground'
        
        ro_sats = sorted([s for s, is_ro in ro_status.items() if is_ro])
        non_ro_sats = sorted([s for s, is_ro in ro_status.items() if not is_ro])
        
        for sat_id in ro_sats:
            item = QListWidgetItem(f"● {sat_id}  [RO]")
            item.setForeground(QColor('#2E7D32'))
            item.setData(Qt.ItemDataRole.UserRole, sat_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, True)  # is_ro
            item.setData(Qt.ItemDataRole.UserRole + 2, 'ground')
            self.addItem(item)
        
        if ro_sats and non_ro_sats:
            sep = QListWidgetItem("─" * 24)
            sep.setFlags(Qt.ItemFlag.NoItemFlags)
            sep.setForeground(QColor('#BDBDBD'))
            self.addItem(sep)
        
        for sat_id in non_ro_sats:
            item = QListWidgetItem(f"○ {sat_id}")
            item.setForeground(QColor('#757575'))
            item.setData(Qt.ItemDataRole.UserRole, sat_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, False)
            item.setData(Qt.ItemDataRole.UserRole + 2, 'ground')
            self.addItem(item)
    
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
    
    def populate_both(self, ground_ro_status: Dict[str, bool], sat_summary_df: pd.DataFrame):
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
        
        ro_sats = sorted([s for s, is_ro in ground_ro_status.items() if is_ro])
        non_ro_sats = sorted([s for s, is_ro in ground_ro_status.items() if not is_ro])
        
        for sat_id in ro_sats:
            item = QListWidgetItem(f"  ● {sat_id}  [RO]")
            item.setForeground(QColor('#2E7D32'))
            item.setData(Qt.ItemDataRole.UserRole, sat_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, True)
            item.setData(Qt.ItemDataRole.UserRole + 2, 'ground')
            self.addItem(item)
        
        for sat_id in non_ro_sats:
            item = QListWidgetItem(f"  ○ {sat_id}")
            item.setForeground(QColor('#757575'))
            item.setData(Qt.ItemDataRole.UserRole, sat_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, False)
            item.setData(Qt.ItemDataRole.UserRole + 2, 'ground')
            self.addItem(item)
        
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
        
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: #424242;")
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
        self.detail_label.setStyleSheet("color: #616161; font-size: 10px;")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
    
    def set_status(self, status: str, detail: str = "", progress: float = 0):
        self.status_label.setText(status)
        self.status_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        self.detail_label.setText(detail)
        self.progress_bar.setValue(int(progress * 100))
    
    def set_complete(self, success: bool, message: str):
        if success:
            self.status_label.setText("Completed")
            self.status_label.setStyleSheet("font-weight: bold; color: #2E7D32;")
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText("Stopped" if "cancel" in message.lower() else "Failed")
            self.status_label.setStyleSheet("font-weight: bold; color: #D32F2F;")
        self.detail_label.setText(message)
    
    def reset(self):
        self.status_label.setText("Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: #424242;")
        self.detail_label.setText("")
        self.progress_bar.setValue(0)


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNSS Radio Occultation Processor v3")
        self.setMinimumSize(1200, 800)
        
        self.input_dir = None
        self.scan_result = None
        self.output_dir = None
        
        # Ground results
        self.ground_intermediate_data = None
        self.ground_ro_status = {}
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
        
        self._setup_ui()
        self._connect_signals()
    
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
        sidebar.setMaximumWidth(350)
        sidebar.setMinimumWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)
        
        # Input directory
        input_group = QGroupBox("Data Directory")
        input_layout = QVBoxLayout(input_group)
        
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Ground: .ubx/.sp3 | Satellite: conPhs_*")
        self.dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.dir_edit)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMaximumWidth(70)
        dir_layout.addWidget(self.browse_btn)
        input_layout.addLayout(dir_layout)
        
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("font-size: 10px;")
        input_layout.addWidget(self.validation_label)
        
        sidebar_layout.addWidget(input_group)
        
        # Station (only for ground)
        self.station_panel = StationInfoPanel()
        self.station_panel.setVisible(False)
        sidebar_layout.addWidget(self.station_panel)
        
        # Run and Stop buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        
        self.run_btn = QPushButton("Start Processing")
        self.run_btn.setEnabled(False)
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1565C0; }
            QPushButton:pressed { background-color: #0D47A1; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #C62828; }
            QPushButton:pressed { background-color: #B71C1C; }
            QPushButton:disabled { background-color: #BDBDBD; }
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
        
        self.result_list = ResultListWidget()
        result_layout.addWidget(self.result_list)
        
        self.legend_label = QLabel("● Success/RO    ○ Failed/No RO")
        self.legend_label.setStyleSheet("color: #757575; font-size: 9px;")
        result_layout.addWidget(self.legend_label)
        
        sidebar_layout.addWidget(result_group, 1)
        
        splitter.addWidget(sidebar)
        
        # MAIN PANEL
        main_panel = QWidget()
        main_panel_layout = QVBoxLayout(main_panel)
        main_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                font-size: 11px;
            }
            QTabBar::tab:selected { font-weight: bold; }
        """)
        
        self.raw_canvas = PlotCanvas()
        self.raw_canvas.show_placeholder("Select an item to view observations")
        self.tab_widget.addTab(self.raw_canvas, "Observations / Raw")
        
        self.derived_canvas = PlotCanvas()
        self.derived_canvas.show_placeholder("Select a successful item to view profiles")
        self.tab_widget.addTab(self.derived_canvas, "Atmospheric Profiles")
        
        main_panel_layout.addWidget(self.tab_widget)
        splitter.addWidget(main_panel)
        
        splitter.setSizes([320, 880])
    
    def _connect_signals(self):
        self.browse_btn.clicked.connect(self._browse_directory)
        self.run_btn.clicked.connect(self._run_pipeline)
        self.stop_btn.clicked.connect(self._stop_pipeline)
        self.result_list.itemClicked.connect(self._on_item_selected)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
    
    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self._validate_directory(directory)
    
    def _validate_directory(self, directory: str):
        self.input_dir = directory
        self.dir_edit.setText(directory)
        
        self.scan_result = scan_input_directory(directory)
        
        msgs = []
        
        # Data type indicator
        if self.scan_result['data_type'] == DataType.GROUND:
            msgs.append("<span style='color:#1976D2; font-weight:bold;'>📡 Ground-based data</span>")
        elif self.scan_result['data_type'] == DataType.SATELLITE:
            msgs.append("<span style='color:#7B1FA2; font-weight:bold;'>🛰 Satellite (LEO) data</span>")
        elif self.scan_result['data_type'] == DataType.BOTH:
            msgs.append("<span style='color:#00796B; font-weight:bold;'>📡🛰 Ground + Satellite data</span>")
        
        # Info messages
        for info in self.scan_result['info']:
            msgs.append(f"<span style='color:#2E7D32'>✓ {info}</span>")
        
        # Warnings
        for w in self.scan_result['warnings']:
            msgs.append(f"<span style='color:#F57C00'>⚠ {w}</span>")
        
        # Errors
        for e in self.scan_result['errors']:
            msgs.append(f"<span style='color:#D32F2F'>✗ {e}</span>")
        
        self.validation_label.setText("<br>".join(msgs))
        
        # Show station panel only for ground data
        has_ground = self.scan_result['data_type'] in (DataType.GROUND, DataType.BOTH)
        self.station_panel.setVisible(has_ground)
        
        # Load metadata if ground data present
        if has_ground and self.scan_result['metadata_file']:
            metadata = load_metadata(self.scan_result['metadata_file'])
            if metadata:
                self.station_panel.load_from_metadata(metadata)
        
        self.run_btn.setEnabled(self.scan_result['valid'])
    

    def _run_pipeline(self):
        data_type = self.scan_result['data_type']
        has_ground = data_type in (DataType.GROUND, DataType.BOTH)
        has_satellite = data_type in (DataType.SATELLITE, DataType.BOTH)
        
        # Validate ground config if needed
        if has_ground:
            station = self.station_panel.get_station_config()
            if not station:
                QMessageBox.warning(self, "Configuration Error",
                                  "Please enter valid station coordinates for ground data.")
                return
            
            if self.scan_result['metadata_file']:
                save_metadata(self.scan_result['metadata_file'], self.station_panel.to_metadata())
        
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
        self.sat_summary_df = None
        
        # Setup multiprocessing
        self.progress_queue = mp.Queue()
        
        # Start ground pipeline
        if has_ground:
            station = self.station_panel.get_station_config()
            station_dict = {
                'latitude': station.latitude,
                'longitude': station.longitude,
                'altitude': station.altitude,
                'name': station.name
            }
            
            self.ground_process = mp.Process(
                target=run_ground_pipeline,
                args=(
                    station_dict,
                    self.scan_result['ubx_dir'],
                    self.scan_result['sp3_file'],
                    self.scan_result['era5_file'],
                    self.ground_output_dir,
                    self.progress_queue
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
                        if success and result_path and os.path.exists(result_path):
                            self.ground_intermediate_data = pd.read_csv(result_path)
                            self.ground_ro_status = evaluate_ro_status(self.ground_intermediate_data)
                    
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
        
        # Populate result list
        data_type = self.scan_result['data_type']
        
        if data_type == DataType.GROUND:
            self.result_list.populate_ground(self.ground_ro_status)
            self.legend_label.setText("● RO detected    ○ Standard tracking")
        elif data_type == DataType.SATELLITE:
            self.result_list.populate_satellite(self.sat_summary_df)
            self.legend_label.setText("● Success    ○ Failed    [VAL] Has validation")
        elif data_type == DataType.BOTH:
            self.result_list.populate_both(self.ground_ro_status, self.sat_summary_df)
            self.legend_label.setText("Ground: ●/○ RO | Satellite: ●/○ Success")
        
        self.raw_canvas.show_placeholder("Select an item to view")
        self.derived_canvas.show_placeholder("Select an item to view profiles")
        
        # Summary message
        msg_parts = []
        if self.ground_success:
            ro_count = sum(1 for v in self.ground_ro_status.values() if v)
            msg_parts.append(f"Ground: {len(self.ground_ro_status)} satellites ({ro_count} RO)")
        if self.sat_success and self.sat_summary_df is not None:
            success_count = self.sat_summary_df['success'].sum()
            msg_parts.append(f"Satellite: {success_count}/{len(self.sat_summary_df)} events")
        
        if msg_parts:
            QMessageBox.information(
                self, "Processing Complete",
                "\n".join(msg_parts) + f"\n\nResults saved to:\n{self.output_dir}"
            )
    
    def _on_item_selected(self, item: QListWidgetItem):
        item_id = item.data(Qt.ItemDataRole.UserRole)
        if not item_id:
            return
        
        is_success = item.data(Qt.ItemDataRole.UserRole + 1)
        item_type = item.data(Qt.ItemDataRole.UserRole + 2)
        
        self._load_plots(item_id, is_success, item_type)
    
    def _on_tab_changed(self, index: int):
        current_item = self.result_list.currentItem()
        if current_item:
            item_id = current_item.data(Qt.ItemDataRole.UserRole)
            is_success = current_item.data(Qt.ItemDataRole.UserRole + 1)
            item_type = current_item.data(Qt.ItemDataRole.UserRole + 2)
            if item_id:
                self._load_plots(item_id, is_success, item_type)
    
    def _load_plots(self, item_id: str, is_success: bool, item_type: str):
        current_tab = self.tab_widget.currentIndex()
        
        if item_type == 'ground':
            plots_dir = os.path.join(self.ground_output_dir, 'plots')
            
            if current_tab == 0:
                raw_path = os.path.join(plots_dir, f'{item_id}_raw.png')
                self.raw_canvas.load_from_png(raw_path)
            elif current_tab == 1:
                if is_success:
                    derived_path = os.path.join(plots_dir, f'{item_id}_derived.png')
                    self.derived_canvas.load_from_png(derived_path)
                else:
                    self.derived_canvas.show_placeholder(
                        f"No radio occultation detected for {item_id}\n\n"
                        "Atmospheric profiles require RO geometry:\n"
                        "• Elevation < 5°\n"
                        "• |Atmospheric Doppler| > 2.5 Hz\n"
                        "• Minimum 10 epochs"
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    if not LoginDialog.authenticate(app):
        sys.exit(0)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
