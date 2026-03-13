"""
PRC 2025 - FINAL: OpenAP with TRUE Dynamic Mass (ALL Flight IDs)
Key fix: Mass is calculated from FLIGHT START, not interval start
- Accounts for all fuel burned from takeoff to current interval
- Mass decreases correctly throughout entire flight
- ✓ ALL 36 AIRCRAFT NOW SUPPORTED (with use_synonym=True)
- ✓ Data density metrics added
- ✓ Results saved to checkpoint_Augmentation folder
"""

import warnings
warnings.filterwarnings('ignore')
import config
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

try:
    from openap import FuelFlow
    HAS_OPENAP = True
except:
    HAS_OPENAP = False
    print("⚠️ OpenAP not available!")


# Paths - Centralized in config.py
DATA_DIR = Path(config.DATA_DIR)
RESULTS_DIR = Path(config.AUGMENTED_DATA_DIR)
PLOTS_DIR = RESULTS_DIR / 'plots'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

AIRCRAFT_DATA = config.AIRCRAFT_DATA

# =============================================================================
# UTILITIES
# =============================================================================

def detect_flight_phase_custom(interval_traj):
    """Custom Flight Phase Detector"""
    try:
        alt_valid = interval_traj['altitude'].dropna()
        gs_valid = interval_traj['groundspeed'].dropna()
        vs_valid = interval_traj['vertical_rate'].dropna()
        
        if len(alt_valid) < 1:
            return 'UNKNOWN', {
                'alt_start_ft': 0, 'alt_end_ft': 0, 'alt_change_ft': 0,
                'alt_avg_ft': 0, 'gs_avg_kts': 0, 'vs_avg_fpm': 0
            }
        
        alt_start = alt_valid.iloc[0] if len(alt_valid) > 0 else 0
        alt_end = alt_valid.iloc[-1] if len(alt_valid) > 0 else 0
        alt_change = alt_end - alt_start
        alt_avg = alt_valid.mean() if len(alt_valid) > 0 else 0
        
        gs_avg = gs_valid.mean() if len(gs_valid) > 0 else 0
        vs_avg = vs_valid.mean() if len(vs_valid) > 0 else 0
        
        phase_info = {
            'alt_start_ft': alt_start,
            'alt_end_ft': alt_end,
            'alt_change_ft': alt_change,
            'alt_avg_ft': alt_avg,
            'gs_avg_kts': gs_avg,
            'vs_avg_fpm': vs_avg,
        }
        
        if alt_avg < 500 and gs_avg < 50:
            phase = 'ON_GROUND'
        elif alt_change > 500 or vs_avg > 300:
            phase = 'CLIMB'
        elif alt_change < -500 or vs_avg < -300:
            phase = 'DESCENT'
        else:
            phase = 'CRUISE'
        
        return phase, phase_info
    except Exception as e:
        return 'UNKNOWN', {
            'alt_start_ft': 0, 'alt_end_ft': 0, 'alt_change_ft': 0,
            'alt_avg_ft': 0, 'gs_avg_kts': 0, 'vs_avg_fpm': 0
        }

def get_segment_type(altitude_ft, phase):
    if phase == 'CLIMB':
        return 'iCO' if altitude_ft < 2000 else 'sCO'
    elif phase == 'DESCENT':
        return 'descent'
    elif phase == 'CRUISE':
        return 'cruise'
    elif phase == 'ON_GROUND':
        return 'on_ground'
    else:
        return 'cruise'

def extract_interval_trajectory(full_traj, interval_start, interval_end):
    traj_copy = full_traj.copy()
    traj_copy['timestamp'] = pd.to_datetime(traj_copy['timestamp'])
    interval_start = pd.Timestamp(interval_start)
    interval_end = pd.Timestamp(interval_end)
    mask = (traj_copy['timestamp'] >= interval_start) & (traj_copy['timestamp'] <= interval_end)
    return traj_copy[mask].copy()

def calculate_missing_data_pct(interval_traj):
    total = len(interval_traj) if len(interval_traj) > 0 else 1
    missing_data = {
        'groundspeed_missing%': (interval_traj['groundspeed'].isna().sum() / total) * 100 if total > 0 else 0,
        'altitude_missing%': (interval_traj['altitude'].isna().sum() / total) * 100 if total > 0 else 0,
        'vertical_rate_missing%': (interval_traj['vertical_rate'].isna().sum() / total) * 100 if total > 0 else 0,
        'total_missing%': 0,
    }
    missing_data['total_missing%'] = np.mean([
        missing_data['groundspeed_missing%'],
        missing_data['altitude_missing%'],
        missing_data['vertical_rate_missing%']
    ])
    return missing_data

def calculate_data_density(interval_traj, interval_duration_sec):
    if len(interval_traj) < 2 or interval_duration_sec <= 0:
        return {'data_points_per_second': 0.0, 'mean_time_between_points_sec': 0.0}
    data_points_per_second = len(interval_traj) / interval_duration_sec
    timestamps = pd.to_datetime(interval_traj['timestamp'])
    time_diffs = (timestamps.diff().dt.total_seconds()).dropna()
    mean_time_between_points = time_diffs.mean() if len(time_diffs) > 0 else 0.0
    return {
        'data_points_per_second': data_points_per_second,
        'mean_time_between_points_sec': mean_time_between_points
    }

def interpolate_trajectory(traj_df):
    traj = traj_df.copy()
    traj['groundspeed'] = traj['groundspeed'].interpolate(method='linear', limit_direction='both')
    traj['altitude'] = traj['altitude'].interpolate(method='linear', limit_direction='both')
    traj['vertical_rate'] = traj['vertical_rate'].interpolate(method='linear', limit_direction='both')
    return traj.fillna(0)

def calculate_flight_fuel_consumption(full_traj, aircraft_type, up_to_timestamp=None):
    if aircraft_type not in AIRCRAFT_DATA:
        return 0.0
    ac_code = aircraft_type.lower()
    try:
        ff_model = FuelFlow(ac=ac_code, use_synonym=True)
    except:
        return 0.0
    ac_data = AIRCRAFT_DATA[aircraft_type]
    mtow = ac_data['mtow_kg']
    oew = ac_data['oew_kg']
    mass_current = mtow * 0.70
    traj = interpolate_trajectory(full_traj.copy())
    if up_to_timestamp:
        up_to_timestamp = pd.Timestamp(up_to_timestamp)
        traj = traj[traj['timestamp'] <= up_to_timestamp]
    if len(traj) < 2:
        return 0.0
    timestamps = pd.to_datetime(traj['timestamp'])
    time_deltas = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    total_fuel_burned = 0.0
    for i in range(len(traj)):
        tas = float(traj.iloc[i]['groundspeed'])
        alt = float(traj.iloc[i]['altitude'])
        vs = float(traj.iloc[i].get('vertical_rate', 0))
        try:
            ff = ff_model.enroute(mass=mass_current, tas=tas, alt=alt, vs=vs)
            if i < len(time_deltas) - 1:
                dt = time_deltas.iloc[i + 1] - time_deltas.iloc[i]
                fuel_burned = ff * dt
                total_fuel_burned += fuel_burned
                mass_current -= fuel_burned
                mass_current = max(mass_current, oew)
        except:
            pass
    return total_fuel_burned

def estimate_with_openap_correct_mass(aircraft_type, full_traj, interval_traj, interval_start, phase):
    try:
        if not HAS_OPENAP or len(interval_traj) < 2:
            return None, "Not available"
        ac_code = aircraft_type.lower()
        try:
            ff_model = FuelFlow(ac=ac_code, use_synonym=True)
        except Exception as e:
            return None, f"FuelFlow error: {str(e)[:150]}"
        timestamps = pd.to_datetime(interval_traj['timestamp'])
        time_deltas = (timestamps - timestamps.iloc[0]).dt.total_seconds()
        ac_data = AIRCRAFT_DATA[aircraft_type]
        mtow = ac_data['mtow_kg']
        oew = ac_data['oew_kg']
        fuel_burned_before_interval = calculate_flight_fuel_consumption(full_traj, aircraft_type, up_to_timestamp=interval_start)
        mass_current = (mtow * 0.70) - fuel_burned_before_interval
        mass_current = max(mass_current, oew)
        fuel_flows = []
        for i in range(len(interval_traj)):
            tas = float(interval_traj.iloc[i]['groundspeed'])
            alt = float(interval_traj.iloc[i]['altitude'])
            vs = float(interval_traj.iloc[i].get('vertical_rate', 0))
            seg = get_segment_type(alt, phase)
            try:
                if phase == 'CLIMB':
                    ff = ff_model.climb(mass=mass_current, tas=tas, alt=alt, vs=vs, seg=seg)
                elif phase == 'DESCENT':
                    ff = ff_model.descent(mass=mass_current, tas=tas, alt=alt, vs=vs)
                elif phase == 'CRUISE':
                    ff = ff_model.cruise(mass=mass_current, tas=tas, alt=alt)
                else:
                    ff = ff_model.cruise(mass=mass_current, tas=tas, alt=alt)
                fuel_flows.append(ff)
                if i < len(time_deltas) - 1:
                    dt = time_deltas.iloc[i + 1] - time_deltas.iloc[i]
                    fuel_burned = ff * dt
                    mass_current -= fuel_burned
                    mass_current = max(mass_current, oew)
            except:
                try:
                    ff = ff_model.enroute(mass=mass_current, tas=tas, alt=alt, vs=vs)
                    fuel_flows.append(ff)
                except:
                    fuel_flows.append(0.0)
        fuel_flows = np.array(fuel_flows)
        dt = np.diff(time_deltas.values, prepend=0)
        total_fuel = np.sum(fuel_flows * dt)
        return {
            'total_fuel_kg': total_fuel,
            'mean_fuel_flow_kg_s': np.mean(fuel_flows),
            'phase_used': phase,
            'starting_mass_kg': (mtow * 0.70) - fuel_burned_before_interval,
        }, "Success"
    except Exception as e:
        return None, str(e)[:150]

# =============================================================================
# RUN PIPELINE STAGE
# =============================================================================

def run(dataset_type='train', force=False):
    output_file = RESULTS_DIR / f'augmented_openap_{dataset_type}.csv'
    
    if output_file.exists() and not force:
        print(f"\n[✓] Results already exist at {output_file}. Skipping augmentation (use --force to rerun).")
        return

    print(f"\n[Loading {dataset_type} data...]")
    fuel_df_raw = pd.read_parquet(DATA_DIR / f'fuel_{dataset_type}.parquet')
    flightlist_df = pd.read_parquet(DATA_DIR / f'flightlist_{dataset_type}.parquet')

    supported_types = list(AIRCRAFT_DATA.keys())
    fuel_df = fuel_df_raw.merge(flightlist_df[['flight_id', 'aircraft_type']], on='flight_id', how='left')
    fuel_df_supported = fuel_df[fuel_df['aircraft_type'].isin(supported_types)].copy()
    TARGET_FLIGHTS = fuel_df_supported['flight_id'].unique().tolist()
    print(f"✓ Found {len(TARGET_FLIGHTS)} flights with supported aircraft")

    fuel_df = fuel_df_supported.copy()
    fuel_df['start_dt'] = pd.to_datetime(fuel_df['start'])
    fuel_df['end_dt'] = pd.to_datetime(fuel_df['end'])

    parquet_files = list((DATA_DIR / f'flights_{dataset_type}' / f'flights_{dataset_type}').glob('*.parquet'))
    flight_data = {}
    print(f"[Loading trajectories...]")
    for file_path in tqdm(parquet_files, desc="Loading flight trajectories"):
        try:
            traj = pd.read_parquet(file_path)
            for flight_id, group in traj.groupby('flight_id'):
                if flight_id in TARGET_FLIGHTS:
                    group = group.sort_values('timestamp').reset_index(drop=True)
                    flight_data[flight_id] = group
        except: pass
    print(f"✓ Loaded {len(flight_data)} flights\n")

    results = []
    flights_processed = 0
    aircraft_errors = {}

    for flight_idx, flight_id in enumerate(tqdm(TARGET_FLIGHTS, desc="Augmenting Flights")):
        if flight_id not in flight_data: continue
        full_traj = flight_data[flight_id]
        flight_info = fuel_df[fuel_df['flight_id'] == flight_id]
        aircraft_type = flight_info['aircraft_type'].iloc[0]
        
        for i, row in flight_info.iterrows():
            interval_start, interval_end, actual_fuel = row['start_dt'], row['end_dt'], row['fuel_kg']
            interval_duration = (pd.to_datetime(interval_end) - pd.to_datetime(interval_start)).total_seconds()
            interval_traj = extract_interval_trajectory(full_traj, interval_start, interval_end)
            data_density = calculate_data_density(interval_traj, interval_duration)
            missing_data = calculate_missing_data_pct(interval_traj)
            
            if len(interval_traj) == 0:
                phase, phase_info = 'UNKNOWN', {'alt_start_ft':0,'alt_end_ft':0,'alt_change_ft':0,'gs_avg_kts':0,'vs_avg_fpm':0}
                interval_traj_interp = pd.DataFrame()
            else:
                interval_traj_interp = interpolate_trajectory(interval_traj)
                phase, phase_info = detect_flight_phase_custom(interval_traj_interp)
            
            openap_result, openap_msg = estimate_with_openap_correct_mass(aircraft_type, full_traj, interval_traj_interp, interval_start, phase)
            
            res = {
                'flight_id': flight_id, 'aircraft': aircraft_type, 'phase': phase,
                'actual_fuel_kg': actual_fuel, 'interval_duration_sec': interval_duration,
                **phase_info, **data_density, **missing_data
            }
            if openap_result:
                res.update({
                    'openap_fuel_kg': openap_result['total_fuel_kg'],
                    'openap_error%': ((openap_result['total_fuel_kg'] - actual_fuel) / actual_fuel) * 100 if actual_fuel > 0 else 0,
                    'openap_status': "Success", 'starting_mass_kg': openap_result['starting_mass_kg']
                })
            else:
                res.update({'openap_fuel_kg': None, 'openap_error%': None, 'openap_status': openap_msg})
            results.append(res)
        flights_processed += 1
        if flights_processed % 100 == 0: print(f"✓ Processed {flights_processed} flights")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to {output_file}")

if __name__ == '__main__':
    run(dataset_type='train')
