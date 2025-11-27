"""
PRC 2025 - FINAL: OpenAP with TRUE Dynamic Mass (ALL Flight IDs)
Key fix: Mass is calculated from FLIGHT START, not interval start
- Accounts for all fuel burned from takeoff to current interval
- Mass decreases correctly throughout entire flight
- ✓ ALL 36 AIRCRAFT NOW SUPPORTED (with use_synonym=True)
- ✓ Data density metrics added
- ✓ Results saved to checkpoint_Augmentation folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


try:
    from openap import FuelFlow
    HAS_OPENAP = True
except:
    HAS_OPENAP = False
    print("⚠️ OpenAP not available!")


print("="*80)
print("PRC 2025 - OpenAP with TRUE Dynamic Mass (Flight-Level Mass Tracking)")
print("✓ Processing ALL Flight IDs from fuel_train.parquet")
print("✓ ALL 36 Aircraft Types Supported (with use_synonym=True)")
print("✓ Data Density Metrics Included")
print("="*80)


# Paths
DATA_DIR = Path('data')
RESULTS_DIR = Path('Results/checkpoint_Augmentation')
PLOTS_DIR = RESULTS_DIR / 'plots'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ALL AIRCRAFT SUPPORTED BY OPENAP (36 types)
# ============================================================================


AIRCRAFT_DATA = {
    # Airbus A320 Family
    'A19N': {'mtow_kg': 79200, 'oew_kg': 45100, 'max_fuel_kg': 27200},
    'A20N': {'mtow_kg': 79000, 'oew_kg': 45400, 'max_fuel_kg': 27200},
    'A21N': {'mtow_kg': 97000, 'oew_kg': 50300, 'max_fuel_kg': 32840},
    'A318': {'mtow_kg': 68000, 'oew_kg': 39500, 'max_fuel_kg': 24210},
    'A319': {'mtow_kg': 75500, 'oew_kg': 40800, 'max_fuel_kg': 24210},
    'A320': {'mtow_kg': 78000, 'oew_kg': 42400, 'max_fuel_kg': 27200},
    'A321': {'mtow_kg': 93500, 'oew_kg': 48700, 'max_fuel_kg': 32840},
    
    # Airbus Wide-Body
    'A332': {'mtow_kg': 233000, 'oew_kg': 119500, 'max_fuel_kg': 139090},
    'A333': {'mtow_kg': 242000, 'oew_kg': 123400, 'max_fuel_kg': 139090},
    'A343': {'mtow_kg': 275000, 'oew_kg': 131000, 'max_fuel_kg': 155040},
    'A359': {'mtow_kg': 280000, 'oew_kg': 142400, 'max_fuel_kg': 138000},
    'A388': {'mtow_kg': 575000, 'oew_kg': 277000, 'max_fuel_kg': 320000},
    
    # Boeing 737 Family
    'B37M': {'mtow_kg': 82200, 'oew_kg': 45100, 'max_fuel_kg': 26020},
    'B38M': {'mtow_kg': 82200, 'oew_kg': 45100, 'max_fuel_kg': 26020},
    'B39M': {'mtow_kg': 88300, 'oew_kg': 46550, 'max_fuel_kg': 26020},
    'B3XM': {'mtow_kg': 89800, 'oew_kg': 50300, 'max_fuel_kg': 26020},
    'B734': {'mtow_kg': 68000, 'oew_kg': 38300, 'max_fuel_kg': 26020},
    'B737': {'mtow_kg': 70100, 'oew_kg': 39800, 'max_fuel_kg': 28600},
    'B738': {'mtow_kg': 79000, 'oew_kg': 41413, 'max_fuel_kg': 28600},
    'B739': {'mtow_kg': 85100, 'oew_kg': 44676, 'max_fuel_kg': 30190},
    
    # Boeing Wide-Body
    'B744': {'mtow_kg': 412775, 'oew_kg': 178100, 'max_fuel_kg': 216840},
    'B748': {'mtow_kg': 447700, 'oew_kg': 197130, 'max_fuel_kg': 243120},
    'B752': {'mtow_kg': 115680, 'oew_kg': 58390, 'max_fuel_kg': 52300},
    'B763': {'mtow_kg': 186880, 'oew_kg': 90010, 'max_fuel_kg': 91380},
    'B772': {'mtow_kg': 297560, 'oew_kg': 145150, 'max_fuel_kg': 171170},
    'B773': {'mtow_kg': 351530, 'oew_kg': 167830, 'max_fuel_kg': 181280},
    'B77W': {'mtow_kg': 351530, 'oew_kg': 167830, 'max_fuel_kg': 181280},
    'B788': {'mtow_kg': 227930, 'oew_kg': 119950, 'max_fuel_kg': 126210},
    'B789': {'mtow_kg': 254010, 'oew_kg': 128850, 'max_fuel_kg': 126370},
    
    # Embraer Regional Jets
    'E145': {'mtow_kg': 22000, 'oew_kg': 12400, 'max_fuel_kg': 6200},
    'E170': {'mtow_kg': 38600, 'oew_kg': 21620, 'max_fuel_kg': 11187},
    'E190': {'mtow_kg': 51800, 'oew_kg': 29540, 'max_fuel_kg': 15200},
    'E195': {'mtow_kg': 52290, 'oew_kg': 29100, 'max_fuel_kg': 15200},
    'E75L': {'mtow_kg': 39380, 'oew_kg': 22010, 'max_fuel_kg': 10300},
    
    # Business Jets
    'C550': {'mtow_kg': 9072, 'oew_kg': 5125, 'max_fuel_kg': 3619},
    'GLF6': {'mtow_kg': 45360, 'oew_kg': 24040, 'max_fuel_kg': 18600},
}


# =============================================================================
# LOAD DATA & EXTRACT ALL FLIGHT IDs
# =============================================================================


print("\n[Loading data...]")


fuel_df_raw = pd.read_parquet(DATA_DIR / 'fuel_train.parquet')
flightlist_df = pd.read_parquet(DATA_DIR / 'flightlist_train.parquet')


# Extract ALL flight IDs
TARGET_FLIGHTS = fuel_df_raw['flight_id'].unique().tolist()
print(f"✓ Found {len(TARGET_FLIGHTS)} unique flights")


# Filter to supported aircraft types
supported_types = list(AIRCRAFT_DATA.keys())
fuel_df = fuel_df_raw.merge(flightlist_df[['flight_id', 'aircraft_type']], 
                            on='flight_id', how='left')
fuel_df_supported = fuel_df[fuel_df['aircraft_type'].isin(supported_types)].copy()


TARGET_FLIGHTS = fuel_df_supported['flight_id'].unique().tolist()
print(f"✓ Filtered to {len(TARGET_FLIGHTS)} flights with supported aircraft")


fuel_df = fuel_df_supported.copy()
fuel_df['start_dt'] = pd.to_datetime(fuel_df['start'])
fuel_df['end_dt'] = pd.to_datetime(fuel_df['end'])


parquet_files = list((DATA_DIR / 'flights_train' / 'flights_train').glob('*.parquet'))


flight_data = {}
print(f"\n[Loading trajectories...]")
for file_path in tqdm(parquet_files, desc="Loading flight trajectories"):
    try:
        traj = pd.read_parquet(file_path)
        for flight_id, group in traj.groupby('flight_id'):
            if flight_id in TARGET_FLIGHTS:
                group = group.sort_values('timestamp').reset_index(drop=True)
                flight_data[flight_id] = group
    except:
        pass


print(f"✓ Loaded {len(flight_data)} flights with trajectory data\n")


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
        
        # Phase detection
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
    """Determine flight segment based on altitude and phase"""
    
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
    """Extract trajectory for specific interval"""
    traj_copy = full_traj.copy()
    traj_copy['timestamp'] = pd.to_datetime(traj_copy['timestamp'])
    interval_start = pd.Timestamp(interval_start)
    interval_end = pd.Timestamp(interval_end)
    
    mask = (traj_copy['timestamp'] >= interval_start) & (traj_copy['timestamp'] <= interval_end)
    return traj_copy[mask].copy()


def calculate_missing_data_pct(interval_traj):
    """Calculate missing data percentages"""
    
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
    """
    ✓ NEW: Calculate data density metrics
    - data_points_per_second: Number of data points per second
    - mean_time_between_points_sec: Average time interval between consecutive data points
    """
    
    if len(interval_traj) < 2 or interval_duration_sec <= 0:
        return {
            'data_points_per_second': 0.0,
            'mean_time_between_points_sec': 0.0
        }
    
    # Data points per second
    data_points_per_second = len(interval_traj) / interval_duration_sec
    
    # Mean time between consecutive points
    timestamps = pd.to_datetime(interval_traj['timestamp'])
    time_diffs = (timestamps.diff().dt.total_seconds()).dropna()
    
    if len(time_diffs) > 0:
        mean_time_between_points = time_diffs.mean()
    else:
        mean_time_between_points = 0.0
    
    return {
        'data_points_per_second': data_points_per_second,
        'mean_time_between_points_sec': mean_time_between_points
    }


def interpolate_trajectory(traj_df):
    """Simple linear interpolation for missing values"""
    traj = traj_df.copy()
    traj['groundspeed'] = traj['groundspeed'].interpolate(method='linear', limit_direction='both')
    traj['altitude'] = traj['altitude'].interpolate(method='linear', limit_direction='both')
    traj['vertical_rate'] = traj['vertical_rate'].interpolate(method='linear', limit_direction='both')
    return traj.fillna(0)


def calculate_flight_fuel_consumption(full_traj, aircraft_type, up_to_timestamp=None):
    """
    ✓ KEY FIX: Calculate fuel burned from FLIGHT START to specified timestamp
    ✓ UPDATED: Uses use_synonym=True for aircraft with missing drag polars
    """
    
    if aircraft_type not in AIRCRAFT_DATA:
        return 0.0
    
    ac_code = aircraft_type.lower()
    
    try:
        ff_model = FuelFlow(ac=ac_code, use_synonym=True)  # ✓ FIXED
    except Exception as e:
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
    """
    ✓ CORRECTED: OpenAP with TRUE dynamic mass
    ✓ UPDATED: Uses use_synonym=True for aircraft with missing drag polars
    """
    try:
        if not HAS_OPENAP or len(interval_traj) < 2:
            return None, "Not available"
        
        ac_code = aircraft_type.lower()
        
        try:
            ff_model = FuelFlow(ac=ac_code, use_synonym=True)  # ✓ FIXED
        except Exception as e:
            error_msg = str(e)[:150]
            return None, f"FuelFlow error: {error_msg}"
        
        timestamps = pd.to_datetime(interval_traj['timestamp'])
        time_deltas = (timestamps - timestamps.iloc[0]).dt.total_seconds()
        
        if aircraft_type not in AIRCRAFT_DATA:
            return None, f"No specs for {aircraft_type}"
        
        ac_data = AIRCRAFT_DATA[aircraft_type]
        mtow = ac_data['mtow_kg']
        oew = ac_data['oew_kg']
        
        fuel_burned_before_interval = calculate_flight_fuel_consumption(
            full_traj, aircraft_type, up_to_timestamp=interval_start
        )
        
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
        mean_ff = np.mean(fuel_flows)
        
        return {
            'total_fuel_kg': total_fuel,
            'mean_fuel_flow_kg_s': mean_ff,
            'phase_used': phase,
            'starting_mass_kg': (mtow * 0.70) - fuel_burned_before_interval,
        }, "Success"
    except Exception as e:
        return None, str(e)[:150]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


results = []
flights_processed = 0
flights_skipped = 0
aircraft_errors = {}


for flight_idx, flight_id in enumerate(TARGET_FLIGHTS):
    print(f"\n{'='*80}")
    print(f"[{flight_idx+1}/{len(TARGET_FLIGHTS)}] FLIGHT: {flight_id}")
    print(f"{'='*80}")
    
    if flight_id not in flight_data:
        print("⚠️ No trajectory data")
        flights_skipped += 1
        continue
    
    full_traj = flight_data[flight_id]
    flight_info = fuel_df[fuel_df['flight_id'] == flight_id]
    if len(flight_info) == 0:
        print("⚠️ No fuel interval data")
        flights_skipped += 1
        continue
    
    aircraft_type = flight_info['aircraft_type'].iloc[0]
    print(f"Aircraft: {aircraft_type}")
    print(f"Intervals: {len(flight_info)} | Processing all...")
    
    intervals_processed = 0
    
    for idx, (i, row) in enumerate(flight_info.iterrows()):
        interval_start = row['start_dt']
        interval_end = row['end_dt']
        actual_fuel = row['fuel_kg']
        interval_duration = (pd.to_datetime(interval_end) - pd.to_datetime(interval_start)).total_seconds()
        
        print(f"  Interval {idx+1}/{len(flight_info)}: {actual_fuel:.1f} kg actual", end=" ")
        
        interval_traj = extract_interval_trajectory(full_traj, interval_start, interval_end)
        data_density = calculate_data_density(interval_traj, interval_duration)
        missing_data = calculate_missing_data_pct(interval_traj)
        
        if len(interval_traj) == 0:
            phase = 'UNKNOWN'
            phase_info = {
                'alt_start_ft': 0, 'alt_end_ft': 0, 'alt_change_ft': 0,
                'alt_avg_ft': 0, 'gs_avg_kts': 0, 'vs_avg_fpm': 0
            }
            interval_traj_interp = pd.DataFrame()
        else:
            interval_traj_interp = interpolate_trajectory(interval_traj)
            phase, phase_info = detect_flight_phase_custom(interval_traj_interp)
        
        openap_result, openap_msg = estimate_with_openap_correct_mass(
            aircraft_type, full_traj, interval_traj_interp, interval_start, phase
        ) if len(interval_traj_interp) > 0 else (None, "Empty")
        
        if openap_result:
            openap_fuel = openap_result['total_fuel_kg']
            openap_error = ((openap_fuel - actual_fuel) / actual_fuel) * 100 if actual_fuel > 0 else 0
            openap_mae = abs(openap_error)
            openap_phase = openap_result.get('phase_used', phase)
            starting_mass = openap_result.get('starting_mass_kg', 0)
            print(f"| Predicted: {openap_fuel:.1f} kg | Error: {openap_error:+.1f}%")
        else:
            openap_fuel = None
            openap_error = None
            openap_mae = None
            openap_phase = phase
            starting_mass = 0
            print(f"| {openap_msg}")
            
            if aircraft_type not in aircraft_errors:
                aircraft_errors[aircraft_type] = []
            aircraft_errors[aircraft_type].append(openap_msg)
        
        results.append({
            'flight_id': flight_id,
            'interval_idx': i,
            'aircraft': aircraft_type,
            'phase': phase,
            'alt_start_ft': phase_info.get('alt_start_ft', 0),
            'alt_end_ft': phase_info.get('alt_end_ft', 0),
            'alt_change_ft': phase_info.get('alt_change_ft', 0),
            'gs_avg_kts': phase_info.get('gs_avg_kts', 0),
            'vs_avg_fpm': phase_info.get('vs_avg_fpm', 0),
            'actual_fuel_kg': actual_fuel,
            'interval_points': len(interval_traj),
            'interval_duration_sec': interval_duration,
            'data_points_per_second': data_density['data_points_per_second'],
            'mean_time_between_points_sec': data_density['mean_time_between_points_sec'],
            'groundspeed_missing%': missing_data['groundspeed_missing%'],
            'altitude_missing%': missing_data['altitude_missing%'],
            'vertical_rate_missing%': missing_data['vertical_rate_missing%'],
            'total_missing%': missing_data['total_missing%'],
            'openap_fuel_kg': openap_fuel,
            'openap_phase_used': openap_phase,
            'openap_error%': openap_error,
            'openap_mae%': openap_mae,
            'openap_status': openap_msg,
            'starting_mass_kg': starting_mass,
        })
        
        intervals_processed += 1
    
    print(f"✓ Processed {intervals_processed} intervals")
    flights_processed += 1
    
    if flights_processed % 100 == 0:
        partial_df = pd.DataFrame(results)
        partial_df.to_csv(RESULTS_DIR / f'augmented_openap_checkpoint_{flights_processed}.csv', index=False)
        print(f"✓ Checkpoint saved: {flights_processed} flights, {len(results)} intervals")
        print(f"   Location: {RESULTS_DIR}/augmented_openap_checkpoint_{flights_processed}.csv")


# =============================================================================
# RESULTS & SUMMARY
# =============================================================================


print(f"\n\n{'='*80}")
print("RESULTS SUMMARY - TRUE Dynamic Mass (Flight-Level Tracking)")
print(f"{'='*80}\n")

print(f"Flights processed: {flights_processed}")
print(f"Flights skipped: {flights_skipped}")

results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(f"Total intervals processed: {len(results_df)}")

successful_openap = results_df['openap_mae%'].notna().sum()
print(f"Successful OpenAP: {successful_openap}/{len(results_df)}\n")

if aircraft_errors:
    print(f"\n{'='*80}")
    print("AIRCRAFT WITH ERRORS")
    print(f"{'='*80}\n")
    for ac_type, errors in sorted(aircraft_errors.items()):
        print(f"{ac_type}: {len(errors)} failures")
        if errors:
            print(f"  First error: {errors[0]}")

print(f"\n{'='*80}")
print("RESULTS BY FLIGHT PHASE")
print(f"{'='*80}\n")

for phase in ['ON_GROUND', 'CLIMB', 'CRUISE', 'DESCENT', 'UNKNOWN']:
    phase_df = results_df[results_df['phase'] == phase]
    if len(phase_df) == 0:
        continue
    
    openap_valid = phase_df['openap_mae%'].notna().sum()
    
    print(f"{phase:12s} ({len(phase_df):3d} intervals, {openap_valid} successful):")
    print(f"  Altitude:        {phase_df['alt_start_ft'].mean():7.0f} → {phase_df['alt_end_ft'].mean():7.0f} ft (Δ {phase_df['alt_change_ft'].mean():+7.0f} ft)")
    print(f"  Fuel range:      {phase_df['actual_fuel_kg'].min():7.1f} - {phase_df['actual_fuel_kg'].max():7.1f} kg")
    
    if openap_valid > 0:
        print(f"  OpenAP MAE:      {phase_df['openap_mae%'].mean():6.2f}% (median: {phase_df['openap_mae%'].median():6.2f}%)")
    else:
        print(f"  OpenAP MAE:      N/A")
    print()

print(f"{'='*80}")
print("OVERALL STATISTICS")
print(f"{'='*80}\n")

openap_mae = results_df['openap_mae%'].dropna()

if len(openap_mae) > 0:
    print(f"OpenAP with TRUE Dynamic Mass (n={len(openap_mae)}/{len(results_df)}):")
    print(f"  Mean MAE:   {openap_mae.mean():.2f}%")
    print(f"  Median MAE: {openap_mae.median():.2f}%")
    print(f"  Std Dev:    {openap_mae.std():.2f}%")
    print(f"  Range:      {openap_mae.min():.2f}% - {openap_mae.max():.2f}%")

results_df.to_csv(RESULTS_DIR / 'augmented_openap_correct_mass_ALL_FLIGHTS.csv', index=False)
print(f"\n✓ Saved final results: augmented_openap_correct_mass_ALL_FLIGHTS.csv")
print(f"  Location: {RESULTS_DIR}/augmented_openap_correct_mass_ALL_FLIGHTS.csv")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE ✓")
print(f"{'='*80}")
print(f"\nAll files saved to: {RESULTS_DIR}/")
