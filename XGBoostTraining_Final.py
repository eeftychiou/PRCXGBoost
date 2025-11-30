import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
import logging
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import time
import json
import sys

warnings.filterwarnings('ignore')

# Aircraft specifications database
AIRCRAFT_DATA = {
    'A19N': {'mtow_kg': 79200, 'oew_kg': 45100, 'mlw_kg': 67400, 'max_fuel_kg': 27200, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A20N': {'mtow_kg': 79000, 'oew_kg': 45400, 'mlw_kg': 67400, 'max_fuel_kg': 27200, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A21N': {'mtow_kg': 97000, 'oew_kg': 50300, 'mlw_kg': 79200, 'max_fuel_kg': 32840, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.025},
    'A318': {'mtow_kg': 68000, 'oew_kg': 39500, 'mlw_kg': 57500, 'max_fuel_kg': 24210, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.025},
    'A319': {'mtow_kg': 75500, 'oew_kg': 40800, 'mlw_kg': 62500, 'max_fuel_kg': 24210, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A320': {'mtow_kg': 78000, 'oew_kg': 42400, 'mlw_kg': 66000, 'max_fuel_kg': 27200, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A321': {'mtow_kg': 93500, 'oew_kg': 48700, 'mlw_kg': 77800, 'max_fuel_kg': 32840, 'type': 'narrowbody',
             'wing_area_m2': 128.0, 'cd0': 0.025},
    'A332': {'mtow_kg': 233000, 'oew_kg': 119500, 'mlw_kg': 182000, 'max_fuel_kg': 139090, 'type': 'widebody',
             'wing_area_m2': 361.6, 'cd0': 0.022},
    'A333': {'mtow_kg': 242000, 'oew_kg': 123400, 'mlw_kg': 187000, 'max_fuel_kg': 139090, 'type': 'widebody',
             'wing_area_m2': 361.6, 'cd0': 0.022},
    'A343': {'mtow_kg': 275000, 'oew_kg': 131000, 'mlw_kg': 192000, 'max_fuel_kg': 155040, 'type': 'widebody',
             'wing_area_m2': 363.1, 'cd0': 0.023},
    'A359': {'mtow_kg': 280000, 'oew_kg': 142400, 'mlw_kg': 205000, 'max_fuel_kg': 138000, 'type': 'widebody',
             'wing_area_m2': 442.0, 'cd0': 0.021},
    'A388': {'mtow_kg': 575000, 'oew_kg': 277000, 'mlw_kg': 427000, 'max_fuel_kg': 320000, 'type': 'widebody',
             'wing_area_m2': 845.0, 'cd0': 0.022},
    'B37M': {'mtow_kg': 82200, 'oew_kg': 45100, 'mlw_kg': 69400, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.023},
    'B38M': {'mtow_kg': 82200, 'oew_kg': 45100, 'mlw_kg': 69400, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.023},
    'B39M': {'mtow_kg': 88300, 'oew_kg': 46550, 'mlw_kg': 72350, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.024},
    'B3XM': {'mtow_kg': 89800, 'oew_kg': 50300, 'mlw_kg': 73940, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 127.4, 'cd0': 0.024},
    'B734': {'mtow_kg': 68000, 'oew_kg': 38300, 'mlw_kg': 56200, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 105.4, 'cd0': 0.025},
    'B737': {'mtow_kg': 70100, 'oew_kg': 39800, 'mlw_kg': 58000, 'max_fuel_kg': 28600, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.024},
    'B738': {'mtow_kg': 79000, 'oew_kg': 41413, 'mlw_kg': 66360, 'max_fuel_kg': 28600, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.023},
    'B739': {'mtow_kg': 85100, 'oew_kg': 44676, 'mlw_kg': 70300, 'max_fuel_kg': 30190, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.024},
    'B744': {'mtow_kg': 412775, 'oew_kg': 178100, 'mlw_kg': 295742, 'max_fuel_kg': 216840, 'type': 'widebody',
             'wing_area_m2': 541.0, 'cd0': 0.023},
    'B748': {'mtow_kg': 447700, 'oew_kg': 197130, 'mlw_kg': 312072, 'max_fuel_kg': 243120, 'type': 'widebody',
             'wing_area_m2': 554.0, 'cd0': 0.022},
    'B752': {'mtow_kg': 115680, 'oew_kg': 58390, 'mlw_kg': 99800, 'max_fuel_kg': 52300, 'type': 'narrowbody',
             'wing_area_m2': 185.3, 'cd0': 0.024},
    'B763': {'mtow_kg': 186880, 'oew_kg': 90010, 'mlw_kg': 145150, 'max_fuel_kg': 91380, 'type': 'widebody',
             'wing_area_m2': 283.3, 'cd0': 0.023},
    'B772': {'mtow_kg': 297560, 'oew_kg': 145150, 'mlw_kg': 213180, 'max_fuel_kg': 171170, 'type': 'widebody',
             'wing_area_m2': 427.8, 'cd0': 0.022},
    'B773': {'mtow_kg': 351530, 'oew_kg': 167830, 'mlw_kg': 251290, 'max_fuel_kg': 181280, 'type': 'widebody',
             'wing_area_m2': 427.8, 'cd0': 0.022},
    'B77W': {'mtow_kg': 351530, 'oew_kg': 167830, 'mlw_kg': 251290, 'max_fuel_kg': 181280, 'type': 'widebody',
             'wing_area_m2': 427.8, 'cd0': 0.022},
    'B788': {'mtow_kg': 227930, 'oew_kg': 119950, 'mlw_kg': 172365, 'max_fuel_kg': 126210, 'type': 'widebody',
             'wing_area_m2': 325.0, 'cd0': 0.021},
    'B789': {'mtow_kg': 254010, 'oew_kg': 128850, 'mlw_kg': 192775, 'max_fuel_kg': 126370, 'type': 'widebody',
             'wing_area_m2': 325.0, 'cd0': 0.021},
    'E145': {'mtow_kg': 22000, 'oew_kg': 12400, 'mlw_kg': 20200, 'max_fuel_kg': 6200, 'type': 'narrowbody',
             'wing_area_m2': 51.2, 'cd0': 0.027},
    'E170': {'mtow_kg': 38600, 'oew_kg': 21620, 'mlw_kg': 35990, 'max_fuel_kg': 11187, 'type': 'narrowbody',
             'wing_area_m2': 72.7, 'cd0': 0.026},
    'E190': {'mtow_kg': 51800, 'oew_kg': 29540, 'mlw_kg': 47790, 'max_fuel_kg': 15200, 'type': 'narrowbody',
             'wing_area_m2': 92.5, 'cd0': 0.025},
    'E195': {'mtow_kg': 52290, 'oew_kg': 29100, 'mlw_kg': 48280, 'max_fuel_kg': 15200, 'type': 'narrowbody',
             'wing_area_m2': 92.5, 'cd0': 0.026},
    'E75L': {'mtow_kg': 39380, 'oew_kg': 22010, 'mlw_kg': 36200, 'max_fuel_kg': 10300, 'type': 'narrowbody',
             'wing_area_m2': 82.0, 'cd0': 0.026},
    'C550': {'mtow_kg': 9072, 'oew_kg': 5125, 'mlw_kg': 8618, 'max_fuel_kg': 3619, 'type': 'narrowbody',
             'wing_area_m2': 31.8, 'cd0': 0.028},
    'GLF6': {'mtow_kg': 45360, 'oew_kg': 24040, 'mlw_kg': 34700, 'max_fuel_kg': 18600, 'type': 'widebody',
             'wing_area_m2': 94.0, 'cd0': 0.022},
}

# WIDEBODY AIRCRAFT LIST
WIDEBODY_AIRCRAFT = ['A332', 'A333', 'A343', 'A359', 'A388', 'B744', 'B748', 
                     'B763', 'B772', 'B773', 'B77W', 'B788', 'B789']

# File paths
DATA_PATH = 'data/AugmentedDataFromOPENAP/augmented_openap_correct_mass_ALL_FLIGHTS_final.csv'
APT_PATH = 'data/apt.parquet'
FLIGHTLIST_PATH = 'data/flightlist_train.parquet'
FUEL_PATH = 'data/fuel_train.parquet'
TEST_CSV_PATH = 'data/AugmentedDataFromOPENAP/augmented_openap_rank_final_ALL_FLIGHTS.csv'
RANK_CSV_PATH = 'data/AugmentedDataFromOPENAP/augmented_openap_submission_ALL_FLIGHTSrank.csv'
FUEL_RANK_PATH = 'data/fuel_final_submission.parquet'
FLIGHTLIST_RANK_PATH = 'data/flightlist_rank.parquet'
FLIGHTLIST_FINAL_PATH = 'data/flightlist_final.parquet'
RESULTS_DIR = 'Results'

FEATURED_DATA_TRAIN = 'data/featured_data_merged.parquet'
FEATURED_DATA_RANK = 'data/featured_data_rank_merged.parquet'
FEATURED_DATA_TEST = 'data/featured_data_final4.parquet'
SYNTHETIC_PATH = os.path.join(RESULTS_DIR, "synthetic_widebody.parquet")
SELECTED_FEATURES_PATH = 'data/selected_features_sfs3.json'


os.makedirs(RESULTS_DIR, exist_ok=True)

log_file = os.path.join(RESULTS_DIR, f'xgboost_top5_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Extended feature list
EXTENDED_FEATURES_FROM_PARQUET = [
    "origin_icao", "origin_name", "destination_icao", "destination_name", "mfc", "pax_high", 
    "fuselage_height", "wing_mac", "wing_t/c", "flaps_type", "flaps_area", "flaps_bf/b", 
    "flaps_Sf/S", "cruise_mach", "engine_default", "drag_cd0", "drag_e", "drag_gears", 
    "fuel_fuel_coef", "limits_OEW", "origin_longitude", "origin_latitude", "origin_elevation", 
    "origin_RWY_1_HEADING_a", "origin_RWY_1_LENGTH", "origin_RWY_2_HEADING_a", "origin_RWY_3_LENGTH", 
    "origin_RWY_4_HEADING_b", "origin_RWY_4_LENGTH", "origin_RWY_5_HEADING_b", "origin_RWY_5_LENGTH", 
    "origin_RWY_8_HEADING_b", "destination_longitude", "destination_latitude", "destination_elevation", 
    "destination_RWY_1_LENGTH", "destination_RWY_2_HEADING_a", "destination_RWY_2_HEADING_b", 
    "destination_RWY_2_LENGTH", "destination_RWY_3_HEADING_a", "destination_RWY_3_HEADING_b", 
    "destination_RWY_3_LENGTH", "destination_RWY_4_HEADING_a", "destination_RWY_4_HEADING_b", 
    "destination_RWY_5_HEADING_b", "destination_RWY_5_LENGTH", "destination_RWY_6_LENGTH", 
    "segment_duration", "seg_latitude_mean", "seg_latitude_std", 
    "seg_longitude_min", "seg_longitude_mean", "seg_longitude_std", "seg_altitude_min", 
    "seg_altitude_max", "seg_altitude_mean", "seg_groundspeed_mean", "seg_track_max", 
    "seg_track_mean", "seg_track_std", "seg_vertical_rate_max", "seg_vertical_rate_std", 
    "seg_mach_min", "seg_mach_max", "seg_mach_mean", "seg_mach_std", "seg_TAS_min", 
    "seg_TAS_max", "seg_TAS_mean", "seg_TAS_std", "seg_calculated_speed_min", 
    "seg_calculated_speed_mean", "seg_calculated_speed_std", "seg_vertical_rate_change_min", 
    "seg_vertical_rate_change_max", "seg_vertical_rate_change_mean", "seg_vertical_rate_change_std", 
    "seg_dist_to_origin_km_min", "seg_dist_to_origin_km_max", "seg_dist_to_origin_km_mean", 
    "seg_dist_to_origin_km_std", "seg_dist_to_dest_km_min", "seg_dist_to_dest_km_max", 
    "seg_dist_to_dest_km_mean", "seg_dist_to_dest_km_std", "phase_fraction_climb", 
    "phase_fraction_cruise", "phase_fraction_descent", "phase_fraction_approach", 
    "start_alt_rev", "end_alt_rev", "departure_rwy_length", "segment_distance_km", 
    "alt_diff_rev", "alt_diff_rev_std", "seg_latitude_delta", "seg_longitude_delta", 
    "seg_altitude_delta", "seg_vertical_rate_delta", "seg_mach_delta", "seg_TAS_delta", 
    "seg_CAS_delta", "seg_calculated_speed_delta", "seg_dist_to_origin_km_delta", 
    "seg_dist_to_dest_km_delta", "takeoff_delta", "landing_delta", "mean_time_in_air", 
    "aircraft_encoded", "start_time_encoded", "end_time_encoded",
    "fuel_consumption_gnd", "fuel_consumption_cl",
    "fuel_consumption_de", "fuel_consumption_lvl", "fuel_consumption_cr", "fuel_consumption_na", "fuel_consumption", "seg_avg_burn_rate",
    "average_load_factor", "estimated_payload_kg", "trip_fuel_kg", "contingency_fuel_kg", "final_reserve_fuel_kg", "estimated_total_fuel_kg",
    "estimated_takeoff_mass"
]


def haversine(lon1, lat1, lon2, lat2):
    if pd.isna([lon1, lat1, lon2, lat2]).any():
        return np.nan
    try:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * 6371
    except:
        return np.nan


# ============================================================================
# SYNTHETIC DATA GENERATION FUNCTION
# ============================================================================
# ============================================================================
# ENHANCED SYNTHETIC DATA GENERATION FUNCTION (25K samples, 25% from long segments)
# ============================================================================
# def generate_synthetic_widebody_data_enhanced(df_train, n_synthetic=25000, long_segment_pct=0.25, random_state=42):
def generate_synthetic_widebody_data_enhanced(df_train, n_synthetic=25000, long_segment_pct=0.25, random_state=42):
    """
    Generate synthetic data for widebody aircraft with emphasis on long segments.
    
    Parameters:
    -----------
    df_train : DataFrame
        Original training data with 'aircraft_type' column
    n_synthetic : int
        Total number of synthetic samples to generate (default: 25000)
    long_segment_pct : float
        Percentage of synthetic samples to generate from long segments (default: 0.25)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    df_synthetic : DataFrame
        Synthetic training data for widebody aircraft
    """
    np.random.seed(random_state)
    
    # Filter widebody aircraft from training data
    df_widebody = df_train[df_train['aircraft_type'].isin(WIDEBODY_AIRCRAFT)].copy()
    
    logger.info(f"Original widebody samples: {len(df_widebody):,}")
    logger.info(f"Generating {n_synthetic:,} synthetic samples...")
    logger.info(f"  - {int(n_synthetic * long_segment_pct):,} from LONG segments ({long_segment_pct*100:.0f}%)")
    logger.info(f"  - {int(n_synthetic * (1-long_segment_pct)):,} from ALL segments ({(1-long_segment_pct)*100:.0f}%)")
    
    # Identify segment duration column
    duration_col = None
    for col in ['segment_duration', 'interval_duration_sec', 'duration']:
        if col in df_widebody.columns:
            duration_col = col
            break
    
    if duration_col is None:
        logger.warning("⚠️  No duration column found, using all data equally")
        long_segment_pct = 0  # Disable long segment emphasis
    else:
        logger.info(f"[+] Using '{duration_col}' to identify long segments")
        
        # Calculate percentile threshold for "long" segments
        duration_75th = df_widebody[duration_col].quantile(0.75)
        duration_90th = df_widebody[duration_col].quantile(0.90)
        
        logger.info(f"    Duration 75th percentile: {duration_75th:.1f}")
        logger.info(f"    Duration 90th percentile: {duration_90th:.1f}")
        
        # Define long segments as top 25% (75th percentile and above)
        df_widebody['is_long_segment'] = df_widebody[duration_col] >= duration_75th
        
        n_long = df_widebody['is_long_segment'].sum()
        logger.info(f"    Long segments identified: {n_long:,} ({n_long/len(df_widebody)*100:.1f}%)")
    
    # Calculate samples per aircraft type (proportional to existing distribution)
    aircraft_counts = df_widebody['aircraft_type'].value_counts()
    aircraft_proportions = aircraft_counts / aircraft_counts.sum()
    
    logger.info("\nOriginal widebody distribution:")
    for aircraft, count in aircraft_counts.items():
        logger.info(f"  {aircraft}: {count:,} ({count/len(df_widebody)*100:.1f}%)")
    
    synthetic_samples = []
    
    # Calculate how many samples from long vs all segments
    n_from_long = int(n_synthetic * long_segment_pct)
    n_from_all = n_synthetic - n_from_long
    
    for aircraft_type in WIDEBODY_AIRCRAFT:
        # Get all samples for this aircraft type
        aircraft_data = df_widebody[df_widebody['aircraft_type'] == aircraft_type].copy()
        
        if len(aircraft_data) == 0:
            logger.info(f"  Warning: No training data for {aircraft_type}, skipping...")
            continue
        
        # Calculate number of synthetic samples for this aircraft
        n_samples_aircraft_total = int(n_synthetic * aircraft_proportions.get(aircraft_type, 1/len(WIDEBODY_AIRCRAFT)))
        
        if n_samples_aircraft_total == 0:
            n_samples_aircraft_total = int(n_synthetic / len(WIDEBODY_AIRCRAFT))
        
        # Split into long-segment and all-segment quotas
        n_aircraft_from_long = int(n_samples_aircraft_total * long_segment_pct)
        n_aircraft_from_all = n_samples_aircraft_total - n_aircraft_from_long
        
        logger.info(f"\n  Generating {n_samples_aircraft_total:,} samples for {aircraft_type}:")
        logger.info(f"    - {n_aircraft_from_long:,} from long segments")
        logger.info(f"    - {n_aircraft_from_all:,} from all segments")
        
        # Get long segments for this aircraft
        if duration_col and 'is_long_segment' in aircraft_data.columns:
            aircraft_data_long = aircraft_data[aircraft_data['is_long_segment']].copy()
            logger.info(f"    - Available long segments: {len(aircraft_data_long):,}")
        else:
            aircraft_data_long = aircraft_data.copy()
        
        # Identify numerical and categorical columns
        numerical_cols = aircraft_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that should not be perturbed
        exclude_cols = ['flight_id', 'interval_idx', 'idx', 'is_long_segment']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Generate samples from LONG segments
        for i in range(n_aircraft_from_long):
            if len(aircraft_data_long) == 0:
                # Fallback to all data if no long segments
                base_sample = aircraft_data.sample(n=1, random_state=random_state+i).iloc[0].copy()
            else:
                base_sample = aircraft_data_long.sample(n=1, random_state=random_state+i).iloc[0].copy()
            
            # Perturb numerical features with small noise
            for col in numerical_cols:
                if col in base_sample.index and pd.notna(base_sample[col]):
                    original_value = base_sample[col]
                    
                    # Calculate standard deviation for this feature in this aircraft type
                    if len(aircraft_data_long) > 1:
                        col_std = aircraft_data_long[col].std()
                    else:
                        col_std = aircraft_data[col].std()
                    
                    if pd.notna(col_std) and col_std > 0:
                        # Add noise: 5-15% of standard deviation
                        noise_factor = np.random.uniform(0.05, 0.15)
                        noise = np.random.normal(0, col_std * noise_factor)
                        new_value = original_value + noise
                        
                        # Ensure physical constraints
                        if col in ['starting_mass_kg', 'actual_fuel_kg', 'fuel_kg']:
                            new_value = max(0, new_value)
                        elif col in ['alt_end_ft', 'alt_avg_ft', 'alt_start_ft']:
                            new_value = max(0, min(new_value, 45000))
                        elif col in ['gs_avg_kts', 'seg_groundspeed_mean', 'seg_TAS_mean']:
                            new_value = max(0, min(new_value, 600))
                        elif col in ['interval_duration_sec', 'segment_duration']:
                            new_value = max(1, new_value)
                        
                        base_sample[col] = new_value
            
            synthetic_samples.append(base_sample)
        
        # Generate samples from ALL segments
        for i in range(n_aircraft_from_all):
            base_sample = aircraft_data.sample(n=1, random_state=random_state+n_aircraft_from_long+i).iloc[0].copy()
            
            # Perturb numerical features with small noise
            for col in numerical_cols:
                if col in base_sample.index and pd.notna(base_sample[col]):
                    original_value = base_sample[col]
                    
                    col_std = aircraft_data[col].std()
                    
                    if pd.notna(col_std) and col_std > 0:
                        # Add noise: 5-15% of standard deviation
                        noise_factor = np.random.uniform(0.05, 0.15)
                        noise = np.random.normal(0, col_std * noise_factor)
                        new_value = original_value + noise
                        
                        # Ensure physical constraints
                        if col in ['starting_mass_kg', 'actual_fuel_kg', 'fuel_kg']:
                            new_value = max(0, new_value)
                        elif col in ['alt_end_ft', 'alt_avg_ft', 'alt_start_ft']:
                            new_value = max(0, min(new_value, 45000))
                        elif col in ['gs_avg_kts', 'seg_groundspeed_mean', 'seg_TAS_mean']:
                            new_value = max(0, min(new_value, 600))
                        elif col in ['interval_duration_sec', 'segment_duration']:
                            new_value = max(1, new_value)
                        
                        base_sample[col] = new_value
            
            synthetic_samples.append(base_sample)
    
    df_synthetic = pd.DataFrame(synthetic_samples)
    
    # Remove the helper column
    if 'is_long_segment' in df_synthetic.columns:
        df_synthetic = df_synthetic.drop(columns=['is_long_segment'])
    
    logger.info(f"\n[+] Generated {len(df_synthetic):,} synthetic widebody samples")
    logger.info(f"[+] Synthetic distribution:")
    synth_counts = df_synthetic['aircraft_type'].value_counts()
    for aircraft, count in synth_counts.items():
        logger.info(f"  {aircraft}: {count:,} ({count/len(df_synthetic)*100:.1f}%)")
    
    # Analyze duration distribution if available
    if duration_col and duration_col in df_synthetic.columns:
        synth_duration_mean = df_synthetic[duration_col].mean()
        orig_duration_mean = df_widebody[duration_col].mean()
        logger.info(f"\n[+] Duration analysis:")
        logger.info(f"    Original mean duration: {orig_duration_mean:.1f}")
        logger.info(f"    Synthetic mean duration: {synth_duration_mean:.1f}")
        logger.info(f"    Increase: {(synth_duration_mean/orig_duration_mean - 1)*100:+.1f}%")
    
    return df_synthetic



def main():
    FORCE_RERUN_SFS = '--force-sfs' in sys.argv
    if FORCE_RERUN_SFS:
        logger.info("⚠️  Force SFS re-run flag detected - will ignore cached features")
    
    logger.info("="*70)
    logger.info("XGBoost FUEL PREDICTION - TOP 5 MODELS TRAINING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # ========================================================================
    # PHASE 1: LOAD AND PREPARE DATA
    # ========================================================================
    logger.info("\nPHASE 1: LOADING TRAINING DATA")

    apt = pd.read_parquet(APT_PATH)
    apt = apt[['icao', 'longitude', 'latitude']]
    logger.info(f"[+] Airports: {len(apt):,}")

    flightlist = pd.read_parquet(FLIGHTLIST_PATH)
    logger.info(f"[+] Flightlist (training): {len(flightlist):,}")

    flightlist = flightlist.merge(apt, left_on='origin_icao', right_on='icao', how='left', suffixes=('', '_origin'))
    flightlist = flightlist.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    flightlist = flightlist.drop(columns=['icao'], errors='ignore')
    flightlist = flightlist.merge(apt, left_on='destination_icao', right_on='icao', how='left', suffixes=('', '_dest'))
    flightlist = flightlist.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    flightlist = flightlist.drop(columns=['icao'], errors='ignore')
    flightlist['great_circle_distance'] = flightlist.apply(
        lambda row: haversine(row.get('origin_lon'), row.get('origin_lat'), 
                             row.get('dest_lon'), row.get('dest_lat')), axis=1
    )

    fuel = pd.read_parquet(FUEL_PATH)
    logger.info(f"[+] Fuel data: {len(fuel):,} intervals")

    logger.info("\nLoading extended feature data from parquets...")
    featured_data = pd.read_parquet(FEATURED_DATA_TRAIN)
    logger.info(f"[+] Featured data (train): {len(featured_data):,} rows, {len(featured_data.columns)} columns")

    featured_data_rank = pd.read_parquet(FEATURED_DATA_TEST)
    logger.info(f"[+] Featured data (test): {len(featured_data_rank):,} rows, {len(featured_data_rank.columns)} columns")

    train_cols = set(featured_data.columns)
    test_cols = set(featured_data_rank.columns)
    common_cols = train_cols.intersection(test_cols)

    logger.info(f"[+] Common columns: {len(common_cols)}")

    available_features = ['flight_id', 'idx']
    for col in EXTENDED_FEATURES_FROM_PARQUET:
        if col in common_cols:
            available_features.append(col)

    logger.info(f"[+] Selected {len(available_features)-2} common features")

    featured_data_selected = featured_data[available_features].copy()
    featured_data_selected = featured_data_selected.rename(columns={'idx': 'interval_idx'})

    featured_data_rank_selected = featured_data_rank[available_features].copy()
    featured_data_rank_selected = featured_data_rank_selected.rename(columns={'idx': 'interval_idx'})

    logger.info(f"\nLoading training CSV: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, delimiter=';', low_memory=False)
    logger.info(f"[+] Training data loaded: {len(df_raw):,} rows")

    flightlist_cols = ['flight_id', 'takeoff', 'landed', 'great_circle_distance',
                       'origin_icao', 'destination_icao', 'aircraft_type',
                       'origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']
    df_raw = df_raw.merge(flightlist[flightlist_cols], on='flight_id', how='left')

    fuel_intervals = fuel[['flight_id', 'idx', 'fuel_kg', 'start', 'end']].copy()
    fuel_intervals = fuel_intervals.rename(columns={'idx': 'interval_idx'})
    df_raw = df_raw.merge(fuel_intervals, on=['flight_id', 'interval_idx'], how='left')

    logger.info("Merging extended features...")
    df_raw = df_raw.merge(featured_data_selected, on=['flight_id', 'interval_idx'], how='left')
    logger.info(f"[+] Total columns: {len(df_raw.columns)}")

    # Build feature list
    base_features = [
        'starting_mass_kg', 'alt_end_ft', 'alt_avg_ft', 'gs_avg_kts', 'vs_avg_fpm',
        'interval_duration_sec', 'altitude_change_rate', 'great_circle_distance',
        'aircraft_type', 'end_hour', 'interval_elapsed_from_flight_start',
    ]

    extended_features_available = [col for col in available_features[2:] if col not in base_features]
    feature_cols_selected = base_features + extended_features_available

    logger.info(f"[+] Total features: {len(feature_cols_selected)}")

    # Add computed columns
    if 'alt_avg_ft' not in df_raw.columns:
        df_raw['alt_avg_ft'] = (df_raw.get('alt_start_ft', 0) + df_raw.get('alt_end_ft', 0)) / 2
    if 'altitude_change_rate' not in df_raw.columns:
        df_raw['altitude_change_rate'] = df_raw.get('alt_change_ft', 0) / (df_raw.get('interval_duration_sec', 60) + 1e-6)
    if 'end_hour' not in df_raw.columns:
        df_raw['end_hour'] = pd.to_datetime(df_raw.get('end'), errors='coerce').dt.hour.fillna(-1).astype(int)
    if 'interval_elapsed_from_flight_start' not in df_raw.columns:
        df_raw['interval_elapsed_from_flight_start'] = 0

    target_col = 'actual_fuel_kg'
    available_feature_cols = [col for col in feature_cols_selected if col in df_raw.columns]
    feature_cols_selected = available_feature_cols

    df_features = df_raw[feature_cols_selected + [target_col]].copy()
    df_features = df_features.dropna(subset=[target_col])
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    logger.info(f"[+] Original dataset: {len(df_features):,} intervals")

    # ========================================================================
    # PHASE 1.5: GENERATE SYNTHETIC WIDEBODY DATA (ENHANCED)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 1.5: GENERATING ENHANCED SYNTHETIC WIDEBODY DATA")
    logger.info("="*70)

    if os.path.exists(SYNTHETIC_PATH):
        logger.info(f"Loading cached synthetic data from {SYNTHETIC_PATH}")
        df_synthetic = pd.read_parquet(SYNTHETIC_PATH)
        logger.info(f"Loaded {len(df_synthetic)} cached synthetic samples")
    else:
        df_synthetic = generate_synthetic_widebody_data_enhanced(
            df_features,
            n_synthetic=25000,        # Generate 25,000 samples
            long_segment_pct=0.25,    # 25% from long segments
            random_state=42
        )
        df_synthetic.to_parquet(SYNTHETIC_PATH, index=False, engine='fastparquet')


    # Combine original and synthetic data
    df_features_augmented = pd.concat([df_features, df_synthetic], ignore_index=True)

    logger.info(f"\n[+] Original training size: {len(df_features):,}")
    logger.info(f"[+] Synthetic samples added: {len(df_synthetic):,}")
    logger.info(f"[+] Augmented training size: {len(df_features_augmented):,}")
    logger.info(f"[+] Augmentation rate: {len(df_synthetic)/len(df_features)*100:.1f}%")

    # Use augmented data for training
    X_full = df_features_augmented[feature_cols_selected]
    y_full = df_features_augmented[target_col].values.astype(np.float32)

    logger.info(f"[+] Full dataset (with synthetic): {len(df_features_augmented):,} intervals")

    # ========================================================================
    # PHASE 2: 80/20 SPLIT FOR VALIDATION
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: 80/20 TRAIN/VALIDATION SPLIT")
    logger.info("="*70)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, shuffle=True
    )

    logger.info(f"[+] Training: {len(X_train):,} intervals ({len(X_train)/len(X_full)*100:.1f}%)")
    logger.info(f"[+] Validation: {len(X_val):,} intervals ({len(X_val)/len(X_full)*100:.1f}%)")

    # ========================================================================
    # PHASE 3: PREPROCESSING (FIT ON TRAIN)
    # ========================================================================
    logger.info("\nPHASE 3: DATA PREPROCESSING (FITTED ON TRAINING SET)")

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()

    numerical_features = []
    categorical_features = []

    for col in feature_cols_selected:
        if X_train_imputed[col].dtype in ['int64', 'float64']:
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    logger.info(f"[+] Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")

    if numerical_features:
        num_imputer = SimpleImputer(strategy='mean')
        X_train_imputed[numerical_features] = num_imputer.fit_transform(X_train_imputed[numerical_features])
        X_val_imputed[numerical_features] = num_imputer.transform(X_val_imputed[numerical_features])

    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_imputed[categorical_features] = cat_imputer.fit_transform(X_train_imputed[categorical_features])
        X_val_imputed[categorical_features] = cat_imputer.transform(X_val_imputed[categorical_features])

        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_imputed[categorical_features] = cat_encoder.fit_transform(X_train_imputed[categorical_features])
        X_val_imputed[categorical_features] = cat_encoder.transform(X_val_imputed[categorical_features])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    logger.info(f"[+] Preprocessing complete")

    # ========================================================================
    # PHASE 4: FEATURE SELECTION (ON TRAIN)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: SEQUENTIAL FEATURE SELECTION (SFS)")
    logger.info("="*70)

    if os.path.exists(SELECTED_FEATURES_PATH) and not FORCE_RERUN_SFS:
        logger.info(f"✓ Found existing selected features: {SELECTED_FEATURES_PATH}")
        logger.info("  Loading pre-selected features (skipping SFS)...")
        
        with open(SELECTED_FEATURES_PATH, 'r') as f:
            feature_data = json.load(f)
        
        selected_features = feature_data['selected_features']
        logger.info(f"[+] Loaded {len(selected_features)} pre-selected features")
        logger.info(f"    Selected on: {feature_data.get('date', 'unknown')}")
        logger.info(f"    Original feature count: {feature_data.get('original_count', 'unknown')}")
        logger.info(f"    SFS took: {feature_data.get('sfs_time_seconds', 0)/60:.2f} minutes")
        
        # Create mask for selected features
        selected_mask = np.array([feat in selected_features for feat in feature_cols_selected])
        
    else:
        if FORCE_RERUN_SFS:
            logger.info("⚠️  Forcing SFS re-run (ignoring cached features)")
        else:
            logger.info("No pre-selected features found. Running SFS...")
        
        base_model_sfs = XGBRegressor(
            random_state=42, objective='reg:squarederror', tree_method='hist', n_jobs=-1,
            n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, verbosity=0
        )

        n_features_to_select = min(45, len(feature_cols_selected))
        logger.info(f"Selecting features from {len(feature_cols_selected)} total...")

        sfs = SequentialFeatureSelector(
            estimator=base_model_sfs, n_features_to_select='auto',
            direction='forward', n_jobs=-1, cv=5, scoring='neg_mean_squared_error'
        )

        logger.info("Running SFS (this may take a while)...")
        sfs_start = time.time()
        sfs.fit(X_train_scaled, y_train_log)
        sfs_time = time.time() - sfs_start

        selected_mask = sfs.get_support()
        selected_features = [feat for feat, selected in zip(feature_cols_selected, selected_mask) if selected]

        logger.info(f"[+] SFS completed in {sfs_time/60:.2f} minutes")
        logger.info(f"[+] SFS selected {len(selected_features)} features")
        
        feature_data = {
            'selected_features': selected_features,
            'original_count': len(feature_cols_selected),
            'selected_count': len(selected_features),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sfs_time_seconds': sfs_time,
            'all_features': feature_cols_selected,
            'sfs_params': {
                'direction': 'forward',
                'cv': 5,
                'scoring': 'neg_mean_squared_error',
                'base_model': 'XGBRegressor'
            }
        }
        
        with open(SELECTED_FEATURES_PATH, 'w') as f:
            json.dump(feature_data, f, indent=2)
        
        logger.info(f"[+] Selected features saved to: {SELECTED_FEATURES_PATH}")
        
        txt_path = SELECTED_FEATURES_PATH.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write("SELECTED FEATURES FROM SEQUENTIAL FEATURE SELECTION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {feature_data['date']}\n")
            f.write(f"Original features: {len(feature_cols_selected)}\n")
            f.write(f"Selected features: {len(selected_features)}\n")
            f.write(f"SFS time: {sfs_time/60:.2f} minutes\n")
            f.write(f"Selection rate: {len(selected_features)/len(feature_cols_selected)*100:.1f}%\n\n")
            f.write("Selected Features (in order of selection):\n")
            f.write("-"*70 + "\n")
            for i, feat in enumerate(selected_features, 1):
                f.write(f"{i:3d}. {feat}\n")
        
        logger.info(f"[+] Human-readable list saved to: {txt_path}")

    logger.info(f"\n[+] Using {len(selected_features)} features for training")
    logger.info("    Top 10 selected features:")
    for i, feat in enumerate(selected_features[:10], 1):
        logger.info(f"      {i:2d}. {feat}")
    if len(selected_features) > 10:
        logger.info(f"      ... and {len(selected_features) - 10} more")

    X_train_sfs = X_train_scaled[:, selected_mask]
    X_val_sfs = X_val_scaled[:, selected_mask]

    # ========================================================================
    # PHASE 5: GRID SEARCH FOR HYPERPARAMETERS ON 80% DATA
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 5: GRID SEARCH FOR OPTIMAL HYPERPARAMETERS")
    logger.info("="*70)

    # param_grid = {
    # # Expand tree structure parameters
    # 'max_depth': [7, 8, 9],
    # 'min_child_weight': [3, 4, 5, 6],
    
    # # Expand learning parameters
    # 'learning_rate': [0.06, 0.065, 0.07, 0.075, 0.08],
    # 'n_estimators': [850, 875, 900, 925, 950],
    
    # # Fine-grain sampling parameters
    # 'subsample': [0.75, 0.775, 0.80, 0.825, 0.85],
    # 'colsample_bytree': [0.57, 0.60, 0.625, 0.65, 0.675, 0.70],
    
    # # Regularization spectrum
    # 'gamma': [0.02, 0.03, 0.04, 0.05, 0.06],
    # 'reg_alpha': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04],
    # 'reg_lambda': [1.0, 1.25, 1.5, 1.75, 2.0]
    # }
    param_grid = {
        'n_estimators': [900],
        'learning_rate': [0.07],
        'max_depth': [8],
        'subsample': [0.80],
        'colsample_bytree': [0.65],
        'gamma': [0.05],
        'reg_alpha': [0.05],
        'reg_lambda': [1.5],
        'min_child_weight': [4]
    }

    # Use RandomizedSearchCV to sample from the grid
    n_iter_search = 200  # Test 200 random combinations
    logger.info(f"Parameter space: 3,240,000 possible combinations")
    logger.info(f"Testing {n_iter_search} random combinations with 5-fold CV")
    logger.info(f"Total model fits: {n_iter_search * 5} = {n_iter_search * 5:,}")

    base_xgb = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1,
        verbosity=0
    )

    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    logger.info("Starting hyperparameter search...")
    start_time = time.time()

    random_search.fit(X_train_sfs, y_train_log)

    elapsed = time.time() - start_time
    logger.info(f"\n[+] Random search completed in {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")

    X_train_sfs_df = pd.DataFrame(X_train_sfs, columns=selected_features)
    X_val_sfs_df = pd.DataFrame(X_val_sfs, columns=selected_features)

    train_processed_path = os.path.join(RESULTS_DIR, 'X_train_processed.csv')
    val_processed_path = os.path.join(RESULTS_DIR, 'X_val_processed.csv')

    X_train_sfs_df.to_csv(train_processed_path, index=False)
    X_val_sfs_df.to_csv(val_processed_path, index=False)

    logger.info(f"[+] Processed training set saved to: {train_processed_path}")
    logger.info(f"[+] Processed validation set saved to: {val_processed_path}")

    # ========================================================================
    # EXTRACT TOP 10 MODELS FROM CV RESULTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXTRACTING TOP 10 MODELS FROM RANDOM SEARCH")
    logger.info("="*70)

    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df = cv_results_df.sort_values('mean_test_score', ascending=False)
    cv_results_df = cv_results_df.reset_index(drop=True)

    # Get top 10 models from CV
    top_10_cv = cv_results_df.head(10).copy()

    logger.info("\nTop 10 models from 5-fold CV on training data:")
    for idx, row in top_10_cv.iterrows():
        cv_rmse = np.sqrt(-row['mean_test_score'])
        cv_std = np.sqrt(row['std_test_score'])
        logger.info(f"\n  CV Rank {idx+1}:")
        logger.info(f"    CV RMSE = {cv_rmse:.4f} ± {cv_std:.4f} kg")
        logger.info(f"    Params: {row['params']}")

    # ========================================================================
    # EVALUATE TOP 10 MODELS ON HELD-OUT 20% VALIDATION SET
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("EVALUATING TOP 10 MODELS ON 20% VALIDATION SET")
    logger.info("="*70)

    validation_results = []

    for rank, (idx, row) in enumerate(top_10_cv.iterrows(), 1):
        logger.info(f"\nEvaluating Model {rank}/10...")
        
        model = XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            verbosity=0,
            **row['params']
        )
        
        model.fit(X_train_sfs, y_train_log)
        
        # Predict on validation set
        val_pred_log = model.predict(X_val_sfs)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.maximum(val_pred, 0.0)
        
        # Calculate validation metrics
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-8))) * 100
        val_r2 = 1 - (np.sum((y_val - val_pred) ** 2) / np.sum((y_val - y_val.mean()) ** 2))
        
        # Calculate training metrics
        train_pred_log = model.predict(X_train_sfs)
        train_pred = np.expm1(train_pred_log)
        train_pred = np.maximum(train_pred, 0.0)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        train_mae = np.mean(np.abs(y_train - train_pred))
        
        validation_results.append({
            'cv_rank': rank,
            'cv_rmse': np.sqrt(-row['mean_test_score']),
            'cv_rmse_std': np.sqrt(row['std_test_score']),
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_mape': val_mape,
            'val_r2': val_r2,
            'overfitting_gap': val_rmse - train_rmse,
            'params': row['params']
        })
        
        logger.info(f"  CV RMSE:    {np.sqrt(-row['mean_test_score']):.4f} kg")
        logger.info(f"  Train RMSE: {train_rmse:.4f} kg")
        logger.info(f"  Val RMSE:   {val_rmse:.4f} kg")
        logger.info(f"  Val MAE:    {val_mae:.4f} kg")
        logger.info(f"  Val R²:     {val_r2:.4f}")
        logger.info(f"  Gap:        {val_rmse - train_rmse:+.4f} kg")

    # ========================================================================
    # RANK BY VALIDATION RMSE
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("TOP 10 MODELS RANKED BY VALIDATION RMSE")
    logger.info("="*70)

    results_df = pd.DataFrame(validation_results)
    results_df = results_df.sort_values('val_rmse', ascending=True)
    results_df['final_rank'] = range(1, len(results_df) + 1)

    # Save detailed results
    results_path = os.path.join(RESULTS_DIR, 'random_search_top10_validation_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"[+] Detailed results saved: {results_path}")

    # Display ranking table
    logger.info("\n| Final | CV   | Val RMSE | Train RMSE | Gap     | Val MAE  | Val R²  |")
    logger.info("|  Rank | Rank |   (kg)   |    (kg)    |  (kg)   |   (kg)   |         |")
    logger.info("|-------|------|----------|------------|---------|----------|---------|")

    for _, row in results_df.iterrows():
        logger.info(
            f"|  {row['final_rank']:2d}   |  {row['cv_rank']:2d}  | "
            f"{row['val_rmse']:8.4f} | {row['train_rmse']:10.4f} | "
            f"{row['overfitting_gap']:7.4f} | {row['val_mae']:8.4f} | "
            f"{row['val_r2']:7.4f} |"
        )

    # ========================================================================
    # PHASE 6: PREPROCESS FULL DATASET
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 6: PREPROCESSING 100% OF DATA FOR FINAL TRAINING")
    logger.info("="*70)

    X_full_imputed = X_full.copy()
    y_full_log = np.log1p(y_full)

    if numerical_features:
        num_imputer_full = SimpleImputer(strategy='mean')
        X_full_imputed[numerical_features] = num_imputer_full.fit_transform(X_full_imputed[numerical_features])

    if categorical_features:
        cat_imputer_full = SimpleImputer(strategy='most_frequent')
        X_full_imputed[categorical_features] = cat_imputer_full.fit_transform(X_full_imputed[categorical_features])
        
        cat_encoder_full = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_full_imputed[categorical_features] = cat_encoder_full.fit_transform(X_full_imputed[categorical_features])

    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full_imputed)
    X_full_sfs = X_full_scaled[:, selected_mask]

    logger.info(f"[+] Full dataset preprocessed: {X_full_sfs.shape}")


    # ========================================================================
    # PHASE 7: DIAGNOSTIC - Check data sources FIRST
    # ========================================================================
    logger.info("🔍 DIAGNOSTIC: Verifying data sources...")

    # Check rank CSV row count
    df_test_rank = pd.read_csv(RANK_CSV_PATH)
    logger.info(f"Rank CSV total rows: {len(df_test_rank):,}")
    logger.info(f"Rank CSV first 5 flight_ids: {df_test_rank['flight_id'].head().tolist()}")

    # Check final CSV row count  
    df_test_final = pd.read_csv(TEST_CSV_PATH)
    logger.info(f"Final CSV total rows: {len(df_test_final):,}")
    logger.info(f"Final CSV first 5 flight_ids: {df_test_final['flight_id'].head().tolist()}")
    logger.info(f"Final CSV rows 24,290+: {len(df_test_final) - 24289:,}")

    # Check featured data differences
    featured_rank = pd.read_parquet(FEATURED_DATA_RANK)
    featured_final = pd.read_parquet(FEATURED_DATA_TEST)
    logger.info(f"Featured RANK shape: {featured_rank.shape}")
    logger.info(f"Featured FINAL shape: {featured_final.shape}")
    logger.info(f"RANK columns (first 10): {featured_rank.columns[:10].tolist()}")
    logger.info(f"FINAL columns (first 10): {featured_final.columns[:10].tolist()}")

    # Check if flight_ids overlap
    rank_flights = set(df_test_rank['flight_id'].unique())
    final_flights = set(df_test_final['flight_id'].unique())
    overlap = len(rank_flights & final_flights)
    logger.info(f"Flight ID overlap: {overlap:,} / {len(rank_flights):,} rank flights")

    logger.info("\n" + "="*70)
    logger.info("PHASE 7: CLEAN UNSCALED TEST DATA - ALL FEATURES FIXED")
    logger.info("="*70)

    N_RANK_ROWS = 24289

    # Load preprocessors
    import joblib
    preprocessors_path = os.path.join(RESULTS_DIR, 'preprocessors_rank.joblib')
    preprocessors = joblib.load(preprocessors_path)
    feature_cols_selected = preprocessors['feature_cols_selected']
    numerical_features = preprocessors['numerical_features']
    categorical_features = preprocessors['categorical_features']
    num_imputer_full = preprocessors['num_imputer_full']
    cat_imputer_full = preprocessors['cat_imputer_full']
    cat_encoder_full = preprocessors['cat_encoder_full']
    scaler_full = preprocessors['scaler_full']
    selected_mask = preprocessors['selected_mask']
    selected_features = preprocessors['selected_features']

    # ========================================================================
    # 1. LOAD CORRECT FUEL SUBMISSION DATA FIRST
    # ========================================================================
    logger.info("🔍 LOADING FUEL SUBMISSION DATA...")
    try:
        fuel_submission = pd.read_parquet('data/fuel_submission_final.parquet')
        logger.info("✅ Loaded fuel_submission_final.parquet")
    except:
        try:
            fuel_submission = pd.read_parquet('data/fuel_submission_final.parquet')
            logger.info("✅ Loaded fuelranksubmission.parquet")
        except:
            fuel_submission = pd.read_parquet(FUEL_RANK_PATH)
            logger.info("✅ Loaded FUEL_RANK_PATH")

    fuel_rank_intervals = fuel_submission[['flight_id', 'idx', 'start', 'end']].rename(columns={'idx': 'interval_idx'})

    # Load other data
    rank_csv = pd.read_csv(RANK_CSV_PATH, delimiter=',', low_memory=False)
    final_csv = pd.read_csv(TEST_CSV_PATH, delimiter=',', low_memory=False)
    featured_data_rank = pd.read_parquet(FEATURED_DATA_RANK).rename(columns={'idx': 'interval_idx'})
    featured_data_final = pd.read_parquet(FEATURED_DATA_TEST).rename(columns={'idx': 'interval_idx'})

    # ========================================================================
    # 2. PROCESS RANK DATA (24,289 rows)
    # ========================================================================
    logger.info("\n🔍 PROCESSING RANK DATA...")
    df_test_rank = rank_csv.head(N_RANK_ROWS).copy()
    if 'idx' in df_test_rank.columns:
        df_test_rank = df_test_rank.rename(columns={'idx': 'interval_idx'})

    df_test_rank = df_test_rank.merge(fuel_rank_intervals, on=['flight_id', 'interval_idx'], how='left')
    logger.info(f"Rank fuel merge: {df_test_rank['end'].notna().sum() if 'end' in df_test_rank.columns else 0:,} / {len(df_test_rank):,} matched")
    df_test_rank = df_test_rank.merge(featured_data_rank, on=['flight_id', 'interval_idx'], how='left')

    # AIRCRAFT_TYPE from featured_data_rank
    if 'aircraft_type' in featured_data_rank.columns:
        aircraft_rank = featured_data_rank[['flight_id', 'aircraft_type']].drop_duplicates(subset=['flight_id'])
        df_test_rank = df_test_rank.merge(aircraft_rank, on='flight_id', how='left', suffixes=('', '_feat'))
        df_test_rank['aircraft_type'] = df_test_rank['aircraft_type'].fillna(df_test_rank['aircraft_type_feat'])
        df_test_rank = df_test_rank.drop(columns=['aircraft_type_feat'], errors='ignore')
    df_test_rank['aircraft_type'] = df_test_rank['aircraft_type'].fillna('A320')
    logger.info(f"✅ Rank aircraft_type: {df_test_rank['aircraft_type'].notna().sum():,} / {len(df_test_rank):,} ({df_test_rank['aircraft_type'].nunique():,} types)")

    # GREAT_CIRCLE_DISTANCE
    logger.info("🔍 Adding RANK great_circle_distance...")
    try:
        flightlist_rank = pd.read_parquet(FLIGHTLIST_RANK_PATH)
        rank_coords = flightlist_rank.merge(apt, left_on='origin_icao', right_on='icao', how='left')
        rank_coords = rank_coords.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'}).drop(columns=['icao'], errors='ignore')
        rank_coords = rank_coords.merge(apt, left_on='destination_icao', right_on='icao', how='left')
        rank_coords = rank_coords.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'}).drop(columns=['icao'], errors='ignore')
        rank_coords['great_circle_distance'] = rank_coords.apply(
            lambda row: haversine(row.origin_lon, row.origin_lat, row.dest_lon, row.dest_lat) if pd.notna(row.origin_lon) and pd.notna(row.dest_lon) else 1000, 
            axis=1
        )
        df_test_rank = df_test_rank.merge(rank_coords[['flight_id', 'great_circle_distance']], on='flight_id', how='left')
        logger.info(f"✅ Rank great_circle_distance: mean={df_test_rank['great_circle_distance'].mean():.0f}km")
    except Exception as e:
        logger.warning(f"⚠️ Rank coordinates failed: {e}")
        df_test_rank['great_circle_distance'] = 1000

    # SAFE COMPUTED COLUMNS
    for col in ['alt_avg_ft', 'altitude_change_rate', 'end_hour', 'interval_elapsed_from_flight_start']:
        if col not in df_test_rank.columns:
            if col == 'alt_avg_ft':
                df_test_rank[col] = (df_test_rank.get('alt_start_ft', 0) + df_test_rank.get('alt_end_ft', 0)) / 2
            elif col == 'altitude_change_rate':
                df_test_rank[col] = df_test_rank.get('alt_change_ft', 0) / (df_test_rank.get('interval_duration_sec', 60) + 1e-6)
            elif col == 'end_hour':
                if 'end' in df_test_rank.columns and df_test_rank['end'].notna().sum() > 0:
                    df_test_rank[col] = pd.to_datetime(df_test_rank['end'], errors='coerce').dt.hour.fillna(12).astype(int)
                else:
                    df_test_rank[col] = 12
            else:  # interval_elapsed_from_flight_start
                if 'start' in df_test_rank.columns and 'end' in df_test_rank.columns:
                    df_test_rank['flight_start'] = pd.to_datetime(df_test_rank['start'], errors='coerce')
                    df_test_rank[col] = (pd.to_datetime(df_test_rank['end'], errors='coerce') - df_test_rank['flight_start']).dt.total_seconds().fillna(3600) / 3600.0
                    df_test_rank = df_test_rank.drop(columns=['flight_start'], errors='ignore')
                else:
                    df_test_rank[col] = 1.0

    logger.info(f"✅ Rank interval_elapsed: mean={df_test_rank.get('interval_elapsed_from_flight_start', 0).mean():.1f}h")

    for col in feature_cols_selected:
        if col not in df_test_rank.columns:
            df_test_rank[col] = 0

    X_test_rank_unscaled = df_test_rank[feature_cols_selected].copy().replace([np.inf, -np.inf], np.nan)
    if numerical_features: 
        X_test_rank_unscaled[numerical_features] = num_imputer_full.transform(X_test_rank_unscaled[numerical_features])
    if categorical_features: 
        X_test_rank_unscaled[categorical_features] = cat_imputer_full.transform(X_test_rank_unscaled[categorical_features])
        X_test_rank_unscaled[categorical_features] = cat_encoder_full.transform(X_test_rank_unscaled[categorical_features])

    X_test_rank_unscaled.to_csv(os.path.join(RESULTS_DIR, 'X_test_unscaled_RANK.csv'), index=False)
    logger.info(f"✅ RANK UNSCALED: {X_test_rank_unscaled.shape}")

    # ========================================================================
    # 3. PROCESS FINAL DATA (61,745 rows)
    # ========================================================================
    logger.info("\n🔍 PROCESSING FINAL DATA...")
    df_test_final = final_csv.copy()
    if 'idx' in df_test_final.columns:
        df_test_final = df_test_final.rename(columns={'idx': 'interval_idx'})

    df_test_final = df_test_final.merge(fuel_rank_intervals, on=['flight_id', 'interval_idx'], how='left')
    logger.info(f"Final fuel merge: {df_test_final['end'].notna().sum() if 'end' in df_test_final.columns else 0:,} / {len(df_test_final):,} matched")
    df_test_final = df_test_final.merge(featured_data_final, on=['flight_id', 'interval_idx'], how='left')

    # AIRCRAFT_TYPE from featured_data_final
    if 'aircraft_type' in featured_data_final.columns:
        aircraft_final = featured_data_final[['flight_id', 'aircraft_type']].drop_duplicates(subset=['flight_id'])
        df_test_final = df_test_final.merge(aircraft_final, on='flight_id', how='left', suffixes=('', '_feat'))
        df_test_final['aircraft_type'] = df_test_final['aircraft_type'].fillna(df_test_final['aircraft_type_feat'])
        df_test_final = df_test_final.drop(columns=['aircraft_type_feat'], errors='ignore')
    df_test_final['aircraft_type'] = df_test_final['aircraft_type'].fillna('A320')
    logger.info(f"✅ Final aircraft_type: {df_test_final['aircraft_type'].notna().sum():,} / {len(df_test_final):,} ({df_test_final['aircraft_type'].nunique():,} types)")

    # GREAT_CIRCLE_DISTANCE
    logger.info("🔍 Adding FINAL great_circle_distance...")
    try:
        flightlist_final = pd.read_parquet(FLIGHTLIST_FINAL_PATH)
        final_coords = flightlist_final.merge(apt, left_on='origin_icao', right_on='icao', how='left')
        final_coords = final_coords.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'}).drop(columns=['icao'], errors='ignore')
        final_coords = final_coords.merge(apt, left_on='destination_icao', right_on='icao', how='left')
        final_coords = final_coords.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'}).drop(columns=['icao'], errors='ignore')
        final_coords['great_circle_distance'] = final_coords.apply(
            lambda row: haversine(row.origin_lon, row.origin_lat, row.dest_lon, row.dest_lat) if pd.notna(row.origin_lon) and pd.notna(row.dest_lon) else 1000, 
            axis=1
        )
        df_test_final = df_test_final.merge(final_coords[['flight_id', 'great_circle_distance']], on='flight_id', how='left')
        logger.info(f"✅ Final great_circle_distance: mean={df_test_final['great_circle_distance'].mean():.0f}km")
    except Exception as e:
        logger.warning(f"⚠️ Final coordinates failed: {e}")
        df_test_final['great_circle_distance'] = 1000

    # SAFE COMPUTED COLUMNS
    for col in ['alt_avg_ft', 'altitude_change_rate', 'end_hour', 'interval_elapsed_from_flight_start']:
        if col not in df_test_final.columns:
            if col == 'alt_avg_ft':
                df_test_final[col] = (df_test_final.get('alt_start_ft', 0) + df_test_final.get('alt_end_ft', 0)) / 2
            elif col == 'altitude_change_rate':
                df_test_final[col] = df_test_final.get('alt_change_ft', 0) / (df_test_final.get('interval_duration_sec', 60) + 1e-6)
            elif col == 'end_hour':
                if 'end' in df_test_final.columns and df_test_final['end'].notna().sum() > 0:
                    df_test_final[col] = pd.to_datetime(df_test_final['end'], errors='coerce').dt.hour.fillna(12).astype(int)
                else:
                    df_test_final[col] = 12
            else:  # interval_elapsed_from_flight_start
                if 'start' in df_test_final.columns and 'end' in df_test_final.columns:
                    df_test_final['flight_start'] = pd.to_datetime(df_test_final['start'], errors='coerce')
                    df_test_final[col] = (pd.to_datetime(df_test_final['end'], errors='coerce') - df_test_final['flight_start']).dt.total_seconds().fillna(3600) / 3600.0
                    df_test_final = df_test_final.drop(columns=['flight_start'], errors='ignore')
                else:
                    df_test_final[col] = 1.0

    logger.info(f"✅ Final interval_elapsed: mean={df_test_final.get('interval_elapsed_from_flight_start', 0).mean():.1f}h")

    for col in feature_cols_selected:
        if col not in df_test_final.columns:
            df_test_final[col] = 0

    X_test_final_unscaled = df_test_final[feature_cols_selected].copy().replace([np.inf, -np.inf], np.nan)
    if numerical_features: 
        X_test_final_unscaled[numerical_features] = num_imputer_full.transform(X_test_final_unscaled[numerical_features])
    if categorical_features: 
        X_test_final_unscaled[categorical_features] = cat_imputer_full.transform(X_test_final_unscaled[categorical_features])
        X_test_final_unscaled[categorical_features] = cat_encoder_full.transform(X_test_final_unscaled[categorical_features])

    X_test_final_unscaled.to_csv(os.path.join(RESULTS_DIR, 'X_test_unscaled_FINAL.csv'), index=False)
    logger.info(f"✅ FINAL UNSCALED: {X_test_final_unscaled.shape}")

    # ========================================================================
    # PHASE 7B: PERFECT HYBRID SCALED X_test_sfs + SUBMISSION TEMPLATE
    # ========================================================================
    logger.info("\n" + "="*50)
    logger.info("PHASE 7B: Testing.py Rank + NEW Final Rows")
    logger.info("="*50)

    # 1. LOAD EXACT Testing.py PROCESSED RANK ROWS (SCALED)
    testing_processed = os.path.join(RESULTS_DIR, 'X_test_processedTesting.csv')
    if os.path.exists(testing_processed):
        X_test_rank_sfs = pd.read_csv(testing_processed).values
        logger.info(f"✅ LOADED Testing.py rank: {X_test_rank_sfs.shape}")
    else:
        logger.warning("⚠️ No Testing.py file - scaling rank data")
        X_test_rank_scaled = scaler_full.transform(X_test_rank_unscaled)
        X_test_rank_sfs = X_test_rank_scaled[:, selected_mask]

    # 2. Scale ONLY NEW FINAL ROWS (37,456 rows from Final CSV 24,290+)
    final_new_unscaled = X_test_final_unscaled.iloc[N_RANK_ROWS:]
    X_test_final_new_scaled = scaler_full.transform(final_new_unscaled)
    X_test_final_new_sfs = X_test_final_new_scaled[:, selected_mask]
    logger.info(f"✅ NEW Final rows: {X_test_final_new_sfs.shape}")

    # 3. PERFECT HYBRID: Testing.py(24k) + NEW Final(37k) = 61,745
    X_test_sfs = np.vstack([X_test_rank_sfs, X_test_final_new_sfs])
    logger.info(f"✅ HYBRID X_test_sfs: {X_test_sfs.shape}")

    # 4. SUBMISSION TEMPLATE
    submission_template = fuel_submission[['idx', 'flight_id', 'start', 'end']].copy()
    logger.info(f"✅ Submission template: {submission_template.shape}")

    # Save scaled test data
    pd.DataFrame(X_test_sfs, columns=selected_features).to_csv(
        os.path.join(RESULTS_DIR, 'X_test_processed_Final.csv'), index=False
    )

    logger.info("\n" + "="*70)
    logger.info("🎯 ALL FEATURES PERFECT:")
    logger.info(f"   • Rows 0-24,288:   Testing.py EXACT ({X_test_rank_sfs.shape[0]:,})")
    logger.info(f"   • Rows 24,289+:    NEW Final     ({X_test_final_new_sfs.shape[0]:,})")
    logger.info(f"   • TOTAL:          {X_test_sfs.shape[0]:,} ✓")
    logger.info(f"   • submission_template: {submission_template.shape[0]:,} ✓")
    logger.info("   • Phase 8 READY!")
    logger.info("="*70)


    # ========================================================================
    # PHASE 8: TRAIN ALL TOP 5 MODELS ON 100% AND GENERATE SUBMISSIONS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 8: TRAINING ALL TOP 5 MODELS ON 100% DATA")
    logger.info("="*70)

    submission_files = []

    for idx, row in results_df.iterrows():
        rank = row['final_rank']
        params = row['params']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING MODEL RANK #{rank}")
        logger.info(f"{'='*70}")
        logger.info(f"Expected Validation RMSE: {row['val_rmse']:.4f} kg")
        logger.info(f"Parameters: {params}")
        
        # Train model on 100% of data
        final_model = XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            verbosity=0,
            **params
        )
        
        logger.info("Training on 100% of data...")
        final_model.fit(X_full_sfs, y_full_log)
        
        # ====================================================================
        # FEATURE IMPORTANCE ANALYSIS
        # ====================================================================
        logger.info("\n" + "-"*70)
        logger.info(f"FEATURE IMPORTANCE ANALYSIS - MODEL RANK #{rank}")
        logger.info("-"*70)
        
        # Get feature importance from the trained model
        feature_importance = final_model.feature_importances_
        
        # Create DataFrame with features and their importance
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': feature_importance,
            'importance_pct': (feature_importance / feature_importance.sum()) * 100
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Add rank column
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Save to CSV
        importance_path = os.path.join(RESULTS_DIR, f'feature_importance_rank{rank}.csv')
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"[+] Feature importance saved to: {importance_path}")
        
        # Log top 20 most important features
        logger.info(f"\n[+] Top 20 Most Important Features:")
        logger.info(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'% Total':<10}")
        logger.info("-"*75)
        for _, imp_row in importance_df.head(20).iterrows():
            logger.info(
                f"{int(imp_row['rank']):<6} {imp_row['feature']:<45} "
                f"{imp_row['importance']:<12.6f} {imp_row['importance_pct']:<10.2f}%"
            )
        
        # Log bottom 10 least important features
        logger.info(f"\n[+] Bottom 10 Least Important Features:")
        logger.info(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'% Total':<10}")
        logger.info("-"*75)
        for _, imp_row in importance_df.tail(10).iterrows():
            logger.info(
                f"{int(imp_row['rank']):<6} {imp_row['feature']:<45} "
                f"{imp_row['importance']:<12.6f} {imp_row['importance_pct']:<10.2f}%"
            )
        
        # Calculate and log cumulative importance
        cumulative_importance = importance_df['importance_pct'].cumsum()
        n_features_90pct = (cumulative_importance <= 90).sum() + 1
        n_features_95pct = (cumulative_importance <= 95).sum() + 1
        
        logger.info(f"\n[+] Cumulative Importance Analysis:")
        logger.info(f"    Top {n_features_90pct} features explain 90% of importance")
        logger.info(f"    Top {n_features_95pct} features explain 95% of importance")
        logger.info(f"    Total features: {len(selected_features)}")
        
        logger.info("-"*70)
        # ====================================================================
        
        # Training performance
        full_pred_log = final_model.predict(X_full_sfs)
        full_pred = np.expm1(full_pred_log)
        full_pred = np.maximum(full_pred, 0.0)
        full_rmse = np.sqrt(np.mean((y_full - full_pred) ** 2))
        
        logger.info(f"[+] Training RMSE (100% data): {full_rmse:.4f} kg")
        
        # Make predictions on test set
        test_pred_log = final_model.predict(X_test_sfs)
        test_pred = np.expm1(test_pred_log)
        test_pred = np.maximum(test_pred, 0.0)
        
        logger.info(f"[+] Test predictions: {len(test_pred):,}")
        logger.info(f"    Range: [{test_pred.min():.2f}, {test_pred.max():.2f}] kg")
        logger.info(f"    Mean: {test_pred.mean():.2f} kg")
        
        # Create submission
        submission_df = submission_template.copy()
        submission_df['fuel_kg'] = test_pred.astype(np.float32)
        submission_df = submission_df[['idx', 'flight_id', 'start', 'end', 'fuel_kg']]
        
        # Save parquet with fastparquet
        parquet_path = os.path.join(RESULTS_DIR, f'submission_rank{rank}_synthetic_valrmse_{row["val_rmse"]:.4f}.parquet')
        submission_df.to_parquet(parquet_path, index=False, engine='fastparquet')
        logger.info(f"[+] Parquet saved: {parquet_path}")
        
        # Save CSV
        csv_path = os.path.join(RESULTS_DIR, f'submission_rank{rank}_synthetic_valrmse_{row["val_rmse"]:.4f}.csv')
        submission_df.to_csv(csv_path, index=False)
        logger.info(f"[+] CSV saved: {csv_path}")
        
        submission_files.append({
            'rank': rank,
            'val_rmse': row['val_rmse'],
            'train_rmse_100pct': full_rmse,
            'parquet_file': parquet_path,
            'csv_file': csv_path,
            'test_mean': test_pred.mean(),
            'test_std': test_pred.std(),
            'params': params
        })
        
        # Save parameters
        params_file = os.path.join(RESULTS_DIR, f'parameters_rank{rank}.txt')
        with open(params_file, 'w') as f:
            f.write(f"MODEL RANK #{rank}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Validation RMSE: {row['val_rmse']:.4f} kg\n")
            f.write(f"Validation MAE:  {row['val_mae']:.4f} kg\n")
            f.write(f"Validation R²:   {row['val_r2']:.4f}\n")
            f.write(f"Training RMSE (100%): {full_rmse:.4f} kg\n\n")
            f.write("Hyperparameters:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")

    # ========================================================================
    # PHASE 9: SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 9: FINAL SUMMARY")
    logger.info("="*70)

    summary_df = pd.DataFrame(submission_files)
    summary_path = os.path.join(RESULTS_DIR, 'top5_models_synthetic_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    logger.info("\n" + "="*70)
    logger.info("TOP 5 MODELS SUMMARY")
    logger.info("="*70)
    logger.info("\n| Rank | Val RMSE | Train RMSE | Test Mean | Test Std  | Submission File |")
    logger.info("|------|----------|------------|-----------|-----------|-----------------|")
    
    for _, row in summary_df.iterrows():
        filename = os.path.basename(row['parquet_file'])
        logger.info(
            f"|  {row['rank']:2d}  | {row['val_rmse']:8.4f} | {row['train_rmse_100pct']:10.4f} | "
            f"{row['test_mean']:9.2f} | {row['test_std']:9.2f} | {filename[:30]}... |"
        )
    
    logger.info("\n" + "="*70)
    logger.info("✓ COMPLETED SUCCESSFULLY WITH SYNTHETIC DATA!")
    logger.info(f"✓ Best Model: Rank #1 (Val RMSE = {summary_df.iloc[0]['val_rmse']:.4f} kg)")
    logger.info(f"✓ All 5 submission parquets generated in: {RESULTS_DIR}")
    logger.info(f"✓ Summary saved: {summary_path}")
    logger.info("="*70)


if __name__ == "__main__":
    main()