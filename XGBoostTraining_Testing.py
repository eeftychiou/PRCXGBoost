import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU only for this script to bypass CUDA context corruption
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
import joblib
import sys
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import generate_paper_plots

# Optimization Modes: 'legacy', 'grid', or 'optuna'
warnings.filterwarnings('ignore')

# Aircraft specifications database - Centralized in config.py
AIRCRAFT_DATA = config.AIRCRAFT_DATA
WIDEBODY_AIRCRAFT = config.WIDEBODY_AIRCRAFT

# File paths - Centralized in config.py
DATA_PATH = config.AUGMENTED_FINAL_CSV
APT_PATH = config.APT_PARQUET
FLIGHTLIST_PATH = config.FLIGHTLIST_TRAIN
FUEL_PATH = config.FUEL_TRAIN
TEST_CSV_PATH = config.AUGMENTED_RANK_CSV
FUEL_RANK_PATH = config.FUEL_RANK
FLIGHTLIST_RANK_PATH = config.FLIGHTLIST_RANK
RESULTS_DIR = config.MODELS_DIR # Changed to models_dir for consistency

FEATURED_DATA_TRAIN = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
FEATURED_DATA_TEST = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_rank.parquet')

SELECTED_FEATURES_PATH = config.SELECTED_FEATURES_PATH
SYNTHETIC_PATH = config.SYNTHETIC_WIDEBODY_PATH
os.makedirs(RESULTS_DIR, exist_ok=True)

log_file = os.path.join(RESULTS_DIR, f'test_xgboost_top5_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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


def save_model_plots(model, X_train, y_train, X_val, y_val, features, output_dir, rank_name):
    """
    Generates and saves diagnostic plots for the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Learning Curves (Loss history)
    results = model.evals_result()
    if results and 'validation_0' in results:
        # Save raw data to CSV
        history_df = pd.DataFrame(results['validation_0']).rename(columns={'rmse': 'train_rmse'})
        if 'validation_1' in results:
            history_df['val_rmse'] = results['validation_1']['rmse']
        
        history_path = os.path.join(output_dir, f'learning_curves_{rank_name}.csv')
        history_df.to_csv(history_path, index_label='epoch')
        
        plt.figure(figsize=(10, 6))
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        
        plt.plot(x_axis, results['validation_0']['rmse'], label='Train', color='royalblue')
        if 'validation_1' in results:
            plt.plot(x_axis, results['validation_1']['rmse'], label='Val', color='orange')
            
        plt.title(f'Learning Curve - {rank_name}')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('RMSE (log-scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'learning_curve_{rank_name}.png'))
        plt.close()

    # 2. Predicted vs. Actual
    y_val_orig = np.expm1(y_val)
    y_pred_log = model.predict(X_val)
    y_pred_orig = np.expm1(y_pred_log)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val_orig, y_pred_orig, alpha=0.3, color='teal', s=10)
    
    # Diagonal line
    max_val = max(y_val_orig.max(), y_pred_orig.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
    
    plt.title(f'Predicted vs Actual - {rank_name}')
    plt.xlabel('Actual Fuel Consumption (kg)')
    plt.ylabel('Predicted Fuel Consumption (kg)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predicted_vs_actual_{rank_name}.png'))
    plt.close()

    # 3. Feature Importance (Top 20)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:20]
    
    plt.figure(figsize=(12, 8))
    plt.barh(np.array(features)[sorted_idx][::-1], importance[sorted_idx][::-1], color='steelblue')
    plt.title(f'Top 20 Feature Importance - {rank_name}')
    plt.xlabel('Relative Importance')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_{rank_name}.png'))
    plt.close()
    
    logger.info(f"Diagnostic plots saved to: {output_dir}")


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



def main(gpu_id=0, force_sfs=False, force_synthetic=False, opt_mode='legacy'):
    FORCE_RERUN_SFS = force_sfs or '--force-sfs' in sys.argv
    FORCE_RERUN_SYNTHETIC = force_synthetic or '--force-synthetic' in sys.argv
    
    # Update RESULTS_DIR to be mode-specific
    global RESULTS_DIR
    RESULTS_DIR = os.path.join(config.MODELS_DIR, opt_mode)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if FORCE_RERUN_SFS:
        logger.info("⚠️  Force SFS re-run flag detected - will ignore cached features")
    
    if FORCE_RERUN_SYNTHETIC:
        logger.info("⚠️  Force synthetic data re-run flag detected - will ignore cached synthetic data")
    
    logger.info("="*70)
    logger.info("XGBoost FUEL PREDICTION - TOP 5 MODELS TRAINING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using GPU device ID: {gpu_id}")
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

    # df_synthetic = generate_synthetic_widebody_data_enhanced(
    #     df_features,
    #     n_synthetic=25000,        # Generate 25,000 samples
    #     long_segment_pct=0.25,    # 25% from long segments
    #     random_state=42
    # )
    if os.path.exists(SYNTHETIC_PATH) and not FORCE_RERUN_SYNTHETIC:
        logger.info(f"Loading cached synthetic data from {SYNTHETIC_PATH}")
        df_synthetic = pd.read_parquet(SYNTHETIC_PATH)
        logger.info(f"Loaded {len(df_synthetic)} cached synthetic samples")
    else:
        if FORCE_RERUN_SYNTHETIC:
            logger.info("⚠️  Regenerating synthetic data (force rerun)...")
        df_synthetic = generate_synthetic_widebody_data_enhanced(
            df_features,
            n_synthetic=25000,        # Generate 25,000 samples
            long_segment_pct=0.25,    # 25% from long segments
            random_state=42
        )
        df_synthetic.to_parquet(SYNTHETIC_PATH, index=False, engine='fastparquet')


    # Combine original and synthetic data
    n_real_rows = len(df_features)
    df_features_augmented = pd.concat([df_features, df_synthetic], ignore_index=True)

    logger.info(f"\n[+] Original training size: {len(df_features):,}")
    logger.info(f"[+] Synthetic samples added: {len(df_synthetic):,}")
    logger.info(f"[+] Augmented training size: {len(df_features_augmented):,}")
    logger.info(f"[+] Augmentation rate: {len(df_synthetic)/len(df_features)*100:.1f}%")

    # Stash per-row metadata needed for OpenAP comparison (real rows only)
    is_real_full = np.array([True] * n_real_rows + [False] * len(df_synthetic))
    # df_features shares the same index as df_raw (post-dropna); use that index to align
    _real_idx = df_features.index
    openap_kg_full     = df_raw.loc[_real_idx, 'openap_fuel_kg'].values  if 'openap_fuel_kg' in df_raw.columns else np.full(n_real_rows, np.nan)
    aircraft_type_full = df_raw.loc[_real_idx, 'aircraft_type'].values   if 'aircraft_type'  in df_raw.columns else np.full(n_real_rows, 'UNK')
    phase_full         = df_raw.loc[_real_idx, 'phase'].values           if 'phase'          in df_raw.columns else np.full(n_real_rows, 'UNK')
    # Pad to combined length (synthetic rows get placeholder)
    openap_kg_full     = np.concatenate([openap_kg_full,     np.full(len(df_synthetic), np.nan)])
    aircraft_type_full = np.concatenate([aircraft_type_full, np.full(len(df_synthetic), 'UNK')])
    phase_full         = np.concatenate([phase_full,         np.full(len(df_synthetic), 'UNK')])

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

    all_indices = np.arange(len(X_full))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.2, random_state=42, shuffle=True
    )
    X_train = X_full.iloc[train_indices]
    X_val   = X_full.iloc[val_indices]
    y_train = y_full[train_indices]
    y_val   = y_full[val_indices]

    # Val metadata
    val_is_real        = is_real_full[val_indices]
    val_openap_kg      = openap_kg_full[val_indices]
    val_aircraft_type  = aircraft_type_full[val_indices]
    val_phase          = phase_full[val_indices]

    logger.info(f"[+] Training: {len(X_train):,} intervals ({len(X_train)/len(X_full)*100:.1f}%)")
    logger.info(f"[+] Validation: {len(X_val):,} intervals ({len(X_val)/len(X_full)*100:.1f}%)")
    logger.info(f"[+] Real rows in val set: {val_is_real.sum():,}")

    # ========================================================================
    # PHASE 3: PREPROCESSING (FIT ON TRAIN)
    # ========================================================================
    logger.info("\nPHASE 3: DATA PREPROCESSING (FITTED ON TRAINING SET)")

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()

    # Drop entirely NaN columns to prevent SimpleImputer shape mismatches
    nan_cols = [col for col in feature_cols_selected if X_train_imputed[col].isna().all()]
    if nan_cols:
        logger.info(f"[-] Dropping {len(nan_cols)} columns that are 100% NaN: {nan_cols}")
        feature_cols_selected = [col for col in feature_cols_selected if col not in nan_cols]
        X_train_imputed = X_train_imputed.drop(columns=nan_cols)
        X_val_imputed = X_val_imputed.drop(columns=nan_cols)

    numerical_features = []
    categorical_features = []

    for col in feature_cols_selected:
        if pd.api.types.is_numeric_dtype(X_train_imputed[col]):
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
        
        # Create mask for selected features (only those present in current data)
        selected_mask = np.array([feat in selected_features for feat in feature_cols_selected])
        missing = [f for f in selected_features if f not in feature_cols_selected]
        if missing:
            logger.warning(f"  ⚠️  {len(missing)} feature(s) from JSON not found in current data and will be skipped: {missing}")
        selected_features = [feat for feat, s in zip(feature_cols_selected, selected_mask) if s]
        
    else:
        if FORCE_RERUN_SFS:
            logger.info("⚠️  Forcing SFS re-run (ignoring cached features)")
        else:
            logger.info("No pre-selected features found. Running SFS...")
        
        base_model_sfs = XGBRegressor(
            random_state=42, objective='reg:squarederror', 
            tree_method='hist',
            device='cpu',
            n_jobs=-1,
            n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, verbosity=0
        )

        n_features_to_select = min(45, len(feature_cols_selected))
        logger.info(f"Selecting features from {len(feature_cols_selected)} total...")

        sfs = SequentialFeatureSelector(
            estimator=base_model_sfs, n_features_to_select='auto',
            direction='forward', n_jobs=1, cv=5, scoring='neg_mean_squared_error'
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
    # PHASE 5: HYPERPARAMETER SELECTION
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info(f"PHASE 5: HYPERPARAMETER SELECTION (Mode: {opt_mode.upper()})")
    logger.info("="*70)

    # 1. Legacy Parameters (previously successful)
    legacy_params = {
        'n_estimators': 1455, 
        'learning_rate': 0.02885922756814833, 
        'max_depth': 9, 
        'min_child_weight': 4, 
        'gamma': 6.24155979490078e-08, 
        'subsample': 0.9991625118585123, 
        'colsample_bytree': 0.6701135673048045, 
        'reg_alpha': 0.004878930563988692, 
        'reg_lambda': 2.3991563444540384e-08
    }

    # 2. Grid Parameters (new combination provided by user)
    grid_params = {
        'n_estimators': 900,
        'learning_rate': 0.07,
        'max_depth': 8,
        'subsample': 0.80,
        'colsample_bytree': 0.65,
        'gamma': 0.05,
        'reg_alpha': 0.05,
        'reg_lambda': 1.5,
        'min_child_weight': 4
    }

    if opt_mode == 'optuna':
        import optuna
        from sklearn.model_selection import KFold
        # ... [Optuna Logic below] ...

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'tree_method': 'hist',
                'device': 'cuda' if gpu_id is not None else 'cpu',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            rmse_scores = []
            
            # Ensure data is C-contiguous
            X_t_arr = np.ascontiguousarray(X_train_sfs)
            y_t_arr = np.ascontiguousarray(np.log1p(y_train))
            
            for train_idx, val_idx in kf.split(X_t_arr):
                X_fold_train, X_fold_val = X_t_arr[train_idx], X_t_arr[val_idx]
                y_fold_train, y_fold_val = y_t_arr[train_idx], y_t_arr[val_idx]
                
                model = XGBRegressor(**params)
                model.fit(X_fold_train, y_fold_train)
                
                preds = model.predict(X_fold_val)
                rmse = np.sqrt(np.mean((y_fold_val - preds) ** 2))
                rmse_scores.append(rmse)
            
            mean_rmse = np.mean(rmse_scores)
            std_rmse = np.std(rmse_scores)
            trial.set_user_attr('cv_std', std_rmse)
            
            return mean_rmse

        logger.info("Starting Optuna study...")
        start_time = time.time()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        elapsed = time.time() - start_time
        logger.info(f"\n[+] Optuna search completed in {elapsed/60:.2f} minutes")
        
        # Save trials
        trials_df = study.trials_dataframe()
        trials_path = os.path.join(RESULTS_DIR, 'test_optuna_trials_history.csv')
        trials_df.to_csv(trials_path, index=False)

        # Extract top 10
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(completed_trials, key=lambda t: t.value)[:10]
        top_10_cv_data = [{
            'mean_test_score': -(t.value**2),
            'std_test_score': t.user_attrs.get('cv_std', 0.0)**2,
            'params': t.params
        } for t in top_trials]
        top_10_cv = pd.DataFrame(top_10_cv_data)
        
    elif opt_mode == 'grid':
        logger.info("[!] Using Manual Grid Hyperparameters as requested.")
        top_10_cv = pd.DataFrame([{
            'mean_test_score': -0.0,
            'std_test_score': 0.0,
            'params': grid_params
        }])
    else:
        logger.info("[!] Using Legacy Hyperparameters as requested.")
        top_10_cv = pd.DataFrame([{
            'mean_test_score': -0.0,
            'std_test_score': 0.0,
            'params': legacy_params
        }])

    # Clean up and save processed data for plotting script
    X_train_sfs_df = pd.DataFrame(X_train_sfs, columns=selected_features)
    X_val_sfs_df = pd.DataFrame(X_val_sfs, columns=selected_features)
    X_train_sfs_df.to_csv(os.path.join(RESULTS_DIR, 'test_X_train_processed.csv'), index=False)
    X_val_sfs_df.to_csv(os.path.join(RESULTS_DIR, 'test_X_val_processed.csv'), index=False)

    # ========================================================================
    # EVALUATE TOP 10 MODELS ON HELD-OUT 20% VALIDATION SET
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("EVALUATING TOP 10 MODELS ON 20% VALIDATION SET")
    logger.info("="*70)

    validation_results = []

    for rank, (idx, row) in enumerate(tqdm(top_10_cv.iterrows(), total=len(top_10_cv), desc="Evaluating Top Models"), 1):
        logger.info(f"\nEvaluating Model {rank}/10...")
        
        model = XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            device='cpu',
            n_jobs=1, # Sequential fit avoids context corruption
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
    # OPENAP COMPARISON ON VAL SET (REAL ROWS ONLY, RANK-1 MODEL)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("OPENAP VS XGBOOST COMPARISON (REAL VAL ROWS, RANK-1 MODEL)")
    logger.info("="*70)

    best_row = results_df.iloc[0]
    best_model_cmp = XGBRegressor(
        random_state=42, objective='reg:squarederror',
        tree_method='hist', device='cpu', n_jobs=-1, verbosity=0,
        **best_row['params']
    )
    best_model_cmp.fit(X_train_sfs, y_train_log)
    best_val_pred_log = best_model_cmp.predict(X_val_sfs)
    best_val_pred     = np.maximum(np.expm1(best_val_pred_log), 0.0)

    # Filter to real-only val rows
    real_mask      = val_is_real
    y_val_real     = y_val[real_mask]
    pred_real      = best_val_pred[real_mask]
    openap_real    = val_openap_kg[real_mask].astype(float)
    aircraft_real  = val_aircraft_type[real_mask]
    phase_real     = val_phase[real_mask]

    def _rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))
    def _mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
    def _mape(a, b): return float(np.mean(np.abs((np.asarray(a) - np.asarray(b)) / (np.asarray(a) + 1e-8))) * 100)

    logger.info(f"\nReal val rows: {real_mask.sum():,}")
    logger.info(f"Overall XGB  RMSE={_rmse(y_val_real,pred_real):.2f}  MAE={_mae(y_val_real,pred_real):.2f}  MAPE={_mape(y_val_real,pred_real):.2f}%")
    valid_oa = ~np.isnan(openap_real)
    if valid_oa.sum() > 0:
        logger.info(f"Overall OpenAP RMSE={_rmse(y_val_real[valid_oa],openap_real[valid_oa]):.2f}  MAE={_mae(y_val_real[valid_oa],openap_real[valid_oa]):.2f}")

    logger.info("\n--- Per-Aircraft (sorted by N) ---")
    ac_rows = []
    for ac in pd.Series(aircraft_real).value_counts().index:
        m = aircraft_real == ac
        if m.sum() < 5:
            continue
        yt, yp = y_val_real[m], pred_real[m]
        yo = openap_real[m]
        valid = ~np.isnan(yo)
        oa_rmse = _rmse(yt[valid], yo[valid]) if valid.sum() > 0 else float('nan')
        ac_rows.append({
            'aircraft': ac, 'N': int(m.sum()),
            'mean_actual_kg': float(np.mean(yt)),
            'xgb_mae': _mae(yt, yp), 'xgb_rmse': _rmse(yt, yp),
            'xgb_mape_pct': _mape(yt, yp),
            'xgb_r2': float(1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)),
            'openap_rmse': oa_rmse,
        })
        logger.info(f"  {ac:6s} N={m.sum():5d}  XGB RMSE={_rmse(yt,yp):7.1f}  XGB MAE={_mae(yt,yp):6.1f}  MAPE={_mape(yt,yp):5.1f}%  OpenAP RMSE={oa_rmse:.1f}")

    ac_df = pd.DataFrame(ac_rows)
    ac_csv = os.path.join(RESULTS_DIR, 'openap_vs_xgb_per_aircraft.csv')
    ac_df.to_csv(ac_csv, index=False)
    logger.info(f"[+] Saved per-aircraft comparison to {ac_csv}")

    logger.info("\n--- Per-Phase ---")
    for ph in ['CLIMB', 'CRUISE', 'DESCENT', 'LEVEL']:
        m = phase_real == ph
        if m.sum() < 5:
            continue
        yt, yp = y_val_real[m], pred_real[m]
        yo = openap_real[m]
        valid = ~np.isnan(yo)
        oa_rmse = _rmse(yt[valid], yo[valid]) if valid.sum() > 0 else float('nan')
        logger.info(f"  {ph:8s} N={m.sum():5d}  XGB RMSE={_rmse(yt,yp):7.1f}  XGB MAE={_mae(yt,yp):6.1f}  MAPE={_mape(yt,yp):5.1f}%  OpenAP RMSE={oa_rmse:.1f}")

    # ========================================================================
    # PHASE 6: PREPROCESS FULL DATASET
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 6: PREPROCESSING 100% OF DATA FOR FINAL TRAINING")
    logger.info("="*70)

    X_full_imputed = X_full[feature_cols_selected].copy()
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

    # Continue with Phase 7 and 8 as in original code...
    # (Test data loading and final model training)
    # The rest remains the same as your original code

    
    # ========================================================================
    # PHASE 7: LOAD AND PREPROCESS TEST DATA
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 7: LOADING AND PREPROCESSING TEST DATA")
    logger.info("="*70)

    logger.info(f"Loading test CSV: {TEST_CSV_PATH}")
    df_test_raw = pd.read_csv(TEST_CSV_PATH, delimiter=',', low_memory=False)
    logger.info(f"[+] Test data: {len(df_test_raw):,} rows")

    fuel_rank = pd.read_parquet(FUEL_RANK_PATH)
    fuel_rank_intervals = fuel_rank[['flight_id', 'idx', 'start', 'end']].copy()
    fuel_rank_intervals = fuel_rank_intervals.rename(columns={'idx': 'interval_idx'})
    submission_template = fuel_rank[['idx', 'flight_id', 'start', 'end']].copy()

    df_test_raw = df_test_raw.merge(fuel_rank_intervals, on=['flight_id', 'interval_idx'], how='left')
    df_test_raw = df_test_raw.merge(featured_data_rank_selected, on=['flight_id', 'interval_idx'], how='left')

    flightlist_rank = pd.read_parquet(FLIGHTLIST_RANK_PATH)
    flightlist_metadata = flightlist_rank[['flight_id', 'aircraft_type', 'origin_icao', 'destination_icao', 'takeoff', 'landed']].drop_duplicates(subset=['flight_id'])

    flightlist_rank_with_coords = flightlist_metadata.copy()
    flightlist_rank_with_coords = flightlist_rank_with_coords.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    flightlist_rank_with_coords = flightlist_rank_with_coords.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    flightlist_rank_with_coords = flightlist_rank_with_coords.drop(columns=['icao'], errors='ignore')
    flightlist_rank_with_coords = flightlist_rank_with_coords.merge(apt, left_on='destination_icao', right_on='icao', how='left')
    flightlist_rank_with_coords = flightlist_rank_with_coords.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    flightlist_rank_with_coords = flightlist_rank_with_coords.drop(columns=['icao'], errors='ignore')
    flightlist_rank_with_coords['great_circle_distance'] = flightlist_rank_with_coords.apply(
        lambda row: haversine(row.get('origin_lon'), row.get('origin_lat'), 
                             row.get('dest_lon'), row.get('dest_lat')), axis=1
    )

    df_test_raw = df_test_raw.merge(
        flightlist_rank_with_coords[['flight_id', 'origin_icao', 'destination_icao', 'great_circle_distance', 
                                      'aircraft_type', 'takeoff', 'landed', 'origin_lon', 'dest_lon']],
        on='flight_id', how='left'
    )

    # Add computed columns
    if 'alt_avg_ft' not in df_test_raw.columns:
        df_test_raw['alt_avg_ft'] = (df_test_raw.get('alt_start_ft', 0) + df_test_raw.get('alt_end_ft', 0)) / 2
    if 'altitude_change_rate' not in df_test_raw.columns:
        df_test_raw['altitude_change_rate'] = df_test_raw.get('alt_change_ft', 0) / (df_test_raw.get('interval_duration_sec', 60) + 1e-6)
    if 'end_hour' not in df_test_raw.columns:
        df_test_raw['end_hour'] = pd.to_datetime(df_test_raw.get('end'), errors='coerce').dt.hour.fillna(-1).astype(int)
    if 'interval_elapsed_from_flight_start' not in df_test_raw.columns:
        df_test_raw['interval_elapsed_from_flight_start'] = 0

    # Handle missing features
    missing_test_features = [col for col in feature_cols_selected if col not in df_test_raw.columns]
    if missing_test_features:
        for col in missing_test_features:
            df_test_raw[col] = 0

    X_test_data = df_test_raw[feature_cols_selected].copy()
    X_test_data = X_test_data.replace([np.inf, -np.inf], np.nan)

    if numerical_features:
        X_test_data[numerical_features] = num_imputer_full.transform(X_test_data[numerical_features])
    if categorical_features:
        X_test_data[categorical_features] = cat_imputer_full.transform(X_test_data[categorical_features])
        X_test_data[categorical_features] = cat_encoder_full.transform(X_test_data[categorical_features])

    X_test_scaled = scaler_full.transform(X_test_data)
    X_test_sfs = X_test_scaled[:, selected_mask]

    logger.info(f"[+] Test data ready: {X_test_sfs.shape}")

    X_test_sfs_df = pd.DataFrame(X_test_sfs, columns=selected_features)
    test_processed_path = os.path.join(RESULTS_DIR, 'test_X_test_processed.csv')
    X_test_sfs_df.to_csv(test_processed_path, index=False)

    logger.info(f"[+] Processed test set saved to: {test_processed_path}")

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
            device='cpu',
            n_jobs=-1,
            verbosity=0,
            **params
        )
        
        # Split for diagnostics
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_full_sfs, y_full_log, test_size=0.1, random_state=42
        )
        
        logger.info("Training with eval_set to capture learning history...")
        final_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_train_final, y_train_final), (X_val_final, y_val_final)],
            verbose=False
        )
        
        # Save plots for this model rank
        # We use a dedicated folder for rank 1 to match evaluate_model.py expectations
        diag_dir = os.path.join(RESULTS_DIR, f'xgb_model_rank{rank}')
        save_model_plots(final_model, X_train_final, y_train_final, X_val_final, y_val_final, 
                         selected_features, diag_dir, f"rank{rank}")
        
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
        parquet_path = os.path.join(RESULTS_DIR, f'test_submission_rank{rank}_synthetic_valrmse_{row["val_rmse"]:.4f}.parquet')
        submission_df.to_parquet(parquet_path, index=False, engine='fastparquet')
        logger.info(f"[+] Parquet saved: {parquet_path}")
        
        # Save CSV
        csv_path = os.path.join(RESULTS_DIR, f'test_submission_rank{rank}_synthetic_valrmse_{row["val_rmse"]:.4f}.csv')
        submission_df.to_csv(csv_path, index=False)
        logger.info(f"[+] CSV saved: {csv_path}")
        
        submission_files.append({
            'rank': rank,
            'val_rmse': row['val_rmse'],
            'val_mae': row.get('val_mae', 0.0),
            'val_r2': row.get('val_r2', 0.0),
            'train_rmse_100pct': full_rmse,
            'parquet_file': parquet_path,
            'csv_file': csv_path,
            'test_mean': test_pred.mean(),
            'test_std': test_pred.std(),
            'params': params
        })
        
        # Save parameters
        params_file = os.path.join(RESULTS_DIR, f'test_parameters_rank{rank}.txt')
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
                
        # --- NEW CONFORMITY BLOCK FOR evaluate_model.py ---
        # The evaluate script expects a folder for each model containing:
        # model.joblib, preprocessor.joblib, selected_features.json
        if rank == 1:
            eval_model_dir = os.path.join(RESULTS_DIR, f'test_xgb_model_rank{rank}')
            os.makedirs(eval_model_dir, exist_ok=True)
            
            # 1. Save model
            joblib.dump(final_model, os.path.join(eval_model_dir, 'model.joblib'))
            
            # 2. Save preprocessor dict
            # We save the individual components so they can be loaded easily if needed,
            # but we also save the unified preprocessors object under this folder for standard predict()
            
            preprocessor_dict = {
                'scaler_full': scaler_full,
                'num_imputer_full': num_imputer_full, 
                'cat_imputer_full': cat_imputer_full,
                'cat_encoder_full': cat_encoder_full,
                'selected_features': selected_features,
                'selected_mask': selected_mask,
                'feature_cols_selected': feature_cols_selected,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features
            }
            joblib.dump(preprocessor_dict, os.path.join(eval_model_dir, 'preprocessor.joblib'))
            
            with open(os.path.join(eval_model_dir, 'selected_features.json'), 'w') as f:
                json.dump(selected_features, f)
            
            logger.info(f"[+] Saved evaluation artifacts to {eval_model_dir}")

            # --- GENERATE PAPER PLOTS ---
            try:
                logger.info("--- Generating Paper Plots for Test/FS results ---")
                generate_paper_plots.run_all(eval_model_dir)
            except Exception as e:
                logger.error(f"[-] Failed to generate paper plots: {e}")

    joblib.dump({
    'scaler_full': scaler_full,
    'num_imputer_full': num_imputer_full, 
    'cat_imputer_full': cat_imputer_full,
    'cat_encoder_full': cat_encoder_full,
    'selected_features': selected_features,
    'selected_mask': selected_mask,
    'feature_cols_selected': feature_cols_selected,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features
    }, os.path.join(RESULTS_DIR, 'test_preprocessors_rank.joblib'))
    # ========================================================================
    # PHASE 9: SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 9: FINAL SUMMARY")
    logger.info("="*70)

    summary_df = pd.DataFrame(submission_files)
    summary_path = os.path.join(RESULTS_DIR, 'test_top5_models_synthetic_summary.csv')
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


def run(gpu_id=0, force_sfs=False, force_synthetic=False, opt_mode='legacy'):
    main(gpu_id=gpu_id, force_sfs=force_sfs, force_synthetic=force_synthetic, opt_mode=opt_mode)

if __name__ == "__main__":
    run()

