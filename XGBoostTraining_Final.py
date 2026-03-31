import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
import joblib
import logging
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import time
import json
import sys
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization modes
warnings.filterwarnings('ignore')

# Aircraft specs from config
AIRCRAFT_DATA = config.AIRCRAFT_DATA
WIDEBODY_AIRCRAFT = config.WIDEBODY_AIRCRAFT

# Config-driven file paths
DATA_PATH = config.AUGMENTED_FINAL_CSV
APT_PATH = config.APT_PARQUET
FLIGHTLIST_PATH = config.FLIGHTLIST_TRAIN
FUEL_PATH = config.FUEL_TRAIN
TEST_CSV_PATH = config.AUGMENTED_FINAL_TEST_CSV # Using actual final test set for final training
RANK_CSV_PATH = config.AUGMENTED_RANK_CSV
FUEL_RANK_PATH = config.FUEL_RANK
FLIGHTLIST_RANK_PATH = config.FLIGHTLIST_RANK
FLIGHTLIST_FINAL_PATH = config.FLIGHTLIST_FINAL
RESULTS_DIR = config.MODELS_DIR

FEATURED_DATA_TRAIN = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
FEATURED_DATA_RANK = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_rank.parquet')
FEATURED_DATA_TEST = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_final.parquet')
SYNTHETIC_PATH = config.SYNTHETIC_WIDEBODY_PATH
SELECTED_FEATURES_PATH = config.SELECTED_FEATURES_PATH

# Ground-truth files (now available after competition reveal)
FUEL_RANK_GT_PATH  = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank.parquet')
FUEL_FINAL_GT_PATH = os.path.join(config.BASE_DATASETS_DIR, 'fuel_final.parquet')


os.makedirs(RESULTS_DIR, exist_ok=True)

log_file = os.path.join(RESULTS_DIR, f'final_xgboost_top5_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Feature space definition
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
    """Generate model diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Learning curves
    results = model.evals_result()
    if results and 'validation_0' in results:
        # Export training metrics
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

    # Regression error analysis
    y_val_orig = np.expm1(y_val)
    y_pred_log = model.predict(X_val)
    y_pred_orig = np.expm1(y_pred_log)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val_orig, y_pred_orig, alpha=0.3, color='teal', s=10)
    
    # Identity line
    max_val = max(y_val_orig.max(), y_pred_orig.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
    
    plt.title(f'Predicted vs Actual - {rank_name}')
    plt.xlabel('Actual Fuel Consumption (kg)')
    plt.ylabel('Predicted Fuel Consumption (kg)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predicted_vs_actual_{rank_name}.png'))
    plt.close()

    # Top-20 feature importance
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
# SYNTHETIC WIDEBODY AUGMENTATION (25K samples; 25% from top-quartile segments)
# ============================================================================
def generate_synthetic_widebody_data_enhanced(df_train, n_synthetic=25000, long_segment_pct=0.25, random_state=42):
    """Gaussian-perturbed widebody augmentation; oversample long segments."""
    np.random.seed(random_state)
    
    # WB-only slice
    df_widebody = df_train[df_train['aircraft_type'].isin(WIDEBODY_AIRCRAFT)].copy()
    
    logger.info(f"Original widebody samples: {len(df_widebody):,}")
    logger.info(f"Generating {n_synthetic:,} synthetic samples...")
    logger.info(f"  - {int(n_synthetic * long_segment_pct):,} from LONG segments ({long_segment_pct*100:.0f}%)")
    logger.info(f"  - {int(n_synthetic * (1-long_segment_pct)):,} from ALL segments ({(1-long_segment_pct)*100:.0f}%)")
    
    # Resolve duration column name
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
        
        # 75th/90th percentile thresholds for "long" segments
        duration_75th = df_widebody[duration_col].quantile(0.75)
        duration_90th = df_widebody[duration_col].quantile(0.90)
        
        logger.info(f"    Duration 75th percentile: {duration_75th:.1f}")
        logger.info(f"    Duration 90th percentile: {duration_90th:.1f}")
        
        # Flag top-quartile duration segments as long
        df_widebody['is_long_segment'] = df_widebody[duration_col] >= duration_75th
        
        n_long = df_widebody['is_long_segment'].sum()
        logger.info(f"    Long segments identified: {n_long:,} ({n_long/len(df_widebody)*100:.1f}%)")
    
    # Per-type sample allocation proportional to original distribution
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
        
        # Perturb-able numerical columns (exclude index/helper cols)
        numerical_cols = aircraft_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['flight_id', 'interval_idx', 'idx', 'is_long_segment']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Long-segment pool sampling
        for i in range(n_aircraft_from_long):
            if len(aircraft_data_long) == 0:
                # Fallback to full pool if no long segments
                base_sample = aircraft_data.sample(n=1, random_state=random_state+i).iloc[0].copy()
            else:
                base_sample = aircraft_data_long.sample(n=1, random_state=random_state+i).iloc[0].copy()
            
            # Gaussian perturbation per numerical feature
            for col in numerical_cols:
                if col in base_sample.index and pd.notna(base_sample[col]):
                    original_value = base_sample[col]
                    col_std = aircraft_data_long[col].std() if len(aircraft_data_long) > 1 else aircraft_data[col].std()
                    if pd.notna(col_std) and col_std > 0:
                        # noise magnitude: U[5%, 15%] × feature std
                        noise_factor = np.random.uniform(0.05, 0.15)
                        noise = np.random.normal(0, col_std * noise_factor)
                        new_value = original_value + noise
                        
                        # Physical domain clamps
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

        # Full-pool sampling
        for i in range(n_aircraft_from_all):
            base_sample = aircraft_data.sample(n=1, random_state=random_state+n_aircraft_from_long+i).iloc[0].copy()
            for col in numerical_cols:
                if col in base_sample.index and pd.notna(base_sample[col]):
                    original_value = base_sample[col]
                    col_std = aircraft_data[col].std()
                    if pd.notna(col_std) and col_std > 0:
                        noise_factor = np.random.uniform(0.05, 0.15)
                        noise = np.random.normal(0, col_std * noise_factor)
                        new_value = original_value + noise
                        # Physical domain clamps
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
    
    # Drop temporary helper column
    if 'is_long_segment' in df_synthetic.columns:
        df_synthetic = df_synthetic.drop(columns=['is_long_segment'])
    
    logger.info(f"\n[+] Generated {len(df_synthetic):,} synthetic widebody samples")
    logger.info(f"[+] Synthetic distribution:")
    synth_counts = df_synthetic['aircraft_type'].value_counts()
    for aircraft, count in synth_counts.items():
        logger.info(f"  {aircraft}: {count:,} ({count/len(df_synthetic)*100:.1f}%)")
    
    # Log duration shift stats
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

    # Mode-specific output directory
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

    # PHASE 1: Data Ingestion
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

    # Load parquet-based feature set
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

    # Join extended features
    df_raw = df_raw.merge(featured_data_selected, on=['flight_id', 'interval_idx'], how='left')
    # Carry flight_duration_seconds for haul binning (non-feature)
    if 'flight_duration_seconds' not in df_raw.columns and 'flight_duration_seconds' in featured_data.columns:
        _dur_tmp = featured_data[['flight_id', 'idx', 'flight_duration_seconds']].rename(
            columns={'idx': 'interval_idx'})
        df_raw = df_raw.merge(_dur_tmp, on=['flight_id', 'interval_idx'], how='left')
    logger.info(f"[+] Total columns: {len(df_raw.columns)}")

    base_features = [
        'starting_mass_kg', 'alt_end_ft', 'alt_avg_ft', 'gs_avg_kts', 'vs_avg_fpm',
        'interval_duration_sec', 'altitude_change_rate', 'great_circle_distance',
        'aircraft_type', 'end_hour', 'interval_elapsed_from_flight_start',
        'openap_fuel_kg',
    ]

    extended_features_available = [col for col in available_features[2:] if col not in base_features]
    feature_cols_selected = base_features + extended_features_available

    logger.info(f"[+] Total features: {len(feature_cols_selected)}")

    # Derived features
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
    # Carry flight_duration_seconds as a non-feature column for haul binning later
    # (merged into df_raw above from featured_data; not part of feature_cols_selected)
    df_features['_flight_dur_s'] = (
        df_raw.loc[df_features.index, 'flight_duration_seconds']
        if 'flight_duration_seconds' in df_raw.columns
        else np.nan
    )

    logger.info(f"[+] Original dataset: {len(df_features):,} intervals")

    # PHASE 1.5: Data Augmentation
    logger.info("\n" + "="*70)
    logger.info("PHASE 1.5: GENERATING ENHANCED SYNTHETIC WIDEBODY DATA")
    logger.info("="*70)

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


    # Concat real + synthetic
    df_features_augmented = pd.concat([df_features, df_synthetic], ignore_index=True)

    logger.info(f"\n[+] Original training size: {len(df_features):,}")
    logger.info(f"[+] Synthetic samples added: {len(df_synthetic):,}")
    logger.info(f"[+] Augmented training size: {len(df_features_augmented):,}")
    logger.info(f"[+] Augmentation rate: {len(df_synthetic)/len(df_features)*100:.1f}%")

    # Full augmented feature/target arrays
    X_full = df_features_augmented[feature_cols_selected]
    y_full = df_features_augmented[target_col].values.astype(np.float32)

    logger.info(f"[+] Full dataset (with synthetic): {len(df_features_augmented):,} intervals")

    # PHASE 2: Validation Strategy
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: 80/20 TRAIN/VALIDATION SPLIT")
    logger.info("="*70)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, shuffle=True
    )

    logger.info(f"[+] Training: {len(X_train):,} intervals ({len(X_train)/len(X_full)*100:.1f}%)")
    logger.info(f"[+] Validation: {len(X_val):,} intervals ({len(X_val)/len(X_full)*100:.1f}%)")

    # PHASE 3: Feature Scaling & Imputation
    logger.info("\nPHASE 3: DATA PREPROCESSING (FITTED ON TRAINING SET)")

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()

    # Remove invariant NaN columns
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

    # PHASE 4: Forward Feature Selection
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
        
        # Intersect loaded features with current feature space
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
            tree_method='auto',
            device=f'cuda:{gpu_id}' if gpu_id is not None else 'cpu',
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

        # Run forward SFS
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
    # Leaderboard: Top 10 features
    for i, feat in enumerate(selected_features[:10], 1):
        logger.info(f"      {i:2d}. {feat}")
    if len(selected_features) > 10:
        logger.info(f"      ... and {len(selected_features) - 10} more")

    X_train_sfs = X_train_scaled[:, selected_mask]
    X_val_sfs = X_val_scaled[:, selected_mask]

    # PHASE 5: Hyperparameter Optimization
    logger.info("\n" + "="*70)
    logger.info(f"PHASE 5: HYPERPARAMETER SELECTION (Mode: {opt_mode.upper()})")
    logger.info("="*70)

    # Legacy tuned parameters
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

    # User-supplied grid parameters
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
        # Optuna optimization routine

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
            
            # Enforce C-contiguity for XGBoost compatibility
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
        trials_path = os.path.join(RESULTS_DIR, 'optuna_trials_history.csv')
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
        logger.info("[!] Using legacy hyperparameters.")
        top_10_cv = pd.DataFrame([{
            'mean_test_score': -0.0,
            'std_test_score': 0.0,
            'params': legacy_params
        }])

    # Save scaled train/val splits for offline analysis
    X_train_sfs_df = pd.DataFrame(X_train_sfs, columns=selected_features)
    X_val_sfs_df = pd.DataFrame(X_val_sfs, columns=selected_features)
    X_train_sfs_df.to_csv(os.path.join(RESULTS_DIR, 'X_train_processed.csv'), index=False)
    X_val_sfs_df.to_csv(os.path.join(RESULTS_DIR, 'X_val_processed.csv'), index=False)

    # Validation phase: Top 10 evaluation
    logger.info("\n" + "="*70)
    logger.info("EVALUATING TOP 10 MODELS ON 20% VALIDATION SET")
    logger.info("="*70)

    validation_results = []
    _best_val_rmse_seen = np.inf
    _best_val_pred_kg   = None   # val predictions for the rank-1 model (lowest val RMSE)

    for rank, (idx, row) in enumerate(tqdm(top_10_cv.iterrows(), total=len(top_10_cv), desc="Training Top Models"), 1):
        logger.info(f"\nEvaluating Model {rank}/10...")
        
        model = XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='auto',
            device=f'cuda:{gpu_id}' if gpu_id is not None else 'cpu',
            n_jobs=-1,
            verbosity=0,
            **row['params']
        )
        
        model.fit(X_train_sfs, y_train_log)
        
        # Predict on held-out val set
        val_pred_log = model.predict(X_val_sfs)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.maximum(val_pred, 0.0)

        # Track rank-1 val predictions for phase/baseline analysis
        _cur_val_rmse = float(np.sqrt(np.mean((y_val - val_pred)**2)))
        if _cur_val_rmse < _best_val_rmse_seen:
            _best_val_rmse_seen = _cur_val_rmse
            _best_val_pred_kg   = val_pred.copy()
        
        # Validation metrics
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-8))) * 100
        val_r2 = 1 - (np.sum((y_val - val_pred) ** 2) / np.sum((y_val - y_val.mean()) ** 2))
        
        # Train diagnostics (overfitting gap)
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

    # Rank models by validation RMSE
    logger.info("\n" + "="*70)
    logger.info("TOP 10 MODELS RANKED BY VALIDATION RMSE")
    logger.info("="*70)

    results_df = pd.DataFrame(validation_results)
    results_df = results_df.sort_values('val_rmse', ascending=True)
    results_df['final_rank'] = range(1, len(results_df) + 1)

    # Save ranked results
    results_path = os.path.join(RESULTS_DIR, 'random_search_top10_validation_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"[+] Detailed results saved: {results_path}")

    # Leaderboard
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

    # PHASE 6: Full Data Preprocessing
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
    # Preserve SFS feature names/mask; Phase 7 will overwrite selected_features
    _training_feat_names = list(selected_features)
    _training_mask = selected_mask.copy()


    # PHASE 7: Data Integrity Diagnostics
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

    # Retrieve pre-trained transformers
    # (Removed to prevent overwriting freshly fitted variables from Phase 6)

    # Load submission template
    logger.info("🔍 LOADING FUEL SUBMISSION DATA...")
    try:
        # For the final evaluation, we need to load the final template for creating the final parquet
        fuel_submission = pd.read_parquet(config.FUEL_FINAL)
        logger.info(f"✅ Loaded {os.path.basename(config.FUEL_FINAL)}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load final template. Falling back to rank: {e}")
        fuel_submission = pd.read_parquet(config.FUEL_RANK)

    fuel_rank_intervals = fuel_submission[['flight_id', 'idx', 'start', 'end']].rename(columns={'idx': 'interval_idx'})

    # Load other data
    rank_csv = pd.read_csv(RANK_CSV_PATH, delimiter=',', low_memory=False)
    final_csv = pd.read_csv(TEST_CSV_PATH, low_memory=False)
    featured_data_rank = pd.read_parquet(FEATURED_DATA_RANK).rename(columns={'idx': 'interval_idx'})
    featured_data_final = pd.read_parquet(FEATURED_DATA_TEST).rename(columns={'idx': 'interval_idx'})

    # Rank segment processing
    logger.info("\n🔍 PROCESSING RANK DATA...")
    df_test_rank = rank_csv.head(N_RANK_ROWS).copy()
    if 'idx' in df_test_rank.columns:
        df_test_rank = df_test_rank.rename(columns={'idx': 'interval_idx'})

    df_test_rank = df_test_rank.merge(fuel_rank_intervals, on=['flight_id', 'interval_idx'], how='left')
    logger.info(f"Rank fuel merge: {df_test_rank['end'].notna().sum() if 'end' in df_test_rank.columns else 0:,} / {len(df_test_rank):,} matched")
    df_test_rank = df_test_rank.merge(featured_data_rank, on=['flight_id', 'interval_idx'], how='left')

    # Aircraft-type infill from featured_data_rank
    if 'aircraft_type' in featured_data_rank.columns:
        aircraft_rank = featured_data_rank[['flight_id', 'aircraft_type']].drop_duplicates(subset=['flight_id'])
        df_test_rank = df_test_rank.merge(aircraft_rank, on='flight_id', how='left', suffixes=('', '_feat'))
        df_test_rank['aircraft_type'] = df_test_rank['aircraft_type'].fillna(df_test_rank['aircraft_type_feat'])
        df_test_rank = df_test_rank.drop(columns=['aircraft_type_feat'], errors='ignore')
    df_test_rank['aircraft_type'] = df_test_rank['aircraft_type'].fillna('A320')
    logger.info(f"✅ Rank aircraft_type: {df_test_rank['aircraft_type'].notna().sum():,} / {len(df_test_rank):,} ({df_test_rank['aircraft_type'].nunique():,} types)")

    # Great-circle distance (rank)
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

    # Derived features (rank)
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

    # Final segment processing
    logger.info("\n🔍 PROCESSING FINAL DATA...")
    df_test_final = final_csv.copy()
    if 'idx' in df_test_final.columns:
        df_test_final = df_test_final.rename(columns={'idx': 'interval_idx'})

    df_test_final = df_test_final.merge(fuel_rank_intervals, on=['flight_id', 'interval_idx'], how='inner')
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

    # Great-circle distance (final)
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

    # Derived features (final)
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

    # Phase 7B: Hybrid test matrix — Testing.py rank rows + new final rows
    logger.info("\n" + "="*50)
    logger.info("PHASE 7B: Testing.py Rank + NEW Final Rows")
    logger.info("="*50)

    # Load Testing.py pre-scaled rank features; recompute from unscaled if columns differ
    testing_processed = os.path.join(RESULTS_DIR, 'test_X_test_processed.csv')
    if os.path.exists(testing_processed):
        _rank_csv_df = pd.read_csv(testing_processed)
        _missing = [c for c in _training_feat_names if c not in _rank_csv_df.columns]
        if _missing:
            logger.warning(f"⚠️ {len(_missing)} training features absent in Testing.py CSV — recomputing rank from unscaled data")
            X_test_rank_scaled = scaler_full.transform(X_test_rank_unscaled)
            X_test_rank_sfs = X_test_rank_scaled[:, _training_mask]
        else:
            X_test_rank_sfs = _rank_csv_df[_training_feat_names].values
        logger.info(f"✅ LOADED Testing.py rank: {X_test_rank_sfs.shape}")
    else:
        logger.warning("⚠️ No Testing.py file - scaling rank data")
        X_test_rank_scaled = scaler_full.transform(X_test_rank_unscaled)
        X_test_rank_sfs = X_test_rank_scaled[:, _training_mask]

    # Scale new final rows (apply training SFS mask)
    final_new_unscaled = X_test_final_unscaled.iloc[N_RANK_ROWS:]
    X_test_final_new_scaled = scaler_full.transform(final_new_unscaled)
    X_test_final_new_sfs = X_test_final_new_scaled[:, _training_mask]
    logger.info(f"✅ NEW Final rows: {X_test_final_new_sfs.shape}")

    # Stack rank + final into hybrid test matrix
    X_test_sfs = np.vstack([X_test_rank_sfs, X_test_final_new_sfs])
    logger.info(f"✅ HYBRID X_test_sfs: {X_test_sfs.shape}")

    # Submission schema
    submission_template = fuel_submission[['idx', 'flight_id', 'start', 'end']].copy()
    logger.info(f"✅ Submission template: {submission_template.shape}")

    # Save scaled test data
    pd.DataFrame(X_test_sfs, columns=_training_feat_names).to_csv(
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


    # PHASE 8: Final Model Training & Submission
    logger.info("\n" + "="*70)
    logger.info("PHASE 8: TRAINING ALL TOP 5 MODELS ON 100% DATA")
    logger.info("="*70)

    # Load GT fuel for post-hoc RMSE evaluation
    gt_rank_df = gt_final_df = None
    if os.path.exists(FUEL_RANK_GT_PATH):
        gt_rank_df = pd.read_parquet(FUEL_RANK_GT_PATH)[['flight_id', 'idx', 'start', 'end', 'fuel_kg']]
        logger.info(f"[+] Ground-truth rank loaded: {len(gt_rank_df):,} segments")
    else:
        logger.warning(f"[!] Ground-truth rank not found: {FUEL_RANK_GT_PATH}")
    if os.path.exists(FUEL_FINAL_GT_PATH):
        gt_final_df = pd.read_parquet(FUEL_FINAL_GT_PATH)[['flight_id', 'idx', 'start', 'end', 'fuel_kg']]
        logger.info(f"[+] Ground-truth final loaded: {len(gt_final_df):,} segments")
    else:
        logger.warning(f"[!] Ground-truth final not found: {FUEL_FINAL_GT_PATH}")

    submission_files = []

    for idx, row in results_df.iterrows():
        rank = row['final_rank']
        params = row['params']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING MODEL RANK #{rank}")
        logger.info(f"{'='*70}")
        logger.info(f"Expected Validation RMSE: {row['val_rmse']:.4f} kg")
        logger.info(f"Parameters: {params}")
        
        # Train on full augmented dataset
        final_model = XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            device='cpu',
            n_jobs=-1,
            verbosity=0,
            **params
        )
        
        # 10% split for learning-curve capture
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
        diag_dir = os.path.join(RESULTS_DIR, f'xgb_model_rank{rank}')
        save_model_plots(final_model, X_train_final, y_train_final, X_val_final, y_val_final,
                         _training_feat_names, diag_dir, f"rank{rank}")
        
        # ── Feature importance (XGBoost gain) ──────────────────────────────
        logger.info("\n" + "-"*70)
        logger.info(f"FEATURE IMPORTANCE ANALYSIS - MODEL RANK #{rank}")
        logger.info("-"*70)
        # Use _training_feat_names (Phase-4 SFS); selected_features overwritten by Phase-7 joblib
        feature_importance = final_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': _training_feat_names,
            'importance': feature_importance,
            'importance_pct': (feature_importance / feature_importance.sum()) * 100
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_path = os.path.join(RESULTS_DIR, f'feature_importance_rank{rank}.csv')
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"[+] Feature importance saved to: {importance_path}")
        
        # Display top-20 contributors
        logger.info(f"\n[+] Top 20 Most Important Features:")
        logger.info(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'% Total':<10}")
        logger.info("-"*75)
        for _, imp_row in importance_df.head(20).iterrows():
            logger.info(
                f"{int(imp_row['rank']):<6} {imp_row['feature']:<45} "
                f"{imp_row['importance']:<12.6f} {imp_row['importance_pct']:<10.2f}%"
            )
        
        # Display bottom-10 contributors
        logger.info(f"\n[+] Bottom 10 Least Important Features:")
        logger.info(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'% Total':<10}")
        logger.info("-"*75)
        for _, imp_row in importance_df.tail(10).iterrows():
            logger.info(
                f"{int(imp_row['rank']):<6} {imp_row['feature']:<45} "
                f"{imp_row['importance']:<12.6f} {imp_row['importance_pct']:<10.2f}%"
            )
        
        # Cumulative importance thresholds
        cumulative_importance = importance_df['importance_pct'].cumsum()
        n_features_90pct = (cumulative_importance <= 90).sum() + 1
        n_features_95pct = (cumulative_importance <= 95).sum() + 1
        
        logger.info(f"\n[+] Cumulative Importance Analysis:")
        logger.info(f"    Top {n_features_90pct} features explain 90% of importance")
        logger.info(f"    Top {n_features_95pct} features explain 95% of importance")
        logger.info(f"    Total features: {len(_training_feat_names)}")
        
        logger.info("-"*70)
        # Training set RMSE
        full_pred_log = final_model.predict(X_full_sfs)
        full_pred = np.expm1(full_pred_log)
        full_pred = np.maximum(full_pred, 0.0)
        full_rmse = np.sqrt(np.mean((y_full - full_pred) ** 2))
        
        logger.info(f"[+] Training RMSE (100% data): {full_rmse:.4f} kg")
        
        # Test set prediction
        test_pred_log = final_model.predict(X_test_sfs)
        test_pred = np.expm1(test_pred_log)
        test_pred = np.maximum(test_pred, 0.0)
        
        logger.info(f"[+] Test predictions: {len(test_pred):,}")
        logger.info(f"    Range: [{test_pred.min():.2f}, {test_pred.max():.2f}] kg")
        logger.info(f"    Mean: {test_pred.mean():.2f} kg")
        
        # Build submission dataframe
        submission_df = submission_template.copy()
        submission_df['fuel_kg'] = test_pred.astype(np.float32)
        submission_df = submission_df[['idx', 'flight_id', 'start', 'end', 'fuel_kg']]

        # ── Ground-truth RMSE evaluation ─────────────────────────────────────
        rank_rmse = final_rmse = combined_rmse = np.nan
        def _rmse(y_true, y_pred):
            return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

        if gt_rank_df is not None:
            merged_rank = submission_df.merge(
                gt_rank_df.rename(columns={'fuel_kg': 'fuel_kg_gt'}),
                on=['flight_id', 'idx', 'start', 'end'], how='inner'
            )
            if len(merged_rank) > 0:
                rank_rmse = _rmse(merged_rank['fuel_kg_gt'], merged_rank['fuel_kg'])
                logger.info(f"[+] RANK  ground-truth RMSE: {rank_rmse:.4f} kg  "
                            f"({len(merged_rank):,} segments matched)")
            else:
                logger.warning("[!] Rank GT merge returned 0 rows — check flight_id/idx alignment")

        if gt_final_df is not None:
            merged_final = submission_df.merge(
                gt_final_df.rename(columns={'fuel_kg': 'fuel_kg_gt'}),
                on=['flight_id', 'idx', 'start', 'end'], how='inner'
            )
            if len(merged_final) > 0:
                final_rmse = _rmse(merged_final['fuel_kg_gt'], merged_final['fuel_kg'])
                logger.info(f"[+] FINAL ground-truth RMSE: {final_rmse:.4f} kg  "
                            f"({len(merged_final):,} segments matched)")
            else:
                logger.warning("[!] Final GT merge returned 0 rows — check flight_id/idx alignment")

        if gt_rank_df is not None and gt_final_df is not None:
            gt_combined = pd.concat([gt_rank_df, gt_final_df], ignore_index=True)
            merged_combined = submission_df.merge(
                gt_combined.rename(columns={'fuel_kg': 'fuel_kg_gt'}),
                on=['flight_id', 'idx', 'start', 'end'], how='inner'
            )
            if len(merged_combined) > 0:
                combined_rmse = _rmse(merged_combined['fuel_kg_gt'], merged_combined['fuel_kg'])
                logger.info(f"[+] COMBINED  ground-truth RMSE: {combined_rmse:.4f} kg  "
                            f"({len(merged_combined):,} segments matched)")
        # ─────────────────────────────────────────────────────────────────────
        
        # Save parquet with fastparquet
        parquet_path = os.path.join(RESULTS_DIR, f'final_submission_rank{rank}_synthetic_valrmse_{row["val_rmse"]:.4f}.parquet')
        submission_df.to_parquet(parquet_path, index=False, engine='fastparquet')
        logger.info(f"[+] Parquet saved: {parquet_path}")
        
        # Export CSV backup
        csv_path = os.path.join(RESULTS_DIR, f'final_submission_rank{rank}_synthetic_valrmse_{row["val_rmse"]:.4f}.csv')
        submission_df.to_csv(csv_path, index=False)
        logger.info(f"[+] CSV saved: {csv_path}")
        
        submission_files.append({
            'rank': rank,
            'val_rmse': row['val_rmse'],
            'val_mae': row.get('val_mae', 0.0),
            'val_r2': row.get('val_r2', 0.0),
            'train_rmse_100pct': full_rmse,
            'gt_rank_rmse': rank_rmse,
            'gt_final_rmse': final_rmse,
            'gt_combined_rmse': combined_rmse,
            'parquet_file': parquet_path,
            'csv_file': csv_path,
            'test_mean': test_pred.mean(),
            'test_std': test_pred.std(),
            'params': params
        })
        
        # Save hyperparameter record
        params_file = os.path.join(RESULTS_DIR, f'final_parameters_rank{rank}.txt')
        with open(params_file, 'w') as f:
            f.write(f"MODEL RANK #{rank}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Validation RMSE: {row['val_rmse']:.4f} kg\n")
            f.write(f"Validation MAE:  {row['val_mae']:.4f} kg\n")
            f.write(f"Validation R²:   {row['val_r2']:.4f}\n")
            f.write(f"Training RMSE (100%): {full_rmse:.4f} kg\n")
            if not np.isnan(rank_rmse):
                f.write(f"Ground-truth Rank  RMSE: {rank_rmse:.4f} kg\n")
            if not np.isnan(final_rmse):
                f.write(f"Ground-truth Final RMSE: {final_rmse:.4f} kg\n")
            if not np.isnan(combined_rmse):
                f.write(f"Ground-truth Combined RMSE: {combined_rmse:.4f} kg\n")
            f.write("\nHyperparameters:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")

        # Save rank-1 artifacts for evaluate.py (model.joblib, preprocessor.joblib, selected_features.json)
        if rank == 1:
            eval_model_dir = os.path.join(RESULTS_DIR, f'final_xgb_model_rank{rank}')
            os.makedirs(eval_model_dir, exist_ok=True)
            joblib.dump(final_model, os.path.join(eval_model_dir, 'model.joblib'))
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

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 8.5: ABLATION STUDY (6 conditions, production pipeline)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("PHASE 8.5: ABLATION STUDY (6 conditions)")
    logger.info("="*70)

    if gt_rank_df is None or gt_final_df is None:
        logger.warning("[ABLATION] GT files not available — skipping ablation study")
    else:
        # ── SFS feature set ───────────────────────────────────────────────
        with open(SELECTED_FEATURES_PATH) as _af:
            _afs = json.load(_af)
        _sfs_feats = (_afs['selected_features'] if isinstance(_afs, dict) else _afs)
        _sfs_feats = [c for c in _sfs_feats if c in feature_cols_selected]
        logger.info(f"  Base SFS feature set: {len(_sfs_feats)} features")

        _LF_COLS = [c for c in [
            'average_load_factor', 'estimated_payload_kg',
            'trip_fuel_kg', 'contingency_fuel_kg', 'final_reserve_fuel_kg',
            'estimated_total_fuel_kg', 'estimated_takeoff_mass',
        ] if c in _sfs_feats]

        _DEVICE = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
        _n_real = len(df_features)

        # ── Helper: refit preprocessors, train on 80% (Val) + 100% (GT) ──
        def _abl_run(label, X_tr_df, feat_list, y_tr,
                     X_ev_r, y_ev_r, at_ev_r,
                     X_ev_f, y_ev_f, at_ev_f,
                     _return_preds=False,
                     X_tr_80_df=None, y_tr_80=None):

            def _prep_fit_predict(Xtr, ytr, *Xev_dfs):
                """Fit XGBoost on Xtr; return kg-scale preds for each Xev_df."""
                f = [c for c in feat_list
                     if c in Xtr.columns and not Xtr[c].isna().all()]
                Xt = Xtr[f].copy().replace([np.inf, -np.inf], np.nan)
                Xevs = [Xe.reindex(columns=f, fill_value=np.nan)
                          .copy().replace([np.inf, -np.inf], np.nan)
                        for Xe in Xev_dfs]
                n_f = [c for c in f if pd.api.types.is_numeric_dtype(Xt[c])]
                c_f = [c for c in f if c not in n_f]
                if n_f:
                    ni = SimpleImputer(strategy='mean')
                    Xt[n_f] = ni.fit_transform(Xt[n_f])
                    for Xe in Xevs:
                        Xe[n_f] = ni.transform(Xe[n_f])
                if c_f:
                    ci = SimpleImputer(strategy='most_frequent')
                    Xt[c_f] = ci.fit_transform(Xt[c_f])
                    for Xe in Xevs:
                        Xe[c_f] = ci.transform(Xe[c_f])
                    ce = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    Xt[c_f] = ce.fit_transform(Xt[c_f])
                    for Xe in Xevs:
                        Xe[c_f] = ce.transform(Xe[c_f])
                sc = StandardScaler()
                Xt_s = sc.fit_transform(Xt)
                Xevs_s = [sc.transform(Xe.values) for Xe in Xevs]
                m = XGBRegressor(
                    random_state=42, objective='reg:squarederror',
                    tree_method='hist', device=_DEVICE, n_jobs=-1, verbosity=0,
                    **dict(legacy_params))
                t0 = time.time()
                m.fit(Xt_s, np.log1p(ytr))
                logger.info(f"  [{label[:35]}] {len(Xt):,} rows, {len(f)} feats → {time.time()-t0:.0f}s")
                return [np.maximum(np.expm1(m.predict(Xe_s)), 0) for Xe_s in Xevs_s]

            def _blk(y_true, y_pred, at_ser, ds):
                rows_ = []
                is_wb = at_ser.isin(WIDEBODY_AIRCRAFT).values
                def _mm(yt_, yp_):
                    mae_  = float(np.mean(np.abs(yt_ - yp_)))
                    rmse_ = float(np.sqrt(np.mean((yt_ - yp_)**2)))
                    mape_ = float(np.mean(np.abs((yt_ - yp_) / (yt_ + 1e-8))) * 100)
                    r2_   = float(1 - np.sum((yt_ - yp_)**2) / np.sum((yt_ - np.mean(yt_))**2))
                    return mae_, rmse_, mape_, r2_
                mae, rmse, mape, r2 = _mm(y_true, y_pred)
                rows_.append({'condition': label, 'dataset': ds, 'split': 'Overall',
                              'n': len(y_true), 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2})
                logger.info(f"    [{ds}] Overall   MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}  N={len(y_true):,}")
                for _sp, _mask in [('Narrowbody', ~is_wb), ('Widebody', is_wb)]:
                    if _mask.sum() > 0:
                        m2, r2b, mp2, r22 = _mm(y_true[_mask], y_pred[_mask])
                        rows_.append({'condition': label, 'dataset': ds, 'split': _sp,
                                      'n': int(_mask.sum()), 'mae': m2, 'rmse': r2b,
                                      'mape': mp2, 'r2': r22})
                        logger.info(f"    [{ds}] {_sp:<12}  MAE={m2:.1f}  RMSE={r2b:.1f}  N={_mask.sum():,}")
                return rows_

            rows = []
            # Val: train on 80%, eval on 20% hold-out
            if X_tr_80_df is not None and y_tr_80 is not None:
                pv = _prep_fit_predict(X_tr_80_df, y_tr_80, X_val)[0]
                rows += _blk(y_val, pv, _at_val, 'Val')
            # GT: train on 100%, eval on combined hidden GT
            pgt = _prep_fit_predict(X_tr_df, y_tr, X_ev_r, X_ev_f)
            pr, pf = pgt[0], pgt[1]
            y_comb  = np.concatenate([y_ev_r, y_ev_f])
            p_comb  = np.concatenate([pr, pf])
            at_comb = pd.concat([at_ev_r.reset_index(drop=True),
                                 at_ev_f.reset_index(drop=True)], ignore_index=True)
            rows += _blk(y_comb, p_comb, at_comb, 'GT')

            if _return_preds:
                return rows, p_comb, y_comb, at_comb
            return rows

        # ── Build GT-aligned eval DataFrames from Phase 7 outputs ─────────
        _gt_r = gt_rank_df[['flight_id', 'idx', 'fuel_kg']].rename(
            columns={'idx': 'interval_idx', 'fuel_kg': '_gt'})
        _gt_f = gt_final_df[['flight_id', 'idx', 'fuel_kg']].rename(
            columns={'idx': 'interval_idx', 'fuel_kg': '_gt'})

        _rank_ev  = df_test_rank.merge(_gt_r,  on=['flight_id', 'interval_idx'], how='inner')
        _final_ev = df_test_final.merge(_gt_f, on=['flight_id', 'interval_idx'], how='inner')
        y_ev_r  = _rank_ev['_gt'].values.astype(np.float32)
        y_ev_f  = _final_ev['_gt'].values.astype(np.float32)
        at_ev_r = _rank_ev.get('aircraft_type',
                                pd.Series(['A320'] * len(_rank_ev), index=_rank_ev.index))
        at_ev_f = _final_ev.get('aircraft_type',
                                 pd.Series(['A320'] * len(_final_ev), index=_final_ev.index))
        logger.info(f"  GT-matched: Rank={len(_rank_ev):,}  Final={len(_final_ev):,}")

        # ── Per-phase GT metrics (production rank-1 model) ────────────────
        logger.info(f"\n{'═'*70}")
        logger.info("  PER-PHASE GT METRICS (Production Rank-1 Model, Combined GT)")
        logger.info(f"{'═'*70}")
        _prod_preds_bl = None  # filled inside try; used by baseline comparison below
        try:
            _rank1_dir = os.path.join(RESULTS_DIR, 'final_xgb_model_rank1')
            _prod_model = joblib.load(os.path.join(_rank1_dir, 'model.joblib'))
            _prep_from_disk = joblib.load(os.path.join(_rank1_dir, 'preprocessor.joblib'))
            _prod_all_feats = _prep_from_disk['feature_cols_selected']  # full set (~125)
            # _training_mask (56 features) is already set from Phase 6 — use it directly
            _prod_num_imp   = _prep_from_disk['num_imputer_full']
            _prod_cat_imp   = _prep_from_disk['cat_imputer_full']
            _prod_cat_enc   = _prep_from_disk['cat_encoder_full']
            _prod_scaler    = _prep_from_disk['scaler_full']
            _prod_num_f     = _prep_from_disk['numerical_features']
            _prod_cat_f     = _prep_from_disk['categorical_features']
            logger.info(f"  Loaded rank-1 model + preprocessors ({len(_training_feat_names)} training features)")

            # ── Assign dominant phase from phase_fraction_* columns ───────
            _pf_map = {
                'phase_fraction_climb':   'CLIMB',
                'phase_fraction_cruise':  'CRUISE',
                'phase_fraction_descent': 'DESCENT',
                'phase_fraction_approach':'DESCENT',   # collapse into DESCENT
                'phase_fraction_landing': 'DESCENT',
                'phase_fraction_takeoff': 'CLIMB',
            }
            _pf_avail = [c for c in _pf_map if c in _rank_ev.columns]
            def _dom_phase(df_):
                if not _pf_avail:
                    return pd.Series(['UNKNOWN'] * len(df_), index=df_.index)
                dom = df_[_pf_avail].idxmax(axis=1).map(_pf_map)
                zero = df_[_pf_avail].sum(axis=1) == 0
                dom[zero] = 'UNKNOWN'
                return dom

            # Combined eval frame (Rank + Final)
            _comb_ev  = pd.concat([_rank_ev, _final_ev], ignore_index=True)
            _comb_gt  = np.concatenate([y_ev_r, y_ev_f])
            _comb_ph  = _dom_phase(_comb_ev).values

            # Apply prod preprocessors: full feature matrix → impute/encode → scale → SFS mask
            _Xp = _comb_ev.reindex(columns=_prod_all_feats,
                                   fill_value=np.nan).copy()
            _Xp.replace([np.inf, -np.inf], np.nan, inplace=True)
            _n_f_p = [c for c in _prod_all_feats if c in _prod_num_f]
            _c_f_p = [c for c in _prod_all_feats if c in _prod_cat_f]
            if _n_f_p:
                _Xp[_n_f_p] = _prod_num_imp.transform(_Xp[_n_f_p])
            if _c_f_p:
                _Xp[_c_f_p] = _prod_cat_imp.transform(_Xp[_c_f_p])
                _Xp[_c_f_p] = _prod_cat_enc.transform(_Xp[_c_f_p])
            # scale on all 125 features, then apply the Phase-4 SFS mask → 56
            _Xp_s = _prod_scaler.transform(_Xp.values)
            _Xp_s = _Xp_s[:, _training_mask]
            _prod_preds = np.maximum(np.expm1(_prod_model.predict(_Xp_s)), 0)
            _prod_preds_bl = _prod_preds  # expose for baseline comparison below

            # OpenAP column (optional, present only if featured_data carried it)
            _oap_col = next((c for c in ['openap_fuel_kg', 'openap_pred_kg',
                                          'openap_segment_fuel'] if c in _comb_ev.columns), None)
            _oap_vals = _comb_ev[_oap_col].values if _oap_col else None

            _ph_rows = []
            for _ph in ['CLIMB', 'CRUISE', 'DESCENT']:
                _msk = (_comb_ph == _ph)
                if _msk.sum() < 5:
                    continue
                _yt_ = _comb_gt[_msk]
                _yp_ = _prod_preds[_msk]
                _n_  = int(_msk.sum())
                _mae_  = float(np.mean(np.abs(_yt_ - _yp_)))
                _rmse_ = float(np.sqrt(np.mean((_yt_ - _yp_)**2)))
                _mape_ = float(np.mean(np.abs((_yt_ - _yp_) / (_yt_ + 1e-8))) * 100)
                _r2_   = float(1 - np.sum((_yt_ - _yp_)**2) /
                               np.sum((_yt_ - np.mean(_yt_))**2))
                _oap_rmse = np.nan
                if _oap_vals is not None:
                    _yo_ = _oap_vals[_msk].astype(float)
                    _vld = ~np.isnan(_yo_)
                    if _vld.sum() > 0:
                        _oap_rmse = float(np.sqrt(np.mean(
                            (_yt_[_vld] - _yo_[_vld])**2)))
                _ph_rows.append({
                    'phase': _ph, 'n': _n_,
                    'mean_actual_kg': float(np.mean(_yt_)),
                    'mae': _mae_, 'rmse': _rmse_, 'mape': _mape_, 'r2': _r2_,
                    'openap_rmse': _oap_rmse,
                })

            _ph_df = pd.DataFrame(_ph_rows)
            _ph_out = os.path.join(RESULTS_DIR, 'per_phase_gt_metrics.csv')
            _ph_df.to_csv(_ph_out, index=False)
            logger.info(f"\n[+] Per-phase GT metrics → {_ph_out}")
            logger.info(f"\n{'─'*80}")
            logger.info(f"  {'Phase':<10} {'N':>7}  {'Mean(kg)':>9}"
                        f"  {'MAE':>7}  {'RMSE':>7}  {'MAPE%':>7}  {'R²':>7}  {'OAP RMSE':>9}")
            logger.info(f"{'─'*80}")
            for _, _rw in _ph_df.iterrows():
                _os = f"{_rw['openap_rmse']:>9.1f}" if not np.isnan(_rw['openap_rmse']) else "      N/A"
                logger.info(f"  {_rw['phase']:<10} {_rw['n']:>7,}  "
                            f"{_rw['mean_actual_kg']:>9.1f}  {_rw['mae']:>7.1f}  "
                            f"{_rw['rmse']:>7.1f}  {_rw['mape']:>7.1f}  "
                            f"{_rw['r2']:>7.4f}  {_os}")
            logger.info(f"{'─'*80}")
            # -- Val per-phase metrics (rank-1 model, 20% hold-out) --
            if _best_val_pred_kg is not None:
                _pf_avail_v = [c for c in _pf_map if c in X_val.columns]
                if _pf_avail_v:
                    _val_ph_dom = X_val[_pf_avail_v].idxmax(axis=1).map(_pf_map)
                    _zero_v = X_val[_pf_avail_v].sum(axis=1) == 0
                    _val_ph_dom[_zero_v] = 'UNKNOWN'
                else:
                    _val_ph_dom = pd.Series(
                        ['UNKNOWN'] * len(X_val), index=X_val.index)
                _val_ph_arr = _val_ph_dom.values
                for _ph_r in _ph_rows:
                    _ph = _ph_r['phase']
                    _msk_v = (_val_ph_arr == _ph)
                    if _msk_v.sum() < 5:
                        _ph_r.update({'val_n': 0, 'val_mae': np.nan,
                                      'val_rmse': np.nan, 'val_mape': np.nan,
                                      'val_r2': np.nan})
                        continue
                    _yt_v = y_val[_msk_v]
                    _yp_v = _best_val_pred_kg[_msk_v]
                    _ph_r['val_n']    = int(_msk_v.sum())
                    _ph_r['val_mae']  = float(np.mean(np.abs(_yt_v - _yp_v)))
                    _ph_r['val_rmse'] = float(np.sqrt(np.mean((_yt_v - _yp_v)**2)))
                    _ph_r['val_mape'] = float(
                        np.mean(np.abs((_yt_v - _yp_v) / (_yt_v + 1e-8))) * 100)
                    _ph_r['val_r2']   = float(
                        1 - np.sum((_yt_v - _yp_v)**2) /
                        np.sum((_yt_v - np.mean(_yt_v))**2))
                _ph_df = pd.DataFrame(_ph_rows)
                _ph_df.to_csv(_ph_out, index=False)  # re-save with val columns
        except Exception as _phe:
            logger.warning(f"[!] Per-phase GT metrics failed: {_phe}")
            import traceback; logger.warning(traceback.format_exc())

        # ── Precompute val-split helpers for dual Val / GT evaluation ─────
        _at_val = (df_features_augmented.loc[X_val.index, 'aircraft_type']
                   .reset_index(drop=True)
                   if 'aircraft_type' in df_features_augmented.columns
                   else pd.Series(['A320'] * len(X_val), name='aircraft_type'))
        _X_aug_80 = X_train.reindex(columns=_sfs_feats, fill_value=np.nan).copy()
        _y_aug_80 = y_train.copy()
        _real_in_train = (X_train.index < _n_real)
        if _real_in_train.any():
            _X_real_80 = X_train.loc[X_train.index[_real_in_train],
                                     _sfs_feats].copy()
            _y_real_80 = y_train[np.asarray(_real_in_train)].copy()
        else:
            _X_real_80, _y_real_80 = _X_aug_80, _y_aug_80
        logger.info(f"  Val helpers: {len(_X_aug_80):,} aug-80% rows, "
                    f"{len(_X_real_80):,} real-80% rows, "
                    f"{len(_at_val):,} val aircraft types")

        # ── Variant training DataFrames ───────────────────────────────────
        # 1 & 2: Base and No-Synthetic
        _X_aug  = df_features_augmented[_sfs_feats].copy()
        _y_aug  = y_full.copy()
        _X_real = df_features[_sfs_feats].copy()
        _y_real = df_features[target_col].values.astype(np.float32)

        # +C1: METAR features — dep_*/arr_* columns from full featured parquet
        _metar_cols = [c for c in featured_data.columns
                       if (c.startswith('dep_') or c.startswith('arr_'))
                       and c not in _sfs_feats]
        _c1_ok = len(_metar_cols) > 0
        if _c1_ok:
            _midx = 'interval_idx' if 'interval_idx' in featured_data.columns else 'idx'
            _ms = featured_data[['flight_id', _midx] + _metar_cols].rename(
                columns={_midx: 'interval_idx'})
            # Build real rows with METAR from df_raw (has flight_id + interval_idx)
            _real_idx = df_features.index
            _dr_real = df_raw.loc[_real_idx, ['flight_id', 'interval_idx', target_col] +
                                              [c for c in _sfs_feats if c in df_raw.columns]].copy()
            _dr_real = _dr_real.merge(_ms, on=['flight_id', 'interval_idx'], how='left')
            _fc_m = [c for c in _sfs_feats if c in _dr_real.columns] + \
                    [c for c in _metar_cols if c in _dr_real.columns]
            _X_real_m = _dr_real[_fc_m].copy()
            _y_real_m = _dr_real[target_col].values.astype(np.float32)
            # Synthetic rows: METAR columns set to NaN (imputed to train mean)
            _syn_m = df_features_augmented.iloc[_n_real:].reindex(columns=_fc_m, fill_value=np.nan)
            _X_aug_m = pd.concat([_X_real_m, _syn_m], ignore_index=True)
            _y_aug_m = np.concatenate([_y_real_m, _y_aug[_n_real:]])
            # Eval frames with METAR columns
            _mr = featured_data_rank[['flight_id', 'interval_idx'] +
                                     [c for c in _metar_cols if c in featured_data_rank.columns]]
            _mf = featured_data_final[['flight_id', 'interval_idx'] +
                                      [c for c in _metar_cols if c in featured_data_final.columns]]
            _rank_ev_m  = _rank_ev.merge(_mr,  on=['flight_id', 'interval_idx'], how='left')
            _final_ev_m = _final_ev.merge(_mf, on=['flight_id', 'interval_idx'], how='left')
            logger.info(f"  C1: {len(_metar_cols)} METAR columns (dep_*/arr_*) added")
        else:
            logger.warning("  C1: No dep_*/arr_* columns in featured_data — skipping")

        # –C3: replace dynamic starting_mass_kg with static MTOW estimate
        _c3_ok = ('starting_mass_kg' in _sfs_feats
                  and 'estimated_takeoff_mass' in df_features_augmented.columns)
        if _c3_ok:
            _X_c3 = _X_aug.copy()
            _X_c3['starting_mass_kg'] = df_features_augmented['estimated_takeoff_mass'].values
            _X_c3_80 = _X_aug_80.copy()
            _X_c3_80['starting_mass_kg'] = df_features_augmented.loc[
                X_train.index, 'estimated_takeoff_mass'].values
            _rank_ev_c3  = _rank_ev.copy()
            _final_ev_c3 = _final_ev.copy()
            if 'estimated_takeoff_mass' in _rank_ev_c3.columns:
                _rank_ev_c3['starting_mass_kg']  = _rank_ev_c3['estimated_takeoff_mass']
            if 'estimated_takeoff_mass' in _final_ev_c3.columns:
                _final_ev_c3['starting_mass_kg'] = _final_ev_c3['estimated_takeoff_mass']
        else:
            logger.warning("  C3: starting_mass_kg or estimated_takeoff_mass not available")

        # +C4: elapsed flight time (not selected by SFS)
        _c4_feat = 'interval_elapsed_from_flight_start'
        _c4_ok = (_c4_feat in df_features_augmented.columns
                  and _c4_feat not in _sfs_feats)
        if _c4_ok:
            _fc_c4 = _sfs_feats + [_c4_feat]
            _X_c4 = df_features_augmented[_fc_c4].copy()
            _X_c4_80 = X_train.reindex(columns=_fc_c4, fill_value=np.nan).copy()
        elif _c4_feat in _sfs_feats:
            logger.info(f"  C4: {_c4_feat} already in SFS — additive test not applicable")
            _c4_ok = False
        else:
            logger.warning(f"  C4: {_c4_feat} not found in training data — skipping")
            _c4_ok = False

        # ── Run all conditions ────────────────────────────────────────────
        _abl_all = []
        _sep60 = '─' * 60

        # ── Augmentation impact: no-aug vs. with-aug ────────────────────
        logger.info(f"\n{'═'*60}")
        logger.info("  WB/NB AUGMENTATION IMPACT (Combined GT — Rank + Final)")
        logger.info(f"{'═'*60}")

        logger.info(f"\n  [Aug A] No WB Augmentation (real data only)")
        _res_noaug = _abl_run(
            'No WB Augmentation', _X_real, _sfs_feats, _y_real,
            _rank_ev, y_ev_r, at_ev_r,
            _final_ev, y_ev_f, at_ev_f,
            _return_preds=True,
            X_tr_80_df=_X_real_80, y_tr_80=_y_real_80)
        _rows_noaug, _preds_noaug, _ytrue_noaug, _at_noaug = _res_noaug

        logger.info(f"\n  [Aug B] With WB Augmentation (SFS + synthetic)")
        _res_aug = _abl_run(
            'With WB Augmentation', _X_aug, _sfs_feats, _y_aug,
            _rank_ev, y_ev_r, at_ev_r,
            _final_ev, y_ev_f, at_ev_f,
            _return_preds=True,
            X_tr_80_df=_X_aug_80, y_tr_80=_y_aug_80)
        _rows_aug, _preds_aug, _ytrue_aug, _at_aug = _res_aug

        _aug_cmp_df = pd.DataFrame(_rows_noaug + _rows_aug)
        _aug_cmp_out = os.path.join(RESULTS_DIR, 'augmentation_wb_nb_gt_results.csv')
        _aug_cmp_df.to_csv(_aug_cmp_out, index=False)
        logger.info(f"\n[+] Augmentation WB/NB GT results → {_aug_cmp_out}")

        # ── Per-aircraft GT metrics (with augmentation) ─────────────────
        _at_aug_arr = _at_aug.values if hasattr(_at_aug, 'values') else np.array(_at_aug)
        _ac_dict = {}
        for _ac, _yt, _yp in zip(_at_aug_arr, _ytrue_aug, _preds_aug):
            _ac_dict.setdefault(_ac, {'yt': [], 'yp': []})
            _ac_dict[_ac]['yt'].append(_yt)
            _ac_dict[_ac]['yp'].append(_yp)

        _ac_rows = []
        for _ac, _vals in sorted(_ac_dict.items(), key=lambda x: -len(x[1]['yt'])):
            _yt_ = np.array(_vals['yt'])
            _yp_ = np.array(_vals['yp'])
            _n_  = len(_yt_)
            _mae_  = float(np.mean(np.abs(_yt_ - _yp_)))
            _rmse_ = float(np.sqrt(np.mean((_yt_ - _yp_)**2)))
            _mape_ = float(np.mean(np.abs((_yt_ - _yp_) / (_yt_ + 1e-8))) * 100)
            _r2_   = float(1 - np.sum((_yt_ - _yp_)**2) /
                           np.sum((_yt_ - np.mean(_yt_))**2)) if _n_ > 1 else np.nan
            _ac_rows.append({
                'aircraft_type': _ac, 'n': _n_,
                'mean_actual_kg':    float(np.mean(_yt_)),
                'mean_predicted_kg': float(np.mean(_yp_)),
                'mae': _mae_, 'rmse': _rmse_, 'mape': _mape_, 'r2': _r2_,
                'is_widebody': _ac in WIDEBODY_AIRCRAFT,
            })

        _ac_df = pd.DataFrame(_ac_rows)
        _ac_out = os.path.join(RESULTS_DIR, 'per_aircraft_gt_metrics.csv')
        _ac_df.to_csv(_ac_out, index=False)
        logger.info(f"[+] Per-aircraft GT metrics → {_ac_out}")
        logger.info(f"\n{'─'*85}")
        logger.info(f"  {'Aircraft':<8} {'Cat':<3} {'N':>6}  {'mean_act':>9}  {'mean_pred':>9}"
                    f"  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}")
        logger.info(f"{'─'*85}")
        for _, _r in _ac_df.iterrows():
            _wb = 'WB' if _r['is_widebody'] else 'NB'
            logger.info(f"  {_r['aircraft_type']:<8} {_wb:<3} {_r['n']:>6,}  "
                        f"{_r['mean_actual_kg']:>9.1f}  {_r['mean_predicted_kg']:>9.1f}  "
                        f"{_r['mae']:>7.1f}  {_r['rmse']:>7.1f}  {_r['r2']:>7.4f}")
        logger.info(f"{'─'*85}")

        # -- Val per-aircraft metrics (rank-1 model, 20% hold-out) --
        if _best_val_pred_kg is not None:
            _at_val_arr = _at_val.values if hasattr(_at_val, 'values') else np.array(_at_val)
            _ac_val_dict = {}
            for _ac_v, _yt_v, _yp_v in zip(_at_val_arr, y_val, _best_val_pred_kg):
                _ac_val_dict.setdefault(_ac_v, {'yt': [], 'yp': []})
                _ac_val_dict[_ac_v]['yt'].append(_yt_v)
                _ac_val_dict[_ac_v]['yp'].append(_yp_v)
            for _ac_r in _ac_rows:
                _ac = _ac_r['aircraft_type']
                _vals_v = _ac_val_dict.get(_ac)
                if _vals_v is None or len(_vals_v['yt']) < 5:
                    _ac_r.update({'val_n': 0, 'val_mae': np.nan,
                                  'val_rmse': np.nan, 'val_mape': np.nan,
                                  'val_r2': np.nan})
                    continue
                _yt_a = np.array(_vals_v['yt'])
                _yp_a = np.array(_vals_v['yp'])
                _ac_r['val_n']    = len(_yt_a)
                _ac_r['val_mae']  = float(np.mean(np.abs(_yt_a - _yp_a)))
                _ac_r['val_rmse'] = float(np.sqrt(np.mean((_yt_a - _yp_a)**2)))
                _ac_r['val_mape'] = float(
                    np.mean(np.abs((_yt_a - _yp_a) / (_yt_a + 1e-8))) * 100)
                _ac_r['val_r2']   = float(
                    1 - np.sum((_yt_a - _yp_a)**2) /
                    np.sum((_yt_a - np.mean(_yt_a))**2))
            _ac_df = pd.DataFrame(_ac_rows)  # re-create with val columns
            _ac_df.to_csv(_ac_out, index=False)

        # ── Mean actual vs predicted per aircraft type ───────────────────
        try:
            _ac_plot = _ac_df[_ac_df['n'] >= 10].sort_values('mean_actual_kg', ascending=False)
            _fig, _ax = plt.subplots(figsize=(14, 6))
            _x  = np.arange(len(_ac_plot))
            _w  = 0.35
            _ax.bar(_x - _w/2, _ac_plot['mean_actual_kg'],    _w,
                    label='Actual',    color='steelblue',  alpha=0.85)
            _ax.bar(_x + _w/2, _ac_plot['mean_predicted_kg'], _w,
                    label='Predicted', color='darkorange', alpha=0.85)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_ac_plot['aircraft_type'], rotation=45, ha='right', fontsize=10)
            _ax.set_ylabel('Mean Fuel Consumption per Segment (kg)', fontsize=12)
            _ax.set_title(
                'Mean Actual vs Predicted Fuel per Aircraft Type\n'
                '(Combined GT — Rank + Final datasets, With WB Augmentation)',
                fontsize=13)
            _ax.legend(fontsize=11)
            _ax.yaxis.grid(True, alpha=0.4)
            _ax.set_axisbelow(True)
            for _xi, (_, _row) in zip(_x, _ac_plot.iterrows()):
                _ymax = max(_row['mean_actual_kg'], _row['mean_predicted_kg'])
                _ax.text(_xi, _ymax * 1.015, f"MAE={_row['mae']:.0f}",
                         ha='center', va='bottom', fontsize=7.5, color='dimgray')
            _fig.tight_layout()
            _ppath = os.path.join(RESULTS_DIR, 'per_aircraft_actual_vs_predicted.png')
            _fig.savefig(_ppath, dpi=150, bbox_inches='tight')
            plt.close(_fig)
            logger.info(f"[+] Per-aircraft plot → {_ppath}")
        except Exception as _pe:
            logger.warning(f"[!] Per-aircraft plot failed: {_pe}")

        # ── Per-haul metrics (GT: combined rank+final) ────────────────────
        try:
            _haul_bins   = [0, 3 * 3600, 6 * 3600, float('inf')]
            _haul_labels = ['Short', 'Medium', 'Long']

            # GT haul bins from _comb_ev
            _gt_dur = _comb_ev.get('flight_duration_seconds',
                                   pd.Series(np.nan, index=_comb_ev.index))
            _gt_haul = pd.cut(_gt_dur, bins=_haul_bins,
                              labels=_haul_labels, right=True)

            _haul_rows = []
            for _hl in _haul_labels:
                _msk_h = (_gt_haul == _hl).values
                if _msk_h.sum() < 5:
                    continue
                _yt_h = _comb_gt[_msk_h]
                _yp_h = _prod_preds[_msk_h]
                _haul_rows.append({
                    'haul': _hl,
                    'n':    int(_msk_h.sum()),
                    'mae':  float(np.mean(np.abs(_yt_h - _yp_h))),
                    'rmse': float(np.sqrt(np.mean((_yt_h - _yp_h)**2))),
                    'mape': float(np.mean(np.abs((_yt_h - _yp_h) / (_yt_h + 1e-8))) * 100),
                    'r2':   float(1 - np.sum((_yt_h - _yp_h)**2) /
                                  np.sum((_yt_h - np.mean(_yt_h))**2)),
                })

            # Val haul bins — use the _flight_dur_s carry-along column
            # (set on df_features before the augment concat, so it's in df_features_augmented)
            _val_dur_s = df_features_augmented.loc[X_val.index, '_flight_dur_s'].values \
                if '_flight_dur_s' in df_features_augmented.columns \
                else np.full(len(X_val), np.nan)
            _val_haul = pd.cut(_val_dur_s, bins=_haul_bins,
                               labels=_haul_labels, right=True)

            if _best_val_pred_kg is not None:
                for _hr in _haul_rows:
                    _hl = _hr['haul']
                    _msk_v = (_val_haul == _hl)
                    if _msk_v.sum() < 5:
                        _hr.update({'val_n': 0, 'val_mae': np.nan,
                                    'val_rmse': np.nan, 'val_r2': np.nan})
                        continue
                    _yt_v = y_val[np.asarray(_msk_v)]
                    _yp_v = _best_val_pred_kg[np.asarray(_msk_v)]
                    _hr['val_n']    = int(_msk_v.sum())
                    _hr['val_mae']  = float(np.mean(np.abs(_yt_v - _yp_v)))
                    _hr['val_rmse'] = float(np.sqrt(np.mean((_yt_v - _yp_v)**2)))
                    _hr['val_r2']   = float(1 - np.sum((_yt_v - _yp_v)**2) /
                                            np.sum((_yt_v - np.mean(_yt_v))**2))

            _haul_df  = pd.DataFrame(_haul_rows)
            _haul_out = os.path.join(RESULTS_DIR, 'per_haul_gt_metrics.csv')
            _haul_df.to_csv(_haul_out, index=False)
            logger.info(f"[+] Per-haul GT metrics → {_haul_out}")
            logger.info(f"\n{'─'*80}")
            logger.info(f"  {'Haul':<8} {'GT N':>7}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}"
                        f"  {'Val N':>7}  {'Val MAE':>8}  {'Val RMSE':>9}  {'Val R²':>7}")
            logger.info(f"{'─'*80}")
            for _, _rh in _haul_df.iterrows():
                logger.info(
                    f"  {_rh['haul']:<8} {int(_rh['n']):>7,}  "
                    f"{_rh['mae']:>7.1f}  {_rh['rmse']:>7.1f}  {_rh['r2']:>7.4f}  "
                    f"{int(_rh.get('val_n',0)):>7,}  {_rh.get('val_mae',float('nan')):>8.1f}  "
                    f"{_rh.get('val_rmse',float('nan')):>9.1f}  "
                    f"{_rh.get('val_r2',float('nan')):>7.4f}")
            logger.info(f"{'─'*80}")
        except Exception as _hle:
            logger.warning(f"[!] Per-haul GT metrics failed: {_hle}")
            import traceback; logger.warning(traceback.format_exc())

        # ── Baseline comparison: Val (20%) + Combined GT ─────────────────
        logger.info(f"\n{'═'*70}")
        logger.info("  BASELINE COMPARISON (Val 20% hold-out  +  Combined GT)")
        logger.info(f"{'═'*70}")
        try:
            from sklearn.linear_model import Ridge as _Ridge
            from sklearn.ensemble import RandomForestRegressor as _RandomForest
            import lightgbm as _lgb

            _bl_comb_ev = pd.concat([_rank_ev, _final_ev], ignore_index=True)
            _bl_gt      = np.concatenate([y_ev_r, y_ev_f])

            def _bl_metrics_row(model_label, split_label, y_true, y_pred):
                _mae_  = float(np.mean(np.abs(y_true - y_pred)))
                _rmse_ = float(np.sqrt(np.mean((y_true - y_pred)**2)))
                _mape_ = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
                _r2_   = float(1 - np.sum((y_true - y_pred)**2) /
                               np.sum((y_true - np.mean(y_true))**2))
                return {'model': model_label, 'split': split_label, 'n': len(y_true),
                        'mae': _mae_, 'rmse': _rmse_, 'mape': _mape_, 'r2': _r2_}

            def _bl_train_pred(Xtr_df, feat_list, y_tr_raw, Xev_df, mdl, log_target=True):
                """Fit mdl on Xtr_df[feat_list], predict on Xev_df; returns kg-scale preds."""
                _f = [c for c in feat_list
                      if c in Xtr_df.columns and not Xtr_df[c].isna().all()]
                _Xt = Xtr_df[_f].copy().replace([np.inf, -np.inf], np.nan)
                _Xe = Xev_df.reindex(columns=_f, fill_value=np.nan).copy().replace([np.inf, -np.inf], np.nan)
                _nf = [c for c in _f if pd.api.types.is_numeric_dtype(_Xt[c])]
                _cf = [c for c in _f if c not in _nf]
                if _nf:
                    _ni = SimpleImputer(strategy='mean')
                    _Xt[_nf] = _ni.fit_transform(_Xt[_nf])
                    _Xe[_nf] = _ni.transform(_Xe[_nf])
                else:
                    _ni = None
                if _cf:
                    _ci = SimpleImputer(strategy='most_frequent')
                    _Xt[_cf] = _ci.fit_transform(_Xt[_cf])
                    _Xe[_cf] = _ci.transform(_Xe[_cf])
                    _ce = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    _Xt[_cf] = _ce.fit_transform(_Xt[_cf])
                    _Xe[_cf] = _ce.transform(_Xe[_cf])
                _sc = StandardScaler()
                _Xt_s = _sc.fit_transform(_Xt)
                _Xe_s = _sc.transform(_Xe)
                _yt = np.log1p(y_tr_raw) if log_target else y_tr_raw
                _t0 = time.time()
                mdl.fit(_Xt_s, _yt)
                logger.info(f"      trained in {time.time()-_t0:.0f}s  ({len(_f)} feats, {len(_Xt):,} rows)")
                _raw = mdl.predict(_Xe_s)
                return np.maximum(np.expm1(_raw), 0) if log_target else np.maximum(_raw, 0)

            # Val: X_train (80% aug) → X_val (20% aug); GT: _X_aug (100%) → combined hidden GT
            _X_train_sfs = X_train[_sfs_feats] if all(c in X_train.columns for c in _sfs_feats) \
                           else X_train.reindex(columns=_sfs_feats, fill_value=np.nan)
            _X_val_sfs   = X_val[_sfs_feats]   if all(c in X_val.columns   for c in _sfs_feats) \
                           else X_val.reindex(columns=_sfs_feats, fill_value=np.nan)
            _X_train_all = X_train[feature_cols_selected] if all(c in X_train.columns for c in feature_cols_selected) \
                           else X_train.reindex(columns=feature_cols_selected, fill_value=np.nan)
            _X_val_all   = X_val[feature_cols_selected]   if all(c in X_val.columns   for c in feature_cols_selected) \
                           else X_val.reindex(columns=feature_cols_selected, fill_value=np.nan)

            _bl_rows = []

            # ── 1. OpenAP FuelFlow (physics, precomputed) ──────────────────
            # Val: look up openap_fuel_kg from df_features_augmented by index
            _oap_val_col = 'openap_fuel_kg' if 'openap_fuel_kg' in df_features_augmented.columns else None
            if _oap_val_col:
                _oap_val_vals = df_features_augmented.loc[X_val.index, _oap_val_col].values.astype(float)
                _vld_v = ~np.isnan(_oap_val_vals)
                if _vld_v.sum() > 0:
                    _bl_rows.append(_bl_metrics_row('OpenAP FuelFlow (physics)', 'Val',
                                                    y_val[_vld_v], _oap_val_vals[_vld_v]))
            # GT: openap from merged eval frame
            _oap_bl_col = next((c for c in ['openap_fuel_kg', 'openap_pred_kg', 'openap_segment_fuel']
                                if c in _bl_comb_ev.columns), None)
            if _oap_bl_col:
                _oap_bl = _bl_comb_ev[_oap_bl_col].values.astype(float)
                _vld_g = ~np.isnan(_oap_bl)
                if _vld_g.sum() > 0:
                    _bl_rows.append(_bl_metrics_row('OpenAP FuelFlow (physics)', 'GT',
                                                    _bl_gt[_vld_g], _oap_bl[_vld_g]))
            if not _oap_val_col and not _oap_bl_col:
                logger.info("  OpenAP column not found — skipping")

            # ── 2. Ridge Regression ────────────────────────────────────────
            logger.info("  Training Ridge Regression (Val) …")
            _bl_rows.append(_bl_metrics_row('Ridge Regression', 'Val', y_val,
                _bl_train_pred(_X_train_sfs, _sfs_feats, y_train, _X_val_sfs, _Ridge(alpha=1.0))))
            logger.info("  Training Ridge Regression (GT) …")
            _bl_rows.append(_bl_metrics_row('Ridge Regression', 'GT', _bl_gt,
                _bl_train_pred(_X_aug, _sfs_feats, _y_aug, _bl_comb_ev, _Ridge(alpha=1.0))))

            # ── 3. Random Forest ───────────────────────────────────────────
            logger.info("  Training Random Forest (Val) …")
            _bl_rows.append(_bl_metrics_row('Random Forest', 'Val', y_val,
                _bl_train_pred(_X_train_sfs, _sfs_feats, y_train, _X_val_sfs,
                               _RandomForest(n_estimators=300, max_depth=20,
                                             min_samples_leaf=2, n_jobs=-1, random_state=42))))
            logger.info("  Training Random Forest (GT) …")
            _bl_rows.append(_bl_metrics_row('Random Forest', 'GT', _bl_gt,
                _bl_train_pred(_X_aug, _sfs_feats, _y_aug, _bl_comb_ev,
                               _RandomForest(n_estimators=300, max_depth=20,
                                             min_samples_leaf=2, n_jobs=-1, random_state=42))))

            # ── 4. LightGBM ────────────────────────────────────────────────
            logger.info("  Training LightGBM (Val) …")
            _bl_rows.append(_bl_metrics_row('LightGBM', 'Val', y_val,
                _bl_train_pred(_X_train_sfs, _sfs_feats, y_train, _X_val_sfs,
                               _lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                                  num_leaves=63, n_jobs=-1, random_state=42,
                                                  verbosity=-1))))
            logger.info("  Training LightGBM (GT) …")
            _bl_rows.append(_bl_metrics_row('LightGBM', 'GT', _bl_gt,
                _bl_train_pred(_X_aug, _sfs_feats, _y_aug, _bl_comb_ev,
                               _lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                                  num_leaves=63, n_jobs=-1, random_state=42,
                                                  verbosity=-1))))

            # ── 5. XGBoost reference (legacy params, no SFS) ───────────────
            logger.info("  Training XGBoost reference (Val) …")
            _bl_rows.append(_bl_metrics_row('XGBoost (reference, no SFS)', 'Val', y_val,
                _bl_train_pred(_X_train_all, feature_cols_selected, y_train, _X_val_all,
                               XGBRegressor(random_state=42, objective='reg:squarederror',
                                            tree_method='hist', device=_DEVICE,
                                            n_jobs=-1, verbosity=0, **dict(legacy_params)))))
            logger.info("  Training XGBoost reference (GT) …")
            _Xtr_all_bl = df_features_augmented[feature_cols_selected].copy()
            _bl_rows.append(_bl_metrics_row('XGBoost (reference, no SFS)', 'GT', _bl_gt,
                _bl_train_pred(_Xtr_all_bl, feature_cols_selected, _y_aug, _bl_comb_ev,
                               XGBRegressor(random_state=42, objective='reg:squarederror',
                                            tree_method='hist', device=_DEVICE,
                                            n_jobs=-1, verbosity=0, **dict(legacy_params)))))

            # ── 6. XGBoost proposed ────────────────────────────────────────
            # Val: reuse rank-1 val predictions from Phase 5; fallback to fresh retrain
            _xgb_val_preds_kg = _best_val_pred_kg \
                if (_best_val_pred_kg is not None and len(_best_val_pred_kg) == len(y_val)) \
                else None
            if _xgb_val_preds_kg is None:
                logger.info("  Training XGBoost proposed (Val, fresh) …")
                _xgb_val_preds_kg = _bl_train_pred(
                    _X_train_sfs, _sfs_feats, y_train, _X_val_sfs,
                    XGBRegressor(random_state=42, objective='reg:squarederror',
                                 tree_method='hist', device=_DEVICE,
                                 n_jobs=-1, verbosity=0, **dict(legacy_params)))
            _bl_rows.append(_bl_metrics_row('XGBoost (proposed)', 'Val', y_val, _xgb_val_preds_kg))
            # GT: production rank-1 predictions, fallback to with-aug ablation preds
            _xgb_gt_preds = _prod_preds_bl if _prod_preds_bl is not None else _preds_aug
            _bl_rows.append(_bl_metrics_row('XGBoost (proposed)', 'GT', _bl_gt, _xgb_gt_preds))

            _bl_df = pd.DataFrame(_bl_rows)
            _bl_out = os.path.join(RESULTS_DIR, 'baseline_comparison_gt.csv')
            _bl_df.to_csv(_bl_out, index=False)
            logger.info(f"\n[+] Baseline comparison (Val + GT) → {_bl_out}")
            logger.info(f"\n{'─'*90}")
            logger.info(f"  {'Model':<35} {'Split':<6} {'N':>7}  {'MAE':>7}  {'RMSE':>7}  {'MAPE%':>7}  {'R²':>7}")
            logger.info(f"{'─'*90}")
            for _, _rw in _bl_df.iterrows():
                logger.info(f"  {_rw['model']:<35} {_rw['split']:<6} {_rw['n']:>7,}  "
                            f"{_rw['mae']:>7.1f}  {_rw['rmse']:>7.1f}  "
                            f"{_rw['mape']:>7.2f}  {_rw['r2']:>7.4f}")
            logger.info(f"{'─'*90}")
        except Exception as _ble:
            logger.warning(f"[!] Baseline comparison failed: {_ble}")
            import traceback; logger.warning(traceback.format_exc())

        # ── Contribution ablation conditions (Combined GT) ────────────────
        logger.info(f"\n{'═'*60}")
        logger.info("  CONTRIBUTION ABLATION (Combined GT — Rank + Final)")
        logger.info(f"{'═'*60}")

        logger.info(f"\n{_sep60}\n  [1/5] No Synthetic (SFS only)")
        _abl_all += _abl_run('No Synthetic (SFS only)',
                             _X_real, _sfs_feats, _y_real,
                             _rank_ev, y_ev_r, at_ev_r,
                             _final_ev, y_ev_f, at_ev_f,
                             X_tr_80_df=_X_real_80, y_tr_80=_y_real_80)

        if _c1_ok:
            logger.info(f"\n{_sep60}\n  [2/5] SFS + Synthetic + METAR (+C1)")
            _abl_all += _abl_run('SFS + Synthetic + METAR (+C1)',
                                 _X_aug_m, _fc_m, _y_aug_m,
                                 _rank_ev_m, y_ev_r, at_ev_r,
                                 _final_ev_m, y_ev_f, at_ev_f,
                                 X_tr_80_df=_X_aug_80, y_tr_80=_y_aug_80)

        if _LF_COLS:
            logger.info(f"\n{_sep60}\n  [3/5] SFS + Synthetic - Load Factor (-C2)")
            _fc_no_lf = [c for c in _sfs_feats if c not in _LF_COLS]
            _abl_all += _abl_run('SFS + Synthetic - Load Factor (-C2)',
                                 _X_aug[_fc_no_lf], _fc_no_lf, _y_aug,
                                 _rank_ev, y_ev_r, at_ev_r,
                                 _final_ev, y_ev_f, at_ev_f,
                                 X_tr_80_df=X_train.reindex(
                                     columns=_fc_no_lf, fill_value=np.nan),
                                 y_tr_80=y_train)

        if _c3_ok:
            logger.info(f"\n{_sep60}\n  [4/5] SFS + Synthetic + Static Est. TOW (-C3)")
            _abl_all += _abl_run('SFS + Synthetic + Static Est. TOW (-C3)',
                                 _X_c3, _sfs_feats, _y_aug,
                                 _rank_ev_c3, y_ev_r, at_ev_r,
                                 _final_ev_c3, y_ev_f, at_ev_f,
                                 X_tr_80_df=_X_c3_80, y_tr_80=_y_aug_80)

        if _c4_ok:
            logger.info(f"\n{_sep60}\n  [5/5] SFS + Synthetic + Corrected Elapsed (+C4)")
            _abl_all += _abl_run('SFS + Synthetic + Corrected Elapsed (+C4)',
                                 _X_c4, _fc_c4, _y_aug,
                                 _rank_ev, y_ev_r, at_ev_r,
                                 _final_ev, y_ev_f, at_ev_f,
                                 X_tr_80_df=_X_c4_80, y_tr_80=_y_aug_80)

        # ── Summary table ─────────────────────────────────────────────────
        _abl_df = pd.DataFrame(_abl_all)
        logger.info("\n" + "=" * 100)
        logger.info(f"{'Condition':<45} {'Dataset':<10} {'Split':<14}"
                    f" {'N':>7} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        logger.info("-" * 100)
        for _, _r in _abl_df.iterrows():
            logger.info(f"{_r['condition']:<45} {_r['dataset']:<10} {_r['split']:<14}"
                        f" {_r['n']:>7,} {_r['mae']:>8.1f} {_r['rmse']:>8.1f} {_r['r2']:>8.4f}")
        logger.info("=" * 100)

        # Delta-MAE vs No Synthetic baseline (Combined Overall)
        _brow = _abl_df[(_abl_df['condition'] == 'No Synthetic (SFS only)') &
                        (_abl_df['dataset'] == 'Combined') &
                        (_abl_df['split'] == 'Overall')]
        if not _brow.empty:
            _bmae = _brow.iloc[0]['mae']
            logger.info(f"\nDelta-MAE vs. No Synthetic [Combined] (Overall):")
            for _, _r in _abl_df[(_abl_df['dataset'] == 'Combined') &
                                  (_abl_df['split'] == 'Overall')].iterrows():
                if _r['condition'] != 'No Synthetic (SFS only)':
                    _d = _r['mae'] - _bmae
                    logger.info(f"  {_r['condition']:<45}  dMAE={_d:+.1f} ({_d/_bmae*100:+.2f}%)")

        _abl_out = os.path.join(RESULTS_DIR, 'ablation_contributions_results.csv')
        _abl_df.to_csv(_abl_out, index=False)
        logger.info(f"\n[+] Ablation results saved to: {_abl_out}")

    # PHASE 9: Performance Summary
    logger.info("\n" + "="*70)
    logger.info("PHASE 9: FINAL SUMMARY")
    logger.info("="*70)

    summary_df = pd.DataFrame(submission_files)
    summary_path = os.path.join(RESULTS_DIR, 'final_top5_models_synthetic_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    logger.info("\n" + "="*90)
    logger.info("TOP 5 MODELS SUMMARY")
    logger.info("="*90)
    logger.info("\n| Rank | Val RMSE | Train RMSE | GT Rank  | GT Final | GT Combined | Test Mean | Submission File |")
    logger.info("|------|----------|------------|----------|----------|-------------|-----------|-----------------|")

    for _, row in summary_df.iterrows():
        filename = os.path.basename(row['parquet_file'])
        def _fmt(v):
            return f"{v:8.4f}" if not (isinstance(v, float) and np.isnan(v)) else "     N/A"
        logger.info(
            f"|  {row['rank']:2d}  | {_fmt(row['val_rmse'])} | {_fmt(row['train_rmse_100pct'])} | "
            f"{_fmt(row['gt_rank_rmse'])} | {_fmt(row['gt_final_rmse'])} | "
            f"  {_fmt(row['gt_combined_rmse'])} | {row['test_mean']:9.2f} | {filename[:30]}... |"
        )

    logger.info("="*90)
    
    # PHASE 10: Visualization & Reporting
    logger.info("\n" + "="*70)
    logger.info("PHASE 10: GENERATING PAPER GRAPHICS AND TABLES")
    logger.info("="*70)
    try:
        import generate_paper_plots
        # Use the Rank 1 model directory for diagnostic plots
        best_model_dir = os.path.join(RESULTS_DIR, 'final_xgb_model_rank1')
        if os.path.exists(best_model_dir):
            generate_paper_plots.run_all(best_model_dir)
            logger.info("[+] Paper graphics and LaTeX tables generated successfully in paper_plots/")
        else:
            # Fallback to automatic selection if specific folder doesn't exist
            generate_paper_plots.run_all()
    except Exception as e:
        logger.error(f"[-] Failed to generate paper graphics: {e}")


def run(gpu_id=0, force_sfs=False, force_synthetic=False, opt_mode='legacy'):
    force_sfs = force_sfs or '--force-sfs' in sys.argv
    force_synthetic = force_synthetic or '--force-synthetic' in sys.argv
    
    # Parse CLI optimization flags
    if '--grid' in sys.argv:
        opt_mode = 'grid'
    elif '--optuna' in sys.argv:
        opt_mode = 'optuna'
        
    main(gpu_id=gpu_id, force_sfs=force_sfs, force_synthetic=force_synthetic, opt_mode=opt_mode)

if __name__ == "__main__":
    run()
