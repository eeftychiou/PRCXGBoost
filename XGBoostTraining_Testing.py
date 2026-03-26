import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU-only to prevent CUDA context corruption
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
TEST_CSV_PATH = config.AUGMENTED_RANK_CSV
FUEL_RANK_PATH = config.FUEL_RANK
FLIGHTLIST_RANK_PATH = config.FLIGHTLIST_RANK
RESULTS_DIR = config.MODELS_DIR # Consistent model directory

FEATURED_DATA_TRAIN = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
FEATURED_DATA_TEST = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_rank.parquet')

SELECTED_FEATURES_PATH = config.SELECTED_FEATURES_PATH
SYNTHETIC_PATH = config.SYNTHETIC_WIDEBODY_PATH
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Ablation study constants (C1–C4) ────────────────────
FLIGHTLIST_COR_PATH = os.path.join(config.PROCESSED_DATA_DIR, 'corrected_flightlist_train.parquet')
ABLATION_OUTPUT_CSV = os.path.join(config.PROCESSED_DATA_DIR, 'ablation_contributions_results.csv')

LOAD_FACTOR_FEATURES = [
    'average_load_factor', 'estimated_payload_kg',
    'trip_fuel_kg', 'contingency_fuel_kg', 'final_reserve_fuel_kg',
    'estimated_total_fuel_kg', 'estimated_takeoff_mass',
]

ABL_BASE_FEATURES = [
    'starting_mass_kg', 'alt_end_ft', 'alt_avg_ft', 'gs_avg_kts', 'vs_avg_fpm',
    'interval_duration_sec', 'altitude_change_rate', 'great_circle_distance',
    'aircraft_type', 'end_hour', 'interval_elapsed_from_flight_start',
]

# Production-invariant ablation parameters
ABL_MODEL_PARAMS = {
    'n_estimators':     1455,
    'learning_rate':    0.02885922756814833,
    'max_depth':        9,
    'min_child_weight': 4,
    'gamma':            6.24155979490078e-08,
    'subsample':        0.9991625118585123,
    'colsample_bytree': 0.6701135673048045,
    'reg_alpha':        0.004878930563988692,
    'reg_lambda':       2.3991563444540384e-08,
    'objective':        'reg:squarederror',
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
    'verbosity':        0,
}

log_file = os.path.join(RESULTS_DIR, f'test_xgboost_top5_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
# SYNTHETIC DATA GENERATION FUNCTION
# ============================================================================
# ============================================================================
# ENHANCED SYNTHETIC DATA GENERATION FUNCTION (25K samples, 25% from long segments)
# ============================================================================
# def generate_synthetic_widebody_data_enhanced(df_train, n_synthetic=25000, long_segment_pct=0.25, random_state=42):
def generate_synthetic_widebody_data_enhanced(df_train, n_synthetic=25000, long_segment_pct=0.25, random_state=42):
    """Generate widebody synthetic data favoring long segments."""
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


# Ablation study: Design contributions (C1–C4)

def _abl_metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    r2   = float(1 - np.sum((y_true - y_pred) ** 2) /
                 np.sum((y_true - np.mean(y_true)) ** 2))
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


def _abl_collect_metrics(condition_name, y_true, y_pred, is_wb):
    rows = []
    rows.append({'condition': condition_name, 'split': 'Overall',
                 'n_segments': len(y_true), **_abl_metrics(y_true, y_pred)})
    nb_mask = ~is_wb
    if nb_mask.sum() > 0:
        rows.append({'condition': condition_name, 'split': 'Narrowbody',
                     'n_segments': int(nb_mask.sum()),
                     **_abl_metrics(y_true[nb_mask], y_pred[nb_mask])})
    if is_wb.sum() > 0:
        rows.append({'condition': condition_name, 'split': 'Widebody',
                     'n_segments': int(is_wb.sum()),
                     **_abl_metrics(y_true[is_wb], y_pred[is_wb])})
    return rows


def _abl_preprocess(X_train_df, X_val_df, feature_cols, X_aug_df=None):
    """Fit preprocessors on train; transform val/synthetic."""
    X_tr = X_train_df[feature_cols].copy()
    X_vl = X_val_df[feature_cols].copy()
    X_au = X_aug_df.reindex(columns=feature_cols, fill_value=np.nan).copy() if X_aug_df is not None else None

    nan_cols = [c for c in feature_cols if X_tr[c].isna().all()]
    if nan_cols:
        feature_cols = [c for c in feature_cols if c not in nan_cols]
        X_tr = X_tr.drop(columns=nan_cols)
        X_vl = X_vl.drop(columns=nan_cols)
        if X_au is not None:
            X_au = X_au.drop(columns=[c for c in nan_cols if c in X_au.columns])

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_tr[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    if num_cols:
        num_imp = SimpleImputer(strategy='mean')
        X_tr[num_cols] = num_imp.fit_transform(X_tr[num_cols])
        X_vl[num_cols] = num_imp.transform(X_vl[num_cols])
        if X_au is not None:
            X_au[num_cols] = num_imp.transform(X_au[num_cols])
    if cat_cols:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X_tr[cat_cols] = cat_imp.fit_transform(X_tr[cat_cols])
        X_vl[cat_cols] = cat_imp.transform(X_vl[cat_cols])
        if X_au is not None:
            X_au[cat_cols] = cat_imp.transform(X_au[cat_cols])
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
        X_vl[cat_cols] = enc.transform(X_vl[cat_cols])
        if X_au is not None:
            X_au[cat_cols] = enc.transform(X_au[cat_cols])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_au_s = scaler.transform(X_au) if X_au is not None else None
    return X_tr_s, X_vl_s, feature_cols, X_au_s


def _abl_train_eval(X_tr_s, X_vl_s, y_train, y_val, is_wb_val, condition_name, gpu_id,
                    X_au_s=None, y_aug=None):
    params = dict(ABL_MODEL_PARAMS)
    if gpu_id is not None:
        params['device'] = f'cuda:{gpu_id}'
    if X_au_s is not None and y_aug is not None:
        X_fit = np.vstack([X_tr_s, X_au_s])
        y_fit = np.log1p(np.concatenate([y_train, y_aug]))
        logger.info(f"  Training [{condition_name}] on {X_fit.shape[0]:,} samples "
                    f"({X_tr_s.shape[0]:,} real + {X_au_s.shape[0]:,} synthetic), "
                    f"{X_fit.shape[1]} features …")
    else:
        X_fit = X_tr_s
        y_fit = np.log1p(y_train)
        logger.info(f"  Training [{condition_name}] on {X_fit.shape[0]:,} samples, "
                    f"{X_fit.shape[1]} features …")
    t0 = time.time()
    model = XGBRegressor(**params)
    model.fit(X_fit, y_fit)
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    y_pred = np.maximum(np.expm1(model.predict(X_vl_s)), 0)
    rows = _abl_collect_metrics(condition_name, y_val, y_pred, is_wb_val)
    for r in rows:
        if r['split'] == 'Overall':
            logger.info(f"  {condition_name:<35s}  MAE={r['mae']:.1f} kg  "
                        f"RMSE={r['rmse']:.1f} kg  MAPE={r['mape']:.2f}%  R²={r['r2']:.4f}")
    return rows


def run_ablation_contributions(gpu_id=None, selected_features=None):
    """Ablation study for contributions C1–C4 using SFS baseline."""
    logger.info("\n" + "=" * 72)
    logger.info("ABLATION: Claimed Design Contributions (C1–C4)")
    logger.info("=" * 72)

    # Data ingestion
    logger.info("\n[ABL-1] Loading data …")

    apt = pd.read_parquet(APT_PATH)[['icao', 'longitude', 'latitude']]

    fl_raw = pd.read_parquet(FLIGHTLIST_PATH)
    fl_raw = fl_raw.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    fl_raw = fl_raw.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    fl_raw = fl_raw.drop(columns=['icao'], errors='ignore')
    fl_raw = fl_raw.merge(apt, left_on='destination_icao', right_on='icao', how='left')
    fl_raw = fl_raw.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    fl_raw = fl_raw.drop(columns=['icao'], errors='ignore')
    fl_raw['great_circle_distance'] = fl_raw.apply(
        lambda r: haversine(r.get('origin_lon'), r.get('origin_lat'),
                            r.get('dest_lon'), r.get('dest_lat')), axis=1)

    # C4: Temporal alignment lookup
    ts_correction_available = os.path.exists(FLIGHTLIST_COR_PATH)
    if ts_correction_available:
        fl_cor = pd.read_parquet(FLIGHTLIST_COR_PATH)[['flight_id', 'takeoff']].copy()
        fl_cor = fl_cor.rename(columns={'takeoff': 'corrected_takeoff'})
        fl_raw_ts = fl_raw[['flight_id', 'takeoff']].copy().rename(
            columns={'takeoff': 'raw_takeoff'})
        ts_lookup = fl_cor.merge(fl_raw_ts, on='flight_id', how='inner')
        ts_lookup['corrected_takeoff'] = pd.to_datetime(
            ts_lookup['corrected_takeoff'], errors='coerce', utc=True)
        ts_lookup['raw_takeoff'] = pd.to_datetime(
            ts_lookup['raw_takeoff'], errors='coerce', utc=True)
        ts_lookup['correction_sec'] = (
            ts_lookup['corrected_takeoff'] - ts_lookup['raw_takeoff']
        ).dt.total_seconds().fillna(0)
        ts_lookup = ts_lookup[['flight_id', 'correction_sec']]
        logger.info(f"  Timestamp correction: {len(ts_lookup):,} flights  "
                    f"(median offset = {ts_lookup['correction_sec'].median():.1f}s, "
                    f"max |offset| = {ts_lookup['correction_sec'].abs().max():.1f}s)")
    else:
        ts_correction_available = False
        logger.warning(f"  corrected_flightlist_train.parquet not found — C4 will be skipped.")

    fuel = pd.read_parquet(FUEL_PATH)
    fuel_intervals = (fuel[['flight_id', 'idx', 'fuel_kg', 'start', 'end']]
                      .copy()
                      .rename(columns={'idx': 'interval_idx'}))

    # Load intersection of parquet schemas
    featured_train = pd.read_parquet(FEATURED_DATA_TRAIN)
    featured_rank  = pd.read_parquet(FEATURED_DATA_TEST)
    common_parquet_cols = set(featured_train.columns) & set(featured_rank.columns)
    extended_cols = [c for c in featured_train.columns
                     if c in common_parquet_cols and c not in ('flight_id', 'idx')]
    feat_slim = featured_train[['flight_id', 'idx'] + extended_cols].copy()
    feat_slim = feat_slim.rename(columns={'idx': 'interval_idx'})

    metar_cols = [c for c in extended_cols if c.startswith('dep_') or c.startswith('arr_')]
    logger.info(f"  Featured parquet: {len(extended_cols)} common columns")
    logger.info(f"  METAR columns detected: {len(metar_cols)}")

    logger.info(f"  Loading augmented CSV: {DATA_PATH} …")
    df_raw = pd.read_csv(DATA_PATH, delimiter=';', low_memory=False)
    logger.info(f"  Augmented CSV: {len(df_raw):,} rows")

    fl_cols = ['flight_id', 'takeoff', 'landed', 'great_circle_distance',
               'origin_icao', 'destination_icao', 'aircraft_type',
               'origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']
    fl_merge_cols = ['flight_id'] + [c for c in fl_cols
                                     if c != 'flight_id'
                                     and c in fl_raw.columns
                                     and c not in df_raw.columns]
    df_raw = df_raw.merge(fl_raw[fl_merge_cols], on='flight_id', how='left')
    df_raw = df_raw.merge(fuel_intervals, on=['flight_id', 'interval_idx'], how='left')
    df_raw = df_raw.merge(feat_slim, on=['flight_id', 'interval_idx'], how='left')

    if 'alt_avg_ft' not in df_raw.columns:
        df_raw['alt_avg_ft'] = (df_raw.get('alt_start_ft', 0) + df_raw.get('alt_end_ft', 0)) / 2
    if 'altitude_change_rate' not in df_raw.columns:
        df_raw['altitude_change_rate'] = (df_raw.get('alt_change_ft', 0) /
                                          (df_raw.get('interval_duration_sec', 60) + 1e-6))
    if 'end_hour' not in df_raw.columns:
        if 'end' in df_raw.columns:
            df_raw['end_hour'] = (pd.to_datetime(df_raw['end'], errors='coerce')
                                  .dt.hour.fillna(-1).astype(int))
        else:
            df_raw['end_hour'] = -1
    if 'interval_elapsed_from_flight_start' not in df_raw.columns:
        df_raw['interval_elapsed_from_flight_start'] = 0

    # C4: pre-compute uncorrected elapsed time
    if ts_correction_available:
        df_raw = df_raw.merge(ts_lookup, on='flight_id', how='left')
        df_raw['interval_elapsed_raw'] = (
            df_raw['interval_elapsed_from_flight_start']
            - df_raw['correction_sec'].fillna(0))
        df_raw = df_raw.drop(columns=['correction_sec'])

    # C3: Aircraft-specific MTOW mapping
    default_mtow = np.median([v['mtow_kg'] for v in AIRCRAFT_DATA.values()])
    if 'aircraft_type' not in df_raw.columns:
        df_raw = df_raw.merge(
            fl_raw[['flight_id', 'aircraft_type']].drop_duplicates('flight_id'),
            on='flight_id', how='left')
    df_raw['static_mass_kg'] = df_raw['aircraft_type'].map(
        {k: v['mtow_kg'] for k, v in AIRCRAFT_DATA.items()}
    ).fillna(default_mtow)

    # Define baseline feature set
    if selected_features is not None:
        # Use exactly the SFS-selected features as the baseline
        baseline_cols = [c for c in selected_features if c in df_raw.columns]
        missing_sfs = [c for c in selected_features if c not in df_raw.columns]
        if missing_sfs:
            logger.warning(f"  {len(missing_sfs)} SFS features not found in ablation data "
                           f"and will be skipped: {missing_sfs[:5]}{'...' if len(missing_sfs)>5 else ''}")
        logger.info(f"  Baseline: {len(baseline_cols)} SFS-selected features")
    else:
        # Fallback: all common parquet cols + base features (original behaviour)
        ext_available = [c for c in extended_cols if c not in ABL_BASE_FEATURES]
        baseline_cols = ABL_BASE_FEATURES + ext_available
        baseline_cols = [c for c in baseline_cols if c in df_raw.columns]
        logger.info(f"  Baseline: {len(baseline_cols)} features (all common parquet cols)")

    # Feature set expansion (C1/C2)
    metar_add = [c for c in metar_cols if c not in baseline_cols and c in df_raw.columns]
    lf_add    = [c for c in LOAD_FACTOR_FEATURES if c not in baseline_cols and c in df_raw.columns]
    logger.info(f"  METAR cols to add for C1: {len(metar_add)}")
    logger.info(f"  Load-factor cols to add for C2: {len(lf_add)}")

    # Feature space union
    full_feature_cols = list(dict.fromkeys(baseline_cols + metar_add + lf_add))
    full_feature_cols = [c for c in full_feature_cols if c in df_raw.columns]

    target_col = 'actual_fuel_kg'
    # Include ablation-only metadata columns alongside the feature cols.
    # interval_elapsed_from_flight_start is always included so the C4 forward
    # path can access it even when SFS did not select it.
    extra_abl_cols = [c for c in ['static_mass_kg', 'interval_elapsed_raw',
                                   'interval_elapsed_from_flight_start']
                      if c in df_raw.columns and c not in full_feature_cols]
    df_all = df_raw[full_feature_cols + [target_col, 'flight_id'] + extra_abl_cols].copy()
    df_all = df_all.dropna(subset=[target_col])
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    logger.info(f"  Usable intervals: {len(df_all):,}  |  Total features: {len(full_feature_cols)}")

    # Partition real data (80/20)
    logger.info("\n[ABL-2] Splitting real data (80/20, random_state=42) …")
    extra_real = df_all.reset_index(drop=True)   # accessor for static_mass_kg / interval_elapsed_raw
    train_real_idx, val_real_idx = train_test_split(
        np.arange(len(extra_real)), test_size=0.2, random_state=42, shuffle=True
    )

    at_val    = extra_real['aircraft_type'].iloc[val_real_idx].values
    is_wb_val = np.isin(at_val, WIDEBODY_AIRCRAFT)
    y_val     = extra_real[target_col].iloc[val_real_idx].values.astype(np.float32)

    X_df_real  = extra_real[full_feature_cols]
    y_arr_real = extra_real[target_col].values.astype(np.float32)
    X_train_df = X_df_real.iloc[train_real_idx]
    X_val_df   = X_df_real.iloc[val_real_idx]
    y_train    = y_arr_real[train_real_idx]

    # Load synthetic train-only cache
    df_syn_aug = None
    y_syn      = None
    if os.path.exists(SYNTHETIC_PATH):
        _df_syn_raw = pd.read_parquet(SYNTHETIC_PATH)
        _all_abl_cols = list(dict.fromkeys(
            full_feature_cols + [c for c in extra_abl_cols if c not in full_feature_cols]
        ))
        df_syn_aug = _df_syn_raw.reindex(columns=_all_abl_cols, fill_value=np.nan).copy()
        # Pre-compute static MTOW for C3 substitution
        if 'aircraft_type' in _df_syn_raw.columns:
            df_syn_aug['static_mass_kg'] = _df_syn_raw['aircraft_type'].map(
                {k: v['mtow_kg'] for k, v in AIRCRAFT_DATA.items()}
            ).fillna(default_mtow)
        else:
            df_syn_aug['static_mass_kg'] = default_mtow
        # For C4: synthetic rows have no timestamp correction → raw = corrected
        _elapsed_key = 'interval_elapsed_from_flight_start'
        df_syn_aug['interval_elapsed_raw'] = (
            df_syn_aug[_elapsed_key].fillna(0)
            if _elapsed_key in df_syn_aug.columns else 0.0
        )
        # Extract target
        for _tc in [target_col, 'actual_fuel_kg', 'fuel_kg']:
            if _tc in _df_syn_raw.columns:
                y_syn = _df_syn_raw[_tc].values.astype(np.float32)
                break
        if y_syn is None:
            logger.warning("  Synthetic data has no target column — not using augmentation.")
            df_syn_aug = None
        else:
            logger.info(f"  +{len(df_syn_aug):,} synthetic rows (training only).")
    else:
        logger.warning("  Synthetic widebody cache not found — training without augmentation.")

    logger.info(
        f"  Real train: {len(train_real_idx):,}  |  Real val: {len(val_real_idx):,}  "
        f"(widebody: {is_wb_val.sum():,})"
        + (f"  |  Synthetic train: {len(df_syn_aug):,}" if df_syn_aug is not None else "")
    )

    # Define ablation scenarios
    # baseline_cols is the SFS feature set (or all-features fallback)
    baseline_avail = [c for c in baseline_cols if c in full_feature_cols]

    # Helper: pull feature slice from synthetic df; handles missing cols via reindex
    def _syn(feat_cols, override_dict=None):
        """Return a copy of df_syn_aug aligned to feat_cols, with optional column overrides."""
        if df_syn_aug is None:
            return None
        s = df_syn_aug.reindex(columns=feat_cols, fill_value=np.nan).copy()
        if override_dict:
            for col, vals in override_dict.items():
                if col in s.columns:
                    s[col] = vals
        return s

    conditions = []  # (label, feat_cols, X_tr_real, X_val_real, X_syn_or_None)

    # A: Baseline (SFS features — what was submitted)
    conditions.append(('Baseline (SFS features)', baseline_avail,
                        X_train_df.copy(), X_val_df.copy(),
                        _syn(baseline_avail)))

    # C1: METAR impact
    if metar_add:
        fc_c1 = baseline_avail + [c for c in metar_add if c in full_feature_cols]
        conditions.append(('SFS + METAR features (+C1)', fc_c1,
                           X_train_df.copy(), X_val_df.copy(),
                           _syn(fc_c1)))
        logger.info(f"  C1: adding {len(metar_add)} METAR columns on top of baseline")
    else:
        logger.warning("  C1 (METAR): all dep_*/arr_* cols already in SFS baseline — "
                       "no new METAR cols to add.  Skipping.")

    # C2: Payload/Load-factor impact
    # If load-factor cols are NOT in baseline → add them (+C2 forward)
    # If they ARE already in baseline → remove them (−C2 backward) to measure their value
    lf_in_baseline = [c for c in LOAD_FACTOR_FEATURES if c in baseline_avail]
    if lf_add:
        fc_c2 = baseline_avail + [c for c in lf_add if c in full_feature_cols]
        conditions.append(('SFS + load-factor features (+C2)', fc_c2,
                           X_train_df.copy(), X_val_df.copy(),
                           _syn(fc_c2)))
        logger.info(f"  C2 forward: adding {len(lf_add)} load-factor columns on top of baseline")
    if lf_in_baseline:
        fc_c2_bwd = [c for c in baseline_avail if c not in lf_in_baseline]
        if fc_c2_bwd:
            conditions.append(('SFS minus load-factor (-C2)', fc_c2_bwd,
                               X_train_df.copy(), X_val_df.copy(),
                               _syn(fc_c2_bwd)))
            logger.info(f"  C2 backward: removing {len(lf_in_baseline)} load-factor cols from baseline")
    if not lf_add and not lf_in_baseline:
        logger.warning("  C2 (load factor): no load-factor columns found anywhere — skipping.")

    # C3: Dynamic mass sensitivity
    if 'starting_mass_kg' in baseline_avail and 'static_mass_kg' in extra_real.columns:
        X_train_static = X_train_df[baseline_avail].copy()
        X_val_static   = X_val_df[baseline_avail].copy()
        X_train_static['starting_mass_kg'] = extra_real['static_mass_kg'].iloc[train_real_idx].values
        X_val_static['starting_mass_kg']   = extra_real['static_mass_kg'].iloc[val_real_idx].values
        # Synthetic C3: substitute starting_mass_kg with pre-computed static MTOW
        _syn_c3_override = ({'starting_mass_kg': df_syn_aug['static_mass_kg'].values}
                            if df_syn_aug is not None and 'static_mass_kg' in df_syn_aug.columns
                            else None)
        conditions.append(('SFS, static MTOW mass (-C3 dynamic tracking)',
                           baseline_avail, X_train_static, X_val_static,
                           _syn(baseline_avail, _syn_c3_override)))
        logger.info("  C3: starting_mass_kg replaced with static MTOW per aircraft type")
    else:
        logger.warning("  C3 (dynamic mass): 'starting_mass_kg' not in SFS baseline — skipping.")

    # C4: Temporal alignment sensitivity
    if ts_correction_available and 'interval_elapsed_raw' in extra_real.columns:
        elapsed_col = 'interval_elapsed_from_flight_start'
        if elapsed_col in baseline_avail:
            # Backward: revert corrected elapsed to raw in the SFS set
            X_train_rawts = X_train_df[baseline_avail].copy()
            X_val_rawts   = X_val_df[baseline_avail].copy()
            X_train_rawts[elapsed_col] = extra_real['interval_elapsed_raw'].iloc[train_real_idx].values
            X_val_rawts[elapsed_col]   = extra_real['interval_elapsed_raw'].iloc[val_real_idx].values
            # Synthetic C4 backward: raw==corrected for synthetic (no correction)
            _syn_c4_override = ({'interval_elapsed_from_flight_start':
                                  df_syn_aug['interval_elapsed_raw'].values}
                                if df_syn_aug is not None else None)
            conditions.append(('SFS, raw timestamps (-C4 correction)',
                               baseline_avail, X_train_rawts, X_val_rawts,
                               _syn(baseline_avail, _syn_c4_override)))
            logger.info("  C4 backward: interval_elapsed_from_flight_start reverted to raw timestamps")
        else:
            # Forward pair: elapsed not in SFS
            fc_c4_cor = baseline_avail + [elapsed_col]
            X_train_cor = X_train_df.reindex(columns=fc_c4_cor, fill_value=0).copy()
            X_val_cor   = X_val_df.reindex(columns=fc_c4_cor, fill_value=0).copy()
            X_train_cor[elapsed_col] = extra_real[elapsed_col].iloc[train_real_idx].values
            X_val_cor[elapsed_col]   = extra_real[elapsed_col].iloc[val_real_idx].values
            # Synthetic C4: Corrected temporal alignment
            _syn_c4_cor_override = ({'interval_elapsed_from_flight_start':
                                      df_syn_aug[elapsed_col].fillna(0).values}
                                    if df_syn_aug is not None and elapsed_col in df_syn_aug.columns
                                    else None)
            conditions.append(('SFS + elapsed corrected (+C4 corrected)',
                               fc_c4_cor, X_train_cor, X_val_cor,
                               _syn(fc_c4_cor, _syn_c4_cor_override)))

            X_train_raw = X_train_df.reindex(columns=fc_c4_cor, fill_value=0).copy()
            X_val_raw   = X_val_df.reindex(columns=fc_c4_cor, fill_value=0).copy()
            X_train_raw[elapsed_col] = extra_real['interval_elapsed_raw'].iloc[train_real_idx].values
            X_val_raw[elapsed_col]   = extra_real['interval_elapsed_raw'].iloc[val_real_idx].values
            # Synthetic C4: Raw temporal alignment (invariant)
            _syn_c4_raw_override = ({'interval_elapsed_from_flight_start':
                                      df_syn_aug['interval_elapsed_raw'].fillna(0).values}
                                    if df_syn_aug is not None else None)
            conditions.append(('SFS + elapsed raw (+C4 uncorrected)',
                               fc_c4_cor, X_train_raw, X_val_raw,
                               _syn(fc_c4_cor, _syn_c4_raw_override)))
            logger.info("  C4 forward: interval_elapsed_from_flight_start not in SFS — "
                        "adding corrected vs raw pair to isolate correction value")
    else:
        logger.warning("  C4 (timestamp correction): corrected flightlist not available — skipping.")

    # Execute ablation matrix
    logger.info(f"\n[ABL-3] Running {len(conditions)} ablation condition(s) …")
    all_rows = []
    for label, feat_cols, X_tr_v, X_vl_v, X_syn_v in conditions:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Condition: {label}")
        logger.info(f"  Features:  {len(feat_cols)}")
        X_tr_s, X_vl_s, _, X_au_s = _abl_preprocess(X_tr_v, X_vl_v, feat_cols, X_syn_v)
        rows = _abl_train_eval(X_tr_s, X_vl_s, y_train, y_val, is_wb_val, label, gpu_id,
                               X_au_s=X_au_s, y_aug=y_syn)
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    # Tabulate metrics
    logger.info("\n" + "=" * 90)
    logger.info(f"{'Condition':<40} {'Split':<14} {'MAE':>10} {'RMSE':>10} "
                f"{'MAPE':>8} {'R²':>8} {'N':>8}")
    logger.info("-" * 90)
    for _, r in results_df.iterrows():
        logger.info(
            f"{r['condition']:<40} {r['split']:<14} {r['mae']:>10.1f} "
            f"{r['rmse']:>10.1f} {r['mape']:>7.2f}% {r['r2']:>8.4f} "
            f"{int(r['n_segments']):>8,}")
    logger.info("=" * 90)

    # Error delta vs baseline (Overall)
    logger.info("\nMAE delta vs. Baseline (SFS features, Overall split):")
    baseline_mae_arr = results_df[
        (results_df['condition'].str.startswith('Baseline')) &
        (results_df['split'] == 'Overall')
    ]['mae'].values
    if len(baseline_mae_arr) > 0:
        baseline_mae = baseline_mae_arr[0]
        for _, r in results_df[results_df['split'] == 'Overall'].iterrows():
            if not r['condition'].startswith('Baseline'):
                delta = r['mae'] - baseline_mae
                pct   = delta / baseline_mae * 100
                # C1/C2: Negative delta denotes performance gain; C3/C4: Positive denotes degradation
                direction = 'degradation vs baseline' if delta > 0 else 'improvement over baseline'
                logger.info(f"  {r['condition']:<42} ΔMAE={delta:+.1f} kg "
                            f"({pct:+.2f}%)  ← {direction}")

    os.makedirs(os.path.dirname(ABLATION_OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(ABLATION_OUTPUT_CSV, index=False)
    logger.info(f"\n[+] Ablation results saved to: {ABLATION_OUTPUT_CSV}")
    return results_df


def main(gpu_id=0, force_sfs=False, force_synthetic=False, opt_mode='legacy'):
    FORCE_RERUN_SFS = force_sfs or '--force-sfs' in sys.argv
    FORCE_RERUN_SYNTHETIC = force_synthetic or '--force-synthetic' in sys.argv
    
    # Mode-specific output directory
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
    logger.info(f"[+] Total columns: {len(df_raw.columns)}")

    # Assemble feature vector
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

    # PHASE 1.5: Data Augmentation
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


    # Merge real and synthetic distributions
    n_real_rows = len(df_features)
    df_features_augmented = pd.concat([df_features, df_synthetic], ignore_index=True)

    logger.info(f"\n[+] Original training size: {len(df_features):,}")
    logger.info(f"[+] Synthetic samples added: {len(df_synthetic):,}")
    logger.info(f"[+] Augmented training size: {len(df_features_augmented):,}")
    logger.info(f"[+] Augmentation rate: {len(df_synthetic)/len(df_features)*100:.1f}%")

    # Cache validation metadata
    is_real_full = np.array([True] * n_real_rows + [False] * len(df_synthetic))
    # Align metadata via dropna-preserved index
    _real_idx = df_features.index
    openap_kg_full     = df_raw.loc[_real_idx, 'openap_fuel_kg'].values  if 'openap_fuel_kg' in df_raw.columns else np.full(n_real_rows, np.nan)
    aircraft_type_full = df_raw.loc[_real_idx, 'aircraft_type'].values   if 'aircraft_type'  in df_raw.columns else np.full(n_real_rows, 'UNK')
    phase_full         = df_raw.loc[_real_idx, 'phase'].values           if 'phase'          in df_raw.columns else np.full(n_real_rows, 'UNK')
    # Pad metadata for combined dataset
    openap_kg_full     = np.concatenate([openap_kg_full,     np.full(len(df_synthetic), np.nan)])
    aircraft_type_full = np.concatenate([aircraft_type_full, np.full(len(df_synthetic), 'UNK')])
    phase_full         = np.concatenate([phase_full,         np.full(len(df_synthetic), 'UNK')])

    # Training set definition
    X_full = df_features_augmented[feature_cols_selected]
    y_full = df_features_augmented[target_col].values.astype(np.float32)

    logger.info(f"[+] Full dataset (with synthetic): {len(df_features_augmented):,} intervals")

    # PHASE 2: Validation Strategy
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

    # Validation set metadata
    val_is_real        = is_real_full[val_indices]
    val_openap_kg      = openap_kg_full[val_indices]
    val_aircraft_type  = aircraft_type_full[val_indices]
    val_phase          = phase_full[val_indices]

    logger.info(f"[+] Training: {len(X_train):,} intervals ({len(X_train)/len(X_full)*100:.1f}%)")
    logger.info(f"[+] Validation: {len(X_val):,} intervals ({len(X_val)/len(X_full)*100:.1f}%)")
    logger.info(f"[+] Real rows in val set: {val_is_real.sum():,}")

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
        
        # Map selected features to current space
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

        # Run SFS (forward search)
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
        
        # Persist Optuna trials
        trials_df = study.trials_dataframe()
        trials_path = os.path.join(RESULTS_DIR, 'test_optuna_trials_history.csv')
        trials_df.to_csv(trials_path, index=False)

        # Retrieve best 10 trials
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

    # Export processed data for visualization
    X_train_sfs_df = pd.DataFrame(X_train_sfs, columns=selected_features)
    X_val_sfs_df = pd.DataFrame(X_val_sfs, columns=selected_features)
    X_train_sfs_df.to_csv(os.path.join(RESULTS_DIR, 'test_X_train_processed.csv'), index=False)
    X_val_sfs_df.to_csv(os.path.join(RESULTS_DIR, 'test_X_val_processed.csv'), index=False)

    # Validation phase: Top 10 evaluation
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
        
        # Validation inference
        val_pred_log = model.predict(X_val_sfs)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.maximum(val_pred, 0.0)
        
        # Compute performance metrics
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        val_mae = np.mean(np.abs(y_val - val_pred))
        val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-8))) * 100
        val_r2 = 1 - (np.sum((y_val - val_pred) ** 2) / np.sum((y_val - y_val.mean()) ** 2))
        
        # Train-set diagnostics
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

    # Export performance metrics
    results_path = os.path.join(RESULTS_DIR, 'random_search_top10_validation_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"[+] Detailed results saved: {results_path}")

    # Log model leaderboard
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

    # Comparative analysis: OpenAP vs XGBoost
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

    # Subset validation to real observations
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

    # Stratification by aircraft type
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

    # Stratification by flight phase
    for ph in ['CLIMB', 'CRUISE', 'DESCENT', 'LEVEL']:
        m = phase_real == ph
        if m.sum() < 5:
            continue
        yt, yp = y_val_real[m], pred_real[m]
        yo = openap_real[m]
        valid = ~np.isnan(yo)
        oa_rmse = _rmse(yt[valid], yo[valid]) if valid.sum() > 0 else float('nan')
        logger.info(f"  {ph:8s} N={m.sum():5d}  XGB RMSE={_rmse(yt,yp):7.1f}  XGB MAE={_mae(yt,yp):6.1f}  MAPE={_mape(yt,yp):5.1f}%  OpenAP RMSE={oa_rmse:.1f}")

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


    
    # PHASE 7: Test Set Preprocessing
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

    # Impute missing test features
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

    # PHASE 8: Final Model Training & Inference
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
        
        # Fit to full augmented dataset
        final_model = XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            device='cpu',
            n_jobs=-1,
            verbosity=0,
            **params
        )
        
        # Diagnostics split
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_full_sfs, y_full_log, test_size=0.1, random_state=42
        )
        
        # Capture learning dynamics
        final_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_train_final, y_train_final), (X_val_final, y_val_final)],
            verbose=False
        )
        
        # Export diagnostic visualizations
        # We use a dedicated folder for rank 1 to match evaluate_model.py expectations
        diag_dir = os.path.join(RESULTS_DIR, f'xgb_model_rank{rank}')
        save_model_plots(final_model, X_train_final, y_train_final, X_val_final, y_val_final, 
                         selected_features, diag_dir, f"rank{rank}")
        
        # Model explainability: Feature importance
        logger.info("\n" + "-"*70)
        logger.info(f"FEATURE IMPORTANCE ANALYSIS - MODEL RANK #{rank}")
        logger.info("-"*70)
        
        # Get feature importance from the trained model
        feature_importance = final_model.feature_importances_
        
        # Aggregate feature weights
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
        
        # Cumulative importance analysis
        cumulative_importance = importance_df['importance_pct'].cumsum()
        n_features_90pct = (cumulative_importance <= 90).sum() + 1
        n_features_95pct = (cumulative_importance <= 95).sum() + 1
        
        logger.info(f"\n[+] Cumulative Importance Analysis:")
        logger.info(f"    Top {n_features_90pct} features explain 90% of importance")
        logger.info(f"    Top {n_features_95pct} features explain 95% of importance")
        logger.info(f"    Total features: {len(selected_features)}")
        
        logger.info("-"*70)
        # ====================================================================
        
        # Full-set training evaluation
        full_pred_log = final_model.predict(X_full_sfs)
        full_pred = np.expm1(full_pred_log)
        full_pred = np.maximum(full_pred, 0.0)
        full_rmse = np.sqrt(np.mean((y_full - full_pred) ** 2))
        
        logger.info(f"[+] Training RMSE (100% data): {full_rmse:.4f} kg")
        
        # Test set inference
        test_pred_log = final_model.predict(X_test_sfs)
        test_pred = np.expm1(test_pred_log)
        test_pred = np.maximum(test_pred, 0.0)
        
        logger.info(f"[+] Test predictions: {len(test_pred):,}")
        logger.info(f"    Range: [{test_pred.min():.2f}, {test_pred.max():.2f}] kg")
        logger.info(f"    Mean: {test_pred.mean():.2f} kg")
        
        # Format submission output
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
        
        # Export model parameters
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
                
        # Export artifacts for downstream evaluation
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

            # Paper-ready visualization
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
    # PHASE 9: Final Summary
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

    # PHASE 10: Ablation Study
    logger.info("\n" + "="*70)
    logger.info("PHASE 10: ABLATION STUDY — DESIGN CONTRIBUTIONS (C1–C4)")
    logger.info("="*70)
    try:
        run_ablation_contributions(gpu_id=gpu_id, selected_features=selected_features)
    except Exception as e:
        logger.error(f"[-] Ablation study failed: {e}", exc_info=True)


def run(gpu_id=0, force_sfs=False, force_synthetic=False, opt_mode='legacy'):
    main(gpu_id=gpu_id, force_sfs=force_sfs, force_synthetic=force_synthetic, opt_mode=opt_mode)

if __name__ == "__main__":
    run()

