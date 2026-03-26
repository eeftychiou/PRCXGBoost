"""
Ablation study: effect of synthetic widebody augmentation on validation MAE.
Output: data/processed/ablation_augmentation_results.csv
"""

import os
import sys
import json
import logging
import argparse
import time
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import config

# Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'ablation_augmentation.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# Paths and Constants
DATA_PATH          = config.AUGMENTED_FINAL_CSV
APT_PATH           = config.APT_PARQUET
FLIGHTLIST_PATH    = config.FLIGHTLIST_TRAIN
FUEL_PATH          = config.FUEL_TRAIN
FEATURED_DATA_TRAIN = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
FEATURED_DATA_TEST  = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_final.parquet')
SYNTHETIC_PATH          = config.SYNTHETIC_WIDEBODY_PATH
SELECTED_FEATURES_PATH  = config.SELECTED_FEATURES_PATH

OUTPUT_CSV = os.path.join(config.PROCESSED_DATA_DIR, 'ablation_augmentation_results.csv')

WIDEBODY_AIRCRAFT  = config.WIDEBODY_AIRCRAFT

# Production hyperparameters
MODEL_PARAMS = {
    'n_estimators': 1455,
    'learning_rate': 0.02885922756814833,
    'max_depth': 9,
    'min_child_weight': 4,
    'gamma': 6.24155979490078e-08,
    'subsample': 0.9991625118585123,
    'colsample_bytree': 0.6701135673048045,
    'reg_alpha': 0.004878930563988692,
    'reg_lambda': 2.3991563444540384e-08,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

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
    "fuel_consumption_de", "fuel_consumption_lvl", "fuel_consumption_cr",
    "fuel_consumption_na", "fuel_consumption", "seg_avg_burn_rate",
    "average_load_factor", "estimated_payload_kg", "trip_fuel_kg",
    "contingency_fuel_kg", "final_reserve_fuel_kg", "estimated_total_fuel_kg",
    "estimated_takeoff_mass",
]


def haversine(lon1, lat1, lon2, lat2):
    if pd.isna([lon1, lat1, lon2, lat2]).any():
        return np.nan
    try:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * 6371
    except Exception:
        return np.nan


def preprocess(X_train_df, X_val_df, X_aug_df):
    """
    Fit imputer / encoder / scaler on X_train_df (real train only), then
    transform val and augmented sets with the same fitted objects.

    Returns (X_train_proc, X_val_proc, X_aug_proc, feature_cols_out)
    After encoding, all columns are numeric.
    """
    feature_cols = list(X_train_df.columns)

    numerical_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train_df[c])]
    categorical_features = [c for c in feature_cols if c not in numerical_features]

    log.info(f"  Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")

    # Feature pruning
    nan_cols = [c for c in numerical_features if X_train_df[c].isna().all()]
    if nan_cols:
        log.info(f"  Dropping all-NaN columns: {nan_cols}")
        numerical_features = [c for c in numerical_features if c not in nan_cols]
        X_train_df = X_train_df.drop(columns=nan_cols)
        X_val_df   = X_val_df.drop(columns=nan_cols)
        X_aug_df   = X_aug_df.drop(columns=nan_cols)

    X_tr = X_train_df.copy()
    X_vl = X_val_df.copy()
    X_au = X_aug_df.copy()

    # Numeric imputation
    if numerical_features:
        num_imp = SimpleImputer(strategy='mean')
        X_tr[numerical_features] = num_imp.fit_transform(X_tr[numerical_features])
        X_vl[numerical_features] = num_imp.transform(X_vl[numerical_features])
        X_au[numerical_features] = num_imp.transform(X_au[numerical_features])

    # Categorical pipeline
    if categorical_features:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X_tr[categorical_features] = cat_imp.fit_transform(X_tr[categorical_features])
        X_vl[categorical_features] = cat_imp.transform(X_vl[categorical_features])
        X_au[categorical_features] = cat_imp.transform(X_au[categorical_features])

        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[categorical_features] = enc.fit_transform(X_tr[categorical_features])
        X_vl[categorical_features] = enc.transform(X_vl[categorical_features])
        X_au[categorical_features] = enc.transform(X_au[categorical_features])

    # Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_au_s = scaler.transform(X_au)

    out_cols = numerical_features + categorical_features
    return X_tr_s, X_vl_s, X_au_s, out_cols


def metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


def run(gpu_id=None):
    device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
    MODEL_PARAMS['device'] = device
    log.info("=" * 70)
    log.info("ABLATION: Synthetic Widebody Augmentation")
    log.info(f"Device: {device}")
    log.info("=" * 70)

    # Data loading
    log.info("\n[1] Loading data...")

    apt = pd.read_parquet(APT_PATH)[['icao', 'longitude', 'latitude']]

    flightlist = pd.read_parquet(FLIGHTLIST_PATH)
    flightlist = flightlist.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    flightlist = flightlist.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    flightlist = flightlist.drop(columns=['icao'], errors='ignore')
    flightlist = flightlist.merge(apt, left_on='destination_icao', right_on='icao', how='left', suffixes=('', '_dest'))
    flightlist = flightlist.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    flightlist = flightlist.drop(columns=['icao', 'icao_dest'], errors='ignore')
    flightlist['great_circle_distance'] = flightlist.apply(
        lambda r: haversine(r.get('origin_lon'), r.get('origin_lat'),
                            r.get('dest_lon'), r.get('dest_lat')), axis=1)

    fuel = pd.read_parquet(FUEL_PATH)

    featured_data = pd.read_parquet(FEATURED_DATA_TRAIN)
    featured_data_final = pd.read_parquet(FEATURED_DATA_TEST)
    common_cols = set(featured_data.columns) & set(featured_data_final.columns)
    available_features = ['flight_id', 'idx'] + [
        c for c in EXTENDED_FEATURES_FROM_PARQUET if c in common_cols
    ]
    feat_slim = featured_data[available_features].copy().rename(columns={'idx': 'interval_idx'})

    log.info(f"  Flightlist: {len(flightlist):,}, Fuel: {len(fuel):,}, Featured cols: {len(available_features)-2}")

    df_raw = pd.read_csv(DATA_PATH, delimiter=';', low_memory=False)
    log.info(f"  Training CSV rows: {len(df_raw):,}")

    flightlist_cols = ['flight_id', 'takeoff', 'landed', 'great_circle_distance',
                       'origin_icao', 'destination_icao', 'aircraft_type',
                       'origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']
    df_raw = df_raw.merge(flightlist[[c for c in flightlist_cols if c in flightlist.columns]],
                          on='flight_id', how='left')

    fuel_intervals = fuel[['flight_id', 'idx', 'fuel_kg', 'start', 'end']].copy()
    fuel_intervals = fuel_intervals.rename(columns={'idx': 'interval_idx'})
    df_raw = df_raw.merge(fuel_intervals, on=['flight_id', 'interval_idx'], how='left')
    df_raw = df_raw.merge(feat_slim, on=['flight_id', 'interval_idx'], how='left')

    base_features = [
        'starting_mass_kg', 'alt_end_ft', 'alt_avg_ft', 'gs_avg_kts', 'vs_avg_fpm',
        'interval_duration_sec', 'altitude_change_rate', 'great_circle_distance',
        'aircraft_type', 'end_hour', 'interval_elapsed_from_flight_start',
    ]
    extended_features_available = [c for c in available_features[2:] if c not in base_features]
    feature_cols = base_features + extended_features_available

    # Computed columns
    if 'alt_avg_ft' not in df_raw.columns:
        df_raw['alt_avg_ft'] = (df_raw.get('alt_start_ft', 0) + df_raw.get('alt_end_ft', 0)) / 2
    if 'altitude_change_rate' not in df_raw.columns:
        df_raw['altitude_change_rate'] = df_raw.get('alt_change_ft', 0) / (df_raw.get('interval_duration_sec', 60) + 1e-6)
    if 'end_hour' not in df_raw.columns:
        df_raw['end_hour'] = pd.to_datetime(df_raw.get('end'), errors='coerce').dt.hour.fillna(-1).astype(int)
    if 'interval_elapsed_from_flight_start' not in df_raw.columns:
        df_raw['interval_elapsed_from_flight_start'] = 0

    target_col = 'actual_fuel_kg'
    feature_cols = [c for c in feature_cols if c in df_raw.columns]

    # SFS feature filtering
    if os.path.exists(SELECTED_FEATURES_PATH):
        try:
            with open(SELECTED_FEATURES_PATH, 'r') as f:
                feat_data = json.load(f)
            selected = feat_data['selected_features'] if isinstance(feat_data, dict) else feat_data
            # Keep only SFS-selected features that are present in feature_cols;
            # always keep aircraft_type for the WB/NB breakdown even if not selected.
            sfs_set = set(selected)
            feature_cols = [c for c in feature_cols if c in sfs_set or c == 'aircraft_type']
            log.info(f"  SFS filter applied: {len(feature_cols)} features retained from {SELECTED_FEATURES_PATH}")
        except Exception as e:
            log.warning(f"  Could not load SFS features: {e}. Using all features.")
    else:
        log.info(f"  {SELECTED_FEATURES_PATH} not found — using all features")

    df_features = df_raw[feature_cols + [target_col]].copy()
    df_features = df_features.dropna(subset=[target_col])
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    log.info(f"  Real training intervals: {len(df_features):,}")

    # Real-data split (80/20)
    log.info("\n[2] Splitting real data into train / val (80/20, random_state=42)...")
    # Keep aircraft_type for post-hoc breakdown — not as a feature at index level
    aircraft_type_series = df_features['aircraft_type'].reset_index(drop=True)

    indices = np.arange(len(df_features))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    X_real = df_features[feature_cols].reset_index(drop=True)
    y_real = df_features[target_col].values.astype(np.float32)

    X_real_train = X_real.iloc[train_idx].reset_index(drop=True)
    X_real_val   = X_real.iloc[val_idx].reset_index(drop=True)
    y_real_train = y_real[train_idx]
    y_real_val   = y_real[val_idx]

    # Aircraft type labels for the val set (for per-type breakdown)
    at_val = aircraft_type_series.iloc[val_idx].reset_index(drop=True)
    is_wb_val = at_val.isin(WIDEBODY_AIRCRAFT).values

    log.info(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")
    log.info(f"  Val widebody segments: {is_wb_val.sum():,}  ({is_wb_val.mean()*100:.1f}%)")

    # Synthetic sample generation
    if os.path.exists(SYNTHETIC_PATH):
        df_synthetic = pd.read_parquet(SYNTHETIC_PATH)
        log.info(f"  Loaded cached synthetic data: {len(df_synthetic):,} rows")
    else:
        log.warning("  synthetic_widebody.parquet not found — generating now (this is slow).")
        # Import generation function from the training script
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from XGBoostTraining_Final import generate_synthetic_widebody_data_enhanced  # noqa
        df_synthetic = generate_synthetic_widebody_data_enhanced(
            df_features[feature_cols + [target_col]],
            n_synthetic=25000, long_segment_pct=0.25, random_state=42)
        df_synthetic.to_parquet(SYNTHETIC_PATH, index=False, engine='fastparquet')
        log.info(f"  Generated and cached {len(df_synthetic):,} synthetic rows")

    # Align synthetic columns to feature_cols + target
    synth_cols = [c for c in feature_cols if c in df_synthetic.columns]
    missing_synth = [c for c in feature_cols if c not in df_synthetic.columns]
    if missing_synth:
        log.warning(f"  Synthetic data missing {len(missing_synth)} feature cols — filling with NaN")
    X_synthetic = df_synthetic.reindex(columns=feature_cols, fill_value=np.nan)
    y_synthetic = df_synthetic[target_col].values.astype(np.float32) if target_col in df_synthetic.columns \
        else df_synthetic.get('actual_fuel_kg', df_synthetic.get('fuel_kg', np.zeros(len(df_synthetic)))).values.astype(np.float32)

    # Augmented training set construction
    log.info("\n[4] Building augmented training set (real train + synthetic)...")
    X_aug_train = pd.concat([X_real_train, X_synthetic], ignore_index=True)
    y_aug_train = np.concatenate([y_real_train, y_synthetic]).astype(np.float32)
    log.info(f"  Augmented train size: {len(X_aug_train):,}  "
             f"(+{len(df_synthetic):,} synthetic = {len(df_synthetic)/len(y_real_train)*100:.1f}%)")

    # Preprocessing
    log.info("\n[5] Preprocessing (fit on real train, transform all sets)...")
    X_tr_s, X_vl_s, X_au_s, _ = preprocess(X_real_train, X_real_val, X_aug_train)

    y_tr_log  = np.log1p(y_real_train)
    y_vl      = y_real_val                            # evaluation on original scale
    y_au_log  = np.log1p(y_aug_train)

    # Baseline training (Train Model A)
    log.info("\n[6] Training Model A (no synthetic)...")
    t0 = time.time()
    model_a = XGBRegressor(**MODEL_PARAMS)
    model_a.fit(X_tr_s, y_tr_log)
    log.info(f"  Done in {time.time()-t0:.1f}s")

    pred_a_log = model_a.predict(X_vl_s)
    pred_a = np.maximum(np.expm1(pred_a_log), 0)

    # Augmented training (Train Model B)
    log.info("\n[7] Training Model B (real + synthetic)...")
    t0 = time.time()
    model_b = XGBRegressor(**MODEL_PARAMS)
    model_b.fit(X_au_s, y_au_log)
    log.info(f"  Done in {time.time()-t0:.1f}s")

    pred_b_log = model_b.predict(X_vl_s)
    pred_b = np.maximum(np.expm1(pred_b_log), 0)

    # Evaluation
    log.info("\n[8] Computing metrics...")

    def collect_metrics(y_true, y_pred, label_prefix, aircraft_mask_wb):
        rows = []
        # Overall
        m = metrics(y_true, y_pred)
        rows.append({'model': label_prefix, 'split': 'Overall', **m,
                     'n_segments': len(y_true)})
        # Narrowbody
        nb_mask = ~aircraft_mask_wb
        if nb_mask.sum() > 0:
            m = metrics(y_true[nb_mask], y_pred[nb_mask])
            rows.append({'model': label_prefix, 'split': 'Narrowbody', **m,
                         'n_segments': nb_mask.sum()})
        # Widebody
        if aircraft_mask_wb.sum() > 0:
            m = metrics(y_true[aircraft_mask_wb], y_pred[aircraft_mask_wb])
            rows.append({'model': label_prefix, 'split': 'Widebody', **m,
                         'n_segments': aircraft_mask_wb.sum()})
        return rows

    rows = []
    rows += collect_metrics(y_vl, pred_a, 'No Augmentation', is_wb_val)
    rows += collect_metrics(y_vl, pred_b, 'With Augmentation', is_wb_val)

    results_df = pd.DataFrame(rows)

    # Pretty-print results table
    log.info("\n" + "=" * 72)
    log.info(f"{'Model':<22} {'Split':<14} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'R²':>8} {'N':>8}")
    log.info("-" * 72)
    for _, r in results_df.iterrows():
        log.info(f"{r['model']:<22} {r['split']:<14} {r['mae']:>10.1f} {r['rmse']:>10.1f} "
                 f"{r['mape']:>7.2f}% {r['r2']:>8.4f} {int(r['n_segments']):>8,}")
    log.info("=" * 72)

    # Improvement summary
    for split in ['Overall', 'Narrowbody', 'Widebody']:
        a = results_df[(results_df['model'] == 'No Augmentation') & (results_df['split'] == split)]
        b = results_df[(results_df['model'] == 'With Augmentation') & (results_df['split'] == split)]
        if len(a) and len(b):
            delta = b.iloc[0]['mae'] - a.iloc[0]['mae']
            pct   = delta / a.iloc[0]['mae'] * 100
            direction = "improvement" if delta < 0 else "degradation"
            log.info(f"  {split:14s} MAE change: {delta:+.1f} kg  ({pct:+.2f}%)  ← {direction}")

    # Persistence
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\n[+] Results saved to: {OUTPUT_CSV}")
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study: synthetic widebody augmentation')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (e.g. 0). Omit for CPU.')
    args = parser.parse_args()
    run(gpu_id=args.gpu)
