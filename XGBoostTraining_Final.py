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
import json  # ✅ ADDED - For saving/loading selected features
import sys   # ✅ ADDED - For command line argument parsing
from skopt import BayesSearchCV
from skopt.space import Real, Integer

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


# File paths
DATA_PATH = 'data/augmented_openap_correct_mass_ALL_FLIGHTS_final.csv'
APT_PATH = 'data/apt.parquet'
FLIGHTLIST_PATH = 'data/flightlist_train.parquet'
FUEL_PATH = 'data/fuel_train.parquet'
TEST_CSV_PATH = 'data/augmented_openap_submission_ALL_FLIGHTSrank.csv'
FINAL_CSV_PATH = 'data/augmented_openap_final_ALL_FLIGHTS.csv'
FUEL_RANK_PATH = 'data/fuel_rank_submission.parquet'
FUEL_FINAL_PATH = 'data/fuel_final_submission.parquet'
FLIGHTLIST_RANK_PATH = 'data/flightlist_rank.parquet'
FLIGHTLIST_FINAL_PATH = 'data/flightlist_final.parquet'
RESULTS_DIR = 'Results'

FEATURED_DATA_TRAIN = 'data/featured_data_merged.parquet'
FEATURED_DATA_TEST = 'data/featured_data_rank_merged.parquet'
FEATURED_DATA_FINAL = 'data/featured_data_matched_structure.parquet'

# ✅ ADDED - Path for caching selected features
SELECTED_FEATURES_PATH = 'Results/selected_features_sfs3.json'


os.makedirs(RESULTS_DIR, exist_ok=True)


log_file = os.path.join(RESULTS_DIR, f'xgboost_top5_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Extended feature list
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
    "aircraft_type_encoded", "segment_duration", "seg_latitude_mean", "seg_latitude_std", 
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


def main():
    # ✅ ADDED - Check for force-rerun flag
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


    X_full = df_features[feature_cols_selected]
    y_full = df_features[target_col].values.astype(np.float32)


    logger.info(f"[+] Full dataset: {len(df_features):,} intervals")


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
        imputed_values = num_imputer.fit_transform(X_train_imputed[numerical_features])
        X_train_imputed[numerical_features] = pd.DataFrame(
            imputed_values, 
            columns=numerical_features, 
            index=X_train_imputed.index
        )
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
    # PHASE 4: FEATURE SELECTION (ON TRAIN) - WITH CACHING ✅ MODIFIED
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: SEQUENTIAL FEATURE SELECTION (SFS)")
    logger.info("="*70)

    # ✅ ADDED - Check if pre-computed features exist
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
        # ✅ MODIFIED - Added logging for force-rerun
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
        sfs_start = time.time()  # ✅ ADDED - Track SFS time
        sfs.fit(X_train_scaled, y_train_log)
        sfs_time = time.time() - sfs_start  # ✅ ADDED

        selected_mask = sfs.get_support()
        selected_features = [feat for feat, selected in zip(feature_cols_selected, selected_mask) if selected]

        logger.info(f"[+] SFS completed in {sfs_time/60:.2f} minutes")
        logger.info(f"[+] SFS selected {len(selected_features)} features")
        
        # ✅ ADDED - Save selected features for future use
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
        
        # ✅ ADDED - Also save a human-readable version
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

    # ✅ ADDED - Log selected features summary
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


    # # Define comprehensive parameter grid
    # param_grid = {
    #     'n_estimators': [500, 700, 800],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'max_depth': [4, 6, 8],
    #     'subsample': [0.7, 0.8, 0.9],
    #     'colsample_bytree': [0.7, 0.8, 0.9],
    #     'gamma': [0, 0.1, 0.5],
    #     'reg_alpha': [0, 0.01, 0.1],
    #     'reg_lambda': [0.5, 1.0, 2.0],
    #     'min_child_weight': [1, 3, 5]
    # }

    # Define comprehensive parameter grid
    # param_grid = {
    #     'n_estimators': [400, 500, 600, 700, 800, 900, 1000, 1200, 1500],
    #     'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.1, 0.15, 0.2],
    #     'max_depth': [4, 6, 7, 8, 9, 10, 11, 12, 13, 15],
    #     'subsample': [0.6, 0.65, 0.7, 0.75, 0.77, 0.8, 0.85, 0.87, 0.9, 0.95],
    #     'colsample_bytree': [0.6, 0.63, 0.7, 0.72, 0.75, 0.8, 0.85, 0.9],
    #     'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 2.0],
    #     'reg_alpha': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    #     'reg_lambda': [0.5, 1.0, 1.5, 2.0, 2.2, 2.5, 2.7, 3.0],
    #     'min_child_weight': [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    # }
    # Ultra-focused grid (smaller, faster)
    # param_grid = {
    #     'n_estimators': [750, 800, 850, 900],
    #     'learning_rate': [0.06, 0.07, 0.08],
    #     'max_depth': [7, 8, 9],
    #     'subsample': [0.82, 0.85, 0.87],
    #     'colsample_bytree': [0.68, 0.70, 0.72, 0.75],
    #     'gamma': [0, 0.05],
    #     'reg_alpha': [0.03, 0.05, 0.07],
    #     'reg_lambda': [2.3, 2.5, 2.7],
    #     'min_child_weight': [2, 3]
    # }

    param_grid = {
        'n_estimators': [850],
        'learning_rate': [0.08],
        'max_depth': [8],
        'subsample': [0.87],
        'colsample_bytree': [0.68],
        'gamma': [0],
        'reg_alpha': [0.05],
        'reg_lambda': [2.3],
        'min_child_weight': [3]
    }



    # Define search space
    # param_grid = {
    #     'n_estimators': Integer(400, 1000),
    #     'learning_rate': Real(0.01, 0.15, prior='log-uniform'),
    #     'max_depth': Integer(4, 12),
    #     'subsample': Real(0.6, 0.95),
    #     'colsample_bytree': Real(0.6, 0.9),
    #     'gamma': Real(1e-8, 1.0, prior='log-uniform'),  # ← Start at tiny positive number
    #     'reg_alpha': Real(1e-8, 1.0, prior='log-uniform'),  # ← Start at tiny positive number
    #     'reg_lambda': Real(0.5, 3.0),
    #     'min_child_weight': Integer(1, 7)
    # }



    # Calculate total combinations
    # total_combinations = np.prod([len(v) for v in param_grid.values()])
    # logger.info(f"Parameter grid size: {total_combinations:,} combinations")


    # Use RandomizedSearchCV for efficiency
    n_iter_search = 50
    logger.info(f"Using RandomizedSearchCV with {n_iter_search} iterations")

    # NEW:
    base_xgb = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1,
        verbosity=0
    )

    # random_search = BayesSearchCV(  # ← Changed name
    #     estimator=base_xgb,
    #     search_spaces=param_grid,  # ← Changed from param_distributions to search_spaces
    #     n_iter=100,  # or whatever number you want
    #     scoring='neg_root_mean_squared_error',
    #     cv=3,
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1
    # )

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
    logger.info(f"\n[+] Grid search completed in {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")


    # ========================================================================
    # EXTRACT TOP 5 MODELS FROM CV RESULTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXTRACTING TOP 5 MODELS FROM GRID SEARCH")
    logger.info("="*70)


    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df = cv_results_df.sort_values('mean_test_score', ascending=False)
    cv_results_df = cv_results_df.reset_index(drop=True)


    # Get top 5 models from CV
    top_5_cv = cv_results_df.head(5).copy()


    logger.info("\nTop 5 models from 5-fold CV on training data:")
    for idx, row in top_5_cv.iterrows():
        cv_rmse = np.sqrt(-row['mean_test_score'])
        cv_std = np.sqrt(row['std_test_score'])
        logger.info(f"\n  CV Rank {idx+1}:")
        logger.info(f"    CV RMSE = {cv_rmse:.4f} ± {cv_std:.4f} kg")
        logger.info(f"    Params: {row['params']}")


    # ========================================================================
    # EVALUATE TOP 5 MODELS ON HELD-OUT 20% VALIDATION SET
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("EVALUATING TOP 5 MODELS ON 20% VALIDATION SET")
    logger.info("="*70)


    validation_results = []


    for rank, (idx, row) in enumerate(top_5_cv.iterrows(), 1):
        logger.info(f"\nEvaluating Model {rank}/5...")
        
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
    logger.info("TOP 5 MODELS RANKED BY VALIDATION RMSE")
    logger.info("="*70)


    results_df = pd.DataFrame(validation_results)
    results_df = results_df.sort_values('val_rmse', ascending=True)
    results_df['final_rank'] = range(1, len(results_df) + 1)


    # Save detailed results
    results_path = os.path.join(RESULTS_DIR, 'grid_search_top5_validation_results.csv')
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
    # PHASE 7: LOADING AND PREPROCESSING TEST + FINAL DATA (SEPARATELY)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 7: LOADING AND PREPROCESSING TEST + FINAL DATA (SEPARATELY)")
    logger.info("="*70)

    # Load COMPLETE submission template (61,745 rows total)
    fuel_final = pd.read_parquet(FUEL_FINAL_PATH)
    submission_template = fuel_final[['idx', 'flight_id', 'start', 'end']].copy()
    logger.info(f"[+] Submission template (FINAL): {len(submission_template):,} rows")

    # # ========================================================================
    # # PROCESS TEST DATASET (Phase 1)
    # # ========================================================================
    # logger.info("\n" + "-"*70)
    # logger.info("PROCESSING TEST DATASET (Phase 1)")
    # logger.info("-"*70)

    # logger.info(f"Loading TEST CSV: {TEST_CSV_PATH}")
    # df_test_raw = pd.read_csv(TEST_CSV_PATH, delimiter=',', low_memory=False)
    # logger.info(f"[+] Test data: {len(df_test_raw):,} rows")

    # # Load test intervals from fuel_rank
    # fuel_rank = pd.read_parquet(FUEL_RANK_PATH)
    # fuel_rank_intervals = fuel_rank[['flight_id', 'idx', 'start', 'end']].copy()
    # fuel_rank_intervals = fuel_rank_intervals.rename(columns={'idx': 'interval_idx'})

    # # Merge test data
    # df_test_raw = df_test_raw.merge(fuel_rank_intervals, on=['flight_id', 'interval_idx'], how='left')
    # df_test_raw = df_test_raw.merge(featured_data_rank_selected, on=['flight_id', 'interval_idx'], how='left')

    # # Load test flightlist
    # flightlist_rank = pd.read_parquet(FLIGHTLIST_RANK_PATH)
    # flightlist_test = flightlist_rank[['flight_id', 'aircraft_type', 'origin_icao', 'destination_icao', 'takeoff', 'landed']].drop_duplicates(subset=['flight_id'])

    # # Add coordinates for test
    # flightlist_test = flightlist_test.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    # flightlist_test = flightlist_test.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    # flightlist_test = flightlist_test.drop(columns=['icao'], errors='ignore')
    # flightlist_test = flightlist_test.merge(apt, left_on='destination_icao', right_on='icao', how='left')
    # flightlist_test = flightlist_test.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    # flightlist_test = flightlist_test.drop(columns=['icao'], errors='ignore')
    # flightlist_test['great_circle_distance'] = flightlist_test.apply(
    #     lambda row: haversine(row.get('origin_lon'), row.get('origin_lat'), 
    #                         row.get('dest_lon'), row.get('dest_lat')), axis=1
    # )

    # df_test_raw = df_test_raw.merge(
    #     flightlist_test[['flight_id', 'origin_icao', 'destination_icao', 'great_circle_distance', 
    #                     'aircraft_type', 'takeoff', 'landed', 'origin_lon', 'dest_lon']],
    #     on='flight_id', how='left', suffixes=('', '_meta')
    # )

    # # Add computed columns for test
    # if 'alt_avg_ft' not in df_test_raw.columns:
    #     df_test_raw['alt_avg_ft'] = (df_test_raw.get('alt_start_ft', 0) + df_test_raw.get('alt_end_ft', 0)) / 2
    # if 'altitude_change_rate' not in df_test_raw.columns:
    #     df_test_raw['altitude_change_rate'] = df_test_raw.get('alt_change_ft', 0) / (df_test_raw.get('interval_duration_sec', 60) + 1e-6)
    # if 'end_hour' not in df_test_raw.columns:
    #     df_test_raw['end_hour'] = pd.to_datetime(df_test_raw.get('end'), errors='coerce').dt.hour.fillna(-1).astype(int)
    # if 'interval_elapsed_from_flight_start' not in df_test_raw.columns:
    #     df_test_raw['interval_elapsed_from_flight_start'] = 0

    # # Handle missing features for test
    # missing_test_features = [col for col in feature_cols_selected if col not in df_test_raw.columns]
    # if missing_test_features:
    #     logger.warning(f"Test: Missing {len(missing_test_features)} features - filling with 0")
    #     for col in missing_test_features:
    #         df_test_raw[col] = 0

    # # Preprocess test data
    # X_test_data = df_test_raw[feature_cols_selected].copy()
    # X_test_data = X_test_data.replace([np.inf, -np.inf], np.nan)

    # if numerical_features:
    #     X_test_data[numerical_features] = num_imputer_full.transform(X_test_data[numerical_features])
    # if categorical_features:
    #     X_test_data[categorical_features] = cat_imputer_full.transform(X_test_data[categorical_features])
    #     X_test_data[categorical_features] = cat_encoder_full.transform(X_test_data[categorical_features])

    # X_test_scaled = scaler_full.transform(X_test_data)
    # X_test_sfs = X_test_scaled[:, selected_mask]

    # logger.info(f"[+] Test data ready: {X_test_sfs.shape}")

    # # ========================================================================
    # # PROCESS FINAL DATASET (Final Phase)
    # # ========================================================================
    # logger.info("\n" + "-"*70)
    # logger.info("PROCESSING FINAL DATASET (Final Phase)")
    # logger.info("-"*70)

    # logger.info(f"Loading FINAL CSV: {FINAL_CSV_PATH}")
    # df_final_raw = pd.read_csv(FINAL_CSV_PATH, delimiter=',', low_memory=False)
    # logger.info(f"[+] Final data: {len(df_final_raw):,} rows")

    # # Load final intervals
    # fuel_final_intervals = fuel_final[['flight_id', 'idx', 'start', 'end']].copy()
    # fuel_final_intervals = fuel_final_intervals.rename(columns={'idx': 'interval_idx'})

    # # Merge final data
    # df_final_raw = df_final_raw.merge(fuel_final_intervals, on=['flight_id', 'interval_idx'], how='left')

    # # Check for separate final featured data
    # if FEATURED_DATA_FINAL and os.path.exists(FEATURED_DATA_FINAL):
    #     logger.info(f"Loading featured data for FINAL: {FEATURED_DATA_FINAL}")
    #     featured_data_final = pd.read_parquet(FEATURED_DATA_FINAL)
        
    #     # Match the same feature selection as test
    #     featured_final_cols = ['flight_id', 'idx'] if 'idx' in featured_data_final.columns else ['flight_id', 'interval_idx']
    #     for col in available_features[2:]:  # Skip flight_id and idx
    #         if col in featured_data_final.columns:
    #             featured_final_cols.append(col)
        
    #     featured_data_final_selected = featured_data_final[featured_final_cols].copy()
    #     if 'idx' in featured_data_final_selected.columns:
    #         featured_data_final_selected = featured_data_final_selected.rename(columns={'idx': 'interval_idx'})
        
    #     df_final_raw = df_final_raw.merge(featured_data_final_selected, on=['flight_id', 'interval_idx'], how='left')
    # else:
    #     logger.warning("No separate FINAL featured data - using TEST features structure")
    #     df_final_raw = df_final_raw.merge(featured_data_rank_selected, on=['flight_id', 'interval_idx'], how='left')

    # # Load final flightlist
    # flightlist_final = pd.read_parquet(FLIGHTLIST_FINAL_PATH)
    # flightlist_final_meta = flightlist_final[['flight_id', 'aircraft_type', 'origin_icao', 'destination_icao', 'takeoff', 'landed']].drop_duplicates(subset=['flight_id'])

    # # Add coordinates for final
    # flightlist_final_meta = flightlist_final_meta.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    # flightlist_final_meta = flightlist_final_meta.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    # flightlist_final_meta = flightlist_final_meta.drop(columns=['icao'], errors='ignore')
    # flightlist_final_meta = flightlist_final_meta.merge(apt, left_on='destination_icao', right_on='icao', how='left')
    # flightlist_final_meta = flightlist_final_meta.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    # flightlist_final_meta = flightlist_final_meta.drop(columns=['icao'], errors='ignore')
    # flightlist_final_meta['great_circle_distance'] = flightlist_final_meta.apply(
    #     lambda row: haversine(row.get('origin_lon'), row.get('origin_lat'), 
    #                         row.get('dest_lon'), row.get('dest_lat')), axis=1
    # )

    # df_final_raw = df_final_raw.merge(
    #     flightlist_final_meta[['flight_id', 'origin_icao', 'destination_icao', 'great_circle_distance', 
    #                         'aircraft_type', 'takeoff', 'landed', 'origin_lon', 'dest_lon']],
    #     on='flight_id', how='left', suffixes=('', '_meta')
    # )

    # # Add computed columns for final
    # if 'alt_avg_ft' not in df_final_raw.columns:
    #     df_final_raw['alt_avg_ft'] = (df_final_raw.get('alt_start_ft', 0) + df_final_raw.get('alt_end_ft', 0)) / 2
    # if 'altitude_change_rate' not in df_final_raw.columns:
    #     df_final_raw['altitude_change_rate'] = df_final_raw.get('alt_change_ft', 0) / (df_final_raw.get('interval_duration_sec', 60) + 1e-6)
    # if 'end_hour' not in df_final_raw.columns:
    #     df_final_raw['end_hour'] = pd.to_datetime(df_final_raw.get('end'), errors='coerce').dt.hour.fillna(-1).astype(int)
    # if 'interval_elapsed_from_flight_start' not in df_final_raw.columns:
    #     df_final_raw['interval_elapsed_from_flight_start'] = 0

    # # Handle missing features for final
    # missing_final_features = [col for col in feature_cols_selected if col not in df_final_raw.columns]
    # if missing_final_features:
    #     logger.warning(f"Final: Missing {len(missing_final_features)} features - filling with 0")
    #     for col in missing_final_features:
    #         df_final_raw[col] = 0

    # # Preprocess final data
    # X_final_data = df_final_raw[feature_cols_selected].copy()
    # X_final_data = X_final_data.replace([np.inf, -np.inf], np.nan)

    # if numerical_features:
    #     X_final_data[numerical_features] = num_imputer_full.transform(X_final_data[numerical_features])
    # if categorical_features:
    #     X_final_data[categorical_features] = cat_imputer_full.transform(X_final_data[categorical_features])
    #     X_final_data[categorical_features] = cat_encoder_full.transform(X_final_data[categorical_features])

    # X_final_scaled = scaler_full.transform(X_final_data)
    # X_final_sfs = X_final_scaled[:, selected_mask]

    # logger.info(f"[+] Final data ready: {X_final_sfs.shape}")

    # # ========================================================================
    # # CONCATENATE TEST + FINAL
    # # ========================================================================
    # logger.info("\n" + "-"*70)
    # logger.info("COMBINING TEST + FINAL PREDICTIONS")
    # logger.info("-"*70)

    # X_test_sfs = np.vstack([X_test_sfs, X_final_sfs])

    # logger.info(f"[+] Combined data ready: {X_test_sfs.shape}")
    # logger.info(f"    Expected submission rows: {len(submission_template):,}")
    # logger.info(f"    Prediction data rows: {len(X_test_sfs):,}")

    # # Verify row count
    # if len(X_test_sfs) != len(submission_template):
    #     logger.error(f"❌ Row count mismatch!")
    #     logger.error(f"   Test: {len(df_test_raw):,}, Final: {len(df_final_raw):,}, Total: {len(X_test_sfs):,}")
    #     logger.error(f"   Expected: {len(submission_template):,}")
    #     raise ValueError(f"Row count mismatch: {len(X_test_sfs)} != {len(submission_template)}")
    # else:
    #     logger.info("✓ Row counts match!")



    # # ========================================================================
    # # PHASE 8: TRAIN ALL TOP 5 MODELS ON 100% AND GENERATE SUBMISSIONS
    # # ========================================================================
    # logger.info("\n" + "="*70)
    # logger.info("PHASE 8: TRAINING ALL TOP 5 MODELS ON 100% DATA")
    # logger.info("="*70)

    # submission_files = []

    # for idx, row in results_df.iterrows():
    #     rank = row['final_rank']
    #     params = row['params']
        
    #     logger.info("="*70)
    #     logger.info(f"TRAINING MODEL RANK {rank}")
    #     logger.info("="*70)
    #     logger.info(f"Expected Validation RMSE: {row['val_rmse']:.4f} kg")
    #     logger.info(f"Parameters: {params}")
        
    #     # Train model on 100% of data
    #     final_model = XGBRegressor(
    #         random_state=42,
    #         objective='reg:squarederror',
    #         tree_method='hist',
    #         n_jobs=-1,
    #         verbosity=0,
    #         **params
    #     )
        
    #     logger.info("Training on 100% of data...")
    #     final_model.fit(X_full_sfs, y_full_log)
        
    #     # Training performance
    #     full_pred_log = final_model.predict(X_full_sfs)
    #     full_pred = np.expm1(full_pred_log)
    #     full_pred = np.maximum(full_pred, 0.0)
    #     full_rmse = np.sqrt(np.mean((y_full - full_pred) ** 2))
    #     logger.info(f"[+] Training RMSE (100% data): {full_rmse:.4f} kg")
        
    #     # Save parameters
    #     params_file = os.path.join(RESULTS_DIR, f'parameters_rank{rank}.txt')
    #     with open(params_file, 'w') as f:
    #         f.write(f"MODEL RANK {rank}\n")
    #         f.write("="*70 + "\n")
    #         f.write(f"Validation RMSE: {row['val_rmse']:.4f} kg\n")
    #         f.write(f"Validation MAE: {row['val_mae']:.4f} kg\n")
    #         f.write(f"Validation R²: {row['val_r2']:.4f}\n")
    #         f.write(f"Training RMSE (100% data): {full_rmse:.4f} kg\n")
    #         f.write("\nHyperparameters:\n")
    #         for param, value in params.items():
    #             f.write(f"  {param}: {value}\n")
        
    #     # ========================================================================
    #     # GET FEATURE IMPORTANCE
    #     # ========================================================================
    #     feature_importance = final_model.feature_importances_
        
    #     # Create DataFrame with features and their importance
    #     importance_df = pd.DataFrame({
    #         'feature': selected_features,
    #         'importance': feature_importance,
    #         'importance_pct': (feature_importance / feature_importance.sum()) * 100
    #     }).sort_values('importance', ascending=False).reset_index(drop=True)
        
    #     # Add rank column
    #     importance_df['rank'] = range(1, len(importance_df) + 1)
        
    #     logger.info("\n" + "="*70)
    #     logger.info(f"FEATURE IMPORTANCE ANALYSIS - MODEL RANK {rank}")
    #     logger.info("="*70)
        
    #     # Log top 20 most important features
    #     logger.info(f"\nTop 20 Most Important Features:")
    #     logger.info(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'% Total':<10}")
    #     logger.info("-"*75)
    #     for _, imp_row in importance_df.head(20).iterrows():
    #         logger.info(f"{int(imp_row['rank']):<6} {imp_row['feature']:<45} {imp_row['importance']:>12.6f} {imp_row['importance_pct']:>10.2f}%")
        
    #     # Save to CSV
    #     importance_path = os.path.join(RESULTS_DIR, f'feature_importance_rank{rank}.csv')
    #     importance_df.to_csv(importance_path, index=False)
    #     logger.info(f"\n[+] Feature importance saved to: {importance_path}")
        
    #     # Log bottom 10 least important features
    #     logger.info(f"\nBottom 10 Least Important Features:")
    #     logger.info(f"{'Rank':<6} {'Feature':<45} {'Importance':<12} {'% Total':<10}")
    #     logger.info("-"*75)
    #     for _, imp_row in importance_df.tail(10).iterrows():
    #         logger.info(f"{int(imp_row['rank']):<6} {imp_row['feature']:<45} {imp_row['importance']:>12.6f} {imp_row['importance_pct']:>10.2f}%")
        
    #     # Calculate and log cumulative importance
    #     cumulative_importance = importance_df['importance_pct'].cumsum()
    #     n_features_90pct = (cumulative_importance <= 90).sum() + 1
    #     n_features_95pct = (cumulative_importance <= 95).sum() + 1
        
    #     logger.info(f"\n[+] Cumulative Importance Analysis:")
    #     logger.info(f"    Top {n_features_90pct} features explain 90% of importance")
    #     logger.info(f"    Top {n_features_95pct} features explain 95% of importance")
    #     logger.info(f"    Total features: {len(selected_features)}")
    #     logger.info("-"*70)
        
    #     # ========================================================================
    #     # MAKE PREDICTIONS ON TEST + FINAL (SEPARATELY, THEN CONCATENATE)
    #     # ========================================================================
    #     logger.info("\n" + "="*70)
    #     logger.info("MAKING PREDICTIONS ON TEST + FINAL DATASETS")
    #     logger.info("="*70)
        
    #     # Predict on TEST dataset
    #     logger.info("\nPredicting on TEST dataset...")
    #     test_pred_log = final_model.predict(X_test_sfs)
    #     test_pred = np.expm1(test_pred_log)
    #     test_pred = np.maximum(test_pred, 0.0)
    #     logger.info(f"[+] Test predictions: {len(test_pred):,} intervals")
    #     logger.info(f"    Range: [{test_pred.min():.2f}, {test_pred.max():.2f}] kg")
    #     logger.info(f"    Mean: {test_pred.mean():.2f} kg")
    #     logger.info(f"    Std: {test_pred.std():.2f} kg")
        
    #     # Predict on FINAL dataset
    #     logger.info("\nPredicting on FINAL dataset...")
    #     final_pred_log = final_model.predict(X_final_sfs)
    #     final_pred = np.expm1(final_pred_log)
    #     final_pred = np.maximum(final_pred, 0.0)
    #     logger.info(f"[+] Final predictions: {len(final_pred):,} intervals")
    #     logger.info(f"    Range: [{final_pred.min():.2f}, {final_pred.max():.2f}] kg")
    #     logger.info(f"    Mean: {final_pred.mean():.2f} kg")
    #     logger.info(f"    Std: {final_pred.std():.2f} kg")
        
    #     # Concatenate TEST + FINAL predictions
    #     logger.info("\nCombining predictions...")
    #     combined_pred = np.concatenate([test_pred, final_pred])
    #     logger.info(f"[+] Combined predictions: {len(combined_pred):,} intervals")
    #     logger.info(f"    Range: [{combined_pred.min():.2f}, {combined_pred.max():.2f}] kg")
    #     logger.info(f"    Mean: {combined_pred.mean():.2f} kg")
    #     logger.info(f"    Std: {combined_pred.std():.2f} kg")
        
    #     # Create submission
    #     submission_df = submission_template.copy()
    #     submission_df['fuel_kg'] = combined_pred.astype(np.float32)
    #     submission_df = submission_df[['idx', 'flight_id', 'start', 'end', 'fuel_kg']]
        
    #     # Verify row count
    #     if len(submission_df) != len(submission_template):
    #         logger.error(f"❌ Submission row mismatch: {len(submission_df):,} != {len(submission_template):,}")
    #         logger.error(f"   Test predictions: {len(test_pred):,}")
    #         logger.error(f"   Final predictions: {len(final_pred):,}")
    #         logger.error(f"   Combined: {len(combined_pred):,}")
    #         logger.error(f"   Template: {len(submission_template):,}")
    #         raise ValueError("Submission row count doesn't match template")
    #     else:
    #         logger.info("✓ Submission row count matches template!")
        
    #     # Save parquet
    #     parquet_path = os.path.join(RESULTS_DIR, f'submission_rank{rank}_valrmse{row["val_rmse"]:.4f}.parquet')
    #     submission_df.to_parquet(parquet_path, index=False, engine='fastparquet')
    #     logger.info(f"\n[+] Parquet saved: {parquet_path}")
        
    #     # Save CSV
    #     csv_path = os.path.join(RESULTS_DIR, f'submission_rank{rank}_valrmse{row["val_rmse"]:.4f}.csv')
    #     submission_df.to_csv(csv_path, index=False)
    #     logger.info(f"[+] CSV saved: {csv_path}")
        
    #     submission_files.append({
    #         'rank': rank,
    #         'val_rmse': row['val_rmse'],
    #         'train_rmse_100pct': full_rmse,
    #         'parquet_file': parquet_path,
    #         'csv_file': csv_path,
    #         'test_mean': combined_pred.mean(),
    #         'test_std': combined_pred.std(),
    #         'params': params
    #     })

    # # ========================================================================
    # # PHASE 9: FINAL SUMMARY
    # # ========================================================================
    # logger.info("\n" + "="*70)
    # logger.info("PHASE 9: FINAL SUMMARY")
    # logger.info("="*70)

    # summary_df = pd.DataFrame(submission_files)
    # summary_path = os.path.join(RESULTS_DIR, 'top5_models_summary.csv')
    # summary_df.to_csv(summary_path, index=False)

    # logger.info("\n" + "="*70)
    # logger.info("TOP 5 MODELS SUMMARY")
    # logger.info("="*70)
    # logger.info(f"\n{'Rank':<6} {'Val RMSE':<10} {'Train RMSE':<12} {'Test Mean':<11} {'Test Std':<11} {'Submission File':<30}")
    # logger.info("-"*90)

    # for _, row in summary_df.iterrows():
    #     filename = os.path.basename(row['parquet_file'])
    #     logger.info(
    #         f"{row['rank']:<6} {row['val_rmse']:<10.4f} {row['train_rmse_100pct']:<12.4f} "
    #         f"{row['test_mean']:<11.2f} {row['test_std']:<11.2f} {filename:<30}"
    #     )

    # logger.info("\n" + "="*70)
    # logger.info("RECOMMENDATIONS")
    # logger.info("="*70)
    # logger.info(f"Best Model: Rank 1 with Validation RMSE = {summary_df.iloc[0]['val_rmse']:.4f} kg")
    # logger.info(f"File: {os.path.basename(summary_df.iloc[0]['parquet_file'])}")
    # logger.info("\nAll 5 submissions generated successfully!")
    # logger.info(f"Summary saved: {summary_path}")

    # # Save comprehensive summary
    # summary_file = os.path.join(RESULTS_DIR, 'comprehensive_summary.txt')
    # with open(summary_file, 'w') as f:
    #     f.write("="*70 + "\n")
    #     f.write("TOP 5 MODELS - COMPREHENSIVE SUMMARY\n")
    #     f.write("="*70 + "\n")
    #     f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
    #     for _, row in summary_df.iterrows():
    #         f.write("="*70 + "\n")
    #         f.write(f"MODEL RANK {row['rank']}\n")
    #         f.write("="*70 + "\n")
    #         f.write(f"Validation RMSE: {row['val_rmse']:.4f} kg\n")
    #         f.write(f"Training RMSE (100%): {row['train_rmse_100pct']:.4f} kg\n")
    #         f.write(f"Test Predictions Mean: {row['test_mean']:.2f} kg\n")
    #         f.write(f"Test Predictions Std: {row['test_std']:.2f} kg\n")
    #         f.write(f"\nSubmission Files:\n")
    #         f.write(f"  Parquet: {row['parquet_file']}\n")
    #         f.write(f"  CSV: {row['csv_file']}\n")
    #         f.write(f"\nHyperparameters:\n")
    #         for param, value in row['params'].items():
    #             f.write(f"  {param}: {value}\n")
    #         f.write("\n")

    # logger.info(f"[+] Comprehensive summary: {summary_file}")

    # logger.info("\n" + "="*70)
    # logger.info("COMPLETED SUCCESSFULLY!")
    # logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # logger.info("="*70)
    # ========================================================================
    # PHASE 7: LOAD TEST PREDICTIONS + PROCESS FINAL DATA
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 7: LOADING TEST PREDICTIONS + PROCESSING FINAL DATA")
    logger.info("="*70)

    # ========================================================================
    # LOAD PRE-COMPUTED TEST PREDICTIONS FROM PARQUET
    # ========================================================================
    logger.info("\nLoading pre-computed test predictions from parquet...")
    TEST_PREDICTIONS_PARQUET = 'data/bright-lobster_v210.parquet'
    df_test_predictions = pd.read_parquet(TEST_PREDICTIONS_PARQUET)
    logger.info(f"[+] Test predictions loaded: {len(df_test_predictions):,} rows")
    logger.info(f"    Columns: {list(df_test_predictions.columns)}")

    # Extract test predictions (assuming column name is 'fuel_kg' or similar)
    if 'fuel_kg' in df_test_predictions.columns:
        test_pred = df_test_predictions['fuel_kg'].values
    elif 'fuel_kg_pred' in df_test_predictions.columns:
        test_pred = df_test_predictions['fuel_kg_pred'].values
    else:
        # Find the prediction column
        pred_cols = [col for col in df_test_predictions.columns if 'fuel' in col.lower() and col not in ['flight_id', 'idx', 'start', 'end']]
        if pred_cols:
            test_pred = df_test_predictions[pred_cols[0]].values
            logger.info(f"    Using prediction column: {pred_cols[0]}")
        else:
            raise ValueError("Could not find fuel prediction column in test parquet")

    logger.info(f"[+] Test predictions: {len(test_pred):,} intervals")
    logger.info(f"    Range: {test_pred.min():.2f} - {test_pred.max():.2f} kg")
    logger.info(f"    Mean: {test_pred.mean():.2f} kg")

    # ========================================================================
    # PROCESS FINAL DATASET
    # ========================================================================
    logger.info("\n" + "-"*70)
    logger.info("PROCESSING FINAL DATASET")
    logger.info("-"*70)

    logger.info(f"Loading FINAL CSV: {FINAL_CSV_PATH}")
    df_final_raw = pd.read_csv(FINAL_CSV_PATH, delimiter=',', low_memory=False)
    logger.info(f"[+] Final data: {len(df_final_raw):,} rows")

    # Load final intervals from fuel_final
    fuel_final = pd.read_parquet(FUEL_FINAL_PATH)
    fuel_final_intervals = fuel_final[['flight_id', 'idx', 'start', 'end']].copy()
    fuel_final_intervals = fuel_final_intervals.rename(columns={'idx': 'interval_idx'})

    # Merge final data
    df_final_raw = df_final_raw.merge(fuel_final_intervals, on=['flight_id', 'interval_idx'], how='left')

    # Load final featured data if available
    if FEATURED_DATA_FINAL and os.path.exists(FEATURED_DATA_FINAL):
        logger.info(f"Loading featured data for FINAL: {FEATURED_DATA_FINAL}")
        featured_data_final = pd.read_parquet(FEATURED_DATA_FINAL)
        featured_data_final = featured_data_final.rename(columns={'idx': 'interval_idx'})
        df_final_raw = df_final_raw.merge(featured_data_final, on=['flight_id', 'interval_idx'], how='left')

    # Load final flightlist
    flightlist_final = pd.read_parquet(FLIGHTLIST_FINAL_PATH)
    flightlist_final_meta = flightlist_final[['flight_id', 'aircraft_type', 'origin_icao', 'destination_icao', 'takeoff', 'landed']].drop_duplicates(subset=['flight_id'])

    # Add coordinates for final
    flightlist_final_meta = flightlist_final_meta.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    flightlist_final_meta = flightlist_final_meta.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    flightlist_final_meta = flightlist_final_meta.drop(columns=['icao'], errors='ignore')
    flightlist_final_meta = flightlist_final_meta.merge(apt, left_on='destination_icao', right_on='icao', how='left')
    flightlist_final_meta = flightlist_final_meta.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    flightlist_final_meta = flightlist_final_meta.drop(columns=['icao'], errors='ignore')
    flightlist_final_meta['great_circle_distance'] = flightlist_final_meta.apply(
        lambda row: haversine(row.get('origin_lon'), row.get('origin_lat'), 
                            row.get('dest_lon'), row.get('dest_lat')), axis=1
    )

    df_final_raw = df_final_raw.merge(
        flightlist_final_meta[['flight_id', 'origin_icao', 'destination_icao', 'great_circle_distance', 
                        'aircraft_type', 'takeoff', 'landed', 'origin_lon', 'dest_lon']],
        on='flight_id', how='left', suffixes=('', '_meta')
    )

    # ========================================================================
    # Add computed columns for final (with proper column existence checks)
    # ========================================================================
    if 'alt_avg_ft' not in df_final_raw.columns:
        if 'alt_start_ft' in df_final_raw.columns and 'alt_end_ft' in df_final_raw.columns:
            df_final_raw['alt_avg_ft'] = (df_final_raw['alt_start_ft'] + df_final_raw['alt_end_ft']) / 2
        else:
            df_final_raw['alt_avg_ft'] = 0

    if 'altitude_change_rate' not in df_final_raw.columns:
        if 'alt_change_ft' in df_final_raw.columns and 'interval_duration_sec' in df_final_raw.columns:
            df_final_raw['altitude_change_rate'] = df_final_raw['alt_change_ft'] / (df_final_raw['interval_duration_sec'] + 1e-6)
        else:
            df_final_raw['altitude_change_rate'] = 0

    if 'end_hour' not in df_final_raw.columns:
        if 'end' in df_final_raw.columns:
            df_final_raw['end_hour'] = pd.to_datetime(df_final_raw['end'], errors='coerce').dt.hour.fillna(-1).astype(int)
        else:
            df_final_raw['end_hour'] = -1

    if 'interval_elapsed_from_flight_start' not in df_final_raw.columns:
        df_final_raw['interval_elapsed_from_flight_start'] = 0

    # Handle missing features for final
    missing_final_features = [col for col in feature_cols_selected if col not in df_final_raw.columns]
    if missing_final_features:
        logger.warning(f"Final: Missing {len(missing_final_features)} features - filling with 0")
        for col in missing_final_features:
            df_final_raw[col] = 0

    # Preprocess final data
    X_final_data = df_final_raw[feature_cols_selected].copy()
    X_final_data = X_final_data.replace([np.inf, -np.inf], np.nan)

    if numerical_features:
        X_final_data[numerical_features] = num_imputer_full.transform(X_final_data[numerical_features])

    if categorical_features:
        X_final_data[categorical_features] = cat_imputer_full.transform(X_final_data[categorical_features])
        X_final_data[categorical_features] = cat_encoder_full.transform(X_final_data[categorical_features])

    X_final_scaled = scaler_full.transform(X_final_data)
    X_final_sfs = X_final_scaled[:, selected_mask]

    logger.info(f"[+] Final data preprocessed: {X_final_sfs.shape}")



    # ========================================================================
    # PHASE 8: TRAINING BEST MODEL ON 100% DATA + PREDICTIONS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 8: TRAINING BEST MODEL + MAKING FINAL PREDICTIONS")
    logger.info("="*70)

    # Get the best model parameters (rank 1)
    best_model_row = results_df.iloc[0]
    best_params = best_model_row['params']

    logger.info(f"Training best model (Rank 1) on 100% of data...")
    logger.info(f"Expected Validation RMSE: {best_model_row['val_rmse']:.4f} kg")
    logger.info(f"Parameters: {best_params}")

    # ========================================================================
    # SAVE TRAINING DATASET INPUT TO CSV (BEFORE TRAINING)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("SAVING TRAINING DATASET INPUT")
    logger.info("="*70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save the preprocessed training dataset input (X_full_sfs)
    training_input_csv = os.path.join(RESULTS_DIR, f'training_dataset_input_{timestamp}.csv')
    training_input_df = pd.DataFrame(X_full_sfs, columns=selected_features)

    # Add target columns
    training_input_df['target_fuel_kg'] = y_full
    training_input_df['target_fuel_kg_log'] = y_full_log

    training_input_df.to_csv(training_input_csv, index=False)
    logger.info(f"[+] Training dataset input saved: {training_input_csv}")
    logger.info(f"    Shape: {training_input_df.shape}")
    logger.info(f"    Rows: {len(training_input_df):,} intervals")
    logger.info(f"    Columns: {len(selected_features)} features + 2 target columns")

    # Save training feature info
    training_feature_info_csv = os.path.join(RESULTS_DIR, f'training_dataset_feature_info_{timestamp}.csv')
    training_feature_info_df = pd.DataFrame({
        'column_index': range(len(selected_features)),
        'feature_name': selected_features,
        'description': [f'Training feature (column {i} in CSV)' for i in range(len(selected_features))]
    })
    training_feature_info_df.to_csv(training_feature_info_csv, index=False)
    logger.info(f"[+] Training feature info saved: {training_feature_info_csv}")

    # Save training feature list as text
    training_feature_list_txt = os.path.join(RESULTS_DIR, f'training_dataset_features_{timestamp}.txt')
    with open(training_feature_list_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING DATASET INPUT FEATURES\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(training_input_df):,}\n")
        f.write(f"Total features: {len(selected_features)}\n\n")
        f.write("CSV Structure:\n")
        f.write("-"*70 + "\n")
        f.write(f"Columns 0-{len(selected_features)-1}: Features (listed below)\n")
        f.write(f"Column {len(selected_features)}: target_fuel_kg (actual fuel consumption)\n")
        f.write(f"Column {len(selected_features)+1}: target_fuel_kg_log (log-transformed target)\n\n")
        f.write("Feature List (in order):\n")
        f.write("-"*70 + "\n")
        for i, feature in enumerate(selected_features, start=1):
            f.write(f"{i:3d}. {feature}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("This is the exact input used to train the model.\n")
        f.write("The model is trained on the log-transformed target (target_fuel_kg_log).\n")
        f.write("="*70 + "\n")

    logger.info(f"[+] Training feature list text saved: {training_feature_list_txt}")

    # Train final model on 100% of data
    final_model = XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1,
        verbosity=0,
        **best_params
    )

    logger.info("\nTraining on 100% of data...")
    final_model.fit(X_full_sfs, y_full_log)

    # Check training performance
    full_pred_log = final_model.predict(X_full_sfs)
    full_pred = np.expm1(full_pred_log)
    full_pred = np.maximum(full_pred, 0.0)
    full_rmse = np.sqrt(np.mean((y_full - full_pred) ** 2))
    logger.info(f"[+] Training RMSE (100% data): {full_rmse:.4f} kg")

    # ========================================================================
    # SAVE FINAL DATASET INPUT TO CSV (BEFORE PREDICTION)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("SAVING FINAL DATASET INPUT FOR INFERENCE")
    logger.info("="*70)

    # Save the preprocessed final dataset input (X_final_sfs)
    final_input_csv = os.path.join(RESULTS_DIR, f'final_dataset_input_{timestamp}.csv')
    final_input_df = pd.DataFrame(X_final_sfs, columns=selected_features)

    # Add metadata columns from df_final_raw
    if len(df_final_raw) == len(final_input_df):
        final_input_df.insert(0, 'flight_id', df_final_raw['flight_id'].values)
        final_input_df.insert(1, 'interval_idx', df_final_raw['interval_idx'].values)

    final_input_df.to_csv(final_input_csv, index=False)
    logger.info(f"[+] Final dataset input saved: {final_input_csv}")
    logger.info(f"    Shape: {final_input_df.shape}")
    logger.info(f"    Rows: {len(final_input_df):,} intervals")
    logger.info(f"    Columns: {len(final_input_df.columns)} (2 metadata + {len(selected_features)} features)")

    # Save final feature info
    final_feature_info_csv = os.path.join(RESULTS_DIR, f'final_dataset_feature_info_{timestamp}.csv')
    final_feature_info_df = pd.DataFrame({
        'column_index': range(len(selected_features)),
        'feature_name': selected_features,
        'description': [f'Inference feature (column {i+2} in CSV)' for i in range(len(selected_features))]
    })
    final_feature_info_df.to_csv(final_feature_info_csv, index=False)
    logger.info(f"[+] Final feature info saved: {final_feature_info_csv}")

    # Save final feature list as text
    final_feature_list_txt = os.path.join(RESULTS_DIR, f'final_dataset_features_{timestamp}.txt')
    with open(final_feature_list_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FINAL DATASET INPUT FEATURES (FOR INFERENCE)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(final_input_df):,}\n")
        f.write(f"Total features: {len(selected_features)}\n\n")
        f.write("CSV Structure:\n")
        f.write("-"*70 + "\n")
        f.write("Column 0: flight_id (metadata)\n")
        f.write("Column 1: interval_idx (metadata)\n")
        f.write("Columns 2+: Features listed below\n\n")
        f.write("Feature List (in order):\n")
        f.write("-"*70 + "\n")
        for i, feature in enumerate(selected_features, start=1):
            f.write(f"{i:3d}. {feature}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("This is the exact input that goes into the model for prediction.\n")
        f.write("="*70 + "\n")

    logger.info(f"[+] Final feature list text saved: {final_feature_list_txt}")

    # ========================================================================
    # PREDICT ON FINAL DATASET
    # ========================================================================
    logger.info("\nPredicting on FINAL dataset...")
    final_pred_log = final_model.predict(X_final_sfs)
    final_pred = np.expm1(final_pred_log)
    final_pred = np.maximum(final_pred, 0.0)

    logger.info(f"[+] Final predictions: {len(final_pred):,} intervals")
    logger.info(f"    Range: {final_pred.min():.2f} - {final_pred.max():.2f} kg")
    logger.info(f"    Mean: {final_pred.mean():.2f} kg")
    logger.info(f"    Std: {final_pred.std():.2f} kg")

    # ========================================================================
    # COMBINE TEST + FINAL PREDICTIONS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("COMBINING TEST + FINAL PREDICTIONS")
    logger.info("="*70)

    # Get the test flight IDs from the loaded predictions
    test_flight_ids = set(df_test_predictions['flight_id'].unique())
    logger.info(f"[+] Test dataset has {len(test_flight_ids)} unique flights")

    # Split fuel_final_intervals into test and final parts
    fuel_final_intervals_with_flag = fuel_final_intervals.copy()
    fuel_final_intervals_with_flag['is_test'] = fuel_final_intervals_with_flag['flight_id'].isin(test_flight_ids)

    test_intervals = fuel_final_intervals_with_flag[fuel_final_intervals_with_flag['is_test']].copy()
    final_intervals = fuel_final_intervals_with_flag[~fuel_final_intervals_with_flag['is_test']].copy()

    logger.info(f"[+] Test intervals: {len(test_intervals):,} rows")
    logger.info(f"[+] Final intervals: {len(final_intervals):,} rows")
    logger.info(f"[+] Total: {len(test_intervals) + len(final_intervals):,} rows")

    # Create test submission dataframe
    df_test_submission = test_intervals.copy()
    df_test_submission = df_test_submission.rename(columns={'interval_idx': 'idx'})
    df_test_submission['fuel_kg'] = test_pred
    df_test_submission = df_test_submission[['idx', 'flight_id', 'start', 'end', 'fuel_kg']]

    # Create final submission dataframe
    df_final_submission = final_intervals.copy()
    df_final_submission = df_final_submission.rename(columns={'interval_idx': 'idx'})
    df_final_submission['fuel_kg'] = final_pred
    df_final_submission = df_final_submission[['idx', 'flight_id', 'start', 'end', 'fuel_kg']]

    logger.info(f"[+] Test submission shape: {df_test_submission.shape}")
    logger.info(f"[+] Final submission shape: {df_final_submission.shape}")

    # Verify lengths match
    if len(df_test_submission) != len(test_pred):
        logger.error(f"Length mismatch! Test intervals: {len(df_test_submission)}, predictions: {len(test_pred)}")
        raise ValueError("Test submission length mismatch")
        
    if len(df_final_submission) != len(final_pred):
        logger.error(f"Length mismatch! Final intervals: {len(df_final_submission)}, predictions: {len(final_pred)}")
        raise ValueError("Final submission length mismatch")

    # Combine: TEST first, then FINAL
    combined_submission = pd.concat([df_test_submission, df_final_submission], ignore_index=True)
    logger.info(f"[+] Combined submission: {len(combined_submission):,} rows")
    logger.info(f"    Column order: {list(combined_submission.columns)}")

    # Save combined submission as parquet
    combined_parquet_path = os.path.join(RESULTS_DIR, f'combined_test_final_{timestamp}.parquet')
    combined_submission.to_parquet(combined_parquet_path, index=False)
    logger.info(f"[+] Combined parquet saved: {combined_parquet_path}")

    # Also save as CSV for verification
    combined_csv_path = os.path.join(RESULTS_DIR, f'combined_test_final_{timestamp}.csv')
    combined_submission.to_csv(combined_csv_path, index=False)
    logger.info(f"[+] Combined CSV saved: {combined_csv_path}")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Test predictions:  {len(test_pred):,} intervals")
    logger.info(f"  Mean: {test_pred.mean():.2f} kg, Std: {test_pred.std():.2f} kg")
    logger.info(f"  Range: [{test_pred.min():.2f}, {test_pred.max():.2f}] kg")
    logger.info("")
    logger.info(f"Final predictions: {len(final_pred):,} intervals")
    logger.info(f"  Mean: {final_pred.mean():.2f} kg, Std: {final_pred.std():.2f} kg")
    logger.info(f"  Range: [{final_pred.min():.2f}, {final_pred.max():.2f}] kg")
    logger.info("")
    logger.info(f"Combined total:    {len(combined_submission):,} intervals")
    logger.info(f"  Mean: {combined_submission['fuel_kg'].mean():.2f} kg")
    logger.info(f"  Std: {combined_submission['fuel_kg'].std():.2f} kg")
    logger.info("")
    logger.info("Files saved:")
    logger.info(f"  TRAINING DATA:")
    logger.info(f"    - Input CSV: {training_input_csv}")
    logger.info(f"    - Feature info: {training_feature_info_csv}")
    logger.info(f"    - Feature list: {training_feature_list_txt}")
    logger.info(f"  FINAL DATA (INFERENCE):")
    logger.info(f"    - Input CSV: {final_input_csv}")
    logger.info(f"    - Feature info: {final_feature_info_csv}")
    logger.info(f"    - Feature list: {final_feature_list_txt}")
    logger.info(f"  PREDICTIONS:")
    logger.info(f"    - Combined parquet: {combined_parquet_path}")
    logger.info(f"    - Combined CSV: {combined_csv_path}")
    logger.info("="*70)
    logger.info("COMPLETED SUCCESSFULLY!")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)






if __name__ == "__main__":
    main()
