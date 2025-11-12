"""
This script handles the model training stage of the ML pipeline using XGBoost.

It can use Sequential Feature Selection (SFS) in forward or backward mode,
or be configured to use all available features.
"""
import os
import pandas as pd
import numpy as np
import datetime
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import config
import json
import logging
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import argparse

# Initialize module logger
logger = logging.getLogger(__name__)


AIRCRAFT_DATA = {
    'A19N': {'mtow_kg': 79200, 'oew_kg': 45100, 'max_fuel_kg': 27200, 'type': 'narrowbody'},
    'A20N': {'mtow_kg': 79000, 'oew_kg': 45400, 'max_fuel_kg': 27200, 'type': 'narrowbody'},
    'A21N': {'mtow_kg': 97000, 'oew_kg': 50300, 'max_fuel_kg': 32840, 'type': 'narrowbody'},
    'A318': {'mtow_kg': 68000, 'oew_kg': 39500, 'max_fuel_kg': 24210, 'type': 'narrowbody'},
    'A319': {'mtow_kg': 75500, 'oew_kg': 40800, 'max_fuel_kg': 24210, 'type': 'narrowbody'},
    'A320': {'mtow_kg': 78000, 'oew_kg': 42400, 'max_fuel_kg': 27200, 'type': 'narrowbody'},
    'A321': {'mtow_kg': 93500, 'oew_kg': 48700, 'max_fuel_kg': 32840, 'type': 'narrowbody'},
    'A332': {'mtow_kg': 233000, 'oew_kg': 119500, 'max_fuel_kg': 139090, 'type': 'widebody'},
    'A333': {'mtow_kg': 242000, 'oew_kg': 123400, 'max_fuel_kg': 139090, 'type': 'widebody'},
    'A343': {'mtow_kg': 275000, 'oew_kg': 131000, 'max_fuel_kg': 155040, 'type': 'widebody'},
    'A359': {'mtow_kg': 280000, 'oew_kg': 142400, 'max_fuel_kg': 138000, 'type': 'widebody'},
    'A388': {'mtow_kg': 575000, 'oew_kg': 277000, 'max_fuel_kg': 320000, 'type': 'widebody'},
    'B37M': {'mtow_kg': 82200, 'oew_kg': 45100, 'max_fuel_kg': 26020, 'type': 'narrowbody'},
    'B38M': {'mtow_kg': 82200, 'oew_kg': 45100, 'max_fuel_kg': 26020, 'type': 'narrowbody'},
    'B39M': {'mtow_kg': 88300, 'oew_kg': 46550, 'max_fuel_kg': 26020, 'type': 'narrowbody'},
    'B3XM': {'mtow_kg': 89800, 'oew_kg': 50300, 'max_fuel_kg': 26020, 'type': 'narrowbody'},
    'B734': {'mtow_kg': 68000, 'oew_kg': 38300, 'max_fuel_kg': 26020, 'type': 'narrowbody'},
    'B737': {'mtow_kg': 70100, 'oew_kg': 39800, 'max_fuel_kg': 28600, 'type': 'narrowbody'},
    'B738': {'mtow_kg': 79000, 'oew_kg': 41413, 'max_fuel_kg': 28600, 'type': 'narrowbody'},
    'B739': {'mtow_kg': 85100, 'oew_kg': 44676, 'max_fuel_kg': 30190, 'type': 'narrowbody'},
    'B744': {'mtow_kg': 412775, 'oew_kg': 178100, 'max_fuel_kg': 216840, 'type': 'widebody'},
    'B748': {'mtow_kg': 447700, 'oew_kg': 197130, 'max_fuel_kg': 243120, 'type': 'widebody'},
    'B752': {'mtow_kg': 115680, 'oew_kg': 58390, 'max_fuel_kg': 52300, 'type': 'narrowbody'},
    'B763': {'mtow_kg': 186880, 'oew_kg': 90010, 'max_fuel_kg': 91380, 'type': 'widebody'},
    'B772': {'mtow_kg': 297560, 'oew_kg': 145150, 'max_fuel_kg': 171170, 'type': 'widebody'},
    'B773': {'mtow_kg': 351530, 'oew_kg': 167830, 'max_fuel_kg': 181280, 'type': 'widebody'},
    'B77W': {'mtow_kg': 351530, 'oew_kg': 167830, 'max_fuel_kg': 181280, 'type': 'widebody'},
    'B788': {'mtow_kg': 227930, 'oew_kg': 119950, 'max_fuel_kg': 126210, 'type': 'widebody'},
    'B789': {'mtow_kg': 254010, 'oew_kg': 128850, 'max_fuel_kg': 126370, 'type': 'widebody'},
    'E145': {'mtow_kg': 22000, 'oew_kg': 12400, 'max_fuel_kg': 6200, 'type': 'narrowbody'},
    'E170': {'mtow_kg': 38600, 'oew_kg': 21620, 'max_fuel_kg': 11187, 'type': 'narrowbody'},
    'E190': {'mtow_kg': 51800, 'oew_kg': 29540, 'max_fuel_kg': 15200, 'type': 'narrowbody'},
    'E195': {'mtow_kg': 52290, 'oew_kg': 29100, 'max_fuel_kg': 15200, 'type': 'narrowbody'},
    'E75L': {'mtow_kg': 39380, 'oew_kg': 22010, 'max_fuel_kg': 10300, 'type': 'narrowbody'},
    'C550': {'mtow_kg': 9072, 'oew_kg': 5125, 'max_fuel_kg': 3619, 'type': 'narrowbody'},
    'GLF6': {'mtow_kg': 45360, 'oew_kg': 24040, 'max_fuel_kg': 18600, 'type': 'widebody'},
}

log_file = os.path.join( f'xgboost_fuel_openap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

def get_aircraft_specs(aircraft_type):
    if pd.isna(aircraft_type):
        return None, None, 'unknown'
    if aircraft_type in AIRCRAFT_DATA:
        data = AIRCRAFT_DATA[aircraft_type]
        return data['mtow_kg'], data.get('oew_kg'), data.get('type', 'unknown')
    else:
        return None, None, 'unknown'

def extract_datetime_features_vectorized(timestamps):
    """Vectorized datetime extraction - much faster than apply()"""
    dt = pd.to_datetime(timestamps, errors='coerce')
    return (
        dt.dt.strftime('%Y-%m-%d'),
        dt.dt.strftime('%H:%M:%S'),
        dt.dt.hour.fillna(-1).astype(int),
        dt.dt.month.fillna(-1).astype(int)
    )


def engineer_features(df, label_encoders=None, fit_encoders=False, is_test=False):
    """Engineer features - OPTIMIZED WITH VECTORIZED DATETIME"""
    df_eng = df.copy()

    if fit_encoders and label_encoders is None:
        label_encoders = {}

    if 'aircraft' not in df_eng.columns:
        logger.warning("No 'aircraft' column found.")
        df_eng['aircraft'] = 'UNKNOWN'

    if fit_encoders:
        le_aircraft = LabelEncoder()
        df_eng['aircraft_encoded'] = le_aircraft.fit_transform(df_eng['aircraft'].astype(str))
        label_encoders['aircraft'] = le_aircraft
    else:
        le_aircraft = label_encoders.get('aircraft')
        if le_aircraft:
            df_eng['aircraft_encoded'] = df_eng['aircraft'].apply(
                lambda x: le_aircraft.transform([str(x)])[0] if str(x) in le_aircraft.classes_ else -1
            )
        else:
            df_eng['aircraft_encoded'] = pd.Categorical(df_eng['aircraft']).codes

    if 'starting_mass_kg' not in df_eng.columns:
        df_eng['starting_mass_kg'] = 50000

    df_eng['mtow_kg'] = df_eng['aircraft'].apply(lambda a: get_aircraft_specs(a)[0])



    df_eng['alt_start_ft'] = df_eng.get('alt_start_ft', 0)
    df_eng['alt_end_ft'] = df_eng.get('alt_end_ft', 0)
    df_eng['alt_change_ft'] = df_eng.get('alt_change_ft', 0)
    df_eng['alt_avg_ft'] = (df_eng['alt_start_ft'] + df_eng['alt_end_ft']) / 2
    df_eng['gs_avg_kts'] = df_eng.get('gs_avg_kts', 0)
    df_eng['vs_avg_fpm'] = df_eng.get('vs_avg_fpm', 0)

    if 'phase' in df_eng.columns:
        df_eng['is_climb'] = (df_eng['phase'].str.upper() == 'CLIMB').astype(int)
        df_eng['is_descent'] = (df_eng['phase'].str.upper() == 'DESCENT').astype(int)
        df_eng['is_cruise'] = (df_eng['phase'].str.upper() == 'CRUISE').astype(int)
        df_eng['is_on_ground'] = (df_eng['phase'].str.upper() == 'ON_GROUND').astype(int)
    else:
        df_eng['is_climb'] = 0
        df_eng['is_descent'] = 0
        df_eng['is_cruise'] = 0
        df_eng['is_on_ground'] = 0

    df_eng['interval_duration_sec'] = df_eng.get('interval_duration_sec', 60)
    df_eng['altitude_change_rate'] = df_eng['alt_change_ft'] / (df_eng['interval_duration_sec'] + 1e-6)

    if 'origin_icao' not in df_eng.columns:
        df_eng['origin_icao'] = 'UNKNOWN'

    if fit_encoders:
        le_origin = LabelEncoder()
        df_eng['origin_icao_encoded'] = le_origin.fit_transform(df_eng['origin_icao'].astype(str))
        label_encoders['origin_icao'] = le_origin
    else:
        le_origin = label_encoders.get('origin_icao')
        if le_origin:
            df_eng['origin_icao_encoded'] = df_eng['origin_icao'].apply(
                lambda x: le_origin.transform([str(x)])[0] if str(x) in le_origin.classes_ else -1
            )
        else:
            df_eng['origin_icao_encoded'] = pd.Categorical(df_eng['origin_icao']).codes

    if 'destination_icao' not in df_eng.columns:
        df_eng['destination_icao'] = 'UNKNOWN'

    if fit_encoders:
        le_dest = LabelEncoder()
        df_eng['destination_icao_encoded'] = le_dest.fit_transform(df_eng['destination_icao'].astype(str))
        label_encoders['destination_icao'] = le_dest
    else:
        le_dest = label_encoders.get('destination_icao')
        if le_dest:
            df_eng['destination_icao_encoded'] = df_eng['destination_icao'].apply(
                lambda x: le_dest.transform([str(x)])[0] if str(x) in le_dest.classes_ else -1
            )
        else:
            df_eng['destination_icao_encoded'] = pd.Categorical(df_eng['destination_icao']).codes

    if 'great_circle_distance' not in df_eng.columns:
        df_eng['great_circle_distance'] = np.nan

    df_eng['flight_duration_hours'] = df_eng['interval_duration_sec'] / 3600

    if 'aircraft_type' not in df_eng.columns:
        df_eng['aircraft_type'] = df_eng['aircraft'].apply(lambda a: get_aircraft_specs(a)[2])

    # VECTORIZED datetime extraction - MUCH FASTER
    logger.info("Extracting datetime features (vectorized)...")
    if 'start' in df_eng.columns:
        start_date, start_time, start_hour, start_month = extract_datetime_features_vectorized(df_eng['start'])
        df_eng['start_date'] = start_date
        df_eng['start_time'] = start_time
        df_eng['start_hour'] = start_hour
        df_eng['start_month'] = start_month
    else:
        df_eng['start_date'] = np.nan
        df_eng['start_time'] = np.nan
        df_eng['start_hour'] = np.nan
        df_eng['start_month'] = np.nan

    if 'end' in df_eng.columns:
        end_date, end_time, end_hour, end_month = extract_datetime_features_vectorized(df_eng['end'])
        df_eng['end_date'] = end_date
        df_eng['end_time'] = end_time
        df_eng['end_hour'] = end_hour
        df_eng['end_month'] = end_month
    else:
        df_eng['end_date'] = np.nan
        df_eng['end_time'] = np.nan
        df_eng['end_hour'] = np.nan
        df_eng['end_month'] = np.nan

    # Encode datetime strings
    if fit_encoders:
        le_start_date = LabelEncoder()
        df_eng['start_date_encoded'] = le_start_date.fit_transform(df_eng['start_date'].astype(str))
        label_encoders['start_date'] = le_start_date

        le_start_time = LabelEncoder()
        df_eng['start_time_encoded'] = le_start_time.fit_transform(df_eng['start_time'].astype(str))
        label_encoders['start_time'] = le_start_time

        le_end_date = LabelEncoder()
        df_eng['end_date_encoded'] = le_end_date.fit_transform(df_eng['end_date'].astype(str))
        label_encoders['end_date'] = le_end_date

        le_end_time = LabelEncoder()
        df_eng['end_time_encoded'] = le_end_time.fit_transform(df_eng['end_time'].astype(str))
        label_encoders['end_time'] = le_end_time
    else:
        # For test data, handle unseen values
        if is_test:
            oe_start_date = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
            df_eng['start_date_encoded'] = oe_start_date.fit_transform(df_eng[['start_date']]).astype(int).squeeze()

            oe_start_time = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
            df_eng['start_time_encoded'] = oe_start_time.fit_transform(df_eng[['start_time']]).astype(int).squeeze()

            oe_end_date = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
            df_eng['end_date_encoded'] = oe_end_date.fit_transform(df_eng[['end_date']]).astype(int).squeeze()

            oe_end_time = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
            df_eng['end_time_encoded'] = oe_end_time.fit_transform(df_eng[['end_time']]).astype(int).squeeze()
        else:
            le_start_date = label_encoders.get('start_date')
            if le_start_date:
                # Safe transform: unseen labels mapped to -1
                known = set(le_start_date.classes_)
                df_eng['start_date_encoded'] = df_eng['start_date'].apply(
                    lambda x: le_start_date.transform([str(x)])[0] if str(x) in known else -1
                )
            else:
                df_eng['start_date_encoded'] = pd.Categorical(df_eng['start_date']).codes

            le_start_time = label_encoders.get('start_time')
            if le_start_time:
                known = set(le_start_time.classes_)
                df_eng['start_time_encoded'] = df_eng['start_time'].apply(
                    lambda x: le_start_time.transform([str(x)])[0] if str(x) in known else -1
                )
            else:
                df_eng['start_time_encoded'] = pd.Categorical(df_eng['start_time']).codes

            le_end_date = label_encoders.get('end_date')
            if le_end_date:
                known = set(le_end_date.classes_)
                df_eng['end_date_encoded'] = df_eng['end_date'].apply(
                    lambda x: le_end_date.transform([str(x)])[0] if str(x) in known else -1
                )
            else:
                df_eng['end_date_encoded'] = pd.Categorical(df_eng['end_date']).codes

            le_end_time = label_encoders.get('end_time')
            if le_end_time:
                known = set(le_end_time.classes_)
                df_eng['end_time_encoded'] = df_eng['end_time'].apply(
                    lambda x: le_end_time.transform([str(x)])[0] if str(x) in known else -1
                )
            else:
                df_eng['end_time_encoded'] = pd.Categorical(df_eng['end_time']).codes

    # Convert zero values to NaN for gs_avg_kts and vs_avg_fpm
    df_eng.loc[df_eng['gs_avg_kts'] == 0, 'gs_avg_kts'] = np.nan
    df_eng.loc[df_eng['vs_avg_fpm'] == 0, 'vs_avg_fpm'] = np.nan

    return df_eng, label_encoders


def train(params_path=None, feature_selection_method='sfs_forward'):
    """Main training function.

    Args:
        params_path (str, optional): Path to a JSON file with hyperparameters.
        feature_selection_method (str): 'sfs_forward', 'sfs_backward', or 'none'.
    """
    print("--- Starting XGBoost Model Training Stage ---")
    print(f"Feature Selection Method: {feature_selection_method}")
    # --- 1. Load Data ---
    if config.TEST_RUN:
        print(f"Test Run:")
        featured_data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data_test.parquet")
    else:
        featured_data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data.parquet")

    if not os.path.exists(featured_data_path):
        print(f"Error: Featured data not found at {featured_data_path}. Please run augment_features.py first.")
        return

    print(f"Loading featured data from {featured_data_path}...")
    df_featured = pd.read_parquet(featured_data_path)
    print(f"Loaded  {featured_data_path} with {len(df_featured)} rows and {len(df_featured.columns)} columns. ")

    df_eng, label_encoders = engineer_features(df_featured, fit_encoders=True)

    # Define target and construct feature list excluding the target
    target_col = 'fuel_kg'
    feature_cols_all = [c for c in df_eng.columns.tolist() if c != target_col]
    # Deduplicate feature columns while preserving order
    feature_cols_all = list(dict.fromkeys(feature_cols_all))

    df_features = df_eng[feature_cols_all + [target_col]].copy()
    df_features = df_features.dropna(subset=[target_col])
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # CRITICAL: Store openap_original AFTER filtering to match df_features length


    X_train_full = df_features[feature_cols_all]
    y_train_full = df_features[target_col].values.astype(np.float32)

    logger.info(f"[+] Training data prepared: {len(df_features):,} intervals")

    logger.info("/nPHASE 3: DATA IMPUTATION")
    y_train_log = np.log1p(y_train_full)

    X_train_imputed = X_train_full.copy()

    # --- 2. Prepare Data for Training ---
    numerical_features =  X_train_imputed.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train_imputed.select_dtypes(exclude=np.number).columns.tolist()

    # Holders for fitted preprocessors
    num_imputer = None
    cat_imputer = None
    cat_encoder = None

    # Drop columns that are entirely NaN before imputation to avoid sklearn warnings and shape mismatches
    drop_all_nan_num = [c for c in numerical_features if X_train_imputed[c].isna().all()]
    if drop_all_nan_num:
        logger.warning(f"Dropping all-NaN numeric features before imputation: {drop_all_nan_num}")
        X_train_imputed.drop(columns=drop_all_nan_num, inplace=True)
        numerical_features = [c for c in numerical_features if c not in drop_all_nan_num]

    drop_all_nan_cat = [c for c in categorical_features if X_train_imputed[c].isna().all()]
    if drop_all_nan_cat:
        logger.warning(f"Dropping all-NaN categorical features before imputation: {drop_all_nan_cat}")
        X_train_imputed.drop(columns=drop_all_nan_cat, inplace=True)
        categorical_features = [c for c in categorical_features if c not in drop_all_nan_cat]

    if numerical_features:
        num_imputer = SimpleImputer(strategy='mean')
        X_train_arr_num = num_imputer.fit_transform(X_train_imputed[numerical_features])
        # Assign back using the same column order
        X_train_imputed.loc[:, numerical_features] = X_train_arr_num
        logger.info(f"[+] Fitted numerical imputer on {len(numerical_features)} features")

    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_arr_cat = cat_imputer.fit_transform(X_train_imputed[categorical_features])
        X_train_imputed.loc[:, categorical_features] = X_train_arr_cat

        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float64)
        X_train_arr_cat_enc = cat_encoder.fit_transform(X_train_imputed[categorical_features])
        X_train_imputed.loc[:, categorical_features] = X_train_arr_cat_enc
        logger.info(f"[+] Fitted categorical imputer and encoder on {len(categorical_features)} features")

    # Optional export of training inputs (without external openap reference)
    try:
        train_export = X_train_imputed.copy()
        train_export['actual_fuel_kg'] = y_train_full
        train_export['log_fuel_kg'] = y_train_log
        train_csv_path = os.path.join('xgboost_training_input.csv')
        train_export.to_csv(train_csv_path, index=False)
        logger.info(f"[+] Training CSV: {train_csv_path} ({len(train_export):,} rows)")
    except Exception as e:
        logger.warning(f"Could not export training CSV: {e}")

    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train_imputed)
    # Convert back to DataFrame to preserve column names for downstream steps
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train_imputed.columns, index=X_train_imputed.index)
    logger.info(f"[+] Scaler fitted")

    # Capture preprocessing spec for inference reproducibility
    imputed_feature_order = X_train_imputed.columns.tolist()
    scaled_feature_order = X_train_scaled.columns.tolist()


    # --- 3. Feature Selection ---
    if feature_selection_method in ['sfs_forward', 'sfs_backward']:
        direction = 'forward' if feature_selection_method == 'sfs_forward' else 'backward'
        print(f"\n--- Starting Feature Selection with SequentialFeatureSelector ({direction.capitalize()}) ---")
        xgb_for_sfs = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
        
        sfs = SequentialFeatureSelector(
            estimator=xgb_for_sfs,
            n_features_to_select='auto',
            direction=direction,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        
        print(f"Fitting SFS ({direction}) to find optimal features (this may take a while)...")
        sfs.fit(X_train_scaled, y_train_log)
        
        selected_features = list(X_train_scaled.columns[sfs.get_support()])
        print(f"SFS selected {len(selected_features)} features out of {len(X_train_scaled.columns)}.")
    
    elif feature_selection_method == 'none':
        print("\n--- Skipping feature selection. Using all available features. ---")
        selected_features = X_train_scaled.columns.tolist()
    
    else:
        raise ValueError(f"Invalid feature_selection_method: '{feature_selection_method}'. Choose 'sfs_forward', 'sfs_backward', or 'none'.")

    # Ensure selected features are unique and present
    selected_features = [c for c in list(dict.fromkeys(selected_features)) if c in X_train_scaled.columns]
    print("Final features to be used:", selected_features)
    X = X_train_scaled[selected_features]
    # Safety: drop any duplicate-named columns in X to avoid xgboost pandas transformer error
    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].unique().tolist()
        logger.warning(f"Dropping duplicate feature columns before training: {dupes}")
        X = X.loc[:, ~X.columns.duplicated()]
    # Define target vector for training and potential validation
    y = y_train_log

    # --- 4. Train Final Model ---
    if config.TEST_RUN:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Test Run: Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
        print(f"Full Run: Training on the entire dataset of {len(X_train)} samples.")

    model_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42, 'n_jobs': -1}
    model_suffix = f"_{feature_selection_method}"

    if params_path:
        try:
            with open(params_path, 'r') as f:
                tuned_params = json.load(f)
            model_params.update(tuned_params)
            model_suffix += "_tuned"
            print(f"Using tuned parameters from {params_path}: {tuned_params}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load params from {params_path}. Using defaults. Error: {e}")

    print("Training final XGBoost Regressor model...")
    xgb_reg = xgb.XGBRegressor(**model_params)
    xgb_reg.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    if X_val is not None and y_val is not None:
        print("\n--- Validation Results ---")
        y_pred = xgb_reg.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
        print(f"R-squared (R²): {r2:.4f}")
    else:
        print("\n--- Full Run Training Complete. No validation metrics to display. ---")

    # --- 6. Feature Importance ---
    print("\n--- Feature Importance ---")
    importances = xgb_reg.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': selected_features, 'importance': importances}).sort_values(by='importance', ascending=False)
    print(feature_importance_df.head(15))

    # --- 7. Save Model and Artifacts ---
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, f"xgb_model{model_suffix}_{run_id}")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(xgb_reg, os.path.join(model_dir, "model.joblib"))
    print(f"\nModel saved to {os.path.join(model_dir, 'model.joblib')}")

    with open(os.path.join(model_dir, "selected_features.json"), 'w') as f:
        json.dump(selected_features, f)
    print(f"Feature list saved to {os.path.join(model_dir, 'selected_features.json')}")

    # Persist preprocessing artifacts for inference-time reuse
    try:
        preprocessors = {
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'cat_encoder': cat_encoder,
            'scaler': scaler,
            'label_encoders': label_encoders or {}
        }
        joblib.dump(preprocessors, os.path.join(model_dir, "preprocessors.joblib"))
        spec = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'imputed_feature_order': imputed_feature_order,
            'scaled_feature_order': scaled_feature_order
        }
        with open(os.path.join(model_dir, "preprocessing_spec.json"), 'w') as sf:
            json.dump(spec, sf)
        logger.info(f"[+] Saved preprocessing artifacts to {model_dir}")
    except Exception as e:
        logger.warning(f"Failed to save preprocessing artifacts: {e}")

    X_train.to_parquet(os.path.join(model_dir, "X_train.parquet"))
    # Save targets as single-column DataFrames to ensure Parquet compatibility across pandas versions
    try:
        pd.DataFrame({'y': y_train}).to_parquet(os.path.join(model_dir, "y_train.parquet"))
    except Exception as e:
        logger.warning(f"Failed to save y_train as Parquet due to: {e}. Falling back to CSV.")
        pd.DataFrame({'y': y_train}).to_csv(os.path.join(model_dir, "y_train.csv"), index=False)

    if X_val is not None:
        X_val.to_parquet(os.path.join(model_dir, "X_val.parquet"))
        try:
            pd.DataFrame({'y': y_val}).to_parquet(os.path.join(model_dir, "y_val.parquet"))
        except Exception as e:
            logger.warning(f"Failed to save y_val as Parquet due to: {e}. Falling back to CSV.")
            pd.DataFrame({'y': y_val}).to_csv(os.path.join(model_dir, "y_val.csv"), index=False)
    print(f"Datasets saved to {model_dir}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, max(8, len(selected_features) * 0.4)))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
    ax.set_title(f"Top 20 Feature Importance for {run_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "feature_importance.png"))
    print(f"Feature importance plot saved to {os.path.join(model_dir, 'feature_importance.png')}")

    print("--- XGBoost Model Training Stage Complete ---")

if __name__ == '__main__':
    # This block is now primarily for direct testing of the train function,
    # as the main entry point is run_pipeline.py
    parser = argparse.ArgumentParser(description="XGBoost Model Training with Feature Selection")
    parser.add_argument('--params', type=str, help="Path to a JSON file with model hyperparameters.")
    parser.add_argument('--feature_selection', type=str, default='sfs_forward', 
                        choices=['sfs_forward', 'sfs_backward', 'none'],
                        help="Feature selection method: 'sfs_forward', 'sfs_backward', or 'none'.")
    
    args = parser.parse_args()
    
    train(params_path=args.params, feature_selection_method=args.feature_selection)
