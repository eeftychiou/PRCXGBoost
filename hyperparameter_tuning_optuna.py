import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
import os
import joblib
import json
import logging
from datetime import datetime
import config

# Setup logging
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', f'optuna_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'device': 'cuda' if trial.suggest_categorical('use_gpu', [True]) else 'cpu',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    preds = model.predict(X_val)
    # Convert back from log-scale for RMSE comparison
    rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(preds)))
    return rmse

def main():
    logger.info("Starting Deep Hyperparameter Optimization with Optuna")
    
    # Load data — prefer augmented CSV to match training column schema
    aug_path = config.AUGMENTED_FINAL_CSV
    data_path = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
    
    if os.path.exists(aug_path):
        logger.info(f"Loading augmented training CSV: {aug_path}")
        df = pd.read_csv(aug_path, delimiter=';', low_memory=False)
        if 'actual_fuel_kg' in df.columns and 'fuel_kg' not in df.columns:
            df = df.rename(columns={'actual_fuel_kg': 'fuel_kg'})
        # Merge parquet extended features
        if os.path.exists(data_path):
            featured = pd.read_parquet(data_path)
            if 'idx' in featured.columns:
                featured = featured.rename(columns={'idx': 'interval_idx'})
            extra_cols = ['flight_id', 'interval_idx'] + [
                c for c in featured.columns if c not in df.columns and c != 'idx'
            ]
            df = df.merge(featured[[c for c in extra_cols if c in featured.columns]],
                          on=['flight_id', 'interval_idx'], how='left')

        # --- Compute derived columns that training script builds on-the-fly ---
        if 'alt_avg_ft' not in df.columns:
            if 'alt_start_ft' in df.columns and 'alt_end_ft' in df.columns:
                df['alt_avg_ft'] = (df['alt_start_ft'] + df['alt_end_ft']) / 2
            elif 'alt_end_ft' in df.columns:
                df['alt_avg_ft'] = df['alt_end_ft']

        if 'altitude_change_rate' not in df.columns:
            if 'alt_change_ft' in df.columns and 'interval_duration_sec' in df.columns:
                df['altitude_change_rate'] = df['alt_change_ft'] / (df['interval_duration_sec'] + 1e-6)
            else:
                df['altitude_change_rate'] = 0.0

        if 'great_circle_distance' not in df.columns:
            try:
                from math import radians, cos, sin, asin, sqrt
                def _haversine_opt(lon1, lat1, lon2, lat2):
                    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [lon1, lat1, lon2, lat2]):
                        return np.nan
                    R = 6371
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                    dlon, dlat = lon2 - lon1, lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    return 2 * R * asin(sqrt(a))
                apt = pd.read_parquet(config.APT_PARQUET)[['icao', 'longitude', 'latitude']]
                fl = pd.read_parquet(config.FLIGHTLIST_TRAIN)[['flight_id', 'origin_icao', 'destination_icao']]
                fl = fl.merge(apt.rename(columns={'icao': 'origin_icao', 'longitude': 'origin_lon', 'latitude': 'origin_lat'}), on='origin_icao', how='left')
                fl = fl.merge(apt.rename(columns={'icao': 'destination_icao', 'longitude': 'dest_lon', 'latitude': 'dest_lat'}), on='destination_icao', how='left')
                fl['great_circle_distance'] = fl.apply(lambda r: _haversine_opt(r.get('origin_lon'), r.get('origin_lat'), r.get('dest_lon'), r.get('dest_lat')), axis=1)
                df = df.merge(fl[['flight_id', 'great_circle_distance']], on='flight_id', how='left')
            except Exception as e:
                logger.warning(f"Could not compute great_circle_distance: {e}. Filling with 0.")
                df['great_circle_distance'] = 0.0

        if 'interval_elapsed_from_flight_start' not in df.columns:
            if 'start' in df.columns and 'end' in df.columns:
                try:
                    df['_fs'] = pd.to_datetime(df['start'], errors='coerce')
                    df['_fe'] = pd.to_datetime(df['end'], errors='coerce')
                    fs_min = df.groupby('flight_id')['_fs'].min().rename('_flight_start')
                    df = df.merge(fs_min, on='flight_id', how='left')
                    df['interval_elapsed_from_flight_start'] = (df['_fe'] - df['_flight_start']).dt.total_seconds().fillna(3600) / 3600.0
                    df.drop(columns=['_fs', '_fe', '_flight_start'], inplace=True)
                except Exception:
                    df['interval_elapsed_from_flight_start'] = 0.0
            else:
                df['interval_elapsed_from_flight_start'] = 0.0

        if 'end_hour' not in df.columns and 'end' in df.columns:
            df['end_hour'] = pd.to_datetime(df['end'], errors='coerce').dt.hour.fillna(-1).astype(int)

    elif os.path.exists(data_path):
        logger.warning(f"Augmented CSV not found. Falling back to {data_path}")
        df = pd.read_parquet(data_path)
        # Apply aliases
        aliases = {
            'estimated_takeoff_mass': 'starting_mass_kg',
            'seg_groundspeed_mean': 'gs_avg_kts',
            'segment_duration': 'interval_duration_sec',
            'great_circle_distance_km': 'great_circle_distance',
            'seg_vertical_rate_mean': 'vs_avg_fpm',
            'seg_altitude_mean': 'alt_avg_ft',
        }
        for src, dst in aliases.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]
    else:
        logger.error(f"No training data found. Checked {aug_path} and {data_path}.")
        return

    df = df.dropna(subset=['fuel_kg'])
    
    # Load selected features if available
    if os.path.exists(config.SELECTED_FEATURES_PATH):
        with open(config.SELECTED_FEATURES_PATH, 'r') as f:
            selected_features = json.load(f)['selected_features']
    else:
        logger.warning("Selected features file not found. Using all numeric columns.")
        selected_features = df.select_dtypes(include=[np.number]).columns.drop('fuel_kg').tolist()

    # Fill any missing feature columns with 0 to avoid KeyError
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    X = df[selected_features]
    y = np.log1p(df['fuel_kg'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna study
    study = optuna.create_study(direction='minimize')
    logger.info("Created Optuna study. Starting trials...")
    
    # Run optimization (e.g., 100 trials)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=100)

    logger.info("Optimization complete.")
    logger.info(f"Best Trial Score: {study.best_value:.4f} (RMSE)")
    logger.info(f"Best Parameters: {json.dumps(study.best_params, indent=2)}")

    # Save best parameters
    output_dir = os.path.join(config.MODELS_DIR, 'optuna_tuning')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'best_optuna_params.json'), 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    logger.info(f"Best parameters saved to {output_dir}")

if __name__ == "__main__":
    main()
