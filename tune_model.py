"""
This script handles the hyperparameter tuning stage of the ML pipeline.

It dynamically lists available models, loads the appropriate data and preprocessor,
and uses RandomizedSearchCV to efficiently find the best hyperparameters.
"""
import os
import pandas as pd
import numpy as np
import datetime
import joblib
import json
import argparse
import logging
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost as xgb
import config

# --- Setup Logging ---
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', 'tune_model.log')
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def select_model_directory():
    """Lists available models and prompts the user to select one."""
    if not os.path.exists(config.MODELS_DIR):
        logging.error(f"Error: Models directory '{config.MODELS_DIR}' not found. Please train a model first.")
        return None

    saved_model_dirs = sorted([d for d in os.listdir(config.MODELS_DIR) if os.path.isdir(os.path.join(config.MODELS_DIR, d)) and d.startswith('xgb_model')])
    if not saved_model_dirs:
        logging.error("Error: No trained XGBoost models found in the models directory.")
        return None

    print("--- Model Selection for Tuning ---")
    for i, model_dir in enumerate(saved_model_dirs):
        print(f"[{i+1}] {model_dir}")
    
    try:
        choice = int(input("\nPlease select a model to tune: ").strip()) - 1
        if not (0 <= choice < len(saved_model_dirs)):
            print("Invalid selection.")
            return None
    except (ValueError, IndexError):
        print("Invalid input.")
        return None
        
    model_dir_name = saved_model_dirs[choice]
    return os.path.join(config.MODELS_DIR, model_dir_name)

def tune():
    """Main tuning function."""
    model_dir_path = select_model_directory()
    if not model_dir_path:
        return

    model_name = os.path.basename(model_dir_path)
    logging.info(f"--- Starting Hyperparameter Tuning for {model_name} ---")

    # --- 1. Load Artifacts and Data ---
    try:
        preprocessor_path = os.path.join(model_dir_path, "preprocessor.joblib")
        features_path = os.path.join(model_dir_path, "selected_features.json")

        preprocessor = joblib.load(preprocessor_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
        
        logging.info("Loaded preprocessor and feature list from the selected model directory.")
    except FileNotFoundError as e:
        logging.error(f"Error loading artifacts from {model_dir_path}: {e}. Please ensure the directory contains preprocessor.joblib and selected_features.json.")
        return

    data_file = 'featured_data_train_test.parquet' if config.TEST_RUN else 'featured_data_train.parquet'
    featured_data_path = os.path.join(config.PROCESSED_DATA_DIR, data_file)
    if not os.path.exists(featured_data_path):
        logging.error(f"Error: Featured data not found at {featured_data_path}. Please run the data preparation stage first.")
        return

    logging.info(f"Loading featured data from {featured_data_path}...")
    df_featured = pd.read_parquet(featured_data_path)
    df_featured.dropna(subset=['fuel_kg'], inplace=True)

    # --- 2. Prepare Data for Tuning ---
    # Align columns to ensure consistency, filling missing ones with 0
    current_cols = set(df_featured.columns)
    required_cols = set(feature_cols)
    missing_cols = required_cols - current_cols
    if missing_cols:
        logging.warning(f"The following required features are missing from the dataset: {sorted(list(missing_cols))}")
        for col in missing_cols:
            df_featured[col] = 0
        logging.warning("Missing columns have been added and filled with 0.")

    X = df_featured[feature_cols]
    y = df_featured['fuel_kg']
    y_log = np.log1p(y) # Use log-transform consistent with training

    # Preprocess the data using the loaded preprocessor
    logging.info("Preprocessing data using the model's original preprocessor...")
    X_processed = preprocessor.transform(X)
    logging.info("Data preprocessing complete.")

    # --- 3. Define Parameter Grid for RandomizedSearch ---
    # A more comprehensive grid for XGBoost
    param_dist = {
        'n_estimators': [300, 500, 700, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [5, 7, 10, 12, 15],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.001, 0.005, 0.01],
        'reg_lambda': [1, 1.5, 2]
    }

    # --- 4. Run RandomizedSearchCV ---
    logging.info("Running RandomizedSearchCV for XGBoost...")
    xgb_reg = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_dist,
        n_iter=50,  # Number of parameter settings that are sampled
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='neg_mean_absolute_error',
        random_state=42
    )
    
    random_search.fit(X_processed, y_log)

    logging.info("\n--- Tuning Complete ---")
    logging.info(f"Best Parameters: {random_search.best_params_}")
    logging.info(f"Best Cross-Validated MAE Score (on log-transformed data): {-random_search.best_score_:.4f}")

    # --- 5. Save Best Parameters ---
    best_params = random_search.best_params_
    params_path = os.path.join(model_dir_path, "best_params.json")
    
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"\nBest parameters saved to {params_path}")
    logging.info("You can now retrain the model using these parameters by running:")
    logging.info(f"python run_pipeline.py train --model xgb --features \"{features_path}\" --params \"{params_path}\"")

if __name__ == '__main__':
    tune()
