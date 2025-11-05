"""
This script handles the hyperparameter tuning stage of the ML pipeline.

It uses GridSearchCV to find the best hyperparameters for a selected model.
"""
import os
import pandas as pd
import numpy as np
import datetime
import joblib
import json
import argparse

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

import config

def tune(model_name):
    """Main tuning function."""
    print(f"--- Starting Hyperparameter Tuning for {model_name.upper()} ---")

    # --- 1. Load Data ---
    featured_data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data{'test' if config.TEST_RUN else ''}.parquet")
    if not os.path.exists(featured_data_path):
        print(f"Error: Featured data not found at {featured_data_path}. Please run the data preparation stage first.")
        return

    print(f"Loading featured data from {featured_data_path}...")
    df_featured = pd.read_parquet(featured_data_path)

    # --- 2. Prepare Data for Training and Validation ---
    features = df_featured.select_dtypes(include=np.number).columns.tolist()
    features.remove('fuel_kg')
    target = 'fuel_kg'

    df_featured.dropna(subset=features + [target], inplace=True)
    X = df_featured[features]
    y = df_featured[target]

    # Split data into training and validation sets for consistent evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Using {len(X_train)} samples for tuning (training set) and {len(X_val)} for validation.")

    # --- 3. Define Models and Parameter Grids ---
    models = {
        'gbr': GradientBoostingRegressor(random_state=42),
        'xgb': xgb.XGBRegressor(random_state=42, n_jobs=-1),
        'rf': RandomForestRegressor(random_state=42, n_jobs=-1)
    }

    # NOTE: These grids are small for demonstration. Expand them for a real tuning run.
    param_grids = {
        'gbr': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5]
        }
    }

    if model_name not in models:
        print(f"Error: Model '{model_name}' is not a valid choice.")
        return

    # --- 4. Run GridSearchCV ---
    print(f"Running GridSearchCV for {model_name.upper()}...")
    grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grids[model_name], 
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
    
    grid_search.fit(X_train, y_train)

    print("\n--- Tuning Complete ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best MAE Score: {-grid_search.best_score_:.2f} kg")

    # --- 5. Save Best Model and Datasets ---
    best_model = grid_search.best_estimator_
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, f"{model_name}_model_tuned_{run_id}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to {model_path}")

    # Save best parameters
    params_path = os.path.join(model_dir, "best_params.json")
    with open(params_path, 'w') as f:
        json.dump(grid_search.best_params_, f)
    print(f"Best parameters saved to {params_path}")

    # Save features
    features_path = os.path.join(model_dir, "features.json")
    with open(features_path, 'w') as f:
        json.dump(features, f)
    print(f"Feature list saved to {features_path}")

    # Save datasets
    X_train.to_parquet(os.path.join(model_dir, "X_train.parquet"))
    y_train.to_frame().to_parquet(os.path.join(model_dir, "y_train.parquet"))
    X_val.to_parquet(os.path.join(model_dir, "X_val.parquet"))
    y_val.to_frame().to_parquet(os.path.join(model_dir, "y_val.parquet"))
    print(f"Training and validation datasets saved to {model_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune ML models.")
    parser.add_argument("model", choices=["gbr", "xgb", "rf"], help="The model to tune.")
    args = parser.parse_args()
    tune(args.model)
