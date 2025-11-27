"""
This script handles the model training stage of the ML pipeline using XGBoost.
It now uses a robust ColumnTransformer for preprocessing.
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import config
import json
import logging
import argparse

# Initialize module logger
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', f'xgboost_fuel_openap_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)


def load_data(stage='train'):
    """
    Load data for a given stage, handling test/full run mode.
    """
    if config.TEST_RUN:
        logger.info("TEST RUN active: loading a fraction of the data.")
        file_name = f"featured_data_{stage}_test.parquet"
    else:
        file_name = f"featured_data_{stage}.parquet"
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for stage '{stage}' not found at {data_path}. Please run the data preparation pipeline first.")

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    return df


def train(params_path=None, selected_features_path=None):
    """
    Main training function.
    """
    print("--- Starting XGBoost Model Training Stage ---")

    # --- 1. Load and Prepare Data ---
    df_train = load_data('train')
    
    target_col = 'fuel_kg'
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in the training data.")

    df_train.dropna(subset=[target_col], inplace=True)
    
    if selected_features_path and os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            feature_cols = json.load(f)
        print(f"Using {len(feature_cols)} features from {selected_features_path}.")
    else:
        print("No feature selection file provided. Using all features except target and identifiers.")
        feature_cols = [col for col in df_train.columns if col not in [target_col, 'flight_id']]

    missing_features = [f for f in feature_cols if f not in df_train.columns]
    if missing_features:
        raise ValueError(f"The following selected features are missing from the training data: {missing_features}")

    X = df_train[feature_cols]
    y = df_train[target_col]
    y_log = np.log1p(y)

    # --- 2. Pre-emptive Data Cleaning ---
    # Drop columns that are entirely NaN to prevent imputer warnings
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"Dropping the following columns because they contain only NaN values: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        # Also update feature_cols to ensure consistency for saving artifacts
        feature_cols = [col for col in feature_cols if col not in all_nan_cols]

    # --- 3. Preprocessing with ColumnTransformer ---
    
    # Identify conditional weather features (impute with 0)
    conditional_weather_features = [col for col in X.columns if 'precip' in col or 'gust' in col or 'storm' in col]

    # Identify standard numeric features (impute with mean), now including booleans
    numeric_features = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    standard_numeric_features = [col for col in numeric_features if col not in conditional_weather_features]
    
    # Identify categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Log columns that are not being used
    handled_features = set(standard_numeric_features + conditional_weather_features + categorical_features)
    all_input_features = set(X.columns)
    unhandled_features = all_input_features - handled_features
    if unhandled_features:
        logger.warning(f"The following columns are present in the data but not handled by the preprocessor, and will be dropped: {sorted(list(unhandled_features))}")

    # Create preprocessing pipelines for each feature type
    standard_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    conditional_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()) # Scale after imputing, as 0 is a meaningful value
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_standard', standard_numeric_transformer, standard_numeric_features),
            ('num_conditional', conditional_transformer, conditional_weather_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop other columns (like timestamps)
    )

    # --- 4. Train Model ---
    if config.TEST_RUN:
        X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
        print(f"Test Run: Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    else:
        X_train, y_train = X, y_log
        X_val, y_val = None, None
        print(f"Full Run: Training on the entire dataset of {len(X_train)} samples.")

    # Fit the preprocessor on the training data and transform it
    print("Fitting preprocessor and transforming training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    model_params = { 'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 0.01, 'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.7, 'random_state': 42, 'n_jobs': -1}
    model_suffix = "_baseline"

    if params_path:
        try:
            with open(params_path, 'r') as f:
                tuned_params = json.load(f)
            model_params.update(tuned_params)
            model_suffix = "_tuned"
            print(f"Using tuned parameters from {params_path}: {tuned_params}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load params from {params_path}. Using defaults. Error: {e}")

    print("Training final XGBoost Regressor model...")
    xgb_reg = xgb.XGBRegressor(**model_params)
    xgb_reg.fit(X_train_processed, y_train)

    # --- 5. Evaluate Model (if in test mode) ---
    if X_val is not None and y_val is not None:
        print("\n--- Validation Results ---")
        X_val_processed = preprocessor.transform(X_val)
        y_pred_log = xgb_reg.predict(X_val_processed)
        y_pred_orig = np.expm1(y_pred_log)
        y_val_orig = np.expm1(y_val)

        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        r2 = r2_score(y_val_orig, y_pred_orig)

        print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
        print(f"R-squared (R²): {r2:.4f}")

    # --- 6. Save Model and Artifacts ---
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, f"xgb_model{model_suffix}_{run_id}")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(xgb_reg, os.path.join(model_dir, "model.joblib"))
    print(f"\nModel saved to {os.path.join(model_dir, 'model.joblib')}")

    # Save the single, powerful preprocessor object
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    print(f"Preprocessing pipeline saved to {os.path.join(model_dir, 'preprocessor.joblib')}")

    # Save the cleaned list of features that was actually used
    with open(os.path.join(model_dir, "selected_features.json"), 'w') as f:
        json.dump(feature_cols, f)
    print(f"Feature list saved to {os.path.join(model_dir, 'selected_features.json')}")

    # --- 7. Feature Importance ---
    print("\n--- Feature Importance (Top 15) ---")
    try:
        # Get feature names from the preprocessor
        ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        
        # Combine all feature names in the correct order
        all_feature_names = standard_numeric_features + conditional_weather_features + list(ohe_feature_names)
        
        importances = xgb_reg.feature_importances_
        
        if len(all_feature_names) != len(importances):
             raise ValueError(f"Mismatch in feature name count ({len(all_feature_names)}) and importance count ({len(importances)}).")

        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        print(feature_importance_df.head(15))

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
        ax.set_title(f"Top 20 Feature Importance for {run_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "feature_importance.png"))
        print(f"Feature importance plot saved to {os.path.join(model_dir, 'feature_importance.png')}")
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot. Error: {e}")


    print("--- XGBoost Model Training Stage Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XGBoost Model Training")
    parser.add_argument('--params', type=str, help="Path to a JSON file with model hyperparameters.")
    parser.add_argument('--features', type=str, help="Path to a JSON file with a list of selected features.")
    
    args = parser.parse_args()
    
    train(params_path=args.params, selected_features_path=args.features)
