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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import config
import json

def train(params_path=None, feature_selection_method='sfs_forward'):
    """Main training function.

    Args:
        params_path (str, optional): Path to a JSON file with hyperparameters.
        feature_selection_method (str): 'sfs_forward', 'sfs_backward', or 'none'.
    """
    print("--- Starting XGBoost Model Training Stage ---")
    print(f"Feature Selection Method: {feature_selection_method}")

    # --- 1. Load Data ---
    featured_data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data.parquet")
    if not os.path.exists(featured_data_path):
        print(f"Error: Featured data not found at {featured_data_path}. Please run augment_features.py first.")
        return

    print(f"Loading featured data from {featured_data_path}...")
    df_featured = pd.read_parquet(featured_data_path)

    # --- 2. Prepare Data for Training ---
    for col in df_featured.select_dtypes(include=['datetime64[ns]']).columns:
        df_featured[col] = df_featured[col].apply(lambda x: x.value if pd.notnull(x) else np.nan)

    features = df_featured.select_dtypes(include=np.number).columns.tolist()
    features.remove('fuel_kg')
    target = 'fuel_kg'
    df_featured.dropna(subset=features + [target], inplace=True)

    X = df_featured[features]
    y = df_featured[target]

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
        sfs.fit(X, y)
        
        selected_features = list(X.columns[sfs.get_support()])
        print(f"SFS selected {len(selected_features)} features out of {len(X.columns)}.")
    
    elif feature_selection_method == 'none':
        print("\n--- Skipping feature selection. Using all available features. ---")
        selected_features = X.columns.tolist()
    
    else:
        raise ValueError(f"Invalid feature_selection_method: '{feature_selection_method}'. Choose 'sfs_forward', 'sfs_backward', or 'none'.")

    print("Final features to be used:", selected_features)
    X = X[selected_features]

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
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, f"xgb_model{model_suffix}_{run_id}")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(xgb_reg, os.path.join(model_dir, "model.joblib"))
    print(f"\nModel saved to {os.path.join(model_dir, 'model.joblib')}")

    with open(os.path.join(model_dir, "selected_features.json"), 'w') as f:
        json.dump(selected_features, f)
    print(f"Feature list saved to {os.path.join(model_dir, 'selected_features.json')}")

    X_train.to_parquet(os.path.join(model_dir, "X_train.parquet"))
    y_train.to_frame().to_parquet(os.path.join(model_dir, "y_train.parquet"))
    if X_val is not None:
        X_val.to_parquet(os.path.join(model_dir, "X_val.parquet"))
        y_val.to_frame().to_parquet(os.path.join(model_dir, "y_val.parquet"))
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
