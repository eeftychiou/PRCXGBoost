"""
This script handles the model training stage of the ML pipeline using XGBoost.

It uses an XGBoost Regressor to train a model on the feature-rich
dataset created by the data preparation stage.
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import config
import json

def train(params_path=None):
    """Main training function.

    Args:
        params_path (str, optional): Path to a JSON file containing hyperparameters.
                                     If None, default hyperparameters are used.
    """
    print("--- Starting XGBoost Model Training Stage ---")

    # --- 1. Load Data ---
    featured_data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data{'test' if config.TEST_RUN else ''}.parquet")
    if not os.path.exists(featured_data_path):
        print(f"Error: Featured data not found at {featured_data_path}. Please run the data preparation stage first.")
        return

    print(f"Loading featured data from {featured_data_path}...")
    df_featured = pd.read_parquet(featured_data_path)

    # --- 2. Prepare Data for Training ---
    # Automatically select all numerical features except for the target
    features = df_featured.select_dtypes(include=np.number).columns.tolist()
    features.remove('fuel_kg')
    target = 'fuel_kg'

    df_featured.dropna(subset=features + [target], inplace=True)

    X = df_featured[features]
    y = df_featured[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # --- 3. Train Model ---
    model_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42, 'n_jobs': -1}
    model_suffix = ""

    if params_path:
        try:
            with open(params_path, 'r') as f:
                tuned_params = json.load(f)
            model_params.update(tuned_params)
            model_suffix = "_tuned"
            print(f"Using tuned parameters from {params_path}: {tuned_params}")
        except FileNotFoundError:
            print(f"Warning: Parameters file not found at {params_path}. Using default parameters.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {params_path}. Using default parameters.")

    print("Training XGBoost Regressor model...")
    xgb_reg = xgb.XGBRegressor(**model_params)
    xgb_reg.fit(X_train, y_train)

    # --- 4. Evaluate Model ---
    print("Evaluating model on validation set...")
    y_pred = xgb_reg.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("\n--- Validation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
    print(f"R-squared (R²): {r2:.4f}")

    # --- 5. Feature Importance ---
    print("\n--- Feature Importance ---")
    importances = xgb_reg.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)
    print(feature_importance_df.head(15))

    # --- 6. Save Model and Datasets ---
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, f"xgb_model{model_suffix}_{run_id}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(xgb_reg, model_path)
    print(f"\nModel saved to {model_path}")

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

    # Save feature importance plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
    ax.set_title(f"Feature Importance for {run_id}")
    plt.tight_layout()
    plot_path = os.path.join(model_dir, "feature_importance.png")
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")

    print("--- XGBoost Model Training Stage Complete ---")

if __name__ == '__main__':
    train()
