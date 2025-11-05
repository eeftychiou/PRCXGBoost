import os
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import config

def main():
    """Main model evaluation function."""
    if not os.path.exists(config.MODELS_DIR):
        print(f"Error: Models directory '{config.MODELS_DIR}' not found. Please train a model first.")
        return

    # Filter for GBR, XGB and RF models, including tuned versions
    saved_model_dirs = [
        d for d in os.listdir(config.MODELS_DIR) 
        if os.path.isdir(os.path.join(config.MODELS_DIR, d)) and 
        (d.startswith('gbr_model') or d.startswith('xgb_model') or d.startswith('rf_model'))
    ]
    if not saved_model_dirs:
        print("Error: No trained models found in the models directory.")
        return

    print("--- Evaluate Model Performance ---")
    for i, model_dir in enumerate(saved_model_dirs):
        print(f"[{i+1}] Evaluate model: {model_dir}")
    
    try:
        choice = input("\nPlease select a model to evaluate: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(saved_model_dirs)):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return

    model_dir_name = saved_model_dirs[int(choice) - 1]
    model_dir_path = os.path.join(config.MODELS_DIR, model_dir_name)
    print(f"\n--- Evaluating model: {model_dir_name} ---")

    try:
        # --- 1. Load Artifacts ---
        print("Loading model and validation data...")
        model_path = os.path.join(model_dir_path, "model.joblib")
        features_path = os.path.join(model_dir_path, "features.json")
        x_val_path = os.path.join(model_dir_path, "X_val.parquet")
        y_val_path = os.path.join(model_dir_path, "y_val.parquet")

        model = joblib.load(model_path)
        X_val = pd.read_parquet(x_val_path)
        y_val = pd.read_parquet(y_val_path)

        with open(features_path, 'r') as f:
            feature_cols = json.load(f)

        # Ensure columns are in the same order as during training
        X_val = X_val[feature_cols]

        print("Artifacts loaded successfully.")

        # Load and display best parameters if available (for tuned models)
        params_path = os.path.join(model_dir_path, "best_params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            print("\n--- Tuned Hyperparameters ---")
            for param, value in best_params.items():
                print(f"{param}: {value}")

        # --- 2. Generate Predictions ---
        print("Generating predictions on the validation set...")
        y_pred = model.predict(X_val)

        # --- 3. Calculate and Display Metrics ---
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print("\n--- Validation Results ---")
        print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
        print(f"R-squared (R²): {r2:.4f}")

        # --- 4. Generate and Save Plots ---
        print("Generating and saving evaluation plots...")
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot 1: Actual vs. Predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_val.iloc[:, 0], y=y_pred, alpha=0.6, ax=ax, label='Predictions')
        perfect_line_start = min(y_val.iloc[:, 0].min(), y_pred.min())
        perfect_line_end = max(y_val.iloc[:, 0].max(), y_pred.max())
        ax.plot([perfect_line_start, perfect_line_end], [perfect_line_start, perfect_line_end], 'r--', label='Perfect Prediction')
        ax.set_xlabel("Actual Fuel Consumption (kg)")
        ax.set_ylabel("Predicted Fuel Consumption (kg)")
        ax.set_title(f"Model Evaluation: Actual vs. Predicted\n({model_dir_name})")
        ax.legend()
        plot_path = os.path.join(model_dir_path, "evaluation_plot.png")
        plt.savefig(plot_path)
        print(f"Evaluation plot saved to {plot_path}")

        # Plot 2: Feature Importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values(by='importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
            ax.set_title(f"Feature Importance for {model_dir_name}")
            plt.tight_layout()
            plot_path = os.path.join(model_dir_path, "feature_importance.png")
            plt.savefig(plot_path)
            print(f"Feature importance plot saved to {plot_path}")

    except FileNotFoundError as e:
        print(f"Error: Missing a required file in the model directory: {e.filename}")
        print("Please ensure the selected model directory contains all required files (model.joblib, .parquet, .json).")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == '__main__':
    main()
