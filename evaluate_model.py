import os
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import config
import argparse
import traceback
import logging

# --- Setup Logging ---
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', 'evaluate_model.log')
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def load_artifacts_for_prediction(model_dir_path):
    """Loads artifacts needed for generating new predictions on raw data."""
    model_path = os.path.join(model_dir_path, "model.joblib")
    preprocessor_path = os.path.join(model_dir_path, "preprocessor.joblib")
    features_path = os.path.join(model_dir_path, "selected_features.json")

    if not all(os.path.exists(p) for p in [model_path, preprocessor_path, features_path]):
        raise FileNotFoundError("One or more required artifacts (model.joblib, preprocessor.joblib, selected_features.json) not found.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)
    
    logging.info("Loaded model, preprocessor, and feature list for prediction.")
    return model, preprocessor, feature_cols

def load_artifacts_for_evaluation(model_dir_path):
    """Loads artifacts from a TEST_RUN for evaluation."""
    model_path = os.path.join(model_dir_path, "model.joblib")
    preprocessor_path = os.path.join(model_dir_path, "preprocessor.joblib")
    # In a TEST_RUN, the validation set is not saved, so this function is deprecated.
    # We will always use the fallback logic in main() for evaluation.
    raise FileNotFoundError(f"Pre-split validation artifacts are no longer used. Evaluation will proceed on a fresh split.")

def generate_predictions(model, data, preprocessor):
    """
    Generates predictions for the given raw data using the saved preprocessor pipeline.
    """
    logging.info("--- Starting Prediction/Preprocessing ---")
    
    # The preprocessor handles everything: selection, imputation, scaling, encoding.
    X_processed = preprocessor.transform(data)
    
    logging.info("Data transformed successfully by the preprocessor.")
    
    logging.info("Generating final predictions.")
    predictions_log = model.predict(X_processed)
    predictions = np.expm1(predictions_log)
    
    return predictions

def evaluate_performance(y_true_orig, y_pred_orig, model_dir_name, model_dir_path, val_identifiers=None):
    """Calculates and displays performance metrics on the original scale."""
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)

    logging.info("\n--- Validation Results (Original Scale) ---")
    logging.info(f"Mean Absolute Error (MAE): {mae:.2f} kg")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
    logging.info(f"R-squared (R²): {r2:.4f}")

    eval_details_df = pd.DataFrame({
        'actual_fuel_kg': y_true_orig,
        'predicted_fuel_kg': y_pred_orig,
        'prediction_error_kg': y_pred_orig - y_true_orig,
        'absolute_error_kg': np.abs(y_pred_orig - y_true_orig),
        'squared_error_kg2': (y_pred_orig - y_true_orig)**2
    })
    
    if val_identifiers is not None:
        # The indices are aligned from the train_test_split, so a direct concat is correct.
        eval_details_df = pd.concat([val_identifiers, eval_details_df], axis=1)
    
    eval_details_path = os.path.join(model_dir_path, "evaluation_details.csv")
    # Save the dataframe's index, which corresponds to the original featured_data index.
    eval_details_df.index.name = 'original_index'
    eval_details_df.to_csv(eval_details_path, index=True)
    logging.info(f"Saved detailed evaluation results to {eval_details_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_true_orig, y=y_pred_orig, alpha=0.5, ax=ax)
    ax.plot([y_true_orig.min(), y_true_orig.max()], [y_true_orig.min(), y_true_orig.max()], 'r--', label='Perfect Prediction')
    ax.set_xlabel("Actual Fuel Consumption (kg)")
    ax.set_ylabel("Predicted Fuel Consumption (kg)")
    ax.set_title(f"Model Evaluation: Actual vs. Predicted\n({model_dir_name})")
    ax.legend()
    plot_path = os.path.join(model_dir_path, "evaluation_plot.png")
    plt.savefig(plot_path)
    logging.info(f"Evaluation plot saved to {plot_path}")

def main(run_type='evaluate'):
    """Main model evaluation and submission generation function."""
    if not os.path.exists(config.MODELS_DIR):
        logging.error(f"Error: Models directory '{config.MODELS_DIR}' not found. Please train a model first.")
        return

    saved_model_dirs = sorted([d for d in os.listdir(config.MODELS_DIR) if os.path.isdir(os.path.join(config.MODELS_DIR, d)) and (d.startswith('gbr_model') or d.startswith('xgb_model') or d.startswith('rf_model'))])
    if not saved_model_dirs:
        logging.error("Error: No trained models found in the models directory.")
        return

    print("--- Model Selection ---")
    for i, model_dir in enumerate(saved_model_dirs):
        print(f"[{i+1}] {model_dir}")
    
    try:
        choice = int(input("\nPlease select a model: ").strip()) - 1
        if not (0 <= choice < len(saved_model_dirs)):
            print("Invalid selection.")
            return
    except (ValueError, IndexError):
        print("Invalid input.")
        return

    model_dir_name = saved_model_dirs[choice]
    model_dir_path = os.path.join(config.MODELS_DIR, model_dir_name)
    print(f"\n--- Using model: {model_dir_name} ---")

    try:
        model, preprocessor, feature_cols = load_artifacts_for_prediction(model_dir_path)

        if run_type == 'evaluate':
            logging.info("Running in evaluation mode...")
            
            data_file = 'featured_data_train_test.parquet' if config.TEST_RUN else 'featured_data_train.parquet'
            df_full = pd.read_parquet(os.path.join(config.PROCESSED_DATA_DIR, data_file))
            df_full.dropna(subset=['fuel_kg'], inplace=True)

            X = df_full[feature_cols]
            y = np.log1p(df_full['fuel_kg'])
            
            # Split data for evaluation
            X_train, X_val, y_train, y_val, identifiers_train, identifiers_val = train_test_split(
                X, y, df_full[['flight_id']], test_size=0.2, random_state=42
            )
            
            # Use the loaded preprocessor to transform the validation set
            X_val_processed = preprocessor.transform(X_val)
            
            y_pred_log = model.predict(X_val_processed)
            
            y_val_orig = np.expm1(y_val)
            y_pred_orig = np.expm1(y_pred_log)
            evaluate_performance(y_val_orig, y_pred_orig, model_dir_name, model_dir_path, identifiers_val)

        elif run_type in ['rank', 'final']:
            logging.info(f"Running in submission mode for '{run_type}' dataset...")
            
            data_file = f"featured_data_{run_type}.parquet"
            prediction_data_path = os.path.join(config.PROCESSED_DATA_DIR, data_file)

            if not os.path.exists(prediction_data_path):
                raise FileNotFoundError(f"Data for submission not found at {prediction_data_path}.")

            df_predict = pd.read_parquet(prediction_data_path)
            
            # Ensure all required feature columns are present
            X_predict = df_predict[feature_cols]
            
            predictions = generate_predictions(model, X_predict, preprocessor)
            
            fuel_file_path = os.path.join(config.BASE_DATASETS_DIR, f"fuel_{run_type}.parquet")
            submission_df = pd.read_parquet(fuel_file_path)
            submission_df['fuel_kg'] = predictions
            
            submission_dir = config.SUBMISSIONS_DIR
            os.makedirs(submission_dir, exist_ok=True)
            submission_path = os.path.join(submission_dir, f"fuel_{run_type}.parquet")
            submission_df.to_parquet(submission_path, index=False)
            logging.info(f"Submission file for '{run_type}' created at: {submission_path}")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Evaluation and Submission Generation")
    parser.add_argument('--run_type', type=str, default='evaluate', choices=['evaluate', 'rank', 'final'],
                        help="The type of run: 'evaluate' for performance metrics, 'rank' or 'final' for submission generation.")
    args = parser.parse_args()
    main(run_type=args.run_type)
