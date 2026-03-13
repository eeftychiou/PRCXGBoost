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
    # Our new preprocessor is a dictionary containing the fitted components
    
    # Extract components
    num_imputer = preprocessor['num_imputer_full']
    cat_imputer = preprocessor['cat_imputer_full']
    cat_encoder = preprocessor['cat_encoder_full']
    scaler = preprocessor['scaler_full']
    selected_mask = preprocessor['selected_mask']
    numerical_features = preprocessor.get('numerical_features', [])
    categorical_features = preprocessor.get('categorical_features', [])
    feature_cols_selected = preprocessor['feature_cols_selected']
    
    # Ensure data matches the expected columns perfectly
    # Use reindex to gracefully fill any missing architectural columns with 0
    X_processed = data.reindex(columns=feature_cols_selected, fill_value=0).copy()
    
    # 1. Imputation
    if numerical_features and num_imputer:
        X_processed[numerical_features] = num_imputer.transform(X_processed[numerical_features])
        
    if categorical_features and cat_imputer:
        X_processed[categorical_features] = cat_imputer.transform(X_processed[categorical_features])
        if cat_encoder:
            X_processed[categorical_features] = cat_encoder.transform(X_processed[categorical_features])
            
    # 2. Scaling
    X_scaled = scaler.transform(X_processed)
    
    # 3. Feature Selection Mask (SFS)
    X_final = X_scaled[:, selected_mask]
    
    logging.info(f"Data transformed successfully: {X_final.shape}")
    
    logging.info("Generating final predictions.")
    predictions_log = model.predict(X_final)
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

    # Check for diagnostic plots and notify user
    if os.path.exists(model_dir_path):
        diag_plots = [f for f in os.listdir(model_dir_path) if f.endswith('.png') and ('learning_curve' in f or 'importance' in f or 'predicted_vs_actual' in f)]
        if diag_plots:
            logging.info(f"✨ Found {len(diag_plots)} diagnostic plots in model directory!")
            for plot in diag_plots:
                logging.info(f"   - {plot}")
        else:
            logging.info("💡 Tip: Diagnostic plots are generated during the 'train' stage and saved to this folder.")

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
            
            # Use the loaded preprocessor dictionary to transform the validation set
            X_val_processed = X_val.copy()
            num_imputer = preprocessor['num_imputer_full']
            cat_imputer = preprocessor['cat_imputer_full']
            cat_encoder = preprocessor['cat_encoder_full']
            scaler      = preprocessor['scaler_full']
            mask        = preprocessor['selected_mask']
            
            num_feas = preprocessor.get('numerical_features', [])
            cat_feas = preprocessor.get('categorical_features', [])
            
            if num_feas and num_imputer:
                X_val_processed[num_feas] = num_imputer.transform(X_val_processed[num_feas])
            if cat_feas and cat_imputer:
                X_val_processed[cat_feas] = cat_imputer.transform(X_val_processed[cat_feas])
                if cat_encoder:
                    X_val_processed[cat_feas] = cat_encoder.transform(X_val_processed[cat_feas])
                    
            X_val_scaled = scaler.transform(X_val_processed)
            X_val_final = X_val_scaled[:, mask]
            
            y_pred_log = model.predict(X_val_final)
            
            y_val_orig = np.expm1(y_val)
            y_pred_orig = np.expm1(y_pred_log)
            evaluate_performance(y_val_orig, y_pred_orig, model_dir_name, model_dir_path, identifiers_val)

        elif run_type in ['rank', 'final']:
            logging.info(f"Running in submission mode for '{run_type}' dataset...")
            # --- DYNAMIC/ENRICHED DATA LOADING ---
            # Try loading fully augmented features first (generated by Aux script)
            # Prioritize the paths defined in config.py
            augmented_data_path = config.AUGMENTED_RANK_CSV if run_type == 'rank' else config.AUGMENTED_FINAL_CSV
            
            data_file = f"featured_data_{run_type}.parquet"
            prediction_data_path = os.path.join(config.PROCESSED_DATA_DIR, data_file)
            
            is_augmented = False
            if os.path.exists(augmented_data_path):
                logging.info(f"Detected fully augmented schema: {os.path.basename(augmented_data_path)}")
                df_predict = pd.read_csv(augmented_data_path)
                is_augmented = True
            elif os.path.exists(prediction_data_path):
                logging.warning(f"Augmented file not found ({augmented_data_path}). Falling back to base {data_file}")
                df_predict = pd.read_parquet(prediction_data_path)
            else:
                raise FileNotFoundError(f"Neither {augmented_data_path} nor {prediction_data_path} exist for submission.")
                
            # --- DYNAMIC FEATURE CALCULATION (Fallback) ---
            # Replicate the feature engineering steps done dynamically if not fully present
            logging.info("Checking/Calculating dynamic features...")
            
            if 'alt_avg_ft' not in df_predict.columns:
                df_predict['alt_avg_ft'] = (df_predict.get('alt_start_ft', 0) + df_predict.get('alt_end_ft', 0)) / 2
                
            if 'altitude_change_rate' not in df_predict.columns:
                df_predict['altitude_change_rate'] = df_predict.get('alt_change_ft', 0) / (df_predict.get('interval_duration_sec', 60) + 1e-6)
                
            if 'end_hour' not in df_predict.columns:
                if 'end' in df_predict.columns:
                    df_predict['end_hour'] = pd.to_datetime(df_predict['end'], errors='coerce').dt.hour.fillna(-1).astype(int)
                else:
                    df_predict['end_hour'] = 12
                    
            if 'interval_elapsed_from_flight_start' not in df_predict.columns:
                if 'start' in df_predict.columns and 'end' in df_predict.columns:
                    df_predict['flight_start'] = pd.to_datetime(df_predict['start'], errors='coerce')
                    df_predict['interval_elapsed_from_flight_start'] = (pd.to_datetime(df_predict['end'], errors='coerce') - df_predict['flight_start']).dt.total_seconds().fillna(3600) / 3600.0
                else:
                    df_predict['interval_elapsed_from_flight_start'] = 0.0
                        
            # Handle any fundamentally missing features (e.g. from missing merge joins)
            # We enforce 0 for numeric missing cols so indexing inside X_predict works flawlessly
            missing_cols = [col for col in feature_cols if col not in df_predict.columns]
            if missing_cols:
                logging.warning(f"Filling {len(missing_cols)} entirely missing training columns with 0 (e.g. {missing_cols[:5]}...)")
                for col in missing_cols:
                    df_predict[col] = 0

            # Ensure all required feature columns are exactly extracted
            X_predict = df_predict[feature_cols]
            
            predictions = generate_predictions(model, X_predict, preprocessor)
            
            # Use the submission template file paths defined in config
            if run_type == 'rank':
                fuel_file_path = config.FUEL_RANK
            else:
                fuel_file_path = config.FUEL_FINAL
                
            if not os.path.exists(fuel_file_path):
                 # Fallback to manual path if config constants differ (unlikely given AGENTS.md)
                 fuel_file_path = os.path.join(config.BASE_DATASETS_DIR, f"fuel_{run_type}_submission.parquet")
                 
            submission_df = pd.read_parquet(fuel_file_path)
            submission_df['fuel_kg'] = predictions
            
            submission_dir = config.SUBMISSIONS_DIR
            os.makedirs(submission_dir, exist_ok=True)
            submission_path = os.path.join(submission_dir, f"fuel_{run_type}.parquet")
            submission_df.to_parquet(submission_path, index=False)
            logging.info(f"Submission file for '{run_type}' created at: {submission_path}")
            
            # --- EVALUATE AGAINST BRIGHT-LOBSTER BASELINE ---
            baseline_filename = f"bright-lobster_{run_type}.parquet"
            # Try multiple locations for the baseline
            possible_baseline_paths = [
                os.path.join(model_dir_path, baseline_filename), # Model folder
                baseline_filename,                                # Current root
                os.path.join("JOAS_Results", baseline_filename)   # User's results folder
            ]
            
            baseline_path = None
            for path in possible_baseline_paths:
                if os.path.exists(path):
                    baseline_path = path
                    break
                    
            if baseline_path:
                logging.info(f"\n--- Comparing with Baseline: {baseline_path} ---")
                baseline_df = pd.read_parquet(baseline_path)
                
                # Merge rigorously on interval idx and flight_id
                comparison_df = submission_df.merge(
                    baseline_df[['idx', 'flight_id', 'fuel_kg']], 
                    on=['idx', 'flight_id'], 
                    suffixes=('_pred', '_base')
                )
                
                if not comparison_df.empty:
                    y_pred = comparison_df['fuel_kg_pred']
                    y_base = comparison_df['fuel_kg_base']
                    
                    mae = mean_absolute_error(y_base, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_base, y_pred))
                    r2 = r2_score(y_base, y_pred)
                    
                    logging.info(f"Comparison Data Points: {len(comparison_df):,}")
                    logging.info(f"Mean Absolute Difference (MAE): {mae:.4f} kg")
                    logging.info(f"Root Mean Squared Difference (RMSE): {rmse:.4f} kg")
                    logging.info(f"Agreement Score (R²): {r2:.4f}\n")
                    
                    # --- CDF Plot of Absolute Errors ---
                    abs_errors = np.abs(y_pred - y_base)
                    sorted_errors = np.sort(abs_errors)
                    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                    
                    plt.style.use('seaborn-v0_8-whitegrid')
                    plt.figure(figsize=(10, 6))
                    plt.plot(sorted_errors, cdf, marker='.', linestyle='none', color='teal', alpha=0.5)
                    plt.step(sorted_errors, cdf, where='post', color='teal', linewidth=1.5)
                    plt.xlabel('Absolute Error (kg)')
                    plt.ylabel('Cumulative Probability')
                    plt.title(f'CDF of Absolute Errors vs Baseline ({run_type})\nModel: {model_dir_name}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    cdf_plot_path = os.path.join(model_dir_path, f"baseline_error_cdf_{run_type}.png")
                    plt.savefig(cdf_plot_path)
                    plt.close()
                    logging.info(f"CDF plot of absolute errors saved to: {cdf_plot_path}")
                    
                    # --- CDF Plot for Distribution Comparison ---
                    # (Plotting the CDF of values themselves to check alignment)
                    sorted_pred = np.sort(y_pred)
                    sorted_base = np.sort(y_base)
                    cdf_vals = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
                    
                    plt.figure(figsize=(10, 6))
                    plt.step(sorted_base, cdf_vals, where='post', color='#CC0000', linewidth=2, label=f'Baseline ({baseline_path})')
                    plt.step(sorted_pred, cdf_vals, where='post', color='#0077CC', linewidth=2, label=f'Model ({model_dir_name})', alpha=0.8)
                    
                    plt.xlabel('Fuel Consumption (kg)')
                    plt.ylabel('Cumulative Probability')
                    plt.title(f'Fuel Distribution CDF Comparison ({run_type})\nModel vs Baseline')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    dist_plot_path = os.path.join(model_dir_path, f"baseline_distribution_cdf_{run_type}.png")
                    plt.savefig(dist_plot_path)
                    plt.close()
                    logging.info(f"Distributional CDF plot saved to: {dist_plot_path}")
                else:
                    logging.warning(f"⚠️ Merge with {baseline_path} yielded 0 rows. Check idx/flight_id alignment.")
            else:
                logging.info(f"\nℹ️ Baseline file '{baseline_path}' not found in root directory. Skipping comparison.")

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
