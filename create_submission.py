import os
import pandas as pd
import numpy as np
import json
import joblib
from tqdm import tqdm
import config
import logging

# --- Setup Logging ---
log_file = 'create_submission.log'
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def find_latest_model_dir(models_base_dir):
    """Finds the most recently created model directory."""
    all_model_dirs = [os.path.join(models_base_dir, d) for d in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, d))]
    if not all_model_dirs:
        return None
    latest_dir = max(all_model_dirs, key=os.path.getmtime)
    return latest_dir

def main():
    """Fills the fuel_rank_submission.parquet file with predictions."""
    try:
        # --- 1. Load and Prepare Data ---
        logging.info("--- Starting Submission Creation ---")
        processed_rank_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'featured_rank_data.parquet')
        
        if not os.path.exists(processed_rank_data_path):
            logging.error(f"Augmented ranking data not found at {processed_rank_data_path}. Please run augment_features.py first.")
            return
            
        logging.info(f"Loading augmented ranking data from {processed_rank_data_path}...")
        df_featured = pd.read_parquet(processed_rank_data_path)

        # Convert timestamps to numerical features, handling potential NaTs
        for col in df_featured.select_dtypes(include=['datetime64[ns]']).columns:
            df_featured[col] = df_featured[col].apply(lambda x: x.value if pd.notnull(x) else np.nan)

        # --- 2. Load Model and Selected Features ---
        logging.info("Finding the latest trained model...")
        latest_model_dir = find_latest_model_dir(config.MODELS_DIR)
        
        if not latest_model_dir:
            logging.error(f"No trained models found in '{config.MODELS_DIR}'. Please train a model first.")
            return
        
        logging.info(f"Using model from: {os.path.basename(latest_model_dir)}")

        # Load the feature list (either 'selected_features.json' or 'features.json')
        features_path = os.path.join(latest_model_dir, "selected_features.json")
        if not os.path.exists(features_path):
            features_path = os.path.join(latest_model_dir, "features.json")
        
        if not os.path.exists(features_path):
            logging.error(f"Feature list ('selected_features.json' or 'features.json') not found in {latest_model_dir}.")
            return

        logging.info(f"Loading feature list from {features_path}...")
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
        
        # Load the model
        model_path = os.path.join(latest_model_dir, "model.joblib")
        if not os.path.exists(model_path):
            logging.error(f"Model file 'model.joblib' not found in {latest_model_dir}.")
            return
            
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        # --- 3. Align Data with Selected Features ---
        logging.info("Aligning data columns with selected features...")
        
        # Ensure all selected features are present in the dataframe, adding them with NaN if missing
        for feature in selected_features:
            if feature not in df_featured.columns:
                logging.warning(f"Feature '{feature}' from training is missing in submission data. Adding it as a column of NaNs.")
                df_featured[feature] = np.nan
        
        # Select only the features the model was trained on
        X_submission = df_featured[selected_features]

        # Impute any remaining NaNs in the submission data using the mean of each column
        # This is a simple but robust strategy for submission.
        if X_submission.isnull().sum().sum() > 0:
            logging.info("Imputing remaining NaN values in submission data with column means...")
            X_submission = X_submission.fillna(X_submission.mean())

        # Final check for NaNs
        if X_submission.isnull().sum().sum() > 0:
            logging.error("NaN values still present after imputation. Aborting.")
            logging.error(X_submission.isnull().sum())
            return

        # --- 4. Generate Predictions ---
        logging.info("Generating predictions...")
        predictions = model.predict(X_submission)

        # --- 5. Create and Save Submission File ---
        submission_template_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')
        df_submission = pd.read_parquet(submission_template_path)
        
        # Ensure the prediction order matches the submission template order
        # This assumes df_featured has the same row order as the original submission template
        df_submission['fuel_kg'] = predictions

        output_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission_final.parquet')
        df_submission.to_parquet(output_path, index=False)

        logging.info(f"\nSuccessfully generated {len(df_submission)} predictions.")
        logging.info(f"Submission file saved to: {output_path}")
        logging.info("\nSubmission Head:")
        logging.info(df_submission.head())

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
