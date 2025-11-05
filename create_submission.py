import os
import pandas as pd
import numpy as np
import json
import joblib
from tqdm import tqdm
import config
import feature_engineering

def main():
    """Fills the fuel_rank_submission.parquet file with predictions."""
    if not os.path.exists(config.MODELS_DIR):
        print(f"Error: Models directory '{config.MODELS_DIR}' not found. Please train a model first.")
        return

    # Filter for GBR, XGB, RF models, including tuned versions
    saved_model_dirs = [
        d for d in os.listdir(config.MODELS_DIR) 
        if os.path.isdir(os.path.join(config.MODELS_DIR, d)) and 
        (d.startswith('gbr_model') or d.startswith('xgb_model') or d.startswith('rf_model'))
    ]
    if not saved_model_dirs:
        print("Error: No trained models found in the models directory.")
        return

    print("--- Create Submission File ---")
    for i, model_dir in enumerate(saved_model_dirs):
        print(f"[{i+1}] Use model: {model_dir}")

    try:
        choice = input("\nPlease select a model to use for prediction: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(saved_model_dirs)):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return

    model_dir_name = saved_model_dirs[int(choice) - 1]
    model_dir_path = os.path.join(config.MODELS_DIR, model_dir_name)
    print(f"\n--- Using model: {model_dir_name} ---")

    try:
        # --- 1. Load Model and Feature List ---
        print("Loading model and feature list...")
        model_path = os.path.join(model_dir_path, "model.joblib")
        features_path = os.path.join(model_dir_path, "features.json")

        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)

        print("Model and feature list loaded successfully.")

        # --- 2. Load Data ---
        print("Loading submission template, flight data, and airport data...")
        submission_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')
        df_submission_template = pd.read_parquet(submission_path)

        flightlist_path = os.path.join(config.BASE_DATASETS_DIR, 'flightlist_rank.parquet')
        df_flightlist_rank = pd.read_parquet(flightlist_path)

        ac_perf_path = os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv')
        df_ac_perf = pd.read_csv(ac_perf_path)

        apt_path = os.path.join(config.BASE_DATASETS_DIR, 'apt.parquet')
        df_apt = pd.read_parquet(apt_path)
        df_apt.rename(columns={'latitude': 'apt_lat', 'longitude': 'apt_lon', 'elevation': 'apt_elev'}, inplace=True)

        print("Data loaded successfully.")

        # --- 3. Merge Data ---
        print("Merging dataframes...")
        # Merge flightlist and ac_perf
        df_merged = pd.merge(df_flightlist_rank, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')

        # Merge with submission template, joining on flight_id
        df_submission = pd.merge(df_submission_template, df_merged, on='flight_id', how='left')

        # Select and rename columns from acPerfOpenAP, mirroring data_preparation.py
        ac_perf_cols = [
            'ICAO_TYPE_CODE', 'mtow', 'mlw', 'oew', 'mfc', 'vmo', 'mmo', 'ceiling', 'pax_max',
            'fuselage_length', 'fuselage_height', 'fuselage_width', 'wing_area', 'wing_span',
            'wing_mac', 'wing_sweep', 'wing_t/c', 'cruise_height', 'cruise_mach', 'cruise_range',
            'engine_number', 'drag_cd0', 'drag_k', 'engine_type', 'engine_mount', 'flaps_type'
        ]
        df_ac_perf_selected = df_ac_perf[ac_perf_cols]

        # Handle categorical features before filling missing numerical values
        categorical_cols = ['engine_type', 'engine_mount', 'flaps_type']
        for col in categorical_cols:
            # Fill with a placeholder 'Unknown' if mode is not suitable or column is entirely NaN
            df_submission[col] = df_submission[col].fillna(df_submission[col].mode()[0] if not df_submission[col].mode().empty else 'Unknown')
        df_submission = pd.get_dummies(df_submission, columns=categorical_cols, prefix=categorical_cols, dtype=float)

        # Handle numerical features
        numerical_cols = df_submission.select_dtypes(include=np.number).columns.tolist()
        # Remove 'fuel_kg' if it somehow appears (it shouldn't in rank data)
        if 'fuel_kg' in numerical_cols: numerical_cols.remove('fuel_kg')

        for col in numerical_cols:
            df_submission[col] = df_submission[col].fillna(df_submission[col].median() if not df_submission[col].isnull().all() else 0)

        print("Dataframes merged successfully.")

        # --- 4. Engineer Features ---
        flights_rank_dir = os.path.join(config.BASE_DATASETS_DIR, 'flights_rank')
        df_featured = feature_engineering.engineer_features(
            df_submission,
            df_apt,
            flights_dir=flights_rank_dir,
            start_col='start',
            end_col='end',
            desc="Engineering Features for Submission"
        )

        # Ensure all one-hot encoded engine columns from training are present
        # and fill any missing numerical features with 0 (or appropriate median/mean)
        for col in feature_cols:
            if col not in df_featured.columns:
                # If a feature is missing, add it and fill with 0 (common for one-hot encoded or new features)
                df_featured[col] = 0.0

        # Reorder columns to match the training features
        X_test = df_featured[feature_cols]

        # --- 5. Generate Predictions ---
        print("Generating predictions...")
        predictions = model.predict(X_test)

        # --- 6. Fill Submission Template ---
        print("Filling submission template with predictions...")
        df_submission_template['fuel_kg'] = predictions

        # --- 7. Save Submission File ---
        print("Saving submission file...")
        df_submission_template.rename(columns={'fuel_kg': 'fuel_consumption'}, inplace=True)
        submission_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')
        df_submission_template.to_parquet(submission_path, index=False)

        print(f"\nSuccessfully generated {len(df_submission_template)} predictions.")
        print(f"Submission file created/updated at: {submission_path}")
        print("\nSubmission Head:")
        print(df_submission_template.head())

    except FileNotFoundError as e:
        print(f"Error: Missing a required file: {e.filename}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
