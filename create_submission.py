import os
import pandas as pd
import numpy as np
import json
import joblib
import datetime
from tqdm import tqdm
import config
import feature_engineering
import introspection
import logging

# --- Setup Logging ---
log_file = 'create_submission.log'
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

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
        logging.info("Loading model and feature list...")
        model_path = os.path.join(model_dir_path, "model.joblib")
        features_path = os.path.join(model_dir_path, "features.json")

        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)

        logging.info("Model and feature list loaded successfully.")

        # --- 2. Load Data ---
        logging.info("Loading submission template, flight data, and airport data...")
        submission_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')
        df_submission_template = pd.read_parquet(submission_path)

        flightlist_path = os.path.join(config.BASE_DATASETS_DIR, 'flightlist_rank.parquet')
        df_flightlist_rank = pd.read_parquet(flightlist_path)

        ac_perf_path = os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv')
        df_ac_perf = pd.read_csv(ac_perf_path)

        apt_path = os.path.join(config.BASE_DATASETS_DIR, 'apt.parquet')
        df_apt = pd.read_parquet(apt_path)

        logging.info("Data loaded successfully.")

        # --- 3. Merge and Preprocess Data ---
        logging.info("Merging and preprocessing dataframes...")
        df_merged = pd.merge(df_submission_template, df_flightlist_rank, on='flight_id', how='left')
        df_merged = pd.merge(df_merged, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
        df_merged = pd.merge(df_merged, df_apt.add_prefix('origin_'), left_on='origin_icao', right_on='origin_icao', how='left')
        df_merged = pd.merge(df_merged, df_apt.add_prefix('destination_'), left_on='destination_icao', right_on='destination_icao', how='left')

        df_merged['start'] = pd.to_datetime(df_merged['start'])
        df_merged['end'] = pd.to_datetime(df_merged['end'])
        df_merged.reset_index(drop=True, inplace=True)

        logging.info("Dataframes merged and preprocessed successfully.")

        # --- 4. Engineer Features ---
        interpolated_flights_rank_dir = os.path.join(config.DATA_DIR, 'interpolated_trajectories/flights_rank')
        if not os.path.exists(interpolated_flights_rank_dir):
            logging.error(f"Error: Interpolated trajectory directory not found: {interpolated_flights_rank_dir}")
            logging.error("Please run the 'interpolate_trajectories' stage first.")
            return

        df_featured = feature_engineering.engineer_features(
            df_merged,
            flights_dir=interpolated_flights_rank_dir,
            start_col='start',
            end_col='end',
            desc="Engineering Features for Submission"
        )

        # --- 5. Drop Columns ---
        logging.info("Dropping columns with high missing values or low correlation...")
        cols_to_drop = [
            'engine_options_A319-113', 'engine_options_A319-114', 'engine_options_A319-111', 'engine_options_A319-112',
            'engine_options_A319-132',
            'engine_options_A319-133', 'engine_options_A319-131', 'engine_options_A319-115', 'engine_options_A330-323',
            'engine_options_A330-322',
            'engine_options_A330-343', 'engine_options_A330-302', 'engine_options_A330-303', 'engine_options_A330-321',
            'engine_options_A330-301',
            'engine_options_A321-251N', 'engine_options_A321-272N', 'engine_options_A321-271N', 'engine_options_A321-252N',
            'engine_options_A321-253N',
            'engine_options_A321-232', 'engine_options_A321-231', 'engine_options_A321-211', 'engine_options_A321-131',
            'engine_options_A321-111',
            'engine_options_A321-212', 'engine_options_A321-213', 'engine_options_A321-112', 'engine_options_A330-243',
            'engine_options_A330-201',
            'engine_options_A330-223', 'engine_options_A330-203', 'engine_options_A330-223F', 'engine_options_A330-202',
            'engine_options',
            'engine_options_A350-941', 'engine_options_A320-214', 'engine_options_A320-211', 'engine_options_A320-111',
            'engine_options_A320-212',
            'engine_options_A320-215', 'engine_options_A320-231', 'engine_options_A320-216', 'engine_options_A320-232',
            'engine_options_A320-233',
            'engine_options_A320-271N', 'engine_options_A320-253N', 'engine_options_A320-272N', 'engine_options_A320-251N',
            'engine_options_A320-252N',
            'engine_options_A320-273N', 'engine_options_A330-341', 'engine_options_A330-342', 'engine_options_A318-112',
            'engine_options_A318-121','engine_options_A318-111','engine_options_A318-122','engine_options_A380-861',
            'engine_options_A380-842', 'engine_options_A380-841','fuel_aircraft','fuel_engine','unknown_notrajectory_fraction',
            'flaps_lambda_f','cruise_height','engine_number','clean_cd0','clean_k','clean_gears',
            'destination_RWY_2_ELEVATION_a',
            'destination_RWY_2_ELEVATION_b',
            'origin_RWY_2_ELEVATION_b',
            'origin_RWY_2_ELEVATION_a',
            'destination_RWY_6_ELEVATION_b',
            'destination_RWY_6_ELEVATION_a',
            'destination_RWY_7_ELEVATION_b',
            'destination_RWY_7_ELEVATION_a',
            'destination_RWY_4_ELEVATION_b',
            'destination_RWY_4_ELEVATION_a',
            'destination_RWY_3_ELEVATION_b',
            'destination_RWY_3_ELEVATION_a',
            'destination_RWY_1_ELEVATION_a',
            'destination_RWY_1_ELEVATION_b',
            'destination_RWY_5_ELEVATION_b',
            'destination_RWY_5_ELEVATION_a',
            'destination_RWY_8_ELEVATION_a',
            'destination_RWY_8_ELEVATION_b',
            'origin_RWY_8_ELEVATION_b',
            'origin_RWY_8_ELEVATION_a',
            'origin_RWY_6_ELEVATION_b',
            'origin_RWY_6_ELEVATION_a',
            'origin_RWY_7_ELEVATION_b',
            'origin_RWY_7_ELEVATION_a',
            'origin_RWY_4_ELEVATION_b',
            'origin_RWY_4_ELEVATION_a',
            'origin_RWY_3_ELEVATION_b',
            'origin_RWY_3_ELEVATION_a',
            'origin_RWY_1_ELEVATION_a',
            'origin_RWY_1_ELEVATION_b',
            'origin_RWY_5_ELEVATION_b',
            'origin_RWY_5_ELEVATION_a',
        ]
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_featured.columns]
        if cols_to_drop_existing:
            df_featured.drop(columns=cols_to_drop_existing, inplace=True)

        # --- 6. Impute Missing Values ---
        logging.info("Imputing missing values...")

        # Impute trajectory data with median
        trajectory_cols = ['std_vertical_rate', 'ending_altitude', 'altitude_difference', 'mean_track', 'std_track',
                               'starting_altitude', 'mean_vertical_rate', 'mean_dist_to_origin_km', 'mean_dist_to_dest_km']
        for col in trajectory_cols:
            if col in df_featured.columns and df_featured[col].isnull().any():
                median_val = df_featured[col].median()
                df_featured[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed missing values in '{col}' with median value: {median_val}")

        # Impute aircraft data with mode
        aircraft_cols = ['engine_default', 'flaps_type', 'engine_mount']
        for col in aircraft_cols:
            if col in df_featured.columns and df_featured[col].isnull().any():
                mode_val = df_featured[col].mode()[0]
                df_featured[col].fillna(mode_val, inplace=True)
                logging.info(f"Imputed missing values in '{col}' with mode value: {mode_val}")

        # --- 7. Generate Introspection Files and Save Data ---
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        introspection.generate_introspection_files(
            df_featured, 
            "create_submission", 
            run_id
        )
        output_filename = "featured_data_submission.parquet"
        output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)
        df_featured.to_parquet(output_path, index=False)
        logging.info(f"Saved feature-rich submission data to {output_path}")

        # --- 8. Align Columns and Predict ---
        logging.info("Aligning columns with the trained model...")
        for col in feature_cols:
            if col not in df_featured.columns:
                df_featured[col] = 0.0
        
        X_test = df_featured[feature_cols]

        logging.info("Generating predictions...")
        predictions = model.predict(X_test)

        # --- 9. Fill Submission Template ---
        logging.info("Filling submission template with predictions...")
        df_submission_template['fuel_kg'] = predictions

        # --- 10. Save Submission File ---
        logging.info("Saving submission file...")
        submission_output_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')
        df_submission_template.to_parquet(submission_output_path, index=False)

        print(f"\nSuccessfully generated {len(df_submission_template)} predictions.")
        print(f"Submission file created/updated at: {submission_output_path}")
        print("\nSubmission Head:")
        print(df_submission_template.head())

    except FileNotFoundError as e:
        print(f"Error: Missing a required file: {e.filename}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
