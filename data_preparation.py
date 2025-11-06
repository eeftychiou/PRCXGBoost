"""
This script handles the data preparation stage of the ML pipeline.

It performs the following steps:
1.  Loads the raw metadata and airport coordinates.
2.  Merges the datasets, including detailed aircraft performance data and all airport data.
3.  If in test mode, it samples a fraction of the data.
4.  Applies the full feature engineering pipeline.
5.  Generates comprehensive introspection files.
6.  Saves the final feature-rich DataFrame for the training stage.
"""
import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import config
import feature_engineering
import introspection

def prepare_data():
    """
    Loads, preprocesses, and saves the data for the pipeline.
    This function now expects trajectory files to be pre-interpolated
    and located in the 'data/interpolated_trajectories' directory.
    """
    print("--- Starting Data Preparation Stage ---")

    # --- 1. Load Data ---
    print("Loading raw data...")
    try:
        df_fuel = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/fuel_train.parquet'))
        df_flightlist = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/flightlist_train.parquet'))
        df_ac_perf = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv'))
        df_apt = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/apt.parquet'))
    except FileNotFoundError as e:
        print(f"Error: Raw data file not found. {e}")
        return

    # --- 2. Preprocess and Merge Data ---
    print("Preprocessing and merging all datasets...")


    # Merge flight list with aircraft performance data
    df_merged = pd.merge(df_flightlist, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
    
    # Merge with fuel data
    df_merged = pd.merge(df_fuel, df_merged, on='flight_id', how='left')

    # Merge with all airport data for origin and destination
    df_merged = pd.merge(df_merged, df_apt.add_prefix('origin_'), left_on='origin_icao', right_on='origin_icao', how='left')
    df_merged = pd.merge(df_merged, df_apt.add_prefix('destination_'), left_on='destination_icao', right_on='destination_icao', how='left')

    df_merged['start'] = pd.to_datetime(df_merged['start'])
    df_merged['end'] = pd.to_datetime(df_merged['end'])
    df_merged.dropna(subset=['fuel_kg', 'aircraft_type'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # --- 3. Handle Test Run ---
    if config.TEST_RUN:
        print(f"Test run enabled. Sampling {config.TEST_RUN_FRACTION:.0%} of the data.")
        df_to_process = df_merged.sample(frac=config.TEST_RUN_FRACTION, random_state=42).copy()
    else:
        print("Full run mode. Processing all data.")
        df_to_process = df_merged

    # --- 4. Engineer Features ---
    # IMPORTANT: Now reading from the interpolated trajectories directory
    interpolated_flights_train_dir = os.path.join(config.DATA_DIR, 'interpolated_trajectories/flights_train')
    if not os.path.exists(interpolated_flights_train_dir):
        print(f"Error: Interpolated trajectory directory not found: {interpolated_flights_train_dir}")
        print("Please run the 'interpolate_trajectories' stage first.")
        return

    df_featured = feature_engineering.engineer_features(
        df_to_process,
        flights_dir=interpolated_flights_train_dir, # Use the interpolated directory
        start_col='start',
        end_col='end',
        desc="Engineering Features"
    )

    # Drop columns with very high (or 100%) missing values and low correlation
    # This list is derived from data_preparation_missing_values.csv and target correlation
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
        'flaps_lambda_f','cruise_height','engine_number','clean_cd0','clean_k','clean_gears'

    ]

    # Filter out columns that don't exist in the DataFrame to avoid errors
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_featured.columns]
    if cols_to_drop_existing:
        df_featured.drop(columns=cols_to_drop_existing, inplace=True)




    # --- 5. Generate Introspection Files ---
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    introspection.generate_introspection_files(
        df_featured, 
        "data_preparation", 
        run_id, 
        target_variable='fuel_kg'
    )

    # --- 6. Save Processed Data ---
    output_filename = f"featured_data{'test' if config.TEST_RUN else ''}.parquet"
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)
    df_featured.to_parquet(output_path, index=False)
    print(f"Saved feature-rich data to {output_path}")

    print("--- Data Preparation Stage Complete ---")

if __name__ == '__main__':
    prepare_data()
