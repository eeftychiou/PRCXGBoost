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

def prepare_data():
    """Loads, preprocesses, and saves the data for the pipeline."""
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
    
    # Select and rename columns from acPerfOpenAP
    ac_perf_cols = [
        'ICAO_TYPE_CODE', 'mtow', 'mlw', 'oew', 'mfc', 'vmo', 'mmo', 'ceiling', 'pax_max', 
        'fuselage_length', 'fuselage_height', 'fuselage_width', 'wing_area', 'wing_span', 
        'wing_mac', 'wing_sweep', 'wing_t/c', 'cruise_height', 'cruise_mach', 'cruise_range', 
        'engine_number', 'drag_cd0', 'drag_k', 'engine_type', 'engine_mount', 'flaps_type'
    ]
    df_ac_perf_selected = df_ac_perf[ac_perf_cols]

    # Merge flight list with aircraft performance data
    df_merged = pd.merge(df_flightlist, df_ac_perf_selected, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
    
    # Merge with fuel data
    df_merged = pd.merge(df_fuel, df_merged, on='flight_id', how='left')

    # Merge with all airport data for origin and destination
    df_merged = pd.merge(df_merged, df_apt.add_prefix('origin_'), left_on='origin_icao', right_on='origin_icao', how='left')
    df_merged = pd.merge(df_merged, df_apt.add_prefix('destination_'), left_on='destination_icao', right_on='destination_icao', how='left')

    # Handle categorical features before filling missing numerical values
    categorical_cols = ['engine_type', 'engine_mount', 'flaps_type']
    for col in categorical_cols:
        if col in df_merged.columns and not df_merged[col].mode().empty:
            df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0])
            
    df_merged = pd.get_dummies(df_merged, columns=categorical_cols, prefix=categorical_cols, dtype=float)

    # Handle numerical features
    numerical_cols = df_merged.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        df_merged[col] = df_merged[col].fillna(df_merged[col].median())

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
    flights_train_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets/flights_train')
    df_featured = feature_engineering.engineer_features(
        df_to_process,
        flights_dir=flights_train_dir,
        start_col='start',
        end_col='end',
        desc="Engineering Features"
    )

    # --- 5. Generate Introspection Files ---
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    introspection_run_dir = os.path.join(config.INTROSPECTION_DIR, run_id)
    os.makedirs(introspection_run_dir, exist_ok=True)
    print(f"Generating introspection files in {introspection_run_dir}")

    df_featured.to_csv(os.path.join(introspection_run_dir, "enhanced_data.csv"), index=False)

    # --- 6. Save Processed Data ---
    output_filename = f"featured_data{'test' if config.TEST_RUN else ''}.parquet"
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)
    df_featured.to_parquet(output_path, index=False)
    print(f"Saved feature-rich data to {output_path}")

    print("--- Data Preparation Stage Complete ---")

if __name__ == '__main__':
    prepare_data()
