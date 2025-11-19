"""
This script handles the data preparation stage of the ML pipeline.

It performs the following steps:
1.  Loads the raw metadata and airport coordinates.
2.  Merges the datasets, including detailed aircraft performance data and all airport data.
3.  If in test mode, it samples a fraction of the data.
4.  Applies the full feature engineering pipeline.
5.  Drops columns with high missing values.
6.  Imputes remaining missing values.
7.  Generates comprehensive introspection files.
8.  Saves the final feature-rich DataFrame for the training stage.
"""
import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import config
import augment_features
import introspection
import correct_date
import glob

import logging

# --- Setup Logging ---
log_file = 'data_preparation.log'
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def _get_columns_to_drop():
    """Returns a list of columns to drop based on high missing values and low correlation."""
    # This list is derived from data_preparation_missing_values.csv and target correlation
    return [
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
        # Added based on 20251111-182738 after-imputation introspection (extremely high missing)
        'phase_fraction_unknown',
        # High missing rate among engineered trajectory dispersion features
        'alt_diff_rev_std'
    ]

def prepare_data():
    """
    Loads, preprocesses, and saves the data for the pipeline.
    This function now expects trajectory files to be pre-interpolated
    and located in the 'data/interpolated_trajectories' directory.
    It processes 'train', 'rank', and 'final' datasets.
    """
    logging.info("--- Starting Data Preparation Stage ---")

    dataset_types = ['train', 'rank', 'final']

    for dataset_type in dataset_types:
        logging.info(f"--- Processing {dataset_type.upper()} Dataset ---")

        # --- 1. Load Data ---
        logging.info(f"Loading raw data for {dataset_type}...")

        try:
            flightlist_path = os.path.join(config.DATA_DIR, f'prc-2025-datasets/flightlist_{dataset_type}.parquet')
            df_flightlist = pd.read_parquet(flightlist_path)

            fuel_path = os.path.join(config.DATA_DIR, f'prc-2025-datasets/fuel_{dataset_type}.parquet')
            df_fuel = pd.read_parquet(fuel_path)

            # --- Handle Test Run ---
            if config.TEST_RUN:
                logging.info(f"Test run enabled. Sampling {config.TEST_RUN_FRACTION:.0%} of the {dataset_type} data from the beginning.")
                df_flightlist = df_flightlist.sample(frac=config.TEST_RUN_FRACTION, random_state=42).copy()
                logging.info(f"Sampled {dataset_type} flightlist has {df_flightlist.shape[0]} rows.")


            df_ac_perf = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv'))
            df_apt = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/apt.parquet'))
            
            # Dynamically set trajectory path
            trajectories_dir = os.path.join(config.INTERPOLATED_TRAJECTORIES_DIR, f'flights_{dataset_type}')
            if not os.path.exists(trajectories_dir):
                logging.error(f"Trajectory directory not found for {dataset_type}: {trajectories_dir}")
                continue

        except FileNotFoundError as e:
            logging.error(f"Error: Raw data file not found for {dataset_type}. {e}")
            continue

        # --- 2. Preprocess and Merge Data ---
        logging.info(f"Preprocessing and merging all datasets for {dataset_type}...")

        # Correct dates on the flight list BEFORE merging
        logging.info("Correcting takeoff and landing times...")
        df_flightlist_corrected = correct_date.joincdates(df_flightlist, df_apt, trajectories_dir)

        # Merge flight list with aircraft performance data
        df_merged = pd.merge(df_flightlist_corrected, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
        
        # Merge with fuel data (right merge to keep all flight segments)
        df_merged = pd.merge(df_merged, df_fuel, on='flight_id', how='right')

        # Merge with all airport data for origin and destination
        df_merged = pd.merge(df_merged, df_apt.add_prefix('origin_'), left_on='origin_icao', right_on='origin_icao', how='left')
        df_merged = pd.merge(df_merged, df_apt.add_prefix('destination_'), left_on='destination_icao', right_on='destination_icao', how='left')

        df_merged['start'] = pd.to_datetime(df_merged['start'])
        df_merged['end'] = pd.to_datetime(df_merged['end'])
        
        if dataset_type == 'train':
            df_merged.dropna(subset=['fuel_kg', 'aircraft_type'], inplace=True)
        else:
            df_merged.dropna(subset=['aircraft_type'], inplace=True)

        df_merged.reset_index(drop=True, inplace=True)

        logging.info(f"Load and Merged {dataset_type}. Dataset has {df_merged.shape[0]} rows and {df_merged.shape[1]} columns")
        
        df_to_process = df_merged

        # --- 4. Engineer Features ---
        df_featured = augment_features.augment_features(
            df_to_process,
            trajectories_folder=trajectories_dir
        )

        # --- 5. Drop Columns with High Missing Values ---
        logging.info("Dropping columns with high missing values...")
        cols_to_drop = _get_columns_to_drop()
        
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_featured.columns]
        if cols_to_drop_existing:
            df_featured.drop(columns=cols_to_drop_existing, inplace=True)
            logging.info(f"Dropped {len(cols_to_drop_existing)} columns for {dataset_type}.")

        # --- 6. Impute Missing Values ---
        logging.info(f"Imputing missing values for {dataset_type}...")

        trajectory_cols = ['std_vertical_rate', 'ending_altitude', 'altitude_difference', 'mean_track', 'std_track',
                               'starting_altitude', 'mean_vertical_rate', 'mean_dist_to_origin_km', 'mean_dist_to_dest_km']
        for col in trajectory_cols:
            if col in df_featured.columns and df_featured[col].isnull().any():
                median_val = df_featured[col].median()
                df_featured[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed missing values in '{col}' with median value: {median_val}")

        seg_cols = [c for c in df_featured.columns if c.startswith('seg_')]
        for col in seg_cols:
            if col in df_featured.columns and pd.api.types.is_numeric_dtype(df_featured[col]) and df_featured[col].isnull().any():
                median_val = df_featured[col].median()
                df_featured[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed missing values in segment feature '{col}' with median value: {median_val}")

        aircraft_cols = ['engine_default', 'flaps_type', 'engine_mount']
        for col in aircraft_cols:
            if col in df_featured.columns and df_featured[col].isnull().any():
                mode_val = df_featured[col].mode()[0]
                df_featured[col].fillna(mode_val, inplace=True)
                logging.info(f"Imputed missing values in '{col}' with mode value: {mode_val}")

        # --- 7. Generate Introspection Files ---
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        introspection.generate_introspection_files(
            df_featured,
            f"data_preparation_after_imputation_{dataset_type}",
            run_id,
            target_variable='fuel_kg' if dataset_type == 'train' else None
        )

        # --- 8. Save Processed Data ---
        logging.info(f"Data Preparation for {dataset_type} Finalized. Dataset has {df_featured.shape[0]} rows and {df_featured.shape[1]} columns")
        
        test_suffix = '_test' if config.TEST_RUN else ''
        output_filename = f"featured_data_{dataset_type}{test_suffix}.parquet"
        output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)
        df_featured.to_parquet(output_path, index=False)
        logging.info(f"Saved feature-rich data for {dataset_type} to {output_path}")

    logging.info("--- Data Preparation Stage Complete ---")

if __name__ == '__main__':
    prepare_data()
