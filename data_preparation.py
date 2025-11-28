"""
This script handles the data preparation stage of the ML pipeline.

It performs the following steps:
1.  Loads the raw metadata and airport coordinates.
2.  Merges the datasets, including detailed aircraft performance data and all airport data.
3.  If in test mode, it samples a fraction of the data.
4.  Applies the full feature engineering pipeline.
5.  Drops columns with high missing values.
6.  Generates comprehensive introspection files.
7.  Saves the final feature-rich DataFrame for the training stage.
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
import metar_utils

import logging

# --- Setup Logging ---
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', 'data_preparation.log')
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def _get_columns_to_drop():
    """Returns a list of columns to drop based on high missing values and low correlation."""
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
        'phase_fraction_unknown',
        'alt_diff_rev_std'
    ]

def correct_timestamps_for_all():
    """
    Corrects takeoff and landing times for all datasets ('train', 'rank', 'final')
    and saves them to intermediate files. This is the new first step.
    """
    logging.info("--- Starting Timestamp Correction Stage ---")
    dataset_types = ['train', 'rank', 'final']
    df_apt = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/apt.parquet'))

    for dataset_type in dataset_types:
        logging.info(f"Correcting timestamps for {dataset_type.upper()} dataset...")
        
        flightlist_path = os.path.join(config.DATA_DIR, f'prc-2025-datasets/flightlist_{dataset_type}.parquet')
        df_flightlist = pd.read_parquet(flightlist_path)
        
        trajectories_dir = os.path.join(config.INTERPOLATED_TRAJECTORIES_DIR, f'flights_{dataset_type}')
        if not os.path.exists(trajectories_dir):
            logging.error(f"Trajectory directory not found for {dataset_type}: {trajectories_dir}. Skipping.")
            continue
            
        df_flightlist_corrected = correct_date.joincdates(df_flightlist, df_apt, trajectories_dir)
        
        output_filename = f"corrected_flightlist_{dataset_type}.parquet"
        output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)
        df_flightlist_corrected.to_parquet(output_path, index=False)
        logging.info(f"Saved corrected flightlist for {dataset_type} to {output_path}")
        
    logging.info("--- Timestamp Correction Stage Complete ---")


def prepare_data():
    """
    Loads, preprocesses, and saves the data for the pipeline.
    This function now starts from the corrected flightlists and uses a flight-keyed weather file.
    """
    logging.info("--- Starting Data Preparation Stage (Single Run) ---")
    
    metar_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_metars.parquet')
    
    try:
        df_metar = pd.read_parquet(metar_path)
        logging.info(f"Successfully loaded processed METAR data from {metar_path}")
    except FileNotFoundError:
        logging.warning(f"Processed METAR file not found at {metar_path}. Weather features will not be added. If this is unexpected, please run the 'prepare_metars' stage first.")
        df_metar = None

    dataset_types = ['final', 'train', 'rank']

    for dataset_type in dataset_types:
        logging.info(f"--- Processing {dataset_type.upper()} Dataset ---")

        # --- 1. Load Corrected and Raw Data ---
        logging.info(f"Loading data for {dataset_type}...")

        try:
            corrected_flightlist_path = os.path.join(config.PROCESSED_DATA_DIR, f'corrected_flightlist_{dataset_type}.parquet')

            df_flightlist_corrected = pd.read_parquet(corrected_flightlist_path)

            if dataset_type=='final':
                rank_corrected_flightlist_path = os.path.join(config.PROCESSED_DATA_DIR, 'corrected_flightlist_rank.parquet')
                df_rank_corrected = pd.read_parquet(rank_corrected_flightlist_path)
                df_flightlist_corrected = pd.concat([df_flightlist_corrected, df_rank_corrected])


            fuel_path = os.path.join(config.DATA_DIR, f'prc-2025-datasets/fuel_{dataset_type}.parquet')
            df_fuel = pd.read_parquet(fuel_path)

            if config.TEST_RUN:
                logging.info(f"Test run enabled. Sampling {config.TEST_RUN_FRACTION:.0%} of the {dataset_type} data.")
                valid_flight_ids = df_fuel['flight_id'].unique()
                df_flightlist_corrected = df_flightlist_corrected[df_flightlist_corrected['flight_id'].isin(valid_flight_ids)]
                
                sampled_flight_ids = df_flightlist_corrected['flight_id'].sample(frac=config.TEST_RUN_FRACTION, random_state=42).unique()
                df_flightlist_corrected = df_flightlist_corrected[df_flightlist_corrected['flight_id'].isin(sampled_flight_ids)].copy()
                df_fuel = df_fuel[df_fuel['flight_id'].isin(sampled_flight_ids)].copy()
                
                logging.info(f"Sampled {dataset_type} flightlist has {df_flightlist_corrected.shape[0]} rows.")

            df_ac_perf = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv'))
            df_apt = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/apt.parquet'))
            
            trajectories_dir = os.path.join(config.INTERPOLATED_TRAJECTORIES_DIR, f'flights_final')

        except FileNotFoundError as e:
            logging.error(f"Error: A required data file was not found for {dataset_type}. {e}. Please ensure 'correct_timestamps' has been run.")
            continue

        # --- 2. Merge Data (starting from corrected flightlist) ---
        logging.info(f"Merging all datasets for {dataset_type}...")

        df_merged = pd.merge(df_flightlist_corrected, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
        df_merged = pd.merge(df_merged, df_fuel, on='flight_id', how='right')
        df_merged = pd.merge(df_merged, df_apt.add_prefix('origin_'), left_on='origin_icao', right_on='origin_icao', how='left')
        df_merged = pd.merge(df_merged, df_apt.add_prefix('destination_'), left_on='destination_icao', right_on='destination_icao', how='left')

        # --- 3. Merge Weather Data (using simple flight_id merge) ---
        if df_metar is not None:
            logging.info("Merging weather data on 'flight_id'...")
            df_merged = pd.merge(df_merged, df_metar, on='flight_id', how='left')
            logging.info("Weather data merged.")
        
        df_to_process = df_merged

        # --- 4. Engineer Features ---
        df_featured = augment_features.augment_features(
            df_to_process,
            trajectories_folder=trajectories_dir
        )

        # --- 5. Drop Columns with High Missing Values ---
        cols_to_drop = _get_columns_to_drop()
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_featured.columns]
        if cols_to_drop_existing:
            df_featured.drop(columns=cols_to_drop_existing, inplace=True)

        # --- 6. Generate Introspection Files ---
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        introspection.generate_introspection_files(
            df_featured,
            f"data_preparation_before_imputation_{dataset_type}",
            run_id,
            target_variable='fuel_kg' if dataset_type == 'train' else None
        )

        # --- 7. Save Processed Data ---
        test_suffix = '_test' if config.TEST_RUN else ''
        output_filename = f"featured_data_{dataset_type}{test_suffix}.parquet"
        output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)
        df_featured.to_parquet(output_path, index=False)
        logging.info(f"Saved feature-rich data for {dataset_type} to {output_path}")

    logging.info("--- Data Preparation Stage Complete ---")

if __name__ == '__main__':
    # This main block can be used for direct testing if needed,
    # but the primary entry point is now run_pipeline.py
    correct_timestamps_for_all()
    prepare_data()
