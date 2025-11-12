import pandas as pd
import os
import logging
import glob
from datetime import datetime
import config # Import the config module
from tqdm import tqdm # Import tqdm for progress bar
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# --- Logger Setup ---
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set the minimum level for the logger to capture all levels of messages

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a FileHandler for logging errors to a file
log_file_path = os.path.join(config.INTROSPECTION_DIR, 'trajectory_interpolation.log')
os.makedirs(config.INTROSPECTION_DIR, exist_ok=True) # Ensure log directory exists
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(config.LOG_LEVEL) # Use LOG_LEVEL from config for file
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a StreamHandler for logging to console
# Suppress console output to only show progress bar and fatal errors
console_handler = logging.StreamHandler()
console_handler.setLevel(config.LOG_LEVEL) # Use LOG_LEVEL from config for console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- End Logger Setup ---

def create_interpolation_diff(df_original, df_interpolated, file_name, columns_to_interpolate):
    """
    Compares original and interpolated dataframes to find differences and saves them to a CSV file.
    """
    diff_list = []
    # Align indices to be safe, assuming row order is preserved.
    df_orig = df_original.reset_index(drop=True)
    df_interp = df_interpolated.reset_index(drop=True)

    for col in columns_to_interpolate:
        if col in df_orig.columns and col in df_interp.columns:
            # Find where original was null and interpolated is not.
            interpolated_mask = df_orig[col].isnull() & df_interp[col].notnull()

            if interpolated_mask.any():
                # Identify columns present in both dataframes for diffing
                diff_cols = ['timestamp', 'flight_id']
                if all(c in df_orig.columns for c in diff_cols):
                    diff_df = pd.DataFrame({
                        'timestamp': df_orig.loc[interpolated_mask, 'timestamp'],
                        'flight_id': df_orig.loc[interpolated_mask, 'flight_id'],
                        'column': col,
                        'original_value': df_orig.loc[interpolated_mask, col],
                        'interpolated_value': df_interp.loc[interpolated_mask, col]
                    })
                    diff_list.append(diff_df)
                else:
                    logger.warning(f"Skipping diff for column '{col}' in {file_name} due to missing 'timestamp' or 'flight_id'.")


    if diff_list:
        full_diff_df = pd.concat(diff_list, ignore_index=True)
        diff_filename = f"interpolation_diff_{os.path.splitext(file_name)[0]}.csv"
        diff_filepath = os.path.join(config.INTROSPECTION_DIR, diff_filename)
        os.makedirs(config.INTROSPECTION_DIR, exist_ok=True) # Ensure dir exists
        full_diff_df.to_csv(diff_filepath, index=False)
        logger.info(f"Saved interpolation differences to {diff_filepath}")
    else:
        logger.info(f"No interpolation differences found to log for {file_name}.")

def interpolate_trajectories(test_mode: bool = False, sample_size: int = 5, diff_all_files: bool = False):
    """
    Processes trajectory files from specified input directories, interpolates missing values,
    and saves the processed files to output directories, using paths from config.py.
    Includes progress tracking and resumability.

    Args:
        test_mode (bool): If True, runs in test mode, processing only a small sample
                          and reporting on missing value reduction.
        sample_size (int): Number of sample files to save for debugging/verification
                           (only applies if not in test_mode).
        diff_all_files (bool): If True, creates a diff file for every processed file.
                               Otherwise, diffs are only created for samples in normal mode.
    """
    input_base_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets') # Assuming raw trajectories are here
    output_base_dir = config.INTERPOLATED_TRAJECTORIES_DIR

    logger.info(f"Starting trajectory interpolation process.")
    logger.info(f"Input base directory: {input_base_dir}")
    logger.info(f"Output base directory: {output_base_dir}")
    if test_mode:
        # In test mode, prompt the user for a filename.
        test_filename = input("Enter the filename of the trajectory file to process (from flights_train): ")
        test_file_path = os.path.join(input_base_dir, 'flights_train', test_filename)

        logger.warning(f"Running in TEST MODE. Processing single file: {test_file_path}")
        if not os.path.exists(test_file_path):
            logger.error(f"Test file not found: {test_file_path}")
            return

        files_to_process = [(test_file_path, 'flights_train')]
        test_output_dir = os.path.join(output_base_dir, 'test_output')
        os.makedirs(test_output_dir, exist_ok=True)
        logger.info(f"Test output will be saved to: {test_output_dir}")
    else:
        files_to_process = []
        sub_dirs = ['flights_train', 'flights_rank']
        for sub_dir in sub_dirs:
            input_dir = os.path.join(input_base_dir, sub_dir)
            trajectory_files = glob.glob(os.path.join(input_dir, '*.parquet'))
            for file_path in trajectory_files:
                files_to_process.append((file_path, sub_dir))

    columns_to_interpolate = [
        'CAS', 'TAS', 'altitude', 'groundspeed', 'latitude', 'longitude',
        'mach', 'track', 'vertical_rate'
    ]

    processed_count = 0
    sample_files_saved = 0

    for file_path, sub_dir in tqdm(files_to_process, desc="Interpolating trajectories", unit="file"):
        file_name = os.path.basename(file_path)
        output_dir = os.path.join(output_base_dir, sub_dir)
        output_file_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_file_path) and not test_mode:
            logger.info(f"Skipping already processed file: {file_name}")
            continue

        try:
            df_original = pd.read_parquet(file_path)
            df = df_original.copy()

            time_indexed = False
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values(by='timestamp')
                if df.duplicated(subset=['timestamp']).any():
                    logger.warning(f"Found duplicate timestamps in {file_name}. Keeping first entry.")
                    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
                df.set_index('timestamp', inplace=True)
                time_indexed = True
            else:
                logger.warning(f"Timestamp column not found in {file_path}. Skipping time-based interpolation.")

            initial_missing_counts = df[columns_to_interpolate].isnull().sum()

            for col in columns_to_interpolate:
                if col in df.columns:
                    # Check if there are enough data points for pchip interpolation
                    if df[col].notna().sum() >= 2:
                        df[col] = df.groupby('flight_id')[col].transform(
                            lambda group: group.interpolate(method='pchip', limit_direction='both')
                        )
                    else:
                        # Fallback to linear interpolation if not enough data points
                        df[col] = df.groupby('flight_id')[col].transform(
                            lambda group: group.interpolate(method='linear', limit_direction='both')
                        )
                    df[col].fillna(0, inplace=True)
                else:
                    logger.warning(f"Column '{col}' not found in {file_path}. Skipping interpolation.")

            if time_indexed:
                df.reset_index(inplace=True)

            if test_mode:
                output_filename = file_name.replace('.parquet', '.csv')
                output_file_path = os.path.join(test_output_dir, output_filename)
                df.to_csv(output_file_path, index=False)
                logger.info(f"Saved test output to {output_file_path}")
            else:
                df.to_parquet(output_file_path, index=False)

            processed_count += 1

            if test_mode:
                create_interpolation_diff(df_original, df, file_name, columns_to_interpolate)
            else:  # not test_mode
                if diff_all_files:
                    create_interpolation_diff(df_original, df, file_name, columns_to_interpolate)
                elif sample_files_saved < sample_size:
                    create_interpolation_diff(df_original, df, file_name, columns_to_interpolate)

            if not test_mode and sample_files_saved < sample_size:
                sample_output_dir = os.path.join(output_base_dir, 'samples')
                os.makedirs(sample_output_dir, exist_ok=True)
                sample_file_name = f"interpolated_sample_{sub_dir}_{file_name}"
                df.head().to_csv(os.path.join(sample_output_dir, sample_file_name.replace('.parquet', '.csv')), index=False)
                logger.info(f"Saved sample of interpolated data to {os.path.join(sample_output_dir, sample_file_name.replace('.parquet', '.csv'))}")
                sample_files_saved += 1

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}", exc_info=True)

    logger.info(f"Finished processing. Processed {processed_count} files.")

if __name__ == '__main__':
    # To run in test mode from the command line:
    # python trajectory_interpolation.py --test
    import argparse
    parser = argparse.ArgumentParser(description='Interpolate trajectory data.')
    parser.add_argument('--test', action='store_true', help='Run in test mode.')
    parser.add_argument('--diff-all', action='store_true', help='Create diff files for all processed files.')
    args = parser.parse_args()

    interpolate_trajectories(test_mode=args.test, diff_all_files=args.diff_all)
