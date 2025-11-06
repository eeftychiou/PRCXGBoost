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
logger.setLevel(logging.INFO) # Set the minimum level for the logger

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a FileHandler for logging to a file
log_file_path = os.path.join(config.INTROSPECTION_DIR, 'trajectory_interpolation.log')
os.makedirs(config.INTROSPECTION_DIR, exist_ok=True) # Ensure log directory exists
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO) # Log INFO and above to file
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a StreamHandler for logging to console (only warnings and errors)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING) # Log WARNING and above to console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- End Logger Setup ---


def interpolate_trajectories(test_mode: bool = False, sample_size: int = 5):
    """
    Processes trajectory files from specified input directories, interpolates missing values,
    and saves the processed files to output directories, using paths from config.py.
    Includes progress tracking and resumability.

    Args:
        test_mode (bool): If True, runs in test mode, processing only a small sample
                          and reporting on missing value reduction.
        sample_size (int): Number of sample files to save for debugging/verification
                           (only applies if not in test_mode).
    """
    input_base_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets') # Assuming raw trajectories are here
    output_base_dir = config.INTERPOLATED_TRAJECTORIES_DIR

    logger.info(f"Starting trajectory interpolation process.")
    logger.info(f"Input base directory: {input_base_dir}")
    logger.info(f"Output base directory: {output_base_dir}")
    if test_mode:
        logger.warning("Running in TEST MODE. Only a limited number of files will be processed for verification.")
        test_output_dir = os.path.join(output_base_dir, 'test_output')
        os.makedirs(test_output_dir, exist_ok=True)
        logger.info(f"Test output will be saved to: {test_output_dir}")
    
    # Columns identified for interpolation from data_profile_report.txt
    columns_to_interpolate = [
        'CAS', 'TAS', 'altitude', 'groundspeed', 'latitude', 'longitude',
        'mach', 'track', 'vertical_rate'
    ]
    
    # Directories to process
    sub_dirs = ['flights_train', 'flights_rank']

    for sub_dir in sub_dirs:
        input_dir = os.path.join(input_base_dir, sub_dir)
        output_dir = os.path.join(output_base_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing directory: {input_dir}")
        
        # Assuming trajectory files are in Parquet format
        trajectory_files = glob.glob(os.path.join(input_dir, '*.parquet'))
        
        if not trajectory_files:
            logger.warning(f"No .parquet files found in {input_dir}. Skipping.")
            continue

        # --- Resumability: Load already processed files ---
        processed_log_path = os.path.join(output_dir, 'processed_files.txt')
        already_processed_files = set()
        if os.path.exists(processed_log_path):
            with open(processed_log_path, 'r') as f:
                for line in f:
                    already_processed_files.add(line.strip())
            logger.info(f"Found {len(already_processed_files)} previously processed files in {processed_log_path}.")

        processed_count = 0
        sample_files_saved = 0

        # Limit files for test_mode
        files_to_process = trajectory_files[:1] if test_mode else trajectory_files # Process only 1 file in test mode

        # --- Progress Bar: Wrap the loop with tqdm ---
        for file_path in tqdm(files_to_process, desc=f"Interpolating {sub_dir}", unit="file"):
            file_name = os.path.basename(file_path)

            # --- Resumability: Check if already processed ---
            if file_name in already_processed_files:
                logger.info(f"Skipping already processed file: {file_name}")
                processed_count += 1 # Count skipped files as processed for overall progress
                continue

            try:
                df_original = pd.read_parquet(file_path) # Keep original for comparison in test mode
                df = df_original.copy() # Work on a copy
                
                # Ensure timestamp is datetime and sort
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values(by=['flight_id', 'timestamp'])
                else:
                    logger.warning(f"Timestamp column not found in {file_path}. Skipping sorting.")

                initial_missing_counts = df[columns_to_interpolate].isnull().sum()
                
                # Apply interpolation per flight_id
                for col in columns_to_interpolate:
                    if col in df.columns:
                        df[col] = df.groupby('flight_id')[col].transform(
                            lambda group: group.interpolate(method='linear', limit_direction='both', limit_area='inside')
                        )
                    else:
                        logger.warning(f"Column '{col}' not found in {file_path}. Skipping interpolation for this column.")
                
                # Handle any remaining NaNs (e.g., if an entire flight has missing values for a column)
                for col in columns_to_interpolate:
                    if col in df.columns and df[col].isnull().any():
                        df[col].fillna(0, inplace=True) # Fill remaining NaNs with 0
                        logger.info(f"Filled remaining NaNs in column '{col}' for {file_name} with 0.")

                final_missing_counts = df[columns_to_interpolate].isnull().sum()
                
                if initial_missing_counts.sum() > 0:
                    logger.info(f"Interpolation summary for {file_name}:")
                    for col in columns_to_interpolate:
                        if col in df.columns:
                            if initial_missing_counts[col] > 0:
                                logger.info(f"  Column '{col}': Initial missing = {initial_missing_counts[col]}, Final missing = {final_missing_counts[col]}")

                # Determine output path
                if test_mode:
                    current_output_dir = test_output_dir
                else:
                    current_output_dir = output_dir
                output_file_path = os.path.join(current_output_dir, file_name)

                # Save processed file
                df.to_parquet(output_file_path, index=False)
                processed_count += 1

                # --- Resumability: Mark file as processed ---
                with open(processed_log_path, 'a') as f:
                    f.write(f"{file_name}\n")
                logger.info(f"Successfully processed and marked as complete: {file_name}")

                # Test mode specific reporting
                if test_mode:
                    print(f"\n--- Test Report for {file_name} ---")
                    print("Original Missing Values:")
                    print(initial_missing_counts[initial_missing_counts > 0])
                    print("\nMissing Values After Interpolation:")
                    print(final_missing_counts[final_missing_counts > 0])
                    
                    all_filled = True
                    for col in columns_to_interpolate:
                        if col in df.columns and initial_missing_counts[col] > 0 and final_missing_counts[col] > 0:
                            print(f"WARNING: Column '{col}' still has {final_missing_counts[col]} missing values after interpolation.")
                            all_filled = False
                    if all_filled:
                        print("SUCCESS: All targeted missing values appear to be filled.")
                    else:
                        print("WARNING: Some missing values remain. Check log for details.")
                    print(f"Processed file saved to: {output_file_path}")
                    print("-------------------------------------------\n")


                # Save sample files for debugging (only if not in test_mode)
                if not test_mode and sample_files_saved < sample_size:
                    sample_output_dir = os.path.join(output_base_dir, 'samples')
                    os.makedirs(sample_output_dir, exist_ok=True)
                    sample_file_name = f"interpolated_sample_{sub_dir}_{file_name}"
                    df.head().to_csv(os.path.join(sample_output_dir, sample_file_name.replace('.parquet', '.csv')), index=False)
                    logger.info(f"Saved sample of interpolated data to {os.path.join(sample_output_dir, sample_file_name.replace('.parquet', '.csv'))}")
                    sample_files_saved += 1

            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}", exc_info=True) # exc_info=True to log traceback
        
        logger.info(f"Finished processing directory: {input_dir}. Processed {processed_count} files.")

    logger.info("Trajectory interpolation process completed.")

if __name__ == '__main__':
    # This part is for testing the module independently
    # Use config paths for example usage
    example_input_base_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets')
    example_output_base_dir = config.INTERPOLATED_TRAJECTORIES_DIR

    # Create dummy input directories and files for testing if they don't exist
    for sd in ['flights_train', 'flights_rank']:
        os.makedirs(os.path.join(example_input_base_dir, sd), exist_ok=True)
        # Create a dummy parquet file with some missing values
        dummy_data = {
            'flight_id': ['flight_A', 'flight_A', 'flight_A', 'flight_B', 'flight_B', 'flight_B'],
            'timestamp': [
                datetime(2025, 1, 1, 10, 0, 0), datetime(2025, 1, 1, 10, 0, 10), datetime(2025, 1, 1, 10, 0, 20),
                datetime(2025, 1, 1, 11, 0, 0), datetime(2025, 1, 1, 11, 0, 10), datetime(2025, 1, 1, 11, 0, 20)
            ],
            'CAS': [100, None, 120, 200, 210, None],
            'TAS': [110, 115, None, None, 220, 230],
            'altitude': [1000, 1100, 1200, 5000, None, 5200],
            'groundspeed': [90, None, 110, 190, 200, None],
            'latitude': [30.0, 30.1, None, 40.0, 40.1, 40.2],
            'longitude': [-90.0, None, -89.8, -100.0, -100.1, None],
            'mach': [0.2, None, 0.25, 0.5, 0.51, None],
            'track': [45, 46, None, 90, None, 92],
            'vertical_rate': [1000, None, 1200, 500, 550, None]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_parquet(os.path.join(example_input_base_dir, sd, f'dummy_trajectory_{sd}.parquet'), index=False)
        logger.info(f"Created dummy file: {os.path.join(example_input_base_dir, sd, f'dummy_trajectory_{sd}.parquet')}")

    # Run in normal mode
    logger.info("--- Running interpolation in normal mode (from __main__) ---")
    interpolate_trajectories() 

    # Run in test mode
    logger.info("\n--- Running interpolation in TEST MODE (from __main__) ---")
    interpolate_trajectories(test_mode=True)
