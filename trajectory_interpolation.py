import pandas as pd
import os
import logging
import glob
from datetime import datetime
import config # Import the config module
from tqdm import tqdm # Import tqdm for progress bar
import warnings
from openap.phase import FlightPhase # Import openap

# Suppress specific warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_path = os.path.join('logs', 'trajectory_interpolation.log')
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(config.LOG_LEVEL)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# --- End Logger Setup ---

def detect_phases_with_openap(df, file_name):
    """Detects flight phases using the openap library, following the correct usage."""
    required_cols = ['altitude', 'groundspeed', 'vertical_rate']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"One or more required columns for OpenAP ({required_cols}) are missing. Skipping phase detection.")
        df['phase'] = 'NA' # Use 'NA' as per openap documentation for unlabeled
        return df

    # Ensure timestamp is a column, not the index
    if df.index.name == 'timestamp':
        df = df.reset_index()

    # Check for empty or all-NaN data before processing
    if df[required_cols].isna().all().all():
        logger.warning(f"Key columns for phase detection are all NaN. Skipping OpenAP in {file_name}.")
        df['phase'] = 'NA'
        return df

    ts = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds()
    alt = df.altitude.values
    spd = df.groundspeed.values
    roc = df.vertical_rate.values

    fp = FlightPhase()
    fp.set_trajectory(ts, alt, spd, roc)
    
    df['phase'] = fp.phaselabel()
    return df

def interpolate_trajectories():
    """
    Processes trajectory files, optionally detects phases, injects segment boundaries, 
    interpolates missing values, and saves the processed files.
    """
    input_base_dir = os.path.join(config.DATA_DIR, 'filtered_trajectories')
    output_base_dir = config.INTERPOLATED_TRAJECTORIES_DIR
    fuel_data_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets')

    logger.info(f"Starting trajectory interpolation process.")
    logger.info(f"Input base directory: {input_base_dir}")
    logger.info(f"Output base directory: {output_base_dir}")
    
    logger.info("Loading all fuel data to get segment boundaries...")
    fuel_files = glob.glob(os.path.join(fuel_data_dir, 'fuel_*.parquet'))
    if not fuel_files:
        raise FileNotFoundError("No fuel files found. Cannot inject segment boundaries.")
    
    all_fuel_data = pd.concat([pd.read_parquet(f) for f in fuel_files], ignore_index=True)
    all_fuel_data['start'] = pd.to_datetime(all_fuel_data['start'])
    all_fuel_data['end'] = pd.to_datetime(all_fuel_data['end'])
    all_fuel_data['flight_id'] = all_fuel_data['flight_id'].astype(str)

    files_to_process = []
    sub_dirs = ['flights_train', 'flights_rank', 'flights_final']
    for sub_dir in sub_dirs:
        input_dir = os.path.join(input_base_dir, sub_dir)
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory not found, skipping: {input_dir}")
            continue
        trajectory_files = glob.glob(os.path.join(input_dir, '*.parquet'))
        for file_path in trajectory_files:
            files_to_process.append((file_path, sub_dir))

    columns_to_interpolate = ['altitude', 'groundspeed', 'latitude', 'longitude', 'track', 'vertical_rate']

    for file_path, sub_dir in tqdm(files_to_process, desc="Interpolating trajectories", unit="file"):
        file_name = os.path.basename(file_path)
        flight_id = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_base_dir, sub_dir)
        output_file_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            df = pd.read_parquet(file_path)
            
            if df.empty:
                logger.warning(f"Skipping empty trajectory file: {file_name}")
                continue

            logger.debug(f"Processing {file_name}: {len(df)} rows.")
            raw_nan_counts = df[columns_to_interpolate].isna().sum()
            logger.debug(f"Raw NaN counts for {file_name}:\n{raw_nan_counts[raw_nan_counts > 0]}")

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            flight_segments = all_fuel_data[all_fuel_data['flight_id'] == flight_id]
            if not flight_segments.empty:
                start_times = flight_segments[['start']].rename(columns={'start': 'timestamp'})
                end_times = flight_segments[['end']].rename(columns={'end': 'timestamp'})
                boundary_times = pd.concat([start_times, end_times], ignore_index=True).drop_duplicates()
                df = pd.concat([df, boundary_times]).drop_duplicates(subset='timestamp', keep='first')
            
            df = df.set_index('timestamp').sort_index()

            # Perform Interpolation
            for col in columns_to_interpolate:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # OpenAP Phase Detection
            df = detect_phases_with_openap(df, file_name)

            df.reset_index().to_parquet(output_file_path, index=False)

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}", exc_info=True)

    logger.info(f"Finished processing all trajectory files.")

if __name__ == '__main__':
    interpolate_trajectories()
