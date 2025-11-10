import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
import config

# --- Constants ---
# Input data file paths
FLIGHTLIST_TRAIN_PATH = os.path.join(config.BASE_DATASETS_DIR, 'flightlist_train.parquet')
FUEL_TRAIN_PATH = os.path.join(config.BASE_DATASETS_DIR, 'fuel_train.parquet')
AC_PERF_PATH = os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv')

# Output data file path
OUTPUT_CSV_PATH = os.path.join(config.PROCESSED_DATA_DIR, 'average_load_factor_by_airport_pair_v3.csv')
LOG_FILE_PATH = os.path.join(config.INTROSPECTION_DIR, 'load_factor_estimation_v3.log')

# --- Setup Logging ---
# Configure logging to write to a file and to the console
# Remove existing handlers to avoid duplicate logs if the script is re-run
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler()
    ]
)

# Enable progress_apply for pandas
tqdm.pandas()


def get_segment_conditions(segment, trajectories_dir):
    """
    For a given fuel segment (row), load its trajectory file and calculate
    the average flight conditions during that segment's time window.
    """
    try:
        flight_id = segment['flight_id']
        start_time = segment['start']
        end_time = segment['end']

        trajectory_path = os.path.join(trajectories_dir, f'{flight_id}.parquet')
        if not os.path.exists(trajectory_path):
            return pd.Series({'avg_altitude': np.nan, 'avg_mach': np.nan, 'avg_vertical_rate': np.nan})

        traj_df = pd.read_parquet(trajectory_path)

        # Ensure timestamp is datetime
        traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])

        # Filter for the specific segment time window
        mask = (traj_df['timestamp'] >= start_time) & (traj_df['timestamp'] <= end_time)
        segment_traj = traj_df.loc[mask]

        if segment_traj.empty:
            return pd.Series({'avg_altitude': np.nan, 'avg_mach': np.nan, 'avg_vertical_rate': np.nan})

        # Calculate average conditions
        avg_altitude = segment_traj['altitude'].mean()
        avg_mach = segment_traj['mach'].mean()
        avg_vertical_rate = segment_traj['vertical_rate'].mean()

        return pd.Series({
            'avg_altitude': avg_altitude,
            'avg_mach': avg_mach,
            'avg_vertical_rate': avg_vertical_rate
        })
    except Exception as e:
        logging.warning(f"Could not process segment for flight {segment.get('flight_id', 'N/A')}: {e}")
        return pd.Series({'avg_altitude': np.nan, 'avg_mach': np.nan, 'avg_vertical_rate': np.nan})


def define_flight_phase(vertical_rate):
    """Categorize the flight phase based on vertical rate."""
    if vertical_rate > 300:
        return 'Climb'
    elif vertical_rate < -300:
        return 'Descent'
    else:
        return 'Cruise'


def estimate_load_factor_per_segment():
    """
    Main function to estimate load factor using a per-segment analysis.
    """
    try:
        # --- 1. Load Data ---
        logging.info("--- Starting Data Loading ---")
        flightlist_train = pd.read_parquet(FLIGHTLIST_TRAIN_PATH)
        fuel_train = pd.read_parquet(FUEL_TRAIN_PATH)
        logging.info(f"Loaded {FLIGHTLIST_TRAIN_PATH} and {FUEL_TRAIN_PATH}")

        # --- 2. Enrich Fuel Segments with Flight Conditions ---
        logging.info("--- Enriching Fuel Segments with Trajectory Data ---")
        logging.info("This step may take a while as it processes each segment...")

        # Apply the function to each row of the fuel_train dataframe
        segment_conditions = fuel_train.progress_apply(get_segment_conditions, trajectories_dir=config.FLIGHTS_TRAIN_DIR,
                                                       axis=1)
        fuel_segments_enriched = pd.concat([fuel_train, segment_conditions], axis=1)

        # Drop segments where trajectory data could not be found or was empty
        fuel_segments_enriched.dropna(subset=['avg_altitude', 'avg_mach', 'avg_vertical_rate'], inplace=True)
        logging.info(f"Successfully enriched {len(fuel_segments_enriched)} segments.")
        logging.info("Head of enriched fuel segments:\n" + fuel_segments_enriched.head().to_string())

        # --- 3. Feature Engineering ---
        logging.info("--- Performing Feature Engineering on Segments ---")

        # Calculate segment duration and fuel flow
        fuel_segments_enriched['duration_s'] = (
                    fuel_segments_enriched['end'] - fuel_segments_enriched['start']).dt.total_seconds()
        # Avoid division by zero for zero-duration segments
        fuel_segments_enriched = fuel_segments_enriched[fuel_segments_enriched['duration_s'] > 0]
        fuel_segments_enriched['fuel_flow_kg_s'] = fuel_segments_enriched['fuel_kg'] / fuel_segments_enriched[
            'duration_s']

        # Define flight phase
        fuel_segments_enriched['flight_phase'] = fuel_segments_enriched['avg_vertical_rate'].apply(define_flight_phase)

        # Bin altitude to create comparable groups
        alt_bins = range(0, 50001, 5000)  # Bins of 5000 feet up to 50k
        alt_labels = [f'{i}-{i + 5000}ft' for i in alt_bins[:-1]]
        fuel_segments_enriched['altitude_bin'] = pd.cut(fuel_segments_enriched['avg_altitude'], bins=alt_bins,
                                                        labels=alt_labels, right=False)

        # Merge with flightlist to get aircraft_type
        fuel_segments_enriched = pd.merge(fuel_segments_enriched, flightlist_train[['flight_id', 'aircraft_type']],
                                          on='flight_id', how='left')
        fuel_segments_enriched.dropna(subset=['aircraft_type', 'altitude_bin'], inplace=True)

        logging.info("Head of feature-engineered segment data:\n" + fuel_segments_enriched[
            ['flight_id', 'aircraft_type', 'fuel_flow_kg_s', 'flight_phase', 'altitude_bin']].head().to_string())

        # --- 4. Normalize Fuel Flow within Groups to Estimate Load Factor ---
        logging.info("--- Normalizing Fuel Flow to Estimate Segment Load Factor ---")
        group_key = ['aircraft_type', 'flight_phase', 'altitude_bin']

        # Use transform to get min/max for each segment's group
        fuel_segments_enriched['min_flow'] = fuel_segments_enriched.groupby(group_key)['fuel_flow_kg_s'].transform(
            'min')
        fuel_segments_enriched['max_flow'] = fuel_segments_enriched.groupby(group_key)['fuel_flow_kg_s'].transform(
            'max')

        # Calculate load factor
        flow_range = fuel_segments_enriched['max_flow'] - fuel_segments_enriched['min_flow']

        # Estimate load factor (handle cases where range is 0)
        fuel_segments_enriched['segment_load_factor'] = np.where(
            flow_range > 0,
            (fuel_segments_enriched['fuel_flow_kg_s'] - fuel_segments_enriched['min_flow']) / flow_range,
            0.5  # Assume median load factor if no variation in the group
        )

        # Clip values to be between 0 and 1
        fuel_segments_enriched['segment_load_factor'] = fuel_segments_enriched['segment_load_factor'].clip(0, 1)
        logging.info("Head of data with estimated segment load factor:\n" + fuel_segments_enriched[
            ['flight_id', 'segment_load_factor']].head().to_string())

        # --- 5. Aggregate to Flight and then to Airport Pair ---
        logging.info("--- Aggregating Load Factors to Flight and Airport Pair ---")

        # Average segment load factors for each flight
        flight_load_factors = fuel_segments_enriched.groupby('flight_id')['segment_load_factor'].mean().reset_index()
        flight_load_factors.rename(columns={'segment_load_factor': 'avg_flight_load_factor'}, inplace=True)

        # Merge with original flightlist to get origin/destination
        final_data = pd.merge(flightlist_train, flight_load_factors, on='flight_id', how='inner')

        # Calculate average load factor for each airport pair
        airport_pair_key = ['origin_icao', 'destination_icao']
        average_load_factor = final_data.groupby(airport_pair_key)['avg_flight_load_factor'].mean().reset_index()
        average_load_factor.rename(columns={'avg_flight_load_factor': 'average_load_factor'}, inplace=True)

        logging.info("Head of final average load factor by airport pair:\n" + average_load_factor.head().to_string())

        # --- 6. Save Output ---
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        logging.info(f"--- Saving Results to {OUTPUT_CSV_PATH} ---")
        average_load_factor.to_csv(OUTPUT_CSV_PATH, index=False)
        logging.info("Process completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: A required file was not found - {e}. Please check your config.py and data directories.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    estimate_load_factor_per_segment()