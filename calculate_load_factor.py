import pandas as pd
import logging
import os
import config

# --- Constants ---
# Input data file paths
FLIGHTLIST_TRAIN_PATH = os.path.join(config.BASE_DATASETS_DIR, 'flightlist_train.parquet')
FUEL_TRAIN_PATH = os.path.join(config.BASE_DATASETS_DIR, 'fuel_train.parquet')
AC_PERF_PATH = os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv')

# Output data file path
OUTPUT_CSV_PATH = os.path.join(config.PROCESSED_DATA_DIR, 'average_load_factor_by_airport_pair.csv')
LOG_FILE_PATH = os.path.join(config.INTROSPECTION_DIR, 'load_factor_estimation.log')

# --- Setup Logging ---
# Configure logging to write to a file and to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler()
    ]
)

def estimate_average_load_factor():
    """
    This program estimates the average load factor for each airport pair based on flight and fuel consumption data.

    The core logic is based on the assumption that for a given aircraft type on a specific route, the amount of fuel
    consumed is primarily influenced by the aircraft's total weight. A heavier aircraft, carrying more payload
    (passengers and cargo), will burn more fuel.

    The methodology is as follows:
    1.  Calculate the total fuel consumed for each flight.
    2.  Merge flight information, total fuel, and aircraft performance data into a single dataset.
    3.  For each unique combination of route (origin-destination) and aircraft type, determine the minimum and
        maximum observed fuel consumption. These values serve as proxies for flights with the lowest and highest
        payloads, respectively.
    4.  Estimate the load factor for each individual flight by normalizing its fuel consumption within the observed
        min-max range for its group. A flight with fuel consumption close to the minimum will have a load factor
        near 0, while a flight with fuel consumption near the maximum will have a load factor close to 1.
    5.  Finally, calculate the average of these estimated load factors for each airport pair.
    """
    try:
        # --- 1. Load Data ---
        logging.info("--- Starting Data Loading ---")

        if not all(os.path.exists(p) for p in [FLIGHTLIST_TRAIN_PATH, FUEL_TRAIN_PATH, AC_PERF_PATH]):
            logging.error("One or more input data files are missing. Please check the paths in config.py.")
            return

        flightlist_train = pd.read_parquet(FLIGHTLIST_TRAIN_PATH)
        logging.info(f"Loaded {FLIGHTLIST_TRAIN_PATH}")
        logging.info("Head of flightlist_train:\n" + flightlist_train.head().to_string())

        fuel_train = pd.read_parquet(FUEL_TRAIN_PATH)
        logging.info(f"Loaded {FUEL_TRAIN_PATH}")
        logging.info("Head of fuel_train:\n" + fuel_train.head().to_string())

        ac_perf = pd.read_csv(AC_PERF_PATH)
        logging.info(f"Loaded {AC_PERF_PATH}")
        logging.info("Head of acPerfOpenAP:\n" + ac_perf.head().to_string())
        logging.info("--- Data Loading Complete ---")

        # --- 2. Preprocess Data ---
        logging.info("--- Starting Data Preprocessing ---")

        # Calculate total fuel consumed for each flight
        logging.info("Calculating total fuel per flight...")
        total_fuel_per_flight = fuel_train.groupby('flight_id')['fuel_kg'].sum().reset_index()
        total_fuel_per_flight.rename(columns={'fuel_kg': 'total_fuel_kg'}, inplace=True)
        logging.info("Head of total_fuel_per_flight:\n" + total_fuel_per_flight.head().to_string())

        # Merge datasets
        logging.info("Merging datasets...")
        # Merge flight list with total fuel
        flight_data = pd.merge(flightlist_train, total_fuel_per_flight, on='flight_id', how='inner')

        # Merge with aircraft performance data
        flight_data = pd.merge(flight_data, ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')

        # Check for flights where aircraft performance data was not found
        missing_ac_perf = flight_data['ICAO_TYPE_CODE'].isnull().sum()
        if missing_ac_perf > 0:
            logging.warning(f"{missing_ac_perf} flights did not have corresponding aircraft performance data and will be excluded.")
            flight_data.dropna(subset=['ICAO_TYPE_CODE'], inplace=True)


        logging.info("Final merged dataset head:\n" + flight_data.head().to_string())
        logging.info("--- Data Preprocessing Complete ---")


        # --- 3. Calculate Load Factor per Flight ---
        logging.info("--- Estimating Load Factor for Each Flight ---")

        # Define the grouping key for identifying similar flights
        group_key = ['origin_icao', 'destination_icao', 'aircraft_type']

        # Calculate min and max fuel consumption for each group
        logging.info("Calculating min and max fuel consumption for each route-aircraft group...")
        fuel_range_by_group = flight_data.groupby(group_key)['total_fuel_kg'].agg(['min', 'max']).reset_index()
        fuel_range_by_group.rename(columns={'min': 'min_fuel_kg', 'max': 'max_fuel_kg'}, inplace=True)

        # Merge the fuel range back into the main flight data
        flight_data = pd.merge(flight_data, fuel_range_by_group, on=group_key, how='left')

        # Calculate the load factor
        logging.info("Calculating normalized load factor for each flight...")

        # Calculate the range of fuel consumption for each group
        fuel_range = flight_data['max_fuel_kg'] - flight_data['min_fuel_kg']

        # Estimate load factor based on where the flight's fuel consumption falls within the range
        # For groups with no variation in fuel (range is 0), we assume a median load factor of 0.5
        flight_data['estimated_load_factor'] = (flight_data['total_fuel_kg'] - flight_data['min_fuel_kg']) / fuel_range
        flight_data['estimated_load_factor'].fillna(0.5, inplace=True) # Handle division by zero by filling with 0.5

        # Ensure load factor is within the logical bounds of 0 and 1
        flight_data['estimated_load_factor'] = flight_data['estimated_load_factor'].clip(0, 1)

        logging.info("Head of data with estimated load factor:\n" + flight_data[['flight_id', 'origin_icao', 'destination_icao', 'aircraft_type', 'total_fuel_kg', 'min_fuel_kg', 'max_fuel_kg', 'estimated_load_factor']].head().to_string())
        logging.info("--- Load Factor Estimation Complete ---")

        # --- 4. Calculate Average Load Factor per Airport Pair ---
        logging.info("--- Calculating Average Load Factor per Airport Pair ---")

        airport_pair_key = ['origin_icao', 'destination_icao']
        average_load_factor = flight_data.groupby(airport_pair_key)['estimated_load_factor'].mean().reset_index()
        average_load_factor.rename(columns={'estimated_load_factor': 'average_load_factor'}, inplace=True)

        logging.info("Head of average load factor by airport pair:\n" + average_load_factor.head().to_string())
        logging.info("--- Average Load Factor Calculation Complete ---")

        # --- 5. Save Output ---
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        logging.info(f"--- Saving Results to {OUTPUT_CSV_PATH} ---")
        average_load_factor.to_csv(OUTPUT_CSV_PATH, index=False)
        logging.info("Process completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found - {e}. Please ensure all required files are in the directory.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    estimate_average_load_factor()