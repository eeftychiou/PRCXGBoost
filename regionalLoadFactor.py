
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
import config

# --- Constants ---
# Input data file paths for all stages
FLIGHTLIST_PATHS = {
    'train': os.path.join(config.BASE_DATASETS_DIR, 'flightlist_train.parquet'),
    'rank': os.path.join(config.BASE_DATASETS_DIR, 'flightlist_rank.parquet'),
    'final': os.path.join(config.BASE_DATASETS_DIR, 'flightlist_final.parquet')
}

# Output data file path
OUTPUT_CSV_PATH = os.path.join(config.PROCESSED_DATA_DIR, 'average_load_factor_by_airport_pair_v3.csv')
LOG_FILE_PATH = os.path.join(config.INTROSPECTION_DIR, 'load_factor_estimation_v3.log')

# --- IATA Data (August 2025 - Page 5 of PDF) ---
# Values are percentages (0.0 - 1.0)
IATA_PLF_2025 = {
    'International': {
        'Africa': 0.797,
        'Asia Pacific': 0.851,
        'Europe': 0.872,
        'Latin America': 0.847,  # Includes Caribbean
        'Middle East': 0.839,
        'North America': 0.875
    },
    'Domestic_Specific': {
        'Australia': 0.834,
        'Brazil': 0.851,
        'China': 0.855,
        'India': 0.832,
        'Japan': 0.896,
        'USA': 0.842
    },
    'Domestic_Global_Average': 0.863  # Fallback for domestic markets not listed above
}

# --- Setup Logging ---
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

tqdm.pandas()


def get_region_and_country_from_icao(icao_code):
    """
    Determines the IATA Region and a pseudo-Country identifier based on ICAO prefixes.

    Returns:
        tuple: (Region_Name, Country_ID)
    """
    if not isinstance(icao_code, str) or len(icao_code) < 2:
        return 'Unknown', 'Unknown'

    prefix_1 = icao_code[0]
    prefix_2 = icao_code[:2]

    # --- North America ---
    if prefix_1 == 'C': return 'North America', 'Canada'
    if prefix_1 == 'K': return 'North America', 'USA'
    # Alaska (PA, PF, PO, PP) and Hawaii (PH) are USA
    if prefix_2 in ['PA', 'PF', 'PO', 'PP', 'PH']: return 'North America', 'USA'

    # --- Europe ---
    # E (North Europe), L (South Europe), B (Greenland/Iceland)
    # U (Russia/CIS - typically grouped with Europe for broad IATA stats or split,
    # but geographically vast. We map to Europe for standard IATA PLF alignment).
    if prefix_1 in ['E', 'L', 'U']: return 'Europe', prefix_2
    if prefix_1 == 'B':
        if prefix_2 == 'BG': return 'North America', 'Greenland'  # Greenland is geographically NA
        return 'Europe', 'Iceland'  # BI is Iceland

    # --- Asia Pacific ---
    # Y (Australia), Z (China), R (Japan/Korea/Taiwan), V (South Asia/Mainland SE Asia),
    # W (Maritime SE Asia), P (Pacific - PT, PK etc, excluding US PAC)
    if prefix_1 == 'Y': return 'Asia Pacific', 'Australia'
    if prefix_1 == 'Z':
        if prefix_2 == 'ZK': return 'Asia Pacific', 'North Korea'  # Rare
        return 'Asia Pacific', 'China'
    if prefix_2 == 'RJ': return 'Asia Pacific', 'Japan'
    if prefix_2 == 'RO': return 'Asia Pacific', 'Japan'  # Okinawa
    if prefix_2 == 'RK': return 'Asia Pacific', 'Korea'
    if prefix_2 == 'RC': return 'Asia Pacific', 'Taiwan'
    if prefix_2 == 'RP': return 'Asia Pacific', 'Philippines'
    if prefix_2 == 'VT': return 'Asia Pacific', 'Thailand'
    if prefix_2 == 'VV': return 'Asia Pacific', 'Vietnam'
    if prefix_2 == 'VO': return 'Asia Pacific', 'India'  # South
    if prefix_2 == 'VA': return 'Asia Pacific', 'India'  # West
    if prefix_2 == 'VE': return 'Asia Pacific', 'India'  # East
    if prefix_2 == 'VI': return 'Asia Pacific', 'India'  # North

    # Catch-all for other Asia codes
    if prefix_1 in ['R', 'V', 'W', 'A', 'N', 'P']: return 'Asia Pacific', prefix_2

    # --- Middle East ---
    # O (Middle East), but OP (Pakistan) and OA (Afghanistan) are Asia in IATA regions
    if prefix_1 == 'O':
        if prefix_2 in ['OP', 'OA']: return 'Asia Pacific', prefix_2
        return 'Middle East', prefix_2

    # --- Africa ---
    if prefix_1 in ['D', 'F', 'H', 'G']: return 'Africa', prefix_2

    # --- Latin America & Caribbean ---
    # S (South America), M (Mexico/Central America), T (Caribbean)
    if prefix_1 == 'S':
        if prefix_2 == 'SB': return 'Latin America', 'Brazil'
        return 'Latin America', prefix_2  # Other South America
    if prefix_1 in ['M', 'T']: return 'Latin America', prefix_2

    return 'Unknown', 'Unknown'


def calculate_load_factor(row):
    """
    Calculates the estimated load factor for a route based on Origin/Dest IATA regions.
    """
    origin = row['origin_icao']
    dest = row['destination_icao']

    reg_o, ctry_o = get_region_and_country_from_icao(origin)
    reg_d, ctry_d = get_region_and_country_from_icao(dest)

    # 1. Handle Unknowns - Fallback to Global International Avg roughly (~86%)
    if reg_o == 'Unknown' or reg_d == 'Unknown':
        return 0.86

    # 2. Check for Domestic Flights
    # Definition: Same Region AND Same Country ID
    if reg_o == reg_d and ctry_o == ctry_d:
        # Check if it is a "Key Domestic Market" defined in IATA report
        if ctry_o == 'Australia': return IATA_PLF_2025['Domestic_Specific']['Australia']
        if ctry_o == 'Brazil': return IATA_PLF_2025['Domestic_Specific']['Brazil']
        if ctry_o == 'China': return IATA_PLF_2025['Domestic_Specific']['China']
        if ctry_o == 'India': return IATA_PLF_2025['Domestic_Specific']['India']
        if ctry_o == 'Japan': return IATA_PLF_2025['Domestic_Specific']['Japan']
        if ctry_o == 'USA': return IATA_PLF_2025['Domestic_Specific']['USA']

        # If not a key market, use the Global Domestic Average
        return IATA_PLF_2025['Domestic_Global_Average']

    # 3. Handle International Flights

    # Case A: Intra-Regional International (e.g., France to Germany)
    if reg_o == reg_d:
        return IATA_PLF_2025['International'].get(reg_o, 0.86)

    # Case B: Inter-Regional International (e.g., Europe to North America)
    # We take the average of the International PLF of the Origin Region and the Destination Region
    # (as traffic flows are bidirectional and reports often average route areas)
    plf_o = IATA_PLF_2025['International'].get(reg_o, 0.86)
    plf_d = IATA_PLF_2025['International'].get(reg_d, 0.86)

    return (plf_o + plf_d) / 2.0


def generate_load_factor_csv():
    try:
        # --- 1. Load Data ---
        logging.info("--- Starting Data Loading ---")
        
        all_flightlists = []
        for stage_name, path in FLIGHTLIST_PATHS.items():
            if not os.path.exists(path):
                logging.warning(f"Flightlist for stage '{stage_name}' not found at {path}. Skipping this stage.")
                continue
            df_stage = pd.read_parquet(path)
            all_flightlists.append(df_stage)
            logging.info(f"Loaded flightlist_{stage_name}.parquet with {len(df_stage)} rows.")

        if not all_flightlists:
            raise FileNotFoundError("No flightlist files were found to process.")

        combined_flightlist = pd.concat(all_flightlists, ignore_index=True)
        logging.info(f"Combined all flightlists into a single DataFrame with {len(combined_flightlist)} rows.")

        # --- 2. Extract Unique Airport Pairs ---
        # We only need one row per route to calculate the IATA average
        logging.info("Extracting unique Origin-Destination pairs...")
        unique_routes = combined_flightlist[['origin_icao', 'destination_icao']].drop_duplicates().copy()
        logging.info(f"Found {len(unique_routes)} unique routes.")

        # --- 3. Apply IATA Load Factor Logic ---
        logging.info("--- Applying IATA Regional Load Factors (August 2025 Data) ---")

        # We use progress_apply for visibility, though it should be fast
        unique_routes['average_load_factor'] = unique_routes.progress_apply(calculate_load_factor, axis=1)

        # Round to 4 decimals for cleanliness
        unique_routes['average_load_factor'] = unique_routes['average_load_factor'].round(4)

        logging.info("Head of processed data:\n" + unique_routes.head().to_string())

        # --- 4. Save Output ---
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        logging.info(f"--- Saving Results to {OUTPUT_CSV_PATH} ---")
        unique_routes.to_csv(OUTPUT_CSV_PATH, index=False)
        logging.info("Process completed successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    generate_load_factor_csv()