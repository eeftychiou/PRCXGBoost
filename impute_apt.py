"""
This script enriches the apt.parquet file by scraping airport data from SkyVector.
It focuses only on airports present in the flightlist_train.parquet and 
flightlist_rank.parquet files, and adds a delay between requests.
"""
import pandas as pd
import shutil
import requests
from bs4 import BeautifulSoup
import re
import os
import logging
import time

# --- Setup Logging ---
log_file = 'impute_apt.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file, mode='w'),
                        logging.StreamHandler()
                    ])

# --- Define Paths ---
apt_file_path = 'data/prc-2025-datasets/apt.parquet'
train_flights_path = 'data/prc-2025-datasets/flightlist_train.parquet'
rank_flights_path = 'data/prc-2025-datasets/flightlist_rank.parquet'
original_apt_file_path = apt_file_path + '.original.parquet'
HTML_DIR = 'htmlfile'  # Directory to save HTML files for debugging

# Create HTML directory if it doesn't exist
os.makedirs(HTML_DIR, exist_ok=True)

def scrape_airport_data():
    # --- Backup Original File ---
    if not os.path.exists(original_apt_file_path):
        shutil.copy(apt_file_path, original_apt_file_path)

    # --- Load Data ---
    apt_df = pd.read_parquet(apt_file_path)

    # Ensure the 'elevation' column exists and is of a numeric type
    if 'elevation' not in apt_df.columns:
        apt_df['elevation'] = pd.NA
    apt_df['elevation'] = pd.to_numeric(apt_df['elevation'], errors='coerce')

    # --- Determine Airports to Process ---
    try:
        train_df = pd.read_parquet(train_flights_path)
        rank_df = pd.read_parquet(rank_flights_path)
        
        relevant_airports = set(train_df['origin_icao'].unique()) | \
                            set(train_df['destination_icao'].unique()) | \
                            set(rank_df['origin_icao'].unique()) | \
                            set(rank_df['destination_icao'].unique())
        
        # Filter for relevant airports
        airports_to_process_df = apt_df[
            apt_df['icao'].isin(relevant_airports)
        ].copy()
        
        logging.info(f"Found {len(relevant_airports)} unique airports in flight lists.")
        logging.info(f"Will process {len(airports_to_process_df)} relevant airports.")

    except FileNotFoundError as e:
        logging.error(f"Flight list file not found: {e}. Processing all airports with missing elevation as a fallback.")
        airports_to_process_df = apt_df[apt_df['elevation'].isnull()].copy()

    # --- Process Airports ---
    updated_data = []

    for index, row in airports_to_process_df.iterrows():
        icao_code = row['icao']
        # Skip if ICAO code is not a 4-letter string
        if not isinstance(icao_code, str) or len(icao_code) != 4:
            logging.warning(f"Skipping invalid ICAO code: {icao_code}")
            continue

        url = f"https://skyvector.com/airport/{icao_code}"
        logging.info(f"Fetching data for {icao_code} from {url}")

        row_update = {'icao': icao_code}
        html_path = os.path.join(HTML_DIR, f"{icao_code}.html")

        try:
            time.sleep(1) # Add a 1-second delay
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save HTML content for debugging
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logging.info(f"Saved HTML for {icao_code} to {html_path}")

            soup = BeautifulSoup(response.text, 'html.parser')

            # --- Extract Airport Elevation ---
            try:
                elevation_text_node = soup.find(string=re.compile(r"Elevation is "))
                if elevation_text_node:
                    elevation_match = re.search(r'(\d+\.?\d*)', elevation_text_node)
                    if elevation_match:
                        elevation_ft = float(elevation_match.group(1))
                        row_update['elevation'] = elevation_ft
                        logging.info(f"  - Elevation: {elevation_ft} ft")
                    else:
                        logging.warning(f"  - Elevation pattern not matched in: {elevation_text_node.strip()}")
                        row_update['elevation'] = pd.NA
                else:
                    logging.warning(f"  - Elevation text not found for {icao_code}")
                    row_update['elevation'] = pd.NA
            except (AttributeError, ValueError) as e:
                logging.error(f"Could not parse elevation for {icao_code}: {e}")
                row_update['elevation'] = pd.NA

            # --- Extract Runway Information ---
            runways_data = []
            runway_headers = soup.find_all('h4', string=re.compile(r'Runway\s'))
            if runway_headers:
                for i, header in enumerate(runway_headers, 1):
                    if len(runways_data) >= 8:
                        logging.warning(f"Found more than 8 runways for {icao_code}, only processing the first 8.")
                        break
                    
                    runway_details = {}
                    logging.info(f"  - Processing Runway {i}")
                    
                    # a) Runway Length
                    try:
                        dimensions_p = header.find_next_sibling('p')
                        dimensions_text = dimensions_p.get_text()
                        length_match_m = re.search(r'/ (\d+) x \d+ meters', dimensions_text)
                        length_match_ft = re.search(r'(\d+) x \d+ feet', dimensions_text)

                        if length_match_m:
                            length = int(length_match_m.group(1))
                            runway_details['LENGTH'] = length
                            logging.info(f"    - Length: {length}m (from meters)")
                        elif length_match_ft:
                            length_ft = int(length_match_ft.group(1))
                            length_m = round(length_ft * 0.3048)
                            runway_details['LENGTH'] = length_m
                            logging.info(f"    - Length: {length_m}m (converted from {length_ft} ft)")
                        else:
                            logging.warning(f"    - Could not parse length for runway {i}. HTML file: {html_path}")

                    except (AttributeError, IndexError, ValueError) as e:
                        logging.warning(f"    - Could not parse length for runway {i}: {e}. HTML file: {html_path}")

                    # Find the table with runway details
                    details_table = header.find_next_sibling('table')
                    if details_table:
                        headings = details_table.find_all('td', string=re.compile(r'\d+°'))
                        elevations = details_table.find_all('td', string=re.compile(r'^\d+(\.\d+)?$'))

                        # b) Runway Headings
                        try:
                            heading_a = int(re.search(r'(\d+)', headings[0].text).group(1))
                            heading_b = int(re.search(r'(\d+)', headings[1].text).group(1))
                            runway_details['HEADING_a'] = heading_a
                            runway_details['HEADING_b'] = heading_b
                            logging.info(f"    - Headings: {heading_a}°, {heading_b}°")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"    - Could not parse headings for runway {i}: {e}. HTML file: {html_path}")

                        # c) Runway Elevation
                        try:
                            elevation_a = float(elevations[0].text)
                            elevation_b = float(elevations[1].text)
                            runway_details['ELEVATION_a'] = elevation_a
                            runway_details['ELEVATION_b'] = elevation_b
                            logging.info(f"    - Elevations: {elevation_a} ft, {elevation_b} ft")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"    - Could not parse elevations for runway {i}: {e}. HTML file: {html_path}")
                    else:
                        logging.warning(f"Runway details table not found for {icao_code}, runway {i}. HTML file: {html_path}")
                    
                    runways_data.append(runway_details)

            if not runways_data:
                logging.warning(f"No runway data found for {icao_code}. HTML file: {html_path}")
            else:
                first_runway_data = runways_data[0]
                for i in range(1, 9):  # For RWY_1 to RWY_8
                    rwy_data_to_use = None
                    if i <= len(runways_data):
                        rwy_data_to_use = runways_data[i-1]
                    else:  # if more runways needed, use the first one
                        rwy_data_to_use = first_runway_data
                        logging.info(f"  - Filling runway {i} with data from runway 1 for {icao_code}")

                    for detail in ['LENGTH', 'HEADING_a', 'HEADING_b', 'ELEVATION_a', 'ELEVATION_b']:
                        col_name = f'RWY_{i}_{detail}'
                        value = rwy_data_to_use.get(detail, pd.NA)
                        row_update[col_name] = value
                        if pd.isna(value):
                            logging.warning(f"Missing detail '{detail}' for runway {i} for {icao_code}. HTML file: {html_path}")


        except requests.exceptions.RequestException as e:
            logging.error(f"Could not fetch data for {icao_code}: {e}")
            logging.warning(f"No HTML file saved for {icao_code} due to fetch error.")
            if 'elevation' not in row_update:
                 row_update['elevation'] = row['elevation'] if pd.notna(row['elevation']) else pd.NA

        logging.info(f"Data extracted for {icao_code}: {row_update}")
        updated_data.append(row_update)

    # --- Update and Save DataFrame ---
    if updated_data:
        update_df = pd.DataFrame(updated_data)
        update_df.set_index('icao', inplace=True)
        apt_df.set_index('icao', inplace=True)

        # Get a list of new columns to add to apt_df
        new_cols = update_df.columns.difference(apt_df.columns)
        if not new_cols.empty:
            # Add new columns to apt_df, initializing with NA
            for col in new_cols:
                apt_df[col] = pd.NA

        # Update apt_df with the new data.
        apt_df.update(update_df)
        apt_df.reset_index(inplace=True)

    # Save the updated DataFrame to the Parquet file
    try:
        apt_df.to_parquet(apt_file_path, index=False)
        logging.info(f"File updated successfully: {apt_file_path}")
    except Exception as e:
        logging.error(f"Failed to save parquet file: {e}")

    logging.info("Enrichment run completed.")

if __name__ == '__main__':
    scrape_airport_data()
