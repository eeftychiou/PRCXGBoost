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
apt_file_path = 'data/apt.parquet'
train_flights_path = 'data/flightlist_train.parquet'
rank_flights_path = 'data/flightlist_rank.parquet'
original_apt_file_path = apt_file_path + '.original.parquet'
HTML_DIR = 'htmlfile'  # Directory to save HTML files for debugging

# Create HTML directory if it doesn't exist
os.makedirs(HTML_DIR, exist_ok=True)

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

    try:
        time.sleep(1) # Add a 1-second delay
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save HTML content for debugging
        html_path = os.path.join(HTML_DIR, f"{icao_code}.html")
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
        runway_headers = soup.find_all('h4', string=re.compile(r'Runway\s'))
        runway_count = 0
        if runway_headers:
            for i, header in enumerate(runway_headers, 1):
                runway_count += 1
                logging.info(f"  - Processing Runway {i}")
                # a) Runway Length
                try:
                    dimensions_p = header.find_next_sibling('p')
                    dimensions_text = dimensions_p.get_text()
                    length_match = re.search(r'/ (\d+) x \d+ meters', dimensions_text)
                    length = int(length_match.group(1))
                    row_update[f'RWY_{i}_LENGTH'] = length
                    logging.info(f"    - Length: {length}m")
                except (AttributeError, IndexError, ValueError) as e:
                    row_update[f'RWY_{i}_LENGTH'] = pd.NA
                    logging.warning(f"    - Could not parse length for runway {i}: {e}")

                # Find the table with runway details
                details_table = header.find_next_sibling('table')
                if details_table:
                    headings = details_table.find_all('td', string=re.compile(r'\d+°'))
                    elevations = details_table.find_all('td', string=re.compile(r'^\d+(\.\d+)?$'))

                    # b) Runway Headings
                    try:
                        heading_a = int(re.search(r'(\d+)', headings[0].text).group(1))
                        heading_b = int(re.search(r'(\d+)', headings[1].text).group(1))
                        row_update[f'RWY_{i}_HEADING_a'] = heading_a
                        row_update[f'RWY_{i}_HEADING_b'] = heading_b
                        logging.info(f"    - Headings: {heading_a}°, {heading_b}°")
                    except (AttributeError, IndexError, ValueError) as e:
                        row_update[f'RWY_{i}_HEADING_a'] = pd.NA
                        row_update[f'RWY_{i}_HEADING_b'] = pd.NA
                        logging.warning(f"    - Could not parse headings for runway {i}: {e}")

                    # c) Runway Elevation
                    try:
                        elevation_a = float(elevations[0].text)
                        elevation_b = float(elevations[1].text)
                        row_update[f'RWY_{i}_ELEVATION_a'] = elevation_a
                        row_update[f'RWY_{i}_ELEVATION_b'] = elevation_b
                        logging.info(f"    - Elevations: {elevation_a} ft, {elevation_b} ft")
                    except (AttributeError, IndexError, ValueError) as e:
                        row_update[f'RWY_{i}_ELEVATION_a'] = pd.NA
                        row_update[f'RWY_{i}_ELEVATION_b'] = pd.NA
                        logging.warning(f"    - Could not parse elevations for runway {i}: {e}")
                else:
                    logging.warning(f"Runway details table not found for {icao_code}, runway {i}")
        else:
            # Fallback parsing for different HTML structure
            runway_divs = soup.find_all('div', class_='aptdata')
            for div in runway_divs:
                title_div = div.find('div', class_='aptdatatitle')
                if title_div and 'Runway' in title_div.text:
                    runway_count += 1
                    logging.info(f"  - Processing Runway (fallback) {runway_count}")
                    table = div.find('table')
                    if table:
                        for tr in table.find_all('tr'):
                            th = tr.find('th')
                            tds = tr.find_all('td')
                            if th and tds:
                                if 'Dimensions' in th.text:
                                    try:
                                        length_match = re.search(r'/ (\d+) x \d+ meters', tds[0].text)
                                        if length_match:
                                            length = int(length_match.group(1))
                                            row_update[f'RWY_{runway_count}_LENGTH'] = length
                                            logging.info(f"    - Length: {length}m")
                                    except (AttributeError, IndexError, ValueError) as e:
                                        logging.warning(f"    - Could not parse length for runway {runway_count} (fallback): {e}")
                                elif 'Runway Heading' in th.text:
                                    try:
                                        headings = [int(re.search(r'(\d+)', h).group(1)) for h in tds[0].text.split('/')]
                                        row_update[f'RWY_{runway_count}_HEADING_a'] = headings[0]
                                        row_update[f'RWY_{runway_count}_HEADING_b'] = headings[1]
                                        logging.info(f"    - Headings: {headings[0]}°, {headings[1]}°")
                                    except (AttributeError, IndexError, ValueError):
                                        try:
                                            heading_a = int(re.search(r'(\d+)', tds[0].text).group(1))
                                            heading_b = int(re.search(r'(\d+)', tds[1].text).group(1))
                                            row_update[f'RWY_{runway_count}_HEADING_a'] = heading_a
                                            row_update[f'RWY_{runway_count}_HEADING_b'] = heading_b
                                            logging.info(f"    - Headings: {heading_a}°, {heading_b}°")
                                        except (AttributeError, IndexError, ValueError) as e:
                                            logging.warning(f"    - Could not parse headings for runway {runway_count} (fallback): {e}")
                                elif 'Elevation' in th.text:
                                    try:
                                        elevation_a = float(tds[0].text)
                                        elevation_b = float(tds[1].text)
                                        row_update[f'RWY_{runway_count}_ELEVATION_a'] = elevation_a
                                        row_update[f'RWY_{runway_count}_ELEVATION_b'] = elevation_b
                                        logging.info(f"    - Elevations: {elevation_a} ft, {elevation_b} ft")
                                    except (AttributeError, IndexError, ValueError) as e:
                                        logging.warning(f"    - Could not parse elevations for runway {runway_count} (fallback): {e}")
        if runway_count == 0:
            logging.warning(f"No runway sections found for {icao_code}")

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
    # This will modify existing columns and fill in the newly added columns.
    apt_df.update(update_df)

    apt_df.reset_index(inplace=True)

# Save the updated DataFrame to the Parquet file
try:
    apt_df.to_parquet(apt_file_path, index=False)
    logging.info(f"File updated successfully: {apt_file_path}")
except Exception as e:
    logging.error(f"Failed to save parquet file: {e}")

logging.info("Enrichment run completed.")
