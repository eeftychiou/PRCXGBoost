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
import pygeomag

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
final_flights_path = 'data/prc-2025-datasets/flightlist_final.parquet'
original_apt_file_path = apt_file_path + '.original.parquet'
HTML_DIR = 'htmlfile'  # Directory to save HTML files for debugging

# Create HTML directory if it doesn't exist
os.makedirs(HTML_DIR, exist_ok=True)

def calculate_true_heading(mag_heading, declination):
    if pd.isna(mag_heading):
        return pd.NA
    true_heading = (mag_heading + declination) % 360
    return round(true_heading)

def scrape_airport_data():
    # Lists to track airports with issues
    airports_no_info = []
    airports_no_runway_details = []

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
        final_df= pd.read_parquet(final_flights_path)
        
        relevant_airports = set(train_df['origin_icao'].unique()) | \
                            set(train_df['destination_icao'].unique()) | \
                            set(rank_df['origin_icao'].unique()) | \
                            set(rank_df['destination_icao'].unique()) |\
                            set(final_df['origin_icao'].unique()) | \
                            set(final_df['destination_icao'].unique())

        airports_to_process_df = apt_df[apt_df['icao'].isin(relevant_airports)].copy()
        
        logging.info(f"Found {len(relevant_airports)} unique airports in flight lists.")
        logging.info(f"Will process {len(airports_to_process_df)} relevant airports.")

    except FileNotFoundError as e:
        logging.error(f"Flight list file not found: {e}. Processing all airports with missing elevation as a fallback.")
        airports_to_process_df = apt_df[apt_df['elevation'].isnull()].copy()

    # --- Process Airports ---
    updated_data = []
    
    for index, row in airports_to_process_df.iterrows():
        icao_code = row['icao']
        if not isinstance(icao_code, str) or len(icao_code) != 4:
            logging.warning(f"Skipping invalid ICAO code: {icao_code}")
            continue

        url = f"https://skyvector.com/airport/{icao_code}"
        logging.info(f"--- Processing {icao_code} ---")
        
        row_update = {'icao': icao_code}
        html_path = os.path.join(HTML_DIR, f"{icao_code}.html")

        # --- Calculate Magnetic Declination ---
        mag_declination = 0
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            try:
                elevation_km = row['elevation'] * 0.0003048 if pd.notna(row['elevation']) else 0
                geo_mag = pygeomag.GeoMag()
                mag = geo_mag.calculate(glat=row['latitude'], glon=row['longitude'], alt=elevation_km, time=2024.0)
                mag_declination = mag.decl
                logging.info(f"  - Calculated Magnetic Declination: {mag_declination:.2f}°")
            except Exception as e:
                logging.error(f"  - Could not calculate magnetic declination for {icao_code}: {e}")
        else:
            logging.warning(f"  - Missing latitude/longitude for {icao_code}, cannot calculate magnetic declination.")

        html_content = None
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            logging.info(f"Using cached HTML for {icao_code} from {html_path}")
        else:
            logging.info(f"Fetching data from {url}")
            try:
                time.sleep(1) # Add a 1-second delay
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                html_content = response.text
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logging.info(f"Saved HTML for {icao_code} to {html_path}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Could not fetch data for {icao_code}: {e}. HTML file: {html_path}")
                logging.warning(f"No HTML file saved for {icao_code} due to fetch error.")
                airports_no_info.append(icao_code)
                # If fetching fails, ensure elevation is carried over if it existed in original apt_df
                if 'elevation' not in row_update:
                     row_update['elevation'] = row['elevation'] if pd.notna(row['elevation']) else pd.NA
                updated_data.append(row_update)
                continue # Skip to next airport if fetching failed

        soup = BeautifulSoup(html_content, 'html.parser')

        # --- Check if airport page is valid ---
        if "Lookup Airport" in soup.title.text or not soup.find('div', class_='aptdatatitle', string=re.compile(r'Location Information for ')):
            logging.warning(f"Airport {icao_code} not found on SkyVector or page is generic lookup page. Skipping. HTML file: {html_path}")
            airports_no_info.append(icao_code)
            updated_data.append(row_update) # Append with just ICAO, other fields will be NA
            continue

        # --- Extract Airport Elevation ---
        airport_elevation = pd.NA
        elevation_text_node = soup.find(string=re.compile(r"Elevation is\s"))
        if elevation_text_node:
            elevation_match = re.search(r'(\d+\.?\d*)', elevation_text_node)
            if elevation_match:
                elevation_ft = float(elevation_match.group(1))
                row_update['elevation'] = elevation_ft
                airport_elevation = elevation_ft
                logging.info(f"  - Found Airport Elevation: {elevation_ft} ft")
            else:
                logging.warning(f"  - Elevation pattern not matched in: {elevation_text_node.strip()}. HTML file: {html_path}")
        else:
            logging.warning(f"  - Airport elevation text not found. HTML file: {html_path}")

        # --- Extract Runway Information ---
        runways_data = []
        logging.info("  - Starting runway extraction...")
        
        # Try to find runways using the 'h4' header structure first
        runway_h4_sections = soup.find_all('h4', string=re.compile(r'Runway\s'))
        if runway_h4_sections:
            logging.info(f"    - Found {len(runway_h4_sections)} 'h4' runway sections.")
            for i, header in enumerate(runway_h4_sections):
                if len(runways_data) >= 8:
                    logging.warning(f"    - Found more than 8 runways for {icao_code} in 'h4' sections, only processing the first 8.")
                    break
                
                runway_details = {}
                logging.info(f"      - Processing 'h4' Runway section {i+1}.")
                
                # Dimensions (Length)
                try:
                    dimensions_p = header.find_next_sibling('p')
                    dimensions_text = dimensions_p.get_text()
                    length_match_m = re.search(r'/ (\d+) x \d+ meters', dimensions_text)
                    length_match_ft = re.search(r'(\d+) x \d+ feet', dimensions_text)

                    if length_match_m:
                        length = int(length_match_m.group(1))
                        runway_details['LENGTH'] = length
                        logging.info(f"        - Parsed Length: {length}m (from meters)")
                    elif length_match_ft:
                        length_ft = int(length_match_ft.group(1))
                        length_m = round(length_ft * 0.3048)
                        runway_details['LENGTH'] = length_m
                        logging.info(f"        - Parsed Length: {length_m}m (converted from {length_ft} ft)")
                    else:
                        logging.warning(f"        - Could not parse length from '{dimensions_text}' for 'h4' runway {i+1}. HTML file: {html_path}")

                except (AttributeError, IndexError, ValueError) as e:
                    logging.warning(f"        - Error parsing length for 'h4' runway {i+1}: {e}. HTML file: {html_path}")

                # Find the table with runway details
                details_table = header.find_next_sibling('table')
                if details_table:
                    # Headings and Elevations are typically in rows within this table
                    heading_row = details_table.find('th', string=re.compile(r'Runway Heading:'))
                    elevation_row = details_table.find('th', string=re.compile(r'Elevation:'))
                    glide_slope_row = details_table.find('th', string=re.compile(r'Glide Slope Indicator'))

                    heading_a, heading_b = pd.NA, pd.NA
                    if heading_row:
                        tds = heading_row.find_next_siblings('td')
                        try:
                            if len(tds) >= 1 and re.search(r'(\d+)', tds[0].text):
                                heading_a = int(re.search(r'(\d+)', tds[0].text).group(1))
                            if len(tds) >= 2 and re.search(r'(\d+)', tds[1].text):
                                heading_b = int(re.search(r'(\d+)', tds[1].text).group(1))
                            
                            if pd.isna(heading_a) and pd.notna(heading_b):
                                heading_a = (heading_b - 180 + 360) % 360
                                logging.info(f"        - Calculated complementary HEADING_a: {heading_a}° for 'h4' runway {i+1}. HTML file: {html_path}")
                            elif pd.notna(heading_a) and pd.isna(heading_b):
                                heading_b = (heading_a + 180) % 360
                                logging.info(f"        - Calculated complementary HEADING_b: {heading_b}° for 'h4' runway {i+1}. HTML file: {html_path}")
                            
                            runway_details['HEADING_a'] = heading_a
                            runway_details['HEADING_b'] = heading_b
                            logging.info(f"        - Parsed Headings: {heading_a}°, {heading_b}°")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"        - Could not parse headings for 'h4' runway {i+1}: {e}. HTML file: {html_path}")
                    else:
                        logging.warning(f"        - Runway Heading row not found for 'h4' runway {i+1}. HTML file: {html_path}")

                    if elevation_row:
                        tds = elevation_row.find_next_siblings('td')
                        try:
                            elevation_a = float(tds[0].text)
                            elevation_b = float(tds[1].text)
                            runway_details['ELEVATION_a'] = elevation_a
                            runway_details['ELEVATION_b'] = elevation_b
                            logging.info(f"        - Parsed Elevations: {elevation_a} ft, {elevation_b} ft")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"        - Could not parse elevations for 'h4' runway {i+1}: {e}. HTML file: {html_path}")
                    elif glide_slope_row:
                        tds = glide_slope_row.find_next_siblings('td')
                        try:
                            # Attempt to parse elevation from PAPI text
                            papi_text = tds[1].text
                            papi_match = re.search(r'(\d+\.?\d*)\s*ft', papi_text)
                            if papi_match:
                                papi_elevation = float(papi_match.group(1))
                                runway_details['ELEVATION_a'] = papi_elevation
                                runway_details['ELEVATION_b'] = papi_elevation
                                logging.info(f"        - Parsed Elevation from PAPI: {papi_elevation} ft")
                            else:
                                runway_details['ELEVATION_a'] = airport_elevation
                                runway_details['ELEVATION_b'] = airport_elevation
                                logging.warning(f"        - Could not parse elevation from PAPI text for 'h4' runway {i+1}. Using airport elevation. HTML file: {html_path}")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"        - Error parsing PAPI for 'h4' runway {i+1}: {e}. HTML file: {html_path}")
                    else:
                        runway_details['ELEVATION_a'] = airport_elevation
                        runway_details['ELEVATION_b'] = airport_elevation
                        logging.warning(f"        - Runway Elevation row not found for 'h4' runway {i+1}. Using airport elevation. HTML file: {html_path}")
                else:
                    logging.warning(f"      - Runway details table not found for 'h4' runway {i+1}. HTML file: {html_path}")
                
                if runway_details:
                    runways_data.append(runway_details)

        # If no runways found with 'h4' headers, try the 'div.aptdata' fallback structure
        if not runways_data:
            logging.info("    - No 'h4' runway sections found, trying 'div.aptdata' fallback.")
            runway_divs = soup.find_all('div', class_='aptdata')
            for i, div in enumerate(runway_divs):
                title_div = div.find('div', class_='aptdatatitle')
                if not title_div or 'Runway' not in title_div.text:
                    continue

                runway_title = title_div.text.strip()
                # Filter out non-runway sections like "Runway Conditions" or general airport data
                if not (re.search(r'\d', runway_title) or '/' in runway_title):
                    logging.info(f"      - Skipping '{runway_title}' as it does not appear to be a specific runway.")
                    continue

                if len(runways_data) >= 8:
                    logging.warning(f"    - Found more than 8 runways for {icao_code} in 'div.aptdata' sections, only processing the first 8.")
                    break
                
                runway_details = {}
                logging.info(f"      - Processing 'div.aptdata' Runway section: '{runway_title}'.")
                table = div.find('table')
                if not table:
                    logging.warning(f"      - No <table> found for runway '{runway_title}'. HTML file: {html_path}")
                    continue

                for tr in table.find_all('tr'):
                    th = tr.find('th')
                    tds = tr.find_all('td')
                    if not th or not tds:
                        continue

                    header_text = th.text.strip()
                    if 'Dimensions' in header_text:
                        try:
                            dimensions_text = tds[0].text
                            length_match_m = re.search(r'/ (\d+) x \d+ meters', dimensions_text)
                            length_match_ft = re.search(r'(\d+) x \d+ feet', dimensions_text)

                            if length_match_m:
                                length = int(length_match_m.group(1))
                                runway_details['LENGTH'] = length
                                logging.info(f"        - Parsed Length: {length}m (from meters)")
                            elif length_match_ft:
                                length_ft = int(length_match_ft.group(1))
                                length_m = round(length_ft * 0.3048)
                                runway_details['LENGTH'] = length_m
                                logging.info(f"        - Parsed Length: {length_m}m (converted from {length_ft} ft)")
                            else:
                                logging.warning(f"        - Could not parse length from '{dimensions_text}' for '{runway_title}'. HTML file: {html_path}")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"        - Error parsing length for '{runway_title}': {e}. HTML file: {html_path}")
                    
                    elif 'Runway Heading' in header_text:
                        heading_a, heading_b = pd.NA, pd.NA
                        try:
                            # Expecting two tds for heading_a and heading_b
                            if len(tds) >= 1 and re.search(r'(\d+)', tds[0].text):
                                heading_a = int(re.search(r'(\d+)', tds[0].text).group(1))
                            if len(tds) >= 2 and re.search(r'(\d+)', tds[1].text):
                                heading_b = int(re.search(r'(\d+)', tds[1].text).group(1))

                            if pd.isna(heading_a) and pd.notna(heading_b):
                                heading_a = (heading_b - 180 + 360) % 360
                                logging.info(f"        - Calculated complementary HEADING_a: {heading_a}° for '{runway_title}'. HTML file: {html_path}")
                            elif pd.notna(heading_a) and pd.isna(heading_b):
                                heading_b = (heading_a + 180) % 360
                                logging.info(f"        - Calculated complementary HEADING_b: {heading_b}° for '{runway_title}'. HTML file: {html_path}")

                            runway_details['HEADING_a'] = heading_a
                            runway_details['HEADING_b'] = heading_b
                            logging.info(f"        - Parsed Headings: {heading_a}°, {heading_b}°")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"        - Error parsing headings for '{runway_title}': {e}. HTML file: {html_path}")
                        
                    elif 'Elevation' in header_text:
                        try:
                            # Expecting two tds for elevation_a and elevation_b
                            if len(tds) >= 2:
                                elevation_a = float(tds[0].text) if tds[0].text.strip() else pd.NA
                                elevation_b = float(tds[1].text) if tds[1].text.strip() else pd.NA
                                runway_details['ELEVATION_a'] = elevation_a
                                runway_details['ELEVATION_b'] = elevation_b
                                logging.info(f"        - Parsed Elevations: {elevation_a} ft, {elevation_b} ft")
                            else:
                                logging.warning(f"        - Could not parse elevations for '{runway_title}', not enough 'td' elements. HTML file: {html_path}")
                        except (AttributeError, IndexError, ValueError) as e:
                            logging.warning(f"        - Error parsing elevations for '{runway_title}': {e}. HTML file: {html_path}")
                    
                if 'ELEVATION_a' not in runway_details or 'ELEVATION_b' not in runway_details:
                    runway_details['ELEVATION_a'] = airport_elevation
                    runway_details['ELEVATION_b'] = airport_elevation
                    logging.warning(f"      - Runway elevation not found for '{runway_title}'. Using airport elevation. HTML file: {html_path}")

                if runway_details:
                    runways_data.append(runway_details)
                else:
                    logging.warning(f"      - No details extracted for runway section '{runway_title}'. HTML file: {html_path}")

        logging.info(f"  - Finished runway extraction. Found {len(runways_data)} valid runway(s).")

        # --- Populate row_update with runway data and fill missing with NA ---
        if runways_data:
            # Check if any runway has the required details (length and heading)
            has_valid_runway = False
            for rwy in runways_data:
                has_length = pd.notna(rwy.get('LENGTH'))
                has_heading = pd.notna(rwy.get('HEADING_a')) or pd.notna(rwy.get('HEADING_b'))
                if has_length and has_heading:
                    has_valid_runway = True
                    break
            
            if not has_valid_runway:
                airports_no_runway_details.append(icao_code)

            # 2. Count the total number of runways available and insert it into a new column called rwy_number.
            num_runways = len(runways_data)
            row_update['rwy_number'] = num_runways

            # 3. Count the total runway lengths and insert into a new column called tot_rwy_len.
            total_length = sum(rwy.get('LENGTH', 0) for rwy in runways_data if pd.notna(rwy.get('LENGTH')))
            row_update['tot_rwy_len'] = total_length

            # 4. Calculate the average runway length.
            if num_runways > 0:
                row_update['avg_rwy_len'] = total_length / num_runways
            else:
                row_update['avg_rwy_len'] = pd.NA

            # Assign runway details, setting to NA if a runway is not available
            for i in range(1, 9):  # For RWY_1 to RWY_8
                if i <= len(runways_data):
                    rwy_data_to_use = runways_data[i-1]
                    logging.info(f"  - Assigning details for Runway {i} from scraped data.")
                    for detail_key in ['LENGTH', 'HEADING_a', 'HEADING_b', 'ELEVATION_a', 'ELEVATION_b']:
                        col_name = f'RWY_{i}_{detail_key}'
                        value = rwy_data_to_use.get(detail_key, pd.NA)
                        row_update[col_name] = value
                        if pd.isna(value):
                            logging.warning(f"    - Missing detail '{detail_key}' for Runway {i} for {icao_code}. Using NA. HTML file: {html_path}")
                    
                    # Calculate and add true headings
                    true_heading_a = calculate_true_heading(rwy_data_to_use.get('HEADING_a'), mag_declination)
                    true_heading_b = calculate_true_heading(rwy_data_to_use.get('HEADING_b'), mag_declination)
                    row_update[f'RWY_{i}_TRUE_HEADING_a'] = true_heading_a
                    row_update[f'RWY_{i}_TRUE_HEADING_b'] = true_heading_b
                    logging.info(f"    - Calculated True Headings for Runway {i}: {true_heading_a}°, {true_heading_b}°")

                else:
                    logging.info(f"  - No data for Runway {i} for {icao_code}. Setting to NA.")
                    for detail_key in ['LENGTH', 'HEADING_a', 'HEADING_b', 'ELEVATION_a', 'ELEVATION_b', 'TRUE_HEADING_a', 'TRUE_HEADING_b']:
                        col_name = f'RWY_{i}_{detail_key}'
                        row_update[col_name] = pd.NA
        else:
            airports_no_runway_details.append(icao_code)
            row_update['rwy_number'] = 0
            row_update['tot_rwy_len'] = 0
            row_update['avg_rwy_len'] = pd.NA
            logging.warning(f"  - No runway data found for {icao_code}. All runway columns will be NA. HTML file: {html_path}")
            # Explicitly set all RWY_x_... columns to NA if no runways are found
            for i in range(1, 9):
                for detail_key in ['LENGTH', 'HEADING_a', 'HEADING_b', 'ELEVATION_a', 'ELEVATION_b', 'TRUE_HEADING_a', 'TRUE_HEADING_b']:
                    col_name = f'RWY_{i}_{detail_key}'
                    row_update[col_name] = pd.NA


        logging.info(f"Data extracted for {icao_code}: {row_update}")
        updated_data.append(row_update)

    # --- Update and Save DataFrame ---
    if updated_data:
        update_df = pd.DataFrame(updated_data)
        update_df.set_index('icao', inplace=True)
        apt_df.set_index('icao', inplace=True)

        new_cols = update_df.columns.difference(apt_df.columns)
        if not new_cols.empty:
            logging.info(f"Adding new columns to apt.parquet: {list(new_cols)}")
            for col in new_cols:
                apt_df[col] = pd.NA

        apt_df.update(update_df)
        apt_df.reset_index(inplace=True)

    try:
        apt_df.to_parquet(apt_file_path, index=False)
        logging.info(f"File updated successfully: {apt_file_path}")
    except Exception as e:
        logging.error(f"Failed to save parquet file: {e}")

    logging.info("Enrichment run completed.")

    # --- Report airports with missing data ---
    if airports_no_info:
        logging.info("\n--- Airports with no information on SkyVector ---")
        logging.info(", ".join(sorted(list(set(airports_no_info)))))

    if airports_no_runway_details:
        logging.info("\n--- Airports where runway details (heading/length) could not be determined ---")
        logging.info(", ".join(sorted(list(set(airports_no_runway_details)))))

def manual_update_airport(icao_code, data_to_update):
    logging.info(f"--- Manually updating data for {icao_code} ---")
    apt_df = pd.read_parquet(apt_file_path)
    
    # --- Calculate Magnetic Declination for manual update ---
    apt_row = apt_df[apt_df['icao'] == icao_code].iloc[0]
    mag_declination = 0
    if pd.notna(apt_row['latitude']) and pd.notna(apt_row['longitude']):
        try:
            elevation_km = apt_row['elevation'] * 0.0003048 if pd.notna(apt_row['elevation']) else 0
            geo_mag = pygeomag.GeoMag()
            mag = geo_mag.calculate(glat=apt_row['latitude'], glon=apt_row['longitude'], alt=elevation_km, time=2024.0)
            mag_declination = mag.decl
            logging.info(f"  - Calculated Magnetic Declination for manual update of {icao_code}: {mag_declination:.2f}°")
        except Exception as e:
            logging.error(f"  - Could not calculate magnetic declination for {icao_code}: {e}")
    else:
        logging.warning(f"  - Missing latitude/longitude for {icao_code}, cannot calculate magnetic declination for manual update.")

    apt_df.set_index('icao', inplace=True)

    if icao_code in apt_df.index:
        # First, set all potential runway columns to NA for this ICAO to clear previous data
        for i in range(1, 9):
            for detail_key in ['LENGTH', 'HEADING_a', 'HEADING_b', 'ELEVATION_a', 'ELEVATION_b', 'TRUE_HEADING_a', 'TRUE_HEADING_b']:
                col_name = f'RWY_{i}_{detail_key}'
                if col_name in apt_df.columns:
                    apt_df.loc[icao_code, col_name] = pd.NA
        
        # Then apply the specific updates
        for col, value in data_to_update.items():
            if col not in apt_df.columns:
                apt_df[col] = pd.NA # Add column if it doesn't exist
            apt_df.loc[icao_code, col] = value
            logging.info(f"  - Updated {col} for {icao_code} with value: {value}")

        # Calculate and add true headings for manual updates
        for i in range(1, 9):
            mag_heading_a_col = f'RWY_{i}_HEADING_a'
            mag_heading_b_col = f'RWY_{i}_HEADING_b'
            if mag_heading_a_col in data_to_update and mag_heading_b_col in data_to_update:
                mag_heading_a = data_to_update[mag_heading_a_col]
                mag_heading_b = data_to_update[mag_heading_b_col]
                
                true_heading_a = calculate_true_heading(mag_heading_a, mag_declination)
                true_heading_b = calculate_true_heading(mag_heading_b, mag_declination)
                
                apt_df.loc[icao_code, f'RWY_{i}_TRUE_HEADING_a'] = true_heading_a
                apt_df.loc[icao_code, f'RWY_{i}_TRUE_HEADING_b'] = true_heading_b
                logging.info(f"  - Manually updated True Headings for Runway {i}: {true_heading_a}°, {true_heading_b}°")

    else:
        logging.warning(f"Airport {icao_code} not found in apt.parquet for manual update.")

    apt_df.reset_index(inplace=True)
    try:
        apt_df.to_parquet(apt_file_path, index=False)
        logging.info(f"Manual update for {icao_code} completed and saved to {apt_file_path}")
    except Exception as e:
        logging.error(f"Failed to save parquet file after manual update for {icao_code}: {e}")


if __name__ == '__main__':
    scrape_airport_data()

    # Manual update for LTDB
    ltdb_data = {
        'elevation': 19, # feet
        'rwy_number': 2,
        'tot_rwy_len': 7000,
        'avg_rwy_len': 3500,
        'RWY_1_LENGTH': 3500,
        'RWY_1_HEADING_a': 3,
        'RWY_1_HEADING_b': 21,
        'RWY_1_ELEVATION_a': 19,
        'RWY_1_ELEVATION_b': 19,
        'RWY_2_LENGTH': 3500,
        'RWY_2_HEADING_a': 3,
        'RWY_2_HEADING_b': 21,
        'RWY_2_ELEVATION_a': 19,
        'RWY_2_ELEVATION_b': 19,
    }

    manual_update_airport('LTDB', ltdb_data)

    # Manual update for ZGBH
    zgbh_data = {
        'elevation': 83, # feet
        'rwy_number': 1,
        'tot_rwy_len': 3214, # meters
        'avg_rwy_len': 3214,
        'RWY_1_LENGTH': 3214,
        'RWY_1_HEADING_a': 13,
        'RWY_1_HEADING_b': 193,
        'RWY_1_ELEVATION_a': 83,
        'RWY_1_ELEVATION_b': 83,
    }
    manual_update_airport('ZGBH', zgbh_data)
