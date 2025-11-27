# PRC: code largely rewritten, like vastly, but originaly I started from https://github.com/akrherz/iem/blob/main/scripts/asos/iem_scraper_example2.py
# MIT License

import datetime
import pandas as pd
import os
import config
import requests
import time

def get_date_range():
    """
    Calculates the start and end dates from all *corrected* flightlist files.
    """
    print("Reading corrected flight lists to determine date range...")
    flight_lists = [
        os.path.join(config.PROCESSED_DATA_DIR, 'corrected_flightlist_train.parquet'),
        os.path.join(config.PROCESSED_DATA_DIR, 'corrected_flightlist_rank.parquet'),
        os.path.join(config.PROCESSED_DATA_DIR, 'corrected_flightlist_final.parquet')
    ]

    min_date = None
    max_date = None

    for file_path in flight_lists:
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                # The corrected files have 'takeoff' and 'landed' columns as datetime objects
                if 'takeoff' in df.columns:
                    current_min = df['takeoff'].min()
                    if pd.notna(current_min):
                        if min_date is None or current_min < min_date:
                            min_date = current_min
                
                if 'landed' in df.columns:
                    current_max = df['landed'].max()
                    if pd.notna(current_max):
                        if max_date is None or current_max > max_date:
                            max_date = current_max
                print(f"Processed {os.path.basename(file_path)}.")
            except Exception as e:
                print(f"Warning: Could not process file {file_path}. Error: {e}")
        else:
            print(f"Info: Corrected flight list not found at {file_path}. Skipping.")

    if not min_date or not max_date:
        raise FileNotFoundError(
            "Could not determine date range. "
            "Please ensure at least one 'corrected_flightlist_*.parquet' file exists "
            "in the processed data directory by running the 'correct_timestamps' stage first."
        )

    # Add a buffer of one day to be safe
    min_date -= datetime.timedelta(days=1)
    max_date += datetime.timedelta(days=1)
    
    print(f"Determined download range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    return min_date, max_date

def download():
    '''Downloads METAR files and stores them in the METAR folder.'''
    try:
        start, end = get_date_range()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    step = datetime.timedelta(hours=24)
    metar_dir = config.METARS_DIR
    os.makedirs(metar_dir, exist_ok=True)
    
    SERVICE_URI = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 10
    
    now = start
    while now < end:
        sdate_str = now.strftime("year1=%Y&month1=%m&day1=%d")
        edate_str = (now + step).strftime("year2=%Y&month2=%m&day2=%d")
        
        uri = (
            f"{SERVICE_URI}?data=all&report_type=3&{sdate_str}&{edate_str}"
            "&tz=UTC&format=comma&missing=empty&latlon=yes&elev=yes"
        )
        
        fname = os.path.join(metar_dir, f"{now:%Y%m%d}.csv")
        
        print(f"Downloading data for {now.strftime('%Y-%m-%d')}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(uri, timeout=60)
                # Raise an exception for bad status codes (4xx or 5xx)
                response.raise_for_status()
                
                # If successful, write to file and break the retry loop
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Successfully saved to {fname}")
                break
                
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} of {MAX_RETRIES} failed: {e}")
                if attempt + 1 == MAX_RETRIES:
                    print(f"Failed to download data for {now.strftime('%Y-%m-%d')} after {MAX_RETRIES} attempts.")
                else:
                    print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
        
        now += step

if __name__ == "__main__":
    download()
