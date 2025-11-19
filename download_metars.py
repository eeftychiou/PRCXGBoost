# PRC: code largely rewritten, like vastly, but originaly I started from https://github.com/akrherz/iem/blob/main/scripts/asos/iem_scraper_example2.py
# MIT License

import datetime
import pandas as pd
from subprocess import call
import os
import config

def get_date_range():
    """
    Calculates the start and end dates from all flightlist files.
    """
    flight_lists = [
        os.path.join(config.DATA_DIR, 'prc-2025-datasets/flightlist_train.parquet'),
        os.path.join(config.DATA_DIR, 'prc-2025-datasets/flightlist_rank.parquet'),
        os.path.join(config.DATA_DIR, 'prc-2025-datasets/flightlist_final.parquet')
    ]

    min_date = None
    max_date = None

    for file_path in flight_lists:
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            if 'takeoff' in df.columns and 'landed' in df.columns:
                current_min = df['takeoff'].min()
                current_max = df['landed'].max()

                if min_date is None or current_min < min_date:
                    min_date = current_min
                if max_date is None or current_max > max_date:
                    max_date = current_max
    
    # Add a buffer of one day to be safe
    if min_date:
        min_date -= datetime.timedelta(days=1)
    if max_date:
        max_date += datetime.timedelta(days=1)
        
    return min_date, max_date

def download():
    ''' donwload METARs files and stores them in the METAR folder as specified in the CONFIG and Makefile files'''
    start, end = get_date_range()
    
    if not start or not end:
        print("Could not determine date range from flight lists. Exiting.")
        return

    step = datetime.timedelta(hours=24)
    metar_dir = os.path.join(config.DATA_DIR, 'METARs')
    os.makedirs(metar_dir, exist_ok=True)
    
    now = start
    while now < end:
        sdate = now.strftime("year1=%Y&month1=%m&day1=%d&")
        edate = (now + step).strftime("year2=%Y&month2=%m&day2=%d&")
        print(f"Downloading: {now}")
        fname = os.path.join(metar_dir,f"{now:%Y%m%d}.csv")
        cmd = f'curl "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?data=all&report_type=3&{sdate}&{edate}&tz=UTC&format=comma&missing=empty&latlon=yes&elev=yes" > {fname}'
        print(cmd)
        call(cmd,shell=True)
        now += step

if __name__ == "__main__":
    download()