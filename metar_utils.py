import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculates haversine distance between two arrays of points in km.
    Returns a pandas Series.
    """
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Ensure index is preserved
    if isinstance(lat2, pd.Series):
        return pd.Series(R * c, index=lat2.index)
    return R * c


def load_raw_metars(metar_dir=config.METARS_DIR):
    """
    Load all raw METAR CSV files from a directory into a single DataFrame.
    """
    print(f"Loading raw METAR data from {metar_dir}...")
    metar_files = list(Path(metar_dir).glob('*.csv'))
    if not metar_files:
        raise FileNotFoundError(f"No METAR files found in {metar_dir}.")
        
    metar_df = pd.concat([pd.read_csv(f, comment='#', na_values=['M', 'null'], low_memory=False) for f in metar_files], ignore_index=True)
    metar_df['valid'] = pd.to_datetime(metar_df['valid'])
    metar_df.rename(columns={'station': 'ICAO_ID'}, inplace=True)
    
    # Drop records with missing essential data
    metar_df.dropna(subset=['ICAO_ID', 'lat', 'lon', 'valid'], inplace=True)
    
    # Sort and set index for efficient lookup
    metar_df.sort_values(['ICAO_ID', 'valid'], inplace=True)
    metar_df.set_index(['ICAO_ID', 'valid'], inplace=True)
    print(f"Loaded {len(metar_df)} METAR records from {len(metar_files)} files.")
    return metar_df

def process_metar_data():
    """
    Pre-processes raw METAR data to create a clean, feature-engineered dataset
    aligned with the flight data from the processed directory.
    """
    print("--- Starting METAR Pre-processing ---")

    # 1. Load all *featured* flight data to get corrected timestamps
    print("Loading featured flight data from processed directory...")
    all_flights = []
    test_suffix = '_test' if config.TEST_RUN else ''
    for dataset_type in ['train', 'rank', 'final']:
        featured_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data_{dataset_type}{test_suffix}.parquet")
        try:
            df_featured = pd.read_parquet(featured_path)
            all_flights.append(df_featured)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Featured data file not found at {featured_path}. "
                "Please run the 'prepare_data' stage at least once before running 'prepare_metars'."
            )
    
    df_flights = pd.concat(all_flights, ignore_index=True)
    # Ensure datetime types
    df_flights['takeoff'] = pd.to_datetime(df_flights['takeoff'])
    df_flights['landed'] = pd.to_datetime(df_flights['landed'])

    # Get unique airports with their coordinates
    origin_airports = df_flights[['origin_icao', 'origin_latitude', 'origin_longitude']].rename(columns={'origin_icao': 'ICAO_ID', 'origin_latitude': 'lat', 'origin_longitude': 'lon'})
    dest_airports = df_flights[['destination_icao', 'destination_latitude', 'destination_longitude']].rename(columns={'destination_icao': 'ICAO_ID', 'destination_latitude': 'lat', 'destination_longitude': 'lon'})
    flight_airports = pd.concat([origin_airports, dest_airports]).drop_duplicates(subset='ICAO_ID').dropna(subset=['ICAO_ID', 'lat', 'lon']).set_index('ICAO_ID')

    # 2. Load Raw METARs
    metar_df = load_raw_metars()
    metar_locations = metar_df.reset_index()[['ICAO_ID', 'lat', 'lon']].drop_duplicates(subset='ICAO_ID').set_index('ICAO_ID')

    # 3. Map Flight Airports to Nearest METAR Stations
    print("Mapping flight airports to nearest METAR stations (within 100nm)...")
    nearest_station_map = {}
    for icao, row in tqdm(flight_airports.iterrows(), total=len(flight_airports)):
        if icao in metar_locations.index:
            nearest_station_map[icao] = icao
            continue

        distances_km = haversine_vectorized(row['lat'], row['lon'], metar_locations['lat'], metar_locations['lon'])
        distances_nm = distances_km * 0.539957  # Convert to nautical miles

        valid_distances = distances_nm[distances_nm <= 100]
        if not valid_distances.empty:
            nearest_station_map[icao] = valid_distances.idxmin()
        else:
            nearest_station_map[icao] = None
    
    # 4. Create a list of required weather lookups
    dep_requests = df_flights[['origin_icao', 'takeoff']].rename(columns={'origin_icao': 'ICAO_ID', 'takeoff': 'time'})
    arr_requests = df_flights[['destination_icao', 'landed']].rename(columns={'destination_icao': 'ICAO_ID', 'landed': 'time'})
    weather_requests = pd.concat([dep_requests, arr_requests]).drop_duplicates().dropna()
    weather_requests = weather_requests[weather_requests['ICAO_ID'].isin(nearest_station_map.keys())]

    # 5. Retrieve and process weather data
    print(f"Retrieving weather for {len(weather_requests)} unique airport-time requests...")
    processed_records = []
    for _, request in tqdm(weather_requests.iterrows(), total=len(weather_requests)):
        station_to_use = nearest_station_map.get(request['ICAO_ID'])
        if not station_to_use:
            continue
        
        try:
            station_weather = metar_df.loc[station_to_use]
            # Find the single closest report in time
            nearest_idx = station_weather.index.get_loc(request['time'], method='nearest')
            weather_data = station_weather.iloc[[nearest_idx]].reset_index() # Keep as DataFrame
            
            # Add original request info
            weather_data['requested_icao'] = request['ICAO_ID']
            weather_data['requested_time'] = request['time']
            processed_records.append(weather_data)
        except (KeyError, IndexError):
            continue # No data for this station

    if not processed_records:
        print("No weather data could be processed. Aborting.")
        return

    df_processed = pd.concat(processed_records, ignore_index=True)

    # 6. Feature Engineering & Encoding
    print("Encoding categorical weather features...")
    
    # Sky Condition: Map to ordinal values
    sky_conditions = ['SKC', 'FEW', 'SCT', 'BKN', 'OVC', 'VV']
    sky_map = {val: i for i, val in enumerate(sky_conditions)}
    for col in ['skyc1', 'skyc2', 'skyc3', 'skyc4']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(sky_map)

    # Weather Codes (wxcodes): One-hot encode common phenomena
    if 'wxcodes' in df_processed.columns:
        df_processed['wxcodes'] = df_processed['wxcodes'].fillna('')
        common_wx = ['RA', 'SN', 'FG', 'BR', 'HZ', 'TS', 'SH', 'FZ']
        for wx in common_wx:
            df_processed[f'wx_{wx}'] = df_processed['wxcodes'].str.contains(wx).astype(int)

    # 7. Select, Clean, and Save
    numeric_features = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti', 'mslp', 'vsby', 'gust', 'ice_accretion_1hr']
    sky_features = ['skyl1', 'skyl2', 'skyl3', 'skyl4', 'skyc1', 'skyc2', 'skyc3', 'skyc4']
    wx_features = [col for col in df_processed.columns if col.startswith('wx_')]
    
    final_cols = ['requested_icao', 'requested_time'] + \
                 [col for col in numeric_features if col in df_processed.columns] + \
                 [col for col in sky_features if col in df_processed.columns] + \
                 wx_features
    
    df_final = df_processed[final_cols].copy()

    # Impute missing numeric values with the median
    for col in numeric_features:
        if col in df_final.columns and df_final[col].isnull().any():
            median_val = df_final[col].median()
            df_final[col].fillna(median_val, inplace=True)
            print(f"Imputed missing values in '{col}' with median: {median_val:.2f}")

    # Rename for merging
    df_final.rename(columns={'requested_icao': 'ICAO_ID', 'requested_time': 'timestamp'}, inplace=True)

    # Save the processed data
    output_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_metars.parquet')
    df_final.to_parquet(output_path, index=False)
    print(f"--- METAR Pre-processing Complete. Saved to {output_path} ---")

if __name__ == '__main__':
    # This allows running the processing independently
    process_metar_data()
