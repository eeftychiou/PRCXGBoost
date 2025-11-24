import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import config
import re

# Comprehensive dictionary for decoding METAR weather phenomena
WX_CODES_DECODED = {
    # Intensity
    '-': 'Light', '+': 'Heavy', 'VC': 'In Vicinity',
    # Descriptor
    'MI': 'Shallow', 'PR': 'Partial', 'BC': 'Patches', 'DR': 'Low Drifting',
    'BL': 'Blowing', 'SH': 'Showers', 'TS': 'Thunderstorm', 'FZ': 'Freezing',
    # Precipitation
    'DZ': 'Drizzle', 'RA': 'Rain', 'SN': 'Snow', 'SG': 'Snow Grains',
    'IC': 'Ice Crystals', 'PL': 'Ice Pellets', 'GR': 'Hail', 'GS': 'Small Hail/Snow Pellets',
    'UP': 'Unknown Precipitation',
    # Obscuration
    'BR': 'Mist', 'FG': 'Fog', 'FU': 'Smoke', 'VA': 'Volcanic Ash',
    'DU': 'Widespread Dust', 'SA': 'Sand', 'HZ': 'Haze',
    # Other
    'PO': 'Well-Developed Dust/Sand Whirls', 'SQ': 'Squalls', 'FC': 'Funnel Cloud',
    'SS': 'Sandstorm', 'DS': 'Duststorm'
}

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
    
    metar_df.dropna(subset=['ICAO_ID', 'lat', 'lon', 'valid'], inplace=True)
    
    metar_df.sort_values(['ICAO_ID', 'valid'], inplace=True)
    metar_df.set_index(['ICAO_ID', 'valid'], inplace=True)
    
    if not metar_df.index.is_unique:
        num_duplicates = metar_df.index.duplicated().sum()
        print(f"DEBUG LOG: Found {num_duplicates} duplicate station-time records in raw METAR files. Keeping the first entry for each.")
        metar_df = metar_df[~metar_df.index.duplicated(keep='first')]

    print(f"Loaded {len(metar_df)} unique METAR records from {len(metar_files)} files.")
    return metar_df

def decode_wxcodes(df):
    """
    Decodes the 'wxcodes' column into human-readable and binary features.
    """
    if 'wxcodes' not in df.columns:
        return df

    df['wxcodes'] = df['wxcodes'].fillna('NSW') # No Significant Weather
    
    df['wx_is_thunderstorm'] = df['wxcodes'].str.contains('TS').astype(int)
    df['wx_is_freezing'] = df['wxcodes'].str.contains('FZ').astype(int)
    df['wx_is_shower'] = df['wxcodes'].str.contains('SH').astype(int)
    df['wx_is_rain'] = df['wxcodes'].str.contains('RA').astype(int)
    df['wx_is_snow'] = df['wxcodes'].str.contains('SN').astype(int)
    df['wx_is_fog_mist'] = df['wxcodes'].str.contains('FG|BR').astype(int)
    df['wx_is_haze_smoke'] = df['wxcodes'].str.contains('HZ|FU').astype(int)

    df['wx_intensity'] = 0 # Default to normal
    df.loc[df['wxcodes'].str.startswith('-'), 'wx_intensity'] = 1 # Light
    df.loc[df['wxcodes'].str.startswith('+'), 'wx_intensity'] = 2 # Heavy

    return df

def get_weather_for_events(events, metar_df, nearest_station_map, event_type):
    """
    Generic function to fetch weather for flight events (departure or arrival).
    """
    print(f"Retrieving weather for {len(events)} {event_type} events...")
    processed_records = []
    
    events['station_to_use'] = events[f'{event_type}_icao'].map(nearest_station_map)
    events.dropna(subset=['station_to_use'], inplace=True)

    for _, event in tqdm(events.iterrows(), total=len(events), desc=f"Fetching {event_type} weather"):
        station_id = event['station_to_use']
        request_time = event[f'{event_type}_time']
        
        try:
            station_weather = metar_df.loc[station_id]
            if not station_weather.index.is_unique:
                station_weather = station_weather[~station_weather.index.duplicated(keep='first')]

            pos = station_weather.index.get_indexer([request_time], method='pad')[0]
            if pos == -1:
                continue

            weather_data = station_weather.iloc[[pos]].reset_index(drop=True)
            weather_data['flight_id'] = event['flight_id']
            processed_records.append(weather_data)
        except (KeyError, IndexError):
            continue
            
    if not processed_records:
        return pd.DataFrame()

    df_weather = pd.concat(processed_records, ignore_index=True)
    
    # Feature Engineering
    df_weather = decode_wxcodes(df_weather)
    
    # Select and clean columns
    id_cols = ['flight_id']
    cols_to_exclude = ['valid', 'lon', 'lat', 'wxcodes', 'metar', 'peak_wind_time', 'ICAO_ID']
    feature_cols = [col for col in df_weather.columns if col not in cols_to_exclude + id_cols]
    
    df_final = df_weather[id_cols + feature_cols].copy()

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_final[col]):
            if df_final[col].isnull().any():
                median_val = df_final[col].median()
                if pd.isna(median_val):
                    df_final[col].fillna(0, inplace=True)
                else:
                    df_final[col].fillna(median_val, inplace=True)
    
    return df_final.add_prefix(f'{event_type}_').rename(columns={f'{event_type}_flight_id': 'flight_id'})


def process_metar_data():
    """
    Pre-processes raw METAR data, creating a single file with weather features
    for both departure and arrival, keyed by flight_id.
    """
    print("--- Starting METAR Pre-processing (keyed by flight_id) ---")

    # 1. Load flight and airport data
    print("Loading corrected flightlist and airport data...")
    all_flights = []
    for dataset_type in ['train', 'rank', 'final']:
        flightlist_path = os.path.join(config.PROCESSED_DATA_DIR, f"corrected_flightlist_{dataset_type}.parquet")
        if not os.path.exists(flightlist_path):
            raise FileNotFoundError(f"Corrected flightlist not found at {flightlist_path}. Run 'correct_timestamps' first.")
        all_flights.append(pd.read_parquet(flightlist_path))
    
    df_flights = pd.concat(all_flights, ignore_index=True)
    df_apt = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/apt.parquet'))
    
    df_flights = pd.merge(df_flights, df_apt[['icao', 'latitude', 'longitude']].add_prefix('origin_'), on='origin_icao', how='left')
    df_flights = pd.merge(df_flights, df_apt[['icao', 'latitude', 'longitude']].add_prefix('destination_'), on='destination_icao', how='left')

    # 2. Load METARs and map stations
    metar_df = load_raw_metars()
    metar_locations = metar_df.reset_index()[['ICAO_ID', 'lat', 'lon']].drop_duplicates(subset='ICAO_ID').set_index('ICAO_ID')
    
    flight_airports = pd.concat([
        df_flights[['origin_icao', 'origin_latitude', 'origin_longitude']].rename(columns={'origin_icao': 'ICAO_ID', 'origin_latitude': 'lat', 'origin_longitude': 'lon'}),
        df_flights[['destination_icao', 'destination_latitude', 'destination_longitude']].rename(columns={'destination_icao': 'ICAO_ID', 'destination_latitude': 'lat', 'destination_longitude': 'lon'})
    ]).drop_duplicates(subset='ICAO_ID').dropna().set_index('ICAO_ID')

    print("Mapping flight airports to nearest METAR stations...")
    nearest_station_map = {icao: (icao if icao in metar_locations.index else None) for icao in flight_airports.index}
    unmapped_airports = [icao for icao, station in nearest_station_map.items() if station is None]
    
    if unmapped_airports:
        unmapped_df = flight_airports.loc[unmapped_airports]
        for icao, row in tqdm(unmapped_df.iterrows(), total=len(unmapped_df), desc="Finding nearby stations"):
            distances_nm = haversine_vectorized(row['lat'], row['lon'], metar_locations['lat'], metar_locations['lon']) * 0.539957
            valid_distances = distances_nm[distances_nm <= 100]
            if not valid_distances.empty:
                nearest_station_map[icao] = valid_distances.idxmin()

    # 3. Process Departure and Arrival Weather Separately
    dep_events = df_flights[['flight_id', 'origin_icao', 'takeoff']].rename(columns={'origin_icao': 'dep_icao', 'takeoff': 'dep_time'})
    arr_events = df_flights[['flight_id', 'destination_icao', 'landed']].rename(columns={'destination_icao': 'arr_icao', 'landed': 'arr_time'})

    df_dep_weather = get_weather_for_events(dep_events, metar_df, nearest_station_map, 'dep')
    df_arr_weather = get_weather_for_events(arr_events, metar_df, nearest_station_map, 'arr')

    # 4. Merge and Save
    if df_dep_weather.empty or df_arr_weather.empty:
        print("Could not process weather for departures or arrivals. Aborting.")
        return
        
    df_final_weather = pd.merge(df_dep_weather, df_arr_weather, on='flight_id', how='outer')

    output_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_metars.parquet')
    df_final_weather.to_parquet(output_path, index=False)
    print(f"--- METAR Pre-processing Complete. Saved flight-keyed weather data to {output_path} ---")

if __name__ == '__main__':
    process_metar_data()
