import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth."""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def classify_flight_phases(traj_df, origin_lat, origin_lon, dest_lat, dest_lon, apt_elevation):
    """Classifies each point in a trajectory into a flight phase based on expert rules."""
    phases = []
    CLIMB_RATE_THRESHOLD, DESCENT_RATE_THRESHOLD = 300, -300
    APPROACH_ALTITUDE_THRESHOLD, TERMINAL_AREA_DISTANCE_THRESHOLD = 10000, 40

    for _, point in traj_df.iterrows():
        alt_agl = point['altitude'] - apt_elevation if apt_elevation else point['altitude']
        dist_to_dest = haversine_distance(point['latitude'], point['longitude'], dest_lat, dest_lon)
        dist_to_origin = haversine_distance(point['latitude'], point['longitude'], origin_lat, origin_lon)

        if dist_to_dest < TERMINAL_AREA_DISTANCE_THRESHOLD and point['vertical_rate'] < 0:
            phases.append('Approach' if alt_agl < APPROACH_ALTITUDE_THRESHOLD else 'Descent')
        elif dist_to_origin < TERMINAL_AREA_DISTANCE_THRESHOLD and point['vertical_rate'] > 0:
            phases.append('Takeoff' if alt_agl < 1500 else 'Climb')
        elif abs(point['vertical_rate']) < CLIMB_RATE_THRESHOLD:
            phases.append('Cruise')
        elif point['vertical_rate'] > CLIMB_RATE_THRESHOLD:
            phases.append('Climb')
        elif point['vertical_rate'] < DESCENT_RATE_THRESHOLD:
            phases.append('Descent')
        else:
            phases.append('Cruise')
    return phases

def engineer_features(df, df_apt, flights_dir, start_col='start', end_col='end', desc="Engineering Features"):
    """Applies the full feature engineering pipeline to the dataframe."""
    print(f"Applying full feature engineering pipeline...")
    enhanced_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        new_row = row.copy()
        origin_apt = df_apt[df_apt['icao'] == row['origin_icao']]
        dest_apt = df_apt[df_apt['icao'] == row['destination_icao']]
        traj_path = os.path.join(flights_dir, f"{row['flight_id']}.parquet")

        if os.path.exists(traj_path) and not origin_apt.empty and not dest_apt.empty:
            origin_lat, origin_lon = origin_apt.iloc[0]['apt_lat'], origin_apt.iloc[0]['apt_lon']
            dest_lat, dest_lon = dest_apt.iloc[0]['apt_lat'], dest_apt.iloc[0]['apt_lon']
            dest_elev = dest_apt.iloc[0]['apt_elev']

            traj_df = pd.read_parquet(traj_path)
            traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
            segment_traj = traj_df[(traj_df['timestamp'] >= new_row[start_col]) & (traj_df['timestamp'] <= new_row[end_col])].copy()

            if len(segment_traj) > 1:
                new_row['segment_duration_s'] = (new_row[end_col] - new_row[start_col]).total_seconds()
                lats, lons = segment_traj['latitude'].values, segment_traj['longitude'].values
                new_row['segment_distance_km'] = np.sum([haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1]) for i in range(len(lats) - 1)])
                new_row['mean_dist_to_origin_km'] = segment_traj.apply(lambda pt: haversine_distance(pt['latitude'], pt['longitude'], origin_lat, origin_lon), axis=1).mean()
                new_row['mean_dist_to_dest_km'] = segment_traj.apply(lambda pt: haversine_distance(pt['latitude'], pt['longitude'], dest_lat, dest_lon), axis=1).mean()

                segment_traj['phase'] = classify_flight_phases(segment_traj, origin_lat, origin_lon, dest_lat, dest_lon, dest_elev)
                phase_fractions = segment_traj['phase'].value_counts(normalize=True)
                for phase in ['Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach']:
                    new_row[f'{phase.lower()}_fraction'] = phase_fractions.get(phase, 0)
            
        enhanced_rows.append(new_row)
    
    return pd.DataFrame(enhanced_rows)
