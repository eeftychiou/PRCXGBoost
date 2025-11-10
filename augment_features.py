import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from tqdm import tqdm

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calculates haversine distance between two arrays of points in km."""
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def is_aligned_vectorized(tracks, runway_headings, tolerance=5):
    """Vectorized check for runway alignment."""
    if not runway_headings:
        return pd.Series(False, index=tracks.index)
    
    alignment_matrix = np.zeros((len(tracks), len(runway_headings)), dtype=bool)
    for i, heading in enumerate(runway_headings):
        diff = np.abs(tracks - heading)
        alignment_matrix[:, i] = np.minimum(diff, 360 - diff) <= tolerance
    
    return np.any(alignment_matrix, axis=1)

def classify_flight_phases_vectorized(traj_df, origin_alt, dest_alt, origin_lat, origin_lon, dest_lat, dest_lon, origin_runways, dest_runways):
    """Classifies flight phases using physical criteria including runway alignment."""
    df = traj_df.copy()

    # --- Constants ---
    NM_TO_KM = 1.852
    DISTANCE_THRESHOLD_KM = 20 * NM_TO_KM
    TAXI_IN_ALT_THRESHOLD_FT = 25
    TAXI_IN_SPEED_THRESHOLD_KNOTS = 100
    TAXI_OUT_ALT_THRESHOLD_FT = 25
    TAXI_OUT_SPEED_THRESHOLD_KNOTS = 30
    MIN_TAXI_SPEED_THRESHOLD_KNOTS = 5
    TAKEOFF_ALT_THRESHOLD_FT = 1000
    LANDING_ALT_THRESHOLD_FT = 2000
    PARKED_SPEED_THRESHOLD_KNOTS = 5
    CRUISE_VR_THRESHOLD = 300
    CLIMB_DESCENT_VR_THRESHOLD = 500
    APPROACH_ALT_THRESHOLD_AGL = 10000

    # --- Prepare helper columns ---
    df['alt_agl_origin'] = df['altitude'] - origin_alt
    df['alt_agl_dest'] = df['altitude'] - dest_alt
    
    if 'groundspeed' in df.columns and df['groundspeed'].notna().any():
        df['speed'] = df['groundspeed'].fillna(df['calculated_speed'])
    else:
        df['speed'] = df['calculated_speed']

    df['dist_to_origin_km'] = haversine_vectorized(df['latitude'], df['longitude'], origin_lat, origin_lon)
    df['dist_to_dest_km'] = haversine_vectorized(df['latitude'], df['longitude'], dest_lat, dest_lon)
    
    aligned_with_origin_rwy = is_aligned_vectorized(df['track'], origin_runways)
    aligned_with_dest_rwy = is_aligned_vectorized(df['track'], dest_runways)

    # --- Initialize phase column ---
    df['phase'] = 'Unknown'

    # --- Phase Classification (in order of precedence) ---
    # 1. Landing
    cond_landing = (df['dist_to_dest_km'] <= DISTANCE_THRESHOLD_KM) & \
                   (df['alt_agl_dest'] < LANDING_ALT_THRESHOLD_FT) & \
                   (df['speed'] < 160) & (df['speed'] > TAXI_IN_SPEED_THRESHOLD_KNOTS) & \
                   (aligned_with_dest_rwy)
    df.loc[cond_landing, 'phase'] = 'Landing'

    # 2. Takeoff
    cond_takeoff = (df['phase'] == 'Unknown') & \
                   (df['dist_to_origin_km'] <= DISTANCE_THRESHOLD_KM) & \
                   (df['speed'] > 30) & \
                   (aligned_with_origin_rwy)
    df.loc[cond_takeoff, 'phase'] = 'Takeoff'

    # 3. Taxi-in
    cond_taxi_in = (df['phase'] == 'Unknown') & (df['alt_agl_dest'] <= TAXI_IN_ALT_THRESHOLD_FT) & (df['speed'] < TAXI_IN_SPEED_THRESHOLD_KNOTS) & (df['dist_to_dest_km'] <= DISTANCE_THRESHOLD_KM) & (df['speed'] > MIN_TAXI_SPEED_THRESHOLD_KNOTS)
    df.loc[cond_taxi_in, 'phase'] = 'Taxi-in'

    # 4. Taxi-out
    cond_taxi_out = (df['phase'] == 'Unknown') & (df['alt_agl_origin'] <= TAXI_OUT_ALT_THRESHOLD_FT) & (df['speed'] < TAXI_OUT_SPEED_THRESHOLD_KNOTS) & (df['speed'] > MIN_TAXI_SPEED_THRESHOLD_KNOTS) & (df['dist_to_origin_km'] <= DISTANCE_THRESHOLD_KM)
    df.loc[cond_taxi_out, 'phase'] = 'Taxi-out'

    # 5. Parked
    cond_parked = df['phase'].isin(['Taxi-in', 'Taxi-out']) & (df['speed'] < PARKED_SPEED_THRESHOLD_KNOTS)
    df.loc[cond_parked, 'phase'] = 'Parked'

    # 6. Air phases
    in_air_mask = (df['phase'] == 'Unknown')
    cond_approach = in_air_mask & (df['vertical_rate'] < -CRUISE_VR_THRESHOLD) & (df['alt_agl_dest'] < APPROACH_ALT_THRESHOLD_AGL)
    df.loc[cond_approach, 'phase'] = 'Approach'
    cond_climb = in_air_mask & (df['phase'] == 'Unknown') & (df['vertical_rate'] > CLIMB_DESCENT_VR_THRESHOLD)
    df.loc[cond_climb, 'phase'] = 'Climb'
    cond_descent = in_air_mask & (df['phase'] == 'Unknown') & (df['vertical_rate'] < -CLIMB_DESCENT_VR_THRESHOLD)
    df.loc[cond_descent, 'phase'] = 'Descent'
    df.loc[in_air_mask & (df['phase'] == 'Unknown'), 'phase'] = 'Cruise'

    return df['phase']

def augment_features(file_path, trajectories_folder):
    output_path = file_path.replace('.parquet', '_augmented.parquet')
    if os.path.exists(output_path):
        print(f"Augmented file already exists, skipping: {output_path}")
        return

    print(f"Processing file: {file_path}")
    df = pd.read_parquet(file_path)

    # Basic Feature Engineering
    for col in ['aircraft_type', 'origin_icao', 'destination_icao']:
        if col in df.columns:
            df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['segment_duration'] = (df['end'] - df['start']).dt.total_seconds()

    # Initialize Columns
    numeric_traj_cols = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'calculated_speed', 'vertical_rate_change']
    aggregations = ['min', 'max', 'mean', 'std']
    for col in numeric_traj_cols:
        for agg in aggregations:
            df[f'seg_{col}_{agg}'] = np.nan
    all_phases = ['Parked', 'Taxi-out', 'Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxi-in']
    for phase in all_phases:
        df[f'phase_fraction_{phase.lower()}'] = 0.0
    df['start_alt_rev'], df['end_alt_rev'] = np.nan, np.nan

    print("Processing trajectory data for each flight...")
    for flight_id, group in tqdm(df.groupby('flight_id'), total=df['flight_id'].nunique()):
        trajectory_file = os.path.join(trajectories_folder, f"{flight_id}.parquet")
        if not os.path.exists(trajectory_file): continue

        traj_df = pd.read_parquet(trajectory_file)
        if traj_df.empty or 'latitude' not in traj_df.columns: continue
        
        traj_df = traj_df.sort_values('timestamp').reset_index(drop=True)
        traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])

        # Calculate New Trajectory Features
        time_diff = traj_df['timestamp'].diff().dt.total_seconds()
        dist_km = haversine_vectorized(traj_df['latitude'].shift(1), traj_df['longitude'].shift(1), traj_df['latitude'], traj_df['longitude'])

        traj_df['calculated_speed'] = (dist_km / time_diff) * 1943.84
        traj_df['calculated_speed'] = traj_df['calculated_speed'].replace([np.inf, -np.inf], np.nan)
        if 'vertical_rate' in traj_df.columns:
            traj_df['vertical_rate_change'] = traj_df['vertical_rate'].diff()

        # Collect Runway and Flight Info
        flight_info = group.iloc[0]
        origin_runways = [flight_info[c] for c in flight_info.index if 'origin_RWY' in c and 'HEADING' in c and pd.notna(flight_info[c])]
        dest_runways = [flight_info[c] for c in flight_info.index if 'destination_RWY' in c and 'HEADING' in c and pd.notna(flight_info[c])]
        
        # Classify Phases
        traj_df['phase'] = classify_flight_phases_vectorized(
            traj_df, flight_info.get('origin_elevation', 0), flight_info.get('destination_elevation', 0),
            flight_info.get('origin_latitude'), flight_info.get('origin_longitude'),
            flight_info.get('destination_latitude'), flight_info.get('destination_longitude'),
            origin_runways, dest_runways
        )
        
        valid_numeric_cols = [col for col in numeric_traj_cols if col in traj_df.columns]

        # Aggregate features for each segment
        for idx, row in group.iterrows():
            segment_traj = traj_df[(traj_df['timestamp'] >= row['start']) & (traj_df['timestamp'] <= row['end'])]
            if segment_traj.empty: continue

            stats = segment_traj[valid_numeric_cols].agg(aggregations)
            for col in valid_numeric_cols:
                for agg in aggregations:
                    df.loc[idx, f'seg_{col}_{agg}'] = stats.loc[agg, col]
            
            phase_counts = segment_traj['phase'].value_counts(normalize=True)
            for phase_name, fraction in phase_counts.items():
                df.loc[idx, f'phase_fraction_{phase_name.lower()}'] = fraction

            start_idx = (traj_df['timestamp'] - row['start']).abs().idxmin()
            df.loc[idx, 'start_alt_rev'] = traj_df.loc[start_idx, 'altitude']
            end_idx = (traj_df['timestamp'] - row['end']).abs().idxmin()
            df.loc[idx, 'end_alt_rev'] = traj_df.loc[end_idx, 'altitude']

    # Final Vectorized Calculations
    print("Performing final vectorized calculations...")
    df['alt_diff_rev'] = df['end_alt_rev'] - df['start_alt_rev']
    df['alt_diff_rev_std'] = df.groupby('flight_id')['alt_diff_rev'].transform('std')
    for col in numeric_traj_cols:
        min_col, max_col = f'seg_{col}_min', f'seg_{col}_max'
        if min_col in df.columns and max_col in df.columns:
            df[f'seg_{col}_delta'] = df[max_col] - df[min_col]

    # Save File
    df.to_parquet(output_path, index=False)
    print(f"Saved augmented file to: {output_path}")

if __name__ == "__main__":
    processed_data_folder = 'data/processed'
    base_trajectories_folder = 'data/interpolated_trajectories'
    
    rank_data_path = os.path.join(processed_data_folder, 'featured_rank_data.parquet')
    if os.path.exists(rank_data_path):
        augment_features(rank_data_path, os.path.join(base_trajectories_folder, 'flights_rank'))

    data_path = os.path.join(processed_data_folder, 'featured_data.parquet')
    if os.path.exists(data_path):
        augment_features(data_path, os.path.join(base_trajectories_folder, 'flights_train'))
