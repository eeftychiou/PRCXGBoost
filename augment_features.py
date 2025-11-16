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

def is_aligned_vectorized(tracks, runway_headings, tolerance=15):
    """Vectorized check for runway alignment."""
    if not runway_headings:
        return pd.Series(False, index=tracks.index)
    
    alignment_matrix = np.zeros((len(tracks), len(runway_headings)), dtype=bool)
    for i, heading in enumerate(runway_headings):
        diff = np.abs(tracks - heading)
        alignment_matrix[:, i] = np.minimum(diff, 360 - diff) <= tolerance
    
    return np.any(alignment_matrix, axis=1)

def is_aligned_with_runway(track, runway_headings, tolerance=15):
    """
    Checks if the given track is aligned with any of the runway headings.
    Track and runway_headings are in degrees.
    """
    if pd.isna(track) or not runway_headings:
        return False
    for heading in runway_headings:
        if pd.notna(heading):
            # Calculate the absolute difference, considering the circular nature of angles
            diff = abs(track - heading)
            if min(diff, 360 - diff) <= tolerance:
                return True
    return False

def classify_flight_phases_vectorized(traj_df, origin_alt, dest_alt, origin_runways, dest_runways, takeoff_time, landed_time):
    """Classifies flight phases using physical criteria including runway alignment."""
    df = traj_df.copy()

    # --- Constants ---
    NM_TO_KM = 1.852
    APPROACH_DISTANCE_THRESHOLD_KM = 20 * NM_TO_KM
    ON_THE_GROUND_ALT_THRESHOLD_FT = 25
    MIN_TOUCHDOWN_SPEED = 100
    MAX_TOUCHDOWN_SPEED = 160
    MAX_TAXI_SPEED = 30
    MIN_TAXI_SPEED = 5
    TAKEOFF_ALT_THRESHOLD_FT = 1000
    LANDING_ALT_THRESHOLD_FT = 3000
    PARKED_DISTANCE_THRESHOLD = 5


    CLIMB_DESCENT_VR_THRESHOLD = 500
    APPROACH_ALT_THRESHOLD_AGL = 10000

    # --- Prepare helper columns ---
    df['alt_agl_origin'] = df['altitude'] - origin_alt
    df['alt_agl_dest'] = df['altitude'] - dest_alt
    
    if 'groundspeed' in df.columns and df['groundspeed'].notna().any():
        df['speed'] = df['groundspeed'].replace(0, np.nan).fillna(df['calculated_speed'])
    else:
        df['speed'] = df['calculated_speed']

    # dist_to_origin_km and dist_to_dest_km are now expected in traj_df
    
    aligned_with_origin_rwy = is_aligned_vectorized(df['track'], origin_runways)
    aligned_with_dest_rwy = is_aligned_vectorized(df['track'], dest_runways)

    # --- Initialize phase column ---
    df['phase'] = 'Unknown'

    # --- Phase Classification (in order of precedence) ---
    # 1. Parked
    if pd.notna(takeoff_time) and pd.notna(landed_time):
        cond_parked_gate =  (df['phase'] == 'Unknown') & ((df['timestamp'] < takeoff_time) | (df['timestamp'] > landed_time)) & \
                            (df['speed'] < MIN_TAXI_SPEED) & ((df['alt_agl_origin'] <= ON_THE_GROUND_ALT_THRESHOLD_FT) | (df['alt_agl_dest'] <= ON_THE_GROUND_ALT_THRESHOLD_FT)) & \
                            ((df['dist_to_origin_km'] < PARKED_DISTANCE_THRESHOLD) | (df['dist_to_dest_km'] < PARKED_DISTANCE_THRESHOLD ))

        df.loc[cond_parked_gate, 'phase'] = 'Parked'


    # 2. Landing
    cond_landing = (df['phase'] == 'Unknown') & (df['dist_to_dest_km'] <= APPROACH_DISTANCE_THRESHOLD_KM) & \
                   (df['alt_agl_dest'] < LANDING_ALT_THRESHOLD_FT) & \
                   (df['speed'] < MAX_TOUCHDOWN_SPEED) & (df['speed'] > MIN_TOUCHDOWN_SPEED) & \
                   (aligned_with_dest_rwy)
    df.loc[cond_landing, 'phase'] = 'Landing'

    # 2. Takeoff
    cond_takeoff = (df['phase'] == 'Unknown') & \
                   (df['dist_to_origin_km'] <= PARKED_DISTANCE_THRESHOLD) & \
                   (df['speed'] > MAX_TAXI_SPEED) & (df['alt_agl_origin'] < TAKEOFF_ALT_THRESHOLD_FT)  & \
                   (aligned_with_origin_rwy)
    df.loc[cond_takeoff, 'phase'] = 'Takeoff'

    # 3. Taxi-in
    cond_taxi_in = ((df['phase'] == 'Unknown') & (df['alt_agl_dest'] <= ON_THE_GROUND_ALT_THRESHOLD_FT) & \
                    (df['speed'] <= (MAX_TAXI_SPEED+70)) & (df['dist_to_dest_km'] <= PARKED_DISTANCE_THRESHOLD) & \
                    (df['speed'] > MIN_TAXI_SPEED))
    df.loc[cond_taxi_in, 'phase'] = 'Taxi-in'

    # 4. Taxi-out
    cond_taxi_out = (df['phase'] == 'Unknown') & (df['alt_agl_origin'] <= ON_THE_GROUND_ALT_THRESHOLD_FT) & (df['speed'] < MAX_TAXI_SPEED) & (df['speed'] > MIN_TAXI_SPEED) & (df['dist_to_origin_km'] <= PARKED_DISTANCE_THRESHOLD)
    df.loc[cond_taxi_out, 'phase'] = 'Taxi-out'

    # 6. Air phases
    in_air_mask = (df['phase'] == 'Unknown')
    cond_approach = in_air_mask & (df['vertical_rate'] < -CLIMB_DESCENT_VR_THRESHOLD) & (df['alt_agl_dest'] < APPROACH_ALT_THRESHOLD_AGL)
    df.loc[cond_approach, 'phase'] = 'Approach'
    cond_climb = in_air_mask & (df['phase'] == 'Unknown') & (df['vertical_rate'] > CLIMB_DESCENT_VR_THRESHOLD) & (df['dist_to_origin_km'] > PARKED_DISTANCE_THRESHOLD)
    df.loc[cond_climb, 'phase'] = 'Climb'
    cond_descent = in_air_mask & (df['phase'] == 'Unknown') & (df['vertical_rate'] < -CLIMB_DESCENT_VR_THRESHOLD)
    df.loc[cond_descent, 'phase'] = 'Descent'
    df.loc[in_air_mask & (df['phase'] == 'Unknown') & ((df['alt_agl_dest'] > ON_THE_GROUND_ALT_THRESHOLD_FT) | (df['alt_agl_origin'] > ON_THE_GROUND_ALT_THRESHOLD_FT)), 'phase'] = 'Cruise'

    return df['phase']

def augment_features(df, trajectories_folder):
    print("Augmenting features...")

    # Basic Feature Engineering
    for col in ['aircraft_type', 'origin_icao', 'destination_icao']:
        if col in df.columns:
            df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col])
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    if 'takeoff' in df.columns:
        df['takeoff'] = pd.to_datetime(df['takeoff'])
    if 'landed' in df.columns:
        df['landed'] = pd.to_datetime(df['landed'])
    df['segment_duration'] = (df['end'] - df['start']).dt.total_seconds()

    # --- Added Features ---
    print("Adding new features: flight duration, great circle distance, day of week, and time of day...")
    if 'takeoff' in df.columns and 'landed' in df.columns:
        df['flight_duration_hours'] = (df['landed'] - df['takeoff']).dt.total_seconds() / 3600

    if all(c in df.columns for c in ['origin_latitude', 'origin_longitude', 'destination_latitude', 'destination_longitude']):
        df['great_circle_distance_km'] = haversine_vectorized(
            df['origin_latitude'], df['origin_longitude'],
            df['destination_latitude'], df['destination_longitude']
        )

    df['day_of_week'] = df['start'].dt.dayofweek
    df['start_time_decimal'] = df['start'].dt.hour + df['start'].dt.minute / 60.0
    df['end_time_decimal'] = df['end'].dt.hour + df['end'].dt.minute / 60.0

    # Initialize Columns
    numeric_traj_cols = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'calculated_speed', 'vertical_rate_change', 'dist_to_origin_km', 'dist_to_dest_km']
    aggregations = ['min', 'max', 'mean', 'std']
    for col in numeric_traj_cols:
        for agg in aggregations:
            df[f'seg_{col}_{agg}'] = np.nan
    all_phases = ['Parked', 'Taxi-out', 'Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxi-in']
    for phase in all_phases:
        df[f'phase_fraction_{phase.lower()}'] = 0.0
    df['start_alt_rev'], df['end_alt_rev'] = np.nan, np.nan
    df['departure_rwy_heading'] = 0.0
    df['departure_rwy_length'] = 0.0
    df['arrival_rwy_heading'] = 0.0
    df['arrival_rwy_length'] = 0.0
    df['segment_distance_km'] = 0.0

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

        if 'vertical_rate' in traj_df.columns:
            traj_df['track_change'] = traj_df['track'].diff()

        # Collect Runway and Flight Info
        flight_info = group.iloc[0]
        origin_runway_data = {}
        destination_runway_data = {}
        for i in range(1, 9):
            for suffix in ['', 'a', 'b']:
                origin_rwy_heading_col = f'origin_RWY_{i}_HEADING_{suffix}'
                origin_rwy_length_col = f'origin_RWY_{i}_LENGTH'
                dest_rwy_heading_col = f'destination_RWY_{i}_HEADING_{suffix}'
                dest_rwy_length_col = f'destination_RWY_{i}_LENGTH'

                if origin_rwy_heading_col in flight_info and pd.notna(flight_info[origin_rwy_heading_col]):
                    origin_runway_data[f'RWY_{i}{suffix}'] = {
                        'heading': flight_info[origin_rwy_heading_col],
                        'length': flight_info.get(origin_rwy_length_col, 0.0)
                    }
                if dest_rwy_heading_col in flight_info and pd.notna(flight_info[dest_rwy_heading_col]):
                    destination_runway_data[f'RWY_{i}{suffix}'] = {
                        'heading': flight_info[dest_rwy_heading_col],
                        'length': flight_info.get(dest_rwy_length_col, 0.0)
                    }
        
        origin_runway_headings = [r['heading'] for r in origin_runway_data.values() if pd.notna(r['heading'])]
        dest_runway_headings = [r['heading'] for r in destination_runway_data.values() if pd.notna(r['heading'])]

        traj_df['dist_to_origin_km'] = haversine_vectorized(traj_df['latitude'], traj_df['longitude'], flight_info.get('origin_latitude'), flight_info.get('origin_longitude'))
        traj_df['dist_to_dest_km'] = haversine_vectorized(traj_df['latitude'], traj_df['longitude'], flight_info.get('destination_latitude'), flight_info.get('destination_longitude'))

        # Classify Phases
        traj_df['phase'] = classify_flight_phases_vectorized(
            traj_df, flight_info.get('origin_elevation', 0), flight_info.get('destination_elevation', 0),
            origin_runway_headings, dest_runway_headings, flight_info.get('takeoff'), flight_info.get('landed')
        )
        
        valid_numeric_cols = [col for col in numeric_traj_cols if col in traj_df.columns]

        # Aggregate features for each segment
        for idx, row in group.iterrows():
            segment_traj = traj_df[(traj_df['timestamp'] >= row['start']) & (traj_df['timestamp'] <= row['end'])]

            # If the segment is empty, find the closest points in the trajectory to represent it
            if segment_traj.empty:
                if traj_df.empty or len(traj_df) < 2:
                    continue  # Not enough trajectory data for this flight

                n_points = 5
                if len(traj_df) < n_points:
                    indices_to_use = traj_df.index
                else:
                    start_indices = (traj_df['timestamp'] - row['start']).abs().nsmallest(n_points).index
                    end_indices = (traj_df['timestamp'] - row['end']).abs().nsmallest(n_points).index
                    combined_indices = start_indices.union(end_indices)
                    indices_to_use = combined_indices
                segment_for_features = traj_df.loc[indices_to_use]
            else:
                segment_for_features = segment_traj

            if segment_for_features.empty:
                continue

            # Calculate segment distance
            valid_dist_points = segment_for_features.dropna(subset=['latitude', 'longitude'])
            if len(valid_dist_points) > 1:
                segment_dist = haversine_vectorized(
                    valid_dist_points['latitude'].shift(1), valid_dist_points['longitude'].shift(1),
                    valid_dist_points['latitude'], valid_dist_points['longitude']
                ).sum()
                df.loc[idx, 'segment_distance_km'] = segment_dist
            else:
                df.loc[idx, 'segment_distance_km'] = 0.0

            # Determine Departure and Arrival Runway
            takeoff_points = segment_for_features[segment_for_features['phase'] == 'Takeoff']
            if not takeoff_points.empty:
                departure_rwy_found = False
                for _, point in takeoff_points.iterrows():
                    for rwy_id, rwy_info in origin_runway_data.items():
                        if is_aligned_with_runway(point['track'], [rwy_info['heading']]):
                            df.loc[idx, 'departure_rwy_heading'] = rwy_info['heading']
                            df.loc[idx, 'departure_rwy_length'] = rwy_info['length']
                            departure_rwy_found = True
                            break
                    if departure_rwy_found:
                        break
            
            landing_points = segment_for_features[segment_for_features['phase'] == 'Landing']
            if not landing_points.empty:
                arrival_rwy_found = False
                for _, point in landing_points.iloc[::-1].iterrows():
                    for rwy_id, rwy_info in destination_runway_data.items():
                        if is_aligned_with_runway(point['track'], [rwy_info['heading']]):
                            df.loc[idx, 'arrival_rwy_heading'] = rwy_info['heading']
                            df.loc[idx, 'arrival_rwy_length'] = rwy_info['length']
                            arrival_rwy_found = True
                            break
                    if arrival_rwy_found:
                        break

            stats = segment_for_features[valid_numeric_cols].agg(aggregations)
            for col in valid_numeric_cols:
                for agg in aggregations:
                    df.loc[idx, f'seg_{col}_{agg}'] = stats.loc[agg, col]
            
            phase_counts = segment_for_features['phase'].value_counts(normalize=True)
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

    # Add Time-Based Features
    if 'takeoff' in df.columns and 'landed' in df.columns:
        print("Calculating time-based features (takeoff_delta, landing_delta, mean_time_in_air)...")
        
        # De-fragment the DataFrame before adding new columns
        df = df.copy()

        df['takeoff_delta'] = (df['start'] - df['takeoff']).dt.total_seconds() / 60
        df['landing_delta'] = (df['landed'] - df['end']).dt.total_seconds() / 60
        segment_midpoint_time = df['start'] + (df['end'] - df['start']) / 2
        mean_time_in_air = (segment_midpoint_time - df['takeoff']).dt.total_seconds() / 60
        df['mean_time_in_air'] = np.maximum(0, mean_time_in_air)

    print("Feature augmentation complete.")
    return df

if __name__ == "__main__":
    # This block is now outdated due to signature change of augment_features.
    # It's left here for reference but will not run correctly.
    pass
