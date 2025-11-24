import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import config

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

def is_aligned_with_runway(track, runway_headings, tolerance=7):
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
    NM_TO_KM = 1.852
    APPROACH_DISTANCE_THRESHOLD_KM = 20 * NM_TO_KM
    ON_THE_GROUND_ALT_THRESHOLD_FT = 25
    MIN_TOUCHDOWN_SPEED, MAX_TOUCHDOWN_SPEED = 100, 160
    MAX_TAXI_SPEED, MIN_TAXI_SPEED = 30, 5
    TAKEOFF_ALT_THRESHOLD_FT, LANDING_ALT_THRESHOLD_FT = 1000, 3000
    PARKED_DISTANCE_THRESHOLD = 5
    CLIMB_DESCENT_VR_THRESHOLD = 500
    APPROACH_ALT_THRESHOLD_AGL = 10000

    df['alt_agl_origin'] = df['altitude'] - origin_alt
    df['alt_agl_dest'] = df['altitude'] - dest_alt
    df['speed'] = df.get('groundspeed', pd.Series(index=df.index)).replace(0, np.nan).fillna(df.get('calculated_speed'))
    
    aligned_with_origin_rwy = is_aligned_vectorized(df['track'], origin_runways)
    aligned_with_dest_rwy = is_aligned_vectorized(df['track'], dest_runways)

    df['phase'] = np.nan
    if pd.notna(takeoff_time) and pd.notna(landed_time):
        cond_parked_gate =  (df['phase'].isna()) & ((df['timestamp'] < takeoff_time) | (df['timestamp'] > landed_time)) & \
                            (df['speed'] < MIN_TAXI_SPEED) & ((df['alt_agl_origin'] <= ON_THE_GROUND_ALT_THRESHOLD_FT) | (df['alt_agl_dest'] <= ON_THE_GROUND_ALT_THRESHOLD_FT)) & \
                            ((df['dist_to_origin_km'] < PARKED_DISTANCE_THRESHOLD) | (df['dist_to_dest_km'] < PARKED_DISTANCE_THRESHOLD ))

        df.loc[cond_parked_gate, 'phase'] = 'Parked'

    # 2. Landing
    cond_landing = (df['phase'].isna()) & (df['dist_to_dest_km'] <= APPROACH_DISTANCE_THRESHOLD_KM) & \
                   (df['alt_agl_dest'] < LANDING_ALT_THRESHOLD_FT) & \
                   (df['speed'] < MAX_TOUCHDOWN_SPEED) & (df['speed'] > MIN_TOUCHDOWN_SPEED) & \
                   (aligned_with_dest_rwy)
    df.loc[cond_landing, 'phase'] = 'Landing'

    # 2. Takeoff
    cond_takeoff = (df['phase'].isna()) & \
                   (df['dist_to_origin_km'] <= PARKED_DISTANCE_THRESHOLD) & \
                   (df['speed'] > MAX_TAXI_SPEED) & (df['alt_agl_origin'] < TAKEOFF_ALT_THRESHOLD_FT)  & \
                   (aligned_with_origin_rwy)
    df.loc[cond_takeoff, 'phase'] = 'Takeoff'

    # 3. Taxi-in
    cond_taxi_in = ((df['phase'].isna()) & (df['alt_agl_dest'] <= ON_THE_GROUND_ALT_THRESHOLD_FT) & \
                    (df['speed'] <= (MAX_TAXI_SPEED+70)) & (df['dist_to_dest_km'] <= PARKED_DISTANCE_THRESHOLD) & \
                    (df['speed'] > MIN_TAXI_SPEED))
    df.loc[cond_taxi_in, 'phase'] = 'Taxi-in'

    # 4. Taxi-out
    cond_taxi_out = (df['phase'].isna()) & (df['alt_agl_origin'] <= ON_THE_GROUND_ALT_THRESHOLD_FT) & (df['speed'] < MAX_TAXI_SPEED) & (df['speed'] > MIN_TAXI_SPEED) & (df['dist_to_origin_km'] <= PARKED_DISTANCE_THRESHOLD)
    df.loc[cond_taxi_out, 'phase'] = 'Taxi-out'

    in_air_mask = df['phase'].isna()
    cond_approach = in_air_mask & (df['vertical_rate'] < -CLIMB_DESCENT_VR_THRESHOLD) & (df['alt_agl_dest'] < APPROACH_ALT_THRESHOLD_AGL)
    df.loc[cond_approach, 'phase'] = 'Approach'
    cond_climb = in_air_mask & (df['phase'].isna()) & (df['vertical_rate'] > CLIMB_DESCENT_VR_THRESHOLD) & (df['dist_to_origin_km'] > PARKED_DISTANCE_THRESHOLD)
    df.loc[cond_climb, 'phase'] = 'Climb'
    cond_descent = in_air_mask & (df['phase'].isna()) & (df['vertical_rate'] < -CLIMB_DESCENT_VR_THRESHOLD)
    df.loc[cond_descent, 'phase'] = 'Descent'
    df.loc[in_air_mask & (df['phase'].isna()) & ((df['alt_agl_dest'] > ON_THE_GROUND_ALT_THRESHOLD_FT) | (df['alt_agl_origin'] > ON_THE_GROUND_ALT_THRESHOLD_FT)), 'phase'] = 'Cruise'

    return df['phase']

def calculate_generic_phase_durations(segment_traj, phase_column_name, phase_names, prefix):
    """Generic function to calculate time spent in each phase for a given trajectory segment."""
    if len(segment_traj) < 2 or phase_column_name not in segment_traj:
        return {f'{prefix}_{phase.lower()}': 0.0 for phase in phase_names}

    segment_traj_copy = segment_traj.copy()
    segment_traj_copy['duration_to_next_point'] = segment_traj_copy['timestamp'].diff().dt.total_seconds().shift(-1).fillna(0)
    
    total_phase_durations = segment_traj_copy.groupby(phase_column_name)['duration_to_next_point'].sum()
    
    durations = {f'{prefix}_{phase.lower()}': total_phase_durations.get(phase, 0.0) for phase in phase_names}
    
    return durations

def calculate_estimated_takeoff_mass(df):
    """
    Calculates an estimated takeoff mass based on OEW, payload (from load factor), and estimated fuel.
    """
    print("Calculating estimated takeoff mass...")
    
    # --- Load and Merge Load Factor Data ---
    load_factor_path = os.path.join(config.PROCESSED_DATA_DIR, 'average_load_factor_by_airport_pair_v3.csv')
    if not os.path.exists(load_factor_path):
        print(f"Warning: Load factor file not found at {load_factor_path}. Skipping takeoff mass estimation.")
        return df
    
    df_load_factor = pd.read_csv(load_factor_path)
    df = df.merge(df_load_factor, on=['origin_icao', 'destination_icao'], how='left')
    
    # Fill missing load factors with a global average (e.g., 0.85)
    df['average_load_factor'].fillna(0.85, inplace=True)

    # --- Define Constants ---
    AVG_PASSENGER_WEIGHT_KG = 100  # Includes baggage
    FINAL_RESERVE_FUEL_MINS = 30
    CONTINGENCY_FUEL_FRACTION = 0.05
    
    # --- Check for necessary columns from acPerf data ---
    required_cols = ['oew', 'mtow', 'pax_max', 'flight_duration_seconds', 'NA']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing one or more required columns for mass estimation (e.g., oew, mtow, pax_max, NA). Skipping.")
        return df

    # --- Payload Estimation ---
    df['estimated_payload_kg'] = df['pax_max'] * df['average_load_factor'] * AVG_PASSENGER_WEIGHT_KG

    # --- Fuel Estimation (Corrected to use seconds and the correct burn rate) ---
    # Use 'NA' (average burn rate for the flight) for total trip fuel estimation
    df['trip_fuel_kg'] = df['flight_duration_seconds'] * df['NA']
    df['contingency_fuel_kg'] = df['trip_fuel_kg'] * CONTINGENCY_FUEL_FRACTION
    df['final_reserve_fuel_kg'] = (FINAL_RESERVE_FUEL_MINS * 60) * df['NA']
    df['estimated_total_fuel_kg'] = df['trip_fuel_kg'] + df['contingency_fuel_kg'] + df['final_reserve_fuel_kg']

    # --- Takeoff Mass Calculation ---
    df['estimated_takeoff_mass'] = df['oew'] + df['estimated_payload_kg'] + df['estimated_total_fuel_kg']
    
    # --- Final Capping ---
    # Ensure the estimated mass does not exceed MTOW
    df['estimated_takeoff_mass'] = df[['estimated_takeoff_mass', 'mtow']].min(axis=1)

    print("Estimated takeoff mass calculated.")
    return df

def create_interaction_features(df):
    """Creates interaction features to capture more complex relationships."""
    print("Creating interaction features...")

    # Prioritize the new estimated mass, fall back to aircraft_mass
    if 'estimated_takeoff_mass' in df.columns:
        weight_proxy = 'estimated_takeoff_mass'
        print(f"Using '{weight_proxy}' for interaction features.")
    elif 'aircraft_mass' in df.columns:
        weight_proxy = 'aircraft_mass'
        print(f"Falling back to '{weight_proxy}' for interaction features.")
    else:
        print("Warning: No weight proxy column found. Skipping weight-based interaction features.")
        return df

    df[weight_proxy] = df[weight_proxy].fillna(df[weight_proxy].median())

    if 'segment_duration' in df.columns:
        df[f'duration_x_{weight_proxy}'] = df['segment_duration'] * df[weight_proxy]
    if 'great_circle_distance_km' in df.columns:
        df[f'distance_x_{weight_proxy}'] = df['great_circle_distance_km'] * df[weight_proxy]
    if 'seg_altitude_mean' in df.columns:
        df[f'alt_x_{weight_proxy}'] = df['seg_altitude_mean'] * df[weight_proxy]
    if 'seg_groundspeed_mean' in df.columns:
        df[f'speed_x_{weight_proxy}'] = df['seg_groundspeed_mean'] * df[weight_proxy]
    if 'segment_duration' in df.columns and 'seg_altitude_mean' in df.columns:
        df['duration_x_altitude'] = df['segment_duration'] * df['seg_altitude_mean']

    print("Interaction features created.")
    return df

def create_polynomial_features(df):
    """Creates polynomial features for key variables to capture non-linearities."""
    print("Creating polynomial features...")
    features_to_poly = {'segment_duration': ['sq'], 'phase_duration_cl': ['sq', 'cub'], 'alt_diff_rev': ['sq']}
    for feature, degrees in features_to_poly.items():
        if feature in df.columns:
            if 'sq' in degrees: df[f'{feature}_sq'] = df[feature]**2
            if 'cub' in degrees: df[f'{feature}_cub'] = df[feature]**3
    print("Polynomial features created.")
    return df

def create_burn_rate_features(df):
    """Creates features based on aircraft performance data (fuel burn rate)."""
    print("Creating fuel burn rate features...")

    # These columns come from the acPerfOpenAP.csv file
    burn_rate_cols = ['GND', 'CL', 'DE', 'LVL', 'CR']
    if not all(col in df.columns for col in burn_rate_cols):
        print("Warning: Not all fuel burn rate columns found. Skipping burn rate feature creation.")
        return df

    # Calculate a weighted average burn rate for the segment.
    phase_duration_cols = [f'phase_duration_{ph.lower()}' for ph in burn_rate_cols]

    # Ensure duration columns exist
    if not all(col in df.columns for col in phase_duration_cols):
        print("Warning: Not all phase duration columns found. Skipping burn rate feature creation.")
        return df

    total_duration = df[phase_duration_cols].sum(axis=1)

    # Calculate weighted average, handling division by zero
    weighted_burn_rate = np.zeros(len(df))
    for i, ph in enumerate(burn_rate_cols):
        weighted_burn_rate += (df[phase_duration_cols[i]] * df[ph])

    # Rename to reflect that this is a segment-specific burn rate
    df['seg_avg_burn_rate'] = np.divide(weighted_burn_rate, total_duration, out=np.zeros_like(weighted_burn_rate), where=total_duration!=0)
    df['seg_avg_burn_rate'].fillna(0, inplace=True)
    print("Segment average burn rate feature created.")
    return df

def augment_features(df, trajectories_folder):
    print("Augmenting features...")

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    if 'takeoff' in df.columns: df['takeoff'] = pd.to_datetime(df['takeoff'])
    if 'landed' in df.columns: df['landed'] = pd.to_datetime(df['landed'])
    df['segment_duration'] = (df['end'] - df['start']).dt.total_seconds()

    print("Adding new features: flight duration, great circle distance, day of week, and time of day...")
    if 'takeoff' in df.columns and 'landed' in df.columns:
        df['flight_duration_seconds'] = (df['landed'] - df['takeoff']).dt.total_seconds()

    if all(c in df.columns for c in ['origin_latitude', 'origin_longitude', 'destination_latitude', 'destination_longitude']):
        df['great_circle_distance_km'] = haversine_vectorized(df['origin_latitude'], df['origin_longitude'], df['destination_latitude'], df['destination_longitude'])

    #segment start and end dates features
    df['seg_start_day_of_week'] = df['start'].dt.dayofweek
    df['seg_end_day_of_week'] = df['end'].dt.dayofweek
    df['seg_start_time_decimal'] = df['start'].dt.hour + df['start'].dt.minute / 60.0
    df['seg_end_time_decimal'] = df['end'].dt.hour + df['end'].dt.minute / 60.0

    #flight takeoff and landing features
    df['flight_start_day_of_week'] = df['takeoff'].dt.dayofweek
    df['flight_end_day_of_week'] = df['landed'].dt.dayofweek
    df['flight_start_time_decimal'] = df['takeoff'].dt.hour + df['takeoff'].dt.minute / 60.0
    df['flight_end_time_decimal'] = df['landed'].dt.hour + df['landed'].dt.minute / 60.0

    #segment relation to flight features
    df['seg_end_to_landing'] = (df['landed'] - df['end']).dt.total_seconds()
    df['seg_start_to_landing'] = (df['landed'] - df['start']).dt.total_seconds()
    df['seg_end_to_takeoff'] = (df['end'] - df['takeoff']).dt.total_seconds()
    df['seg_start_to_takeoff'] = (df['start'] - df['takeoff']).dt.total_seconds()

    # --- Refactored Column Initialization to Prevent Fragmentation ---
    new_cols = {}

    # Trajectory aggregation columns
    numeric_traj_cols = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'calculated_speed', 'vertical_rate_change', 'dist_to_origin_km', 'dist_to_dest_km']
    aggregations = ['min', 'max', 'mean', 'std']
    for col in numeric_traj_cols:
        for agg in aggregations:
            new_cols[f'seg_{col}_{agg}'] = np.nan

    # Phase fraction and duration columns
    ee_phases = ['Parked', 'Taxi-out', 'Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxi-in']
    openAP_phases = ['GND', 'CL', 'DE', 'LVL', 'CR', 'NA']
    for phase in ee_phases:
        new_cols[f'phase_fraction_{phase.lower()}'] = 0.0
        new_cols[f'ee_phase_duration_{phase.lower()}'] = 0.0
    for phase in openAP_phases:
        new_cols[f'phase_duration_{phase.lower()}'] = 0.0

    # Other miscellaneous columns
    new_cols['start_alt_rev'] = np.nan
    new_cols['end_alt_rev'] = np.nan
    new_cols['departure_rwy_heading'] = 0.0
    new_cols['departure_rwy_length'] = 0.0
    new_cols['arrival_rwy_heading'] = 0.0
    new_cols['arrival_rwy_length'] = 0.0
    new_cols['segment_distance_km'] = 0.0
    new_cols['segment_point_count'] = 0
    new_cols['departure_runway_found'] = False
    new_cols['arrival_runway_found'] = False

    df = df.assign(**new_cols)

    print("Processing trajectory data for each flight...")
    for flight_id, group in tqdm(df.groupby('flight_id'), total=df['flight_id'].nunique()):
        trajectory_file = os.path.join(trajectories_folder, f"{flight_id}.parquet")
        if not os.path.exists(trajectory_file): continue
        traj_df = pd.read_parquet(trajectory_file)
        if traj_df.empty or 'latitude' not in traj_df.columns: continue
        
        traj_df = traj_df.sort_values('timestamp').reset_index(drop=True)
        traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])

        time_diff = traj_df['timestamp'].diff().dt.total_seconds()
        dist_km = haversine_vectorized(traj_df['latitude'].shift(1), traj_df['longitude'].shift(1), traj_df['latitude'], traj_df['longitude'])
        traj_df['calculated_speed'] = (dist_km / time_diff) * 1943.84
        traj_df['calculated_speed'] = traj_df['calculated_speed'].replace([np.inf, -np.inf], np.nan)
        if 'vertical_rate' in traj_df.columns: traj_df['vertical_rate_change'] = traj_df['vertical_rate'].diff()
        if 'track' in traj_df.columns: traj_df['track_change'] = traj_df['track'].diff()

        # Collect Runway and Flight Info
        flight_info = group.iloc[0]
        origin_runway_data, destination_runway_data = {}, {}
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
        traj_df['phase_ee'] = classify_flight_phases_vectorized(traj_df, flight_info.get('origin_elevation', 0), flight_info.get('destination_elevation', 0), origin_runway_headings, dest_runway_headings, flight_info.get('takeoff'), flight_info.get('landed'))
        valid_numeric_cols = [col for col in numeric_traj_cols if col in traj_df.columns]

        # Aggregate features for each segment
        for idx, row in group.iterrows():
            segment_traj = traj_df[(traj_df['timestamp'] >= row['start']) & (traj_df['timestamp'] <= row['end'])]

            segment_for_features = segment_traj
            if segment_traj.empty:
                if traj_df.empty or len(traj_df) < 2:
                    df.loc[idx, 'segment_point_count'] = 0
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

            df.loc[idx, 'segment_point_count'] = len(segment_for_features)

            if segment_for_features.empty:
                continue

            # Calculate segment distance
            if len(segment_for_features.dropna(subset=['latitude', 'longitude'])) > 1:
                df.loc[idx, 'segment_distance_km'] = haversine_vectorized(segment_for_features['latitude'].shift(1), segment_for_features['longitude'].shift(1), segment_for_features['latitude'], segment_for_features['longitude']).sum()
            
            #determine arrival and departure runway
            takeoff_points = segment_for_features[segment_for_features['phase_ee'] == 'Takeoff']
            departure_rwy_found = False
            if not takeoff_points.empty:
                for _, point in takeoff_points.iterrows():
                    for rwy_id, rwy_info in origin_runway_data.items():
                        if is_aligned_with_runway(point['track'], [rwy_info['heading']]):
                            df.loc[idx, 'departure_rwy_heading'] = rwy_info['heading']
                            df.loc[idx, 'departure_rwy_length'] = rwy_info['length']
                            departure_rwy_found = True
                            break
                    if departure_rwy_found:
                        break
            df.loc[idx, 'departure_runway_found'] = departure_rwy_found
            
            landing_points = segment_for_features[segment_for_features['phase_ee'] == 'Landing']
            arrival_rwy_found = False
            if not landing_points.empty:
                for _, point in landing_points.iloc[::-1].iterrows():
                    for rwy_id, rwy_info in destination_runway_data.items():
                        if is_aligned_with_runway(point['track'], [rwy_info['heading']]):
                            df.loc[idx, 'arrival_rwy_heading'] = rwy_info['heading']
                            df.loc[idx, 'arrival_rwy_length'] = rwy_info['length']
                            arrival_rwy_found = True
                            break
                    if arrival_rwy_found:
                        break
            df.loc[idx, 'arrival_runway_found'] = arrival_rwy_found

            stats = segment_for_features[valid_numeric_cols].agg(aggregations)
            for col in valid_numeric_cols:
                for agg in aggregations:
                    df.loc[idx, f'seg_{col}_{agg}'] = stats.loc[agg, col]
            
            phase_counts = segment_for_features['phase_ee'].value_counts(normalize=True)
            for phase_name, fraction in phase_counts.items():
                if pd.notna(phase_name): df.loc[idx, f'phase_fraction_{phase_name.lower()}'] = fraction
            
            # Calculate durations for both sets of phases
            openap_durations = calculate_generic_phase_durations(segment_for_features, 'phase', openAP_phases, 'phase_duration')
            for col, duration in openap_durations.items(): df.loc[idx, col] = duration
            
            ee_durations = calculate_generic_phase_durations(segment_for_features, 'phase_ee', ee_phases, 'ee_phase_duration')
            for col, duration in ee_durations.items(): df.loc[idx, col] = duration

            start_idx = (traj_df['timestamp'] - row['start']).abs().idxmin()
            df.loc[idx, 'start_alt_rev'] = traj_df.loc[start_idx, 'altitude']
            end_idx = (traj_df['timestamp'] - row['end']).abs().idxmin()
            df.loc[idx, 'end_alt_rev'] = traj_df.loc[end_idx, 'altitude']

    # Final Vectorized Calculations
    print("Performing final vectorized calculations...")
    df['alt_diff_rev'] = df['end_alt_rev'] - df['start_alt_rev']
    df['alt_diff_rev_std'] = df.groupby('flight_id')['alt_diff_rev'].transform('std')

    # Calculate delta features for numeric trajectory columns to avoid fragmentation
    delta_cols = {}
    for col in numeric_traj_cols:
        min_col, max_col = f'seg_{col}_min', f'seg_{col}_max'
        if min_col in df.columns and max_col in df.columns:
            delta_cols[f'seg_{col}_delta'] = df[max_col] - df[min_col]
    df = df.assign(**delta_cols)

    # Add Time-Based Features
    if 'takeoff' in df.columns and 'landed' in df.columns:
        print("Calculating time-based features...")
        df['takeoff_delta'] = (df['start'] - df['takeoff']).dt.total_seconds()
        df['landing_delta'] = (df['landed'] - df['end']).dt.total_seconds()
        segment_midpoint_time = df['start'] + (df['end'] - df['start']) / 2
        mean_time_in_air = (segment_midpoint_time - df['takeoff']).dt.total_seconds()
        df['mean_time_in_air'] = np.maximum(0, mean_time_in_air)


    df['fuel_consumption_gnd'] = df['segment_duration'] * df['GND']
    df['fuel_consumption_cl'] = df['segment_duration'] * df['CL']
    df['fuel_consumption_de'] = df['segment_duration'] * df['DE']
    df['fuel_consumption_lvl'] = df['segment_duration'] * df['LVL']
    df['fuel_consumption_cr'] = df['segment_duration'] * df['CR']
    df['fuel_consumption_na'] = df['segment_duration'] * df['NA']

    df['fuel_consumption'] =df['fuel_consumption_gnd'] + df ['fuel_consumption_cl'] + df['fuel_consumption_de'] +\
                            df['fuel_consumption_lvl'] + df['fuel_consumption_cr'] +  df['fuel_consumption_na']



    df = create_burn_rate_features(df)
    df = calculate_estimated_takeoff_mass(df)
    df = create_interaction_features(df)
    df = create_polynomial_features(df)

    print("Feature augmentation complete.")
    return df

if __name__ == "__main__":
    pass
