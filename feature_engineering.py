import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
import datetime

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

def classify_flight_phases(traj_df, origin_lat, origin_lon, dest_lat, dest_lon,
                           origin_apt_elevation, dest_apt_elevation,
                           start_time, takeoff_time, landed_time, end_time,
                           origin_runway_data, destination_runway_data):
    """Classifies each point in a trajectory into a flight phase based on expert rules."""
    phases = []
    CLIMB_RATE_THRESHOLD, DESCENT_RATE_THRESHOLD = 300, -300
    APPROACH_ALTITUDE_THRESHOLD, TERMINAL_AREA_DISTANCE_THRESHOLD = 10000, 40 # feet, km

    TAKEOFF_LANDING_TIME_WINDOW_MIN = 5 # minutes
    RUNWAY_ALIGNMENT_TOLERANCE_DEG = 15 # degrees
    LANDING_ALTITUDE_THRESHOLD_AGL = 5000 # feet AGL
    TAKEOFF_ALTITUDE_THRESHOLD_AGL = 1500 # feet AGL for specific takeoff phase

    origin_runway_headings = [r['heading'] for r in origin_runway_data.values() if pd.notna(r['heading'])]
    destination_runway_headings = [r['heading'] for r in destination_runway_data.values() if pd.notna(r['heading'])]

    for _, point in traj_df.iterrows():
        current_timestamp = point['timestamp']
        
        alt_agl_origin = point['altitude'] - origin_apt_elevation if pd.notna(origin_apt_elevation) else point['altitude']
        alt_agl_dest = point['altitude'] - dest_apt_elevation if pd.notna(dest_apt_elevation) else point['altitude']

        dist_to_dest = haversine_distance(point['latitude'], point['longitude'], dest_lat, dest_lon)
        dist_to_origin = haversine_distance(point['latitude'], point['longitude'], origin_lat, origin_lon)

        # 1. Specific Takeoff Phase (aligned with runway, climbing, low alt AGL)
        if (takeoff_time <= current_timestamp <= takeoff_time + datetime.timedelta(minutes=TAKEOFF_LANDING_TIME_WINDOW_MIN)) and \
           point['vertical_rate'] > 0 and \
           alt_agl_origin < TAKEOFF_ALTITUDE_THRESHOLD_AGL and \
           is_aligned_with_runway(point['track'], origin_runway_headings, RUNWAY_ALIGNMENT_TOLERANCE_DEG):
            phases.append('Takeoff')
            continue

        # 2. Specific Landing Phase (aligned with runway, descending, low alt AGL, close to dest)
        if (landed_time - datetime.timedelta(minutes=TAKEOFF_LANDING_TIME_WINDOW_MIN) <= current_timestamp < landed_time) and \
           point['vertical_rate'] < 0 and \
           dist_to_dest < TERMINAL_AREA_DISTANCE_THRESHOLD and \
           alt_agl_dest < LANDING_ALTITUDE_THRESHOLD_AGL and \
           is_aligned_with_runway(point['track'], destination_runway_headings, RUNWAY_ALIGNMENT_TOLERANCE_DEG):
            phases.append('Landing')
            continue

        # 3. Taxiing Phase
        if (start_time <= current_timestamp < takeoff_time) or \
           (landed_time < current_timestamp <= end_time):
            phases.append('Taxiing')
            continue

        # 4. General Climb/Descent/Approach/Cruise
        if point['vertical_rate'] > CLIMB_RATE_THRESHOLD:
            phases.append('Climb')
        elif point['vertical_rate'] < DESCENT_RATE_THRESHOLD:
            phases.append('Descent')
        elif dist_to_dest < TERMINAL_AREA_DISTANCE_THRESHOLD and point['vertical_rate'] < 0:
            phases.append('Approach')
        elif dist_to_origin < TERMINAL_AREA_DISTANCE_THRESHOLD and point['vertical_rate'] > 0:
            phases.append('Climb') # General climb near origin, if not specific takeoff
        elif abs(point['vertical_rate']) < CLIMB_RATE_THRESHOLD:
            phases.append('Cruise')
        else:
            phases.append('Cruise') # Fallback
    return phases

def engineer_features(df, flights_dir, start_col='start', end_col='end', desc="Engineering Features"):
    """Applies the full feature engineering pipeline to the dataframe."""
    print(f"Applying full feature engineering pipeline...")
    enhanced_rows = []

    # Constants for runway determination
    RUNWAY_ALIGNMENT_TOLERANCE_DEG = 15 # degrees

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        new_row = row.copy()
        traj_path = os.path.join(flights_dir, f"{row['flight_id']}.parquet")

        # Initialize new runway columns to 0.0
        new_row['arrival_rwy_heading'] = 0.0
        new_row['departure_rwy_heading'] = 0.0
        new_row['arrival_rwy_length'] = 0.0
        new_row['departure_rwy_length'] = 0.0

        # Initialize new segment features to NaN or 0
        new_row['starting_altitude'] = np.nan
        new_row['ending_altitude'] = np.nan
        new_row['altitude_difference'] = np.nan
        new_row['mean_vertical_rate'] = np.nan
        new_row['std_vertical_rate'] = np.nan
        new_row['mean_track'] = np.nan
        new_row['std_track'] = np.nan
        new_row['num_trajectory_points'] = 0
        new_row['points_per_second'] = 0.0

        # Extract times
        takeoff_time = row['takeoff']
        landed_time = row['landed']
        start_time = new_row[start_col]
        end_time = new_row[end_col]

        # Always get origin/destination coordinates for distance calculations
        origin_lat, origin_lon = row['origin_latitude'], row['origin_longitude']
        dest_lat, dest_lon = row['destination_latitude'], row['destination_longitude']
        
        # Calculate overall flight distance for use when trajectory is missing
        great_circle_distance_origin_to_dest = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)

        if os.path.exists(traj_path):
            origin_elev = row['origin_elevation']
            dest_elev = row['destination_elevation']

            # Collect runway data (heading and length) for origin and destination
            origin_runway_data = {}
            destination_runway_data = {}
            for i in range(1, 9): # Assuming up to 8 runways
                for suffix in ['', 'a', 'b']: # Include empty suffix for single runways, then 'a', 'b'
                    origin_rwy_heading_col = f'origin_RWY_{i}_HEADING_{suffix}'
                    origin_rwy_length_col = f'origin_RWY_{i}_LENGTH'
                    dest_rwy_heading_col = f'destination_RWY_{i}_HEADING_{suffix}'
                    dest_rwy_length_col = f'destination_RWY_{i}_LENGTH'

                    if origin_rwy_heading_col in row and pd.notna(row[origin_rwy_heading_col]):
                        origin_runway_data[f'RWY_{i}{suffix}'] = {
                            'heading': row[origin_rwy_heading_col],
                            'length': row[origin_rwy_length_col] if origin_rwy_length_col in row and pd.notna(row[origin_rwy_length_col]) else 0.0
                        }
                    if dest_rwy_heading_col in row and pd.notna(row[dest_rwy_heading_col]):
                        destination_runway_data[f'RWY_{i}{suffix}'] = {
                            'heading': row[dest_rwy_heading_col],
                            'length': row[dest_rwy_length_col] if dest_rwy_length_col in row and pd.notna(row[dest_rwy_length_col]) else 0.0
                        }

            traj_df = pd.read_parquet(traj_path)
            traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
            segment_traj = traj_df[(traj_df['timestamp'] >= start_time) & (traj_df['timestamp'] <= end_time)].copy()

            if len(segment_traj) > 1:
                new_row['segment_duration_s'] = (end_time - start_time).total_seconds()

                # Filter out NaN lat/lon for distance calculations and mean distance to origin/destination
                valid_traj_points_for_dist = segment_traj.dropna(subset=['latitude', 'longitude'])

                if len(valid_traj_points_for_dist) > 1:
                    lats_dist = valid_traj_points_for_dist['latitude'].values
                    lons_dist = valid_traj_points_for_dist['longitude'].values
                    
                    # Calculate segment distance by summing distances between consecutive valid points
                    segment_distances = []
                    for i in range(len(lats_dist) - 1):
                        dist = haversine_distance(lats_dist[i], lons_dist[i], lats_dist[i+1], lons_dist[i+1])
                        if pd.notna(dist):
                            segment_distances.append(dist)
                    new_row['segment_distance_km'] = np.sum(segment_distances) if segment_distances else 0.0
                    
                    # Calculate mean distances to origin/destination using only valid points
                    new_row['mean_dist_to_origin_km'] = valid_traj_points_for_dist.apply(
                        lambda pt: haversine_distance(pt['latitude'], pt['longitude'], origin_lat, origin_lon), axis=1
                    ).mean()
                    new_row['mean_dist_to_dest_km'] = valid_traj_points_for_dist.apply(
                        lambda pt: haversine_distance(pt['latitude'], pt['longitude'], dest_lat, dest_lon), axis=1
                    ).mean()
                else:
                    # Not enough valid points for distance calculations
                    new_row['segment_distance_km'] = 0.0
                    new_row['mean_dist_to_origin_km'] = np.nan
                    new_row['mean_dist_to_dest_km'] = np.nan

                # Other features (altitude, vertical rate, track, num_points) are calculated on the original segment_traj
                # as the request specifically mentioned lat/lon NaNs for "ignore and use next available"
                # and mean/std functions already handle NaNs by default.
                new_row['starting_altitude'] = segment_traj['altitude'].iloc[0] if not segment_traj['altitude'].empty else np.nan
                new_row['ending_altitude'] = segment_traj['altitude'].iloc[-1] if not segment_traj['altitude'].empty else np.nan
                new_row['altitude_difference'] = new_row['ending_altitude'] - new_row['starting_altitude']
                new_row['mean_vertical_rate'] = segment_traj['vertical_rate'].mean()
                new_row['std_vertical_rate'] = segment_traj['vertical_rate'].std()
                new_row['mean_track'] = segment_traj['track'].mean()
                new_row['std_track'] = segment_traj['track'].std()
                new_row['num_trajectory_points'] = len(segment_traj)
                if new_row['segment_duration_s'] > 0:
                    new_row['points_per_second'] = len(segment_traj) / new_row['segment_duration_s']
                else:
                    new_row['points_per_second'] = 0.0

                segment_traj['phase'] = classify_flight_phases(
                    segment_traj,
                    origin_lat, origin_lon,
                    dest_lat, dest_lon,
                    origin_elev, dest_elev,
                    start_time, takeoff_time, landed_time, end_time,
                    origin_runway_data, destination_runway_data # Pass runway data
                )
                phase_fractions = segment_traj['phase'].value_counts(normalize=True)
                all_possible_phases = ['Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxiing'] # Added 'Landing'
                for phase in all_possible_phases:
                    new_row[f'{phase.lower()}_fraction'] = phase_fractions.get(phase, 0)

                # Determine Departure Runway
                departure_rwy_found = False
                for idx, point in segment_traj.iterrows():
                    if point['phase'] == 'Takeoff':
                        for rwy_id, rwy_info in origin_runway_data.items():
                            if is_aligned_with_runway(point['track'], [rwy_info['heading']], RUNWAY_ALIGNMENT_TOLERANCE_DEG):
                                new_row['departure_rwy_heading'] = rwy_info['heading']
                                new_row['departure_rwy_length'] = rwy_info['length']
                                departure_rwy_found = True
                                break
                    if departure_rwy_found:
                        break

                # Determine Arrival Runway
                arrival_rwy_found = False
                # Iterate backwards to find the landing phase closer to the actual landing
                for idx, point in segment_traj.iloc[::-1].iterrows(): 
                    if point['phase'] == 'Landing':
                        for rwy_id, rwy_info in destination_runway_data.items():
                            if is_aligned_with_runway(point['track'], [rwy_info['heading']], RUNWAY_ALIGNMENT_TOLERANCE_DEG):
                                new_row['arrival_rwy_heading'] = rwy_info['heading']
                                new_row['arrival_rwy_length'] = rwy_info['length']
                                arrival_rwy_found = True
                                break
                    if arrival_rwy_found:
                        break

            else: # No trajectory points for this segment or only one point
                # Determine the phase for the entire segment based on timestamps
                inferred_phase = 'Unknown_NoTrajectory' # Default if no clear phase can be assigned
                
                # Check for taxiing before takeoff
                if end_time <= takeoff_time:
                    inferred_phase = 'Taxiing'
                    new_row['mean_dist_to_origin_km'] = 0.0
                    new_row['mean_dist_to_dest_km'] = great_circle_distance_origin_to_dest
                # Check for taxiing after landing
                elif start_time >= landed_time:
                    inferred_phase = 'Taxiing'
                    new_row['mean_dist_to_origin_km'] = great_circle_distance_origin_to_dest
                    new_row['mean_dist_to_dest_km'] = 0.0
                else: # Unknown_NoTrajectory
                    new_row['mean_dist_to_origin_km'] = np.nan
                    new_row['mean_dist_to_dest_km'] = np.nan
                
                # Assign 1.0 fraction to the inferred phase, 0 to others
                all_possible_phases = ['Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxiing', 'Unknown_NoTrajectory'] # Added 'Landing'
                for phase in all_possible_phases:
                    new_row[f'{phase.lower()}_fraction'] = 0
                new_row[f'{inferred_phase.lower()}_fraction'] = 1.0
                
                # Also set other segment-level features to NaN or 0 as no trajectory data
                new_row['segment_duration_s'] = (end_time - start_time).total_seconds()
                new_row['segment_distance_km'] = 0.0 # No movement recorded
        else: # No trajectory file found at all
            inferred_phase = 'Unknown_NoTrajectory'
            
            # Check for taxiing before takeoff
            if end_time <= takeoff_time:
                inferred_phase = 'Taxiing'
                new_row['mean_dist_to_origin_km'] = 0.0
                new_row['mean_dist_to_dest_km'] = great_circle_distance_origin_to_dest
            # Check for taxiing after landing
            elif start_time >= landed_time:
                inferred_phase = 'Taxiing'
                new_row['mean_dist_to_origin_km'] = great_circle_distance_origin_to_dest
                new_row['mean_dist_to_dest_km'] = 0.0
            else: # Unknown_NoTrajectory
                new_row['mean_dist_to_origin_km'] = np.nan
                new_row['mean_dist_to_dest_km'] = np.nan

            # Assign 1.0 fraction to the inferred phase, 0 to others
            all_possible_phases = ['Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Landing', 'Taxiing', 'Unknown_NoTrajectory'] # Added 'Landing'
            for phase in all_possible_phases:
                new_row[f'{phase.lower()}_fraction'] = 0
            new_row[f'{inferred_phase.lower()}_fraction'] = 1.0
            
            # Also set other segment-level features to NaN or 0 as no trajectory data
            new_row['segment_duration_s'] = (end_time - start_time).total_seconds()
            new_row['segment_distance_km'] = 0.0 # No movement recorded

        enhanced_rows.append(new_row)
    
    return pd.DataFrame(enhanced_rows)
