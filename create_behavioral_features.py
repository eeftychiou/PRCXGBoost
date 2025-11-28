"""
This script performs the heavy lifting of creating a "behavioral signature" for
every aircraft type found in the trajectory data. It should be run once to generate
the features needed for the imputation model.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import config

def create_behavioral_features():
    """
    Analyzes all trajectory and fuel data to create a feature vector summarizing
    the typical flight behavior for each unique aircraft type.
    """
    print("--- Creating Behavioral Features from All Flight Data ---")

    try:
        df_fuel = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/fuel_train.parquet'))
        df_flightlist = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/flightlist_train.parquet'))
    except FileNotFoundError as e:
        print(f"Error: Base data file not found. {e}")
        return

    df_segments = pd.merge(df_fuel, df_flightlist, on='flight_id', how='left')
    df_segments['duration_s'] = (df_segments['end'] - df_segments['start']).dt.total_seconds()
    df_segments['fuel_flow_kg_s'] = df_segments['fuel_kg'] / df_segments['duration_s'].where(df_segments['duration_s'] > 0)

    source_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets/flights_train')
    segment_details = []

    # Limit to a smaller set for faster processing if in TEST_RUN mode
    if config.TEST_RUN:
        unique_flights = df_segments['flight_id'].unique()
        num_test_flights = int(len(unique_flights) * config.TEST_RUN_FRACTION)
        np.random.seed(42)
        test_flight_ids = np.random.choice(unique_flights, size=num_test_flights, replace=False)
        df_segments = df_segments[df_segments['flight_id'].isin(test_flight_ids)]
        print(f"\n*** TEST_RUN MODE: Analyzing {len(test_flight_ids)} flights ***")

    for _, segment in tqdm(df_segments.iterrows(), total=len(df_segments), desc="Analyzing Flight Segments"):
        traj_path = os.path.join(source_dir, f"{segment['flight_id']}.parquet")
        if not os.path.exists(traj_path): continue

        try:
            traj_df = pd.read_parquet(traj_path)
            mask = (traj_df['timestamp'] >= segment['start']) & (traj_df['timestamp'] <= segment['end'])
            segment_traj = traj_df[mask]

            if len(segment_traj) < 2: continue

            avg_gs_kts = segment_traj['groundspeed'].mean()
            avg_gs_km_s = avg_gs_kts * 1.852 / 3600
            efficiency_kg_km = segment['fuel_flow_kg_s'] / avg_gs_km_s if avg_gs_km_s > 0 else np.nan

            segment_details.append({
                'aircraft_type': segment['aircraft_type'],
                'avg_gs': avg_gs_kts,
                'avg_alt': segment_traj['altitude'].mean(),
                'avg_vr': segment_traj['vertical_rate'].mean(),
                'fuel_flow': segment['fuel_flow_kg_s'],
                'efficiency_kg_km': efficiency_kg_km
            })
        except Exception as e:
            print(f"Skipping segment for flight {segment['flight_id']} due to error: {e}")
            continue

    if not segment_details:
        print("\nNo valid segment details could be extracted.")
        return

    df_segment_details = pd.DataFrame(segment_details)
    df_behavioral = df_segment_details.groupby('aircraft_type').agg(
        avg_cruise_gs=('avg_gs', 'mean'),
        avg_cruise_alt=('avg_alt', 'mean'),
        avg_climb_rate=('avg_vr', 'mean'),
        avg_fuel_flow=('fuel_flow', 'mean'),
        avg_efficiency_kg_km=('efficiency_kg_km', 'mean')
    ).dropna()

    output_path = "behavioral_features.csv"
    df_behavioral.to_csv(output_path)

    print(f"\n--- Behavioral Feature Extraction Complete ---")
    print(f"Created behavioral signatures for {len(df_behavioral)} aircraft types.")
    print(f"Saved features to: {output_path}")

if __name__ == '__main__':
    create_behavioral_features()
