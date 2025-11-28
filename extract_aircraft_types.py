"""
This script scans all trajectory files from both the 'flights_train' and 'flights_rank'
directories to find and list all unique aircraft types.
"""
import os
import pandas as pd
from tqdm import tqdm

import config

def find_unique_aircraft_types():
    """Scans all trajectory files and outputs a CSV of unique aircraft types."""
    print("--- Starting Aircraft Type Extraction ---")

    # --- 1. Define Source Directories ---
    train_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets/flights_train')
    rank_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets/flights_rank')
    source_dirs = [train_dir, rank_dir]

    all_files = []
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found at {source_dir}. Skipping.")
            continue
        
        files_in_dir = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.parquet')]
        all_files.extend(files_in_dir)
        print(f"Found {len(files_in_dir)} files in {os.path.basename(source_dir)}.")

    if not all_files:
        print(f"No trajectory files found in the specified directories.")
        return

    print(f"\nFound a total of {len(all_files)} trajectory files to process.")

    # --- 2. Process Files and Extract Types ---
    unique_aircraft_types = set()

    for file_path in tqdm(all_files, desc="Processing Trajectories"):
        try:
            traj_df = pd.read_parquet(file_path)

            if 'typecode' in traj_df.columns:
                unique_types_in_file = traj_df['typecode'].unique()
                unique_aircraft_types.update(unique_types_in_file)

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            continue

    print(f"\nFound {len(unique_aircraft_types)} unique aircraft types.")

    # --- 3. Save to CSV ---
    output_df = pd.DataFrame(sorted(list(unique_aircraft_types)), columns=['aircraft_type'])
    output_path = "unique_aircraft_types.csv"
    output_df.to_csv(output_path, index=False)

    print(f"--- Extraction Complete ---")
    print(f"List of unique aircraft types saved to: {output_path}")

if __name__ == '__main__':
    find_unique_aircraft_types()
