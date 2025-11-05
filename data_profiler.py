"""
This script performs a deep, assumption-free analysis of all data sources
to build a complete data dictionary and understanding of the data model.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import config

def profile_dataframe(name, df, f):
    """Helper function to write a detailed profile of a dataframe to the report file."""
    f.write(f"\n" + '='*50 + f"\nProfiling: {name}\n" + '='*50 + "\n")
    f.write(f"Shape: {df.shape}\n")
    f.write("\n--- Column Data Dictionary ---\n")
    
    # Get a small sample for value examples, handling small dataframes
    sample_size = min(3, len(df))
    sample = df.sample(sample_size) if sample_size > 0 else pd.DataFrame()

    for col in df.columns:
        f.write(f"\n  Column: {col}\n")
        f.write(f"    - DType: {df[col].dtype}\n")
        missing_pct = df[col].isnull().sum() / len(df) * 100 if len(df) > 0 else 0
        f.write(f"    - Missing: {missing_pct:.2f}%\n")
        if not sample.empty:
            sample_values = sample[col].tolist()
            f.write(f"    - Sample Values: {sample_values}\n")

def profile_trajectories(name, path, f):
    """Dynamically profiles a directory of trajectory files without assumptions."""
    f.write("\n" + '='*50 + f"\nProfiling Trajectories: {name}\n" + '='*50 + "\n")
    if not os.path.exists(path):
        f.write("Directory not found.\n")
        return

    trajectory_files = [p for p in os.listdir(path) if p.endswith('.parquet')]
    if not trajectory_files:
        f.write("No trajectory files found.\n")
        return

    sample_size = min(500, len(trajectory_files))
    f.write(f"Analyzing a sample of {sample_size} out of {len(trajectory_files)} trajectory files...\n")

    # --- Dynamically discover all possible columns ---
    all_columns = set()
    for filename in np.random.choice(trajectory_files, min(100, len(trajectory_files)), replace=False):
        try:
            df_traj_sample = pd.read_parquet(os.path.join(path, filename))
            all_columns.update(df_traj_sample.columns)
        except Exception:
            continue
    
    f.write(f"\nDiscovered Columns: {sorted(list(all_columns))}\n")

    # --- Profile the discovered columns ---
    col_availability = {col: 0 for col in all_columns}
    col_has_data = {col: 0 for col in all_columns}

    for filename in tqdm(np.random.choice(trajectory_files, sample_size, replace=False), desc=f"Analyzing {name}"):
        try:
            df_traj = pd.read_parquet(os.path.join(path, filename))
            for col in all_columns:
                if col in df_traj:
                    col_availability[col] += 1
                    if df_traj[col].notna().any():
                        col_has_data[col] += 1
        except Exception:
            continue

    f.write("\n--- Column Availability ---\n(% of files in sample where column exists)\n")
    for col, count in sorted(col_availability.items()):
        f.write(f"  - {col}: {count / sample_size * 100:.2f}%\n")

    f.write("\n--- Column Data Presence ---\n(% of files in sample where column has at least one non-null value)\n")
    for col, count in sorted(col_has_data.items()):
        f.write(f"  - {col}: {count / sample_size * 100:.2f}%\n")

def profile_data():
    """Generates a comprehensive report on all project data sources."""
    print("--- Starting Comprehensive Data Profiling ---")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    profile_dir = os.path.join(config.INTROSPECTION_DIR, f"data_profile_{run_id}")
    os.makedirs(profile_dir, exist_ok=True)
    report_path = os.path.join(profile_dir, "data_profile_report.txt")

    base_data_path = os.path.join(config.DATA_DIR, 'prc-2025-datasets')

    with open(report_path, 'w') as f:
        f.write(f"Data Profile Report - {run_id}\n")
        
        file_paths = {
            "flightlist_train": os.path.join(base_data_path, 'flightlist_train.parquet'),
            "flightlist_rank": os.path.join(base_data_path, 'flightlist_rank.parquet'),
            "fuel_train": os.path.join(base_data_path, 'fuel_train.parquet'),
            "apt": os.path.join(base_data_path, 'apt.parquet'),
            "acPerfOpenAP": os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv'),
            "fuel_rank_submission.parquet" :os.path.join(base_data_path, 'fuel_rank_submission.parquet')
        }

        for name, path in file_paths.items():
            try:
                df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
                profile_dataframe(name, df, f)
            except Exception as e:
                f.write(f"\n--- Profiling: {name} ---\nERROR: {e}\n")

        profile_trajectories("flights_train", os.path.join(base_data_path, 'flights_train'), f)
        profile_trajectories("flights_rank", os.path.join(base_data_path, 'flights_rank'), f)

    print(f"\nProfiling complete. Report saved to: {report_path}")

if __name__ == '__main__':
    profile_data()
