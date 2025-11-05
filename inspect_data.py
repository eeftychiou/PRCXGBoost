import pandas as pd
import os

# --- Configuration ---
# File Paths
BASE_PATH = r"data/prc-2025-datasets/"
AC_PERF_PATH = r"data/acPerf/acPerfOpenAP.csv"

def inspect_training_data():
    """Loads, merges, and prints samples and info about the training data."""
    print("--- Inspecting Training Data ---")
    
    # --- Load Data ---
    print("\nLoading and preprocessing full training dataset...")
    try:
        df_fuel = pd.read_parquet(os.path.join(BASE_PATH, 'fuel_train.parquet'))
        df_flightlist = pd.read_parquet(os.path.join(BASE_PATH, 'flightlist_train.parquet'))
        df_ac_perf = pd.read_csv(AC_PERF_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Preprocess Data (mirroring fuel_pinn.py) ---
    df_merged = pd.merge(df_flightlist, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
    df_merged = pd.merge(df_fuel, df_merged, on='flight_id', how='left')
    df_merged = pd.get_dummies(df_merged, columns=['ICAO_ENGINE_DESC'], prefix='eng', dtype=float)
    df_merged['start'] = pd.to_datetime(df_merged['start'])
    df_merged['end'] = pd.to_datetime(df_merged['end'])
    df_merged.dropna(subset=['MASS', 'fuel_kg', 'aircraft_type'], inplace=True)
    
    # --- Explicitly add all possible engine columns to prevent KeyErrors ---
    all_eng_cols = ['eng_J', 'eng_P', 'eng_T']
    for col in all_eng_cols:
        if col not in df_merged.columns:
            df_merged[col] = 0.0

    df_merged.reset_index(drop=True, inplace=True)
    print(f"Loaded and processed {len(df_merged)} total training records.")

    # --- Display Data Samples and Info ---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n\n--- DataFrame Head (First 5 Rows) ---")
    print(df_merged.head())

    print("\n\n--- DataFrame Info ---")
    df_merged.info()

if __name__ == '__main__':
    inspect_training_data()
