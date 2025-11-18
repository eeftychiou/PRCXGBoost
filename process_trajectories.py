import pandas as pd
import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import config
import readers
from interpolate import interpolate

def process_all_trajectories(smooth: float):
    """
    Processes and interpolates all trajectory files from the input directories
    and saves them to the output directory.

    Args:
        smooth (float): The smoothing factor for the interpolation.
    """
    input_base_dir = os.path.join(config.DATA_DIR, 'filtered_trajectories')
    output_base_dir = config.INTERPOLATED_TRAJECTORIES_DIR
    sub_dirs = ['flights_train', 'flights_rank', 'flights_final']

    for sub_dir in sub_dirs:
        input_dir = os.path.join(input_base_dir, sub_dir)
        output_dir = os.path.join(output_base_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        trajectory_files = glob.glob(os.path.join(input_dir, '*.parquet'))

        for file_path in tqdm(trajectory_files, desc=f"Processing {sub_dir}", unit="file"):
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(output_dir, file_name)

            if os.path.exists(output_file_path):
                continue

            df = pd.read_parquet(file_path)
            df.columns = [col.lower() for col in df.columns]

            # Sort by flight and time, and remove duplicates to prevent assertion errors
            df = df.sort_values(by=['flight_id', 'timestamp'])
            df = df.drop_duplicates(subset=['flight_id', 'timestamp'], keep='first')

            df = readers.convert_from_SI(readers.add_features_trajectories(readers.convert_to_SI(df)))
            df_interpolated = df.groupby("flight_id").apply(lambda x: interpolate(x, smooth), include_groups=False).reset_index()
            
            if "level_1" in df_interpolated.columns:
                df_interpolated = df_interpolated.drop(columns="level_1")

            df_interpolated.to_parquet(output_file_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate all trajectory files.'
    )
    parser.add_argument("--smooth", type=float, default=0.1, help="Smoothing factor for interpolation.")
    args = parser.parse_args()

    process_all_trajectories(args.smooth)
