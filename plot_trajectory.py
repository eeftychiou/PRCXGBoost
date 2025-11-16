"""
This script visualizes flight trajectories using the traffic library.

It takes a flight ID and a data source (original or interpolated) as command-line
arguments and generates a plot showing the horizontal trajectory, vertical profile,
and flight phases.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from traffic.core import Flight, Traffic
import config

def find_and_load_trajectory(flight_id, use_interpolated):
    """
    Finds and loads a trajectory file for a given flight ID from the specified data source.
    It checks both training and ranking directories.

    Args:
        flight_id (str): The ID of the flight to load.
        use_interpolated (bool): If True, loads from the interpolated data directory.
                                 Otherwise, loads from the original data directories.

    Returns:
        pandas.DataFrame: The loaded trajectory data, or None if the file is not found.
    """
    if use_interpolated:
        base_dirs = [
            os.path.join(config.INTERPOLATED_TRAJECTORIES_DIR, "flights_train"),
            os.path.join(config.INTERPOLATED_TRAJECTORIES_DIR, "flights_rank")
        ]
        data_source_name = "interpolated"
    else:
        base_dirs = [config.FLIGHTS_TRAIN_DIR, config.FLIGHTS_RANK_DIR]
        data_source_name = "original"

    for base_dir in base_dirs:
        file_path = os.path.join(base_dir, f"{flight_id}.parquet")
        if os.path.exists(file_path):
            print(f"Loading {data_source_name} trajectory for flight ID {flight_id} from {file_path}...")
            return pd.read_parquet(file_path)

    print(f"Error: Trajectory file for flight ID {flight_id} not found in any of the following directories: {base_dirs}")
    return None

def main():
    """
    Main function to parse arguments, load data, and generate plots.
    """
    parser = argparse.ArgumentParser(description="Plot flight trajectory, vertical profile, and flight phases.")
    parser.add_argument("flight_id", type=str, help="The ID of the flight to plot (e.g., 'prc786592201').")
    parser.add_argument("--interpolated", action="store_true", help="Use interpolated trajectory data.")
    args = parser.parse_args()

    df = find_and_load_trajectory(args.flight_id, args.interpolated)

    if df is None:
        return

    # The 'traffic' library expects specific column names.
    # We rename them here if they are different in the source file.
    rename_mapping = {
        'latitude': 'latitude',
        'longitude': 'longitude',
        'altitude': 'altitude',
        'timestamp': 'timestamp',
        'ground_speed': 'groundspeed',
        'true_airspeed': 'TAS',
        'calibrated_airspeed': 'CAS'
    }
    df = df.rename(columns=rename_mapping)
    
    # Ensure timestamp is in the correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    flight = Flight(df)
    
    # --- Calculate distance and phases ---
    # This needs to be done before plotting
    # flight = flight.assign_distance(inplace=False)
    # flight = flight.assign_phase(inplace=False)


    # Create a figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # 1. Horizontal Trajectory
    ax1 = axes[0]
    ax1.set_title(f"Flight {args.flight_id} - Horizontal Trajectory")
    flight.plot(ax1, marker=".")

    # 2. Vertical Profile
    ax2 = axes[1]
    ax2.set_title(f"Flight {args.flight_id} - Vertical Profile")
    flight.plot(ax=ax2, x='distance', y='altitude')
    
    # 3. Flight Phases
    ax3 = axes[2]
    ax3.set_title(f"Flight {args.flight_id} - Flight Phases")
    flight.phases.plot(ax=ax3)

    plt.tight_layout()

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    data_source_suffix = "interpolated" if args.interpolated else "original"
    output_path = os.path.join(output_dir, f"flight_{args.flight_id}_{data_source_suffix}.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
