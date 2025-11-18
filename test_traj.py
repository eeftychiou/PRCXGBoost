"""
This script visualizes flight trajectories using the traffic library.

It takes a flight ID and a data source (original or interpolated) as command-line
arguments and generates a plot showing the horizontal trajectory, vertical profile,
and flight phases.

This script is revised based on the traffic library's official documentation and tutorials.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # Import cartopy for map projections
from traffic.core import Flight
# You may need to create a config.py file or replace these with actual paths
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

    print(
        f"Error: Trajectory file for flight ID {flight_id} not found in any of the following directories: {base_dirs}")
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
    rename_mapping = {
        'latitude': 'latitude',
        'longitude': 'longitude',
        'altitude': 'altitude',
        'timestamp': 'timestamp',
        'ground_speed': 'groundspeed',
    }
    df = df.rename(columns=rename_mapping)

    # Ensure timestamp is in the correct format and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    flight = Flight(df)

    # -- CORRECTION --
    # The erroneous `assign_distance()` and `assign_phase()` calls have been removed.
    # The library computes these attributes automatically when requested by other methods.

    # Use the 'traffic' style context for better plot aesthetics
    with plt.style.context("traffic"):
        fig = plt.figure(figsize=(15, 22))
        fig.suptitle(f"Analysis for Flight {args.flight_id}", fontsize=18)

        # 1. Horizontal Trajectory on a map
        ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
        ax1.set_title("Horizontal Trajectory")
        flight.plot(ax1, marker=".")
        ax1.coastlines()
        ax1.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)

        # 2. Vertical Profile (Altitude vs. Distance)
        # The 'distance' column is computed by the library when requested here.
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.set_title("Vertical Profile")
        flight.plot(ax=ax2, x='distance', y='altitude')
        ax2.set_ylabel("Altitude (ft)")
        ax2.set_xlabel("Cumulative Distance (nm)")
        ax2.grid(True, linestyle='--', alpha=0.6)

        # 3. Flight Phases
        # As seen in the documentation, `.phases()` is a method that returns
        # an iterable object which has its own `plot()` method.
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.set_title("Flight Phases")
        flight.phases().plot(ax=ax3)
        ax3.set_ylabel("Flight Phase")
        ax3.set_xlabel("Time (UTC)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save the plot
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        data_source_suffix = "interpolated" if args.interpolated else "original"
        output_path = os.path.join(output_dir, f"flight_{args.flight_id}_{data_source_suffix}.png")
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()