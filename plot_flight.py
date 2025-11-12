import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def plot_flight_segments_on_map():
    """
    Reads flight data, detects discontinuities, and plots flight segments
    on an interactive globe and a separate vertical profile chart.
    """
    pd.set_option('display.max_columns', None)

    # --- Step 1: Load Supporting Data ---
    base_path = r"data/prc-2025-datasets/"
    try:
        df_apt = pd.read_parquet(os.path.join(base_path, 'apt.parquet'))
        df_fuel_train = pd.read_parquet(os.path.join(base_path, 'fuel_train.parquet'))
        df_flightlist_train = pd.read_parquet(os.path.join(base_path, 'flightlist_train.parquet'))
        print("\nSuccessfully loaded apt.parquet, fuel_train.parquet, and flightlist_train.parquet.")
    except Exception as e:
        print(f"An error occurred while reading the supporting Parquet files: {e}")
        return

    # --- Step 2: Select and Load Trajectory File ---
    default_path = os.path.join(base_path, 'flights_train')
    path_input = input(f"Enter directory path for trajectory files or a specific file path (press Enter for default: '{default_path}'): ")

    if not path_input:
        path_input = default_path
        print(f"Using default path: '{path_input}'")

    if not os.path.exists(path_input):
        print(f"Error: Path '{path_input}' not found.")
        return

    file_path = None

    if os.path.isdir(path_input):
        directory_path = path_input
        try:
            parquet_files = sorted([f for f in os.listdir(directory_path) if f.endswith(('.parquet', '.parq'))])
            if not parquet_files:
                print(f"No Parquet files found in '{directory_path}'.")
                return
        except OSError as e:
            print(f"Error accessing directory: {e}")
            return

        print("\nAvailable Trajectory files:")
        for i, file_name in enumerate(parquet_files):
            print(f"  [{i}] {file_name}")

        selected_file = None
        while selected_file is None:
            identifier = input("\nEnter the number or name of the file to load: ")
            try:
                file_index = int(identifier)
                if 0 <= file_index < len(parquet_files):
                    selected_file = parquet_files[file_index]
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                if identifier in parquet_files:
                    selected_file = identifier
                else:
                    print(f"Invalid file name '{identifier}'. Please try again.")
        
        file_path = os.path.join(directory_path, selected_file)

    elif os.path.isfile(path_input):
        if path_input.endswith(('.parquet', '.parq')):
            file_path = path_input
        else:
            print(f"Error: '{path_input}' is not a Parquet file.")
            return
    else:
        print(f"Error: Path '{path_input}' is not a valid directory or file.")
        return

    if not file_path:
        print("No file selected to load.")
        return

    try:
        flight_df = pd.read_parquet(file_path)
        print(f"\nSuccessfully loaded {os.path.basename(file_path)}.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if flight_df.empty or 'flight_id' not in flight_df.columns:
        print("Loaded DataFrame is empty or missing 'flight_id' column.")
        return

    flight_id_to_plot = flight_df['flight_id'].iloc[0]

    # --- Step 3: Merge Supporting Data ---
    flight_details_row = df_flightlist_train[df_flightlist_train['flight_id'] == flight_id_to_plot]
    flight_details = flight_details_row.iloc[0] if not flight_details_row.empty else None

    flight_fuel_df = df_fuel_train[df_fuel_train['flight_id'] == flight_id_to_plot].copy()
    if not flight_fuel_df.empty:
        flight_fuel_df['start'] = pd.to_datetime(flight_fuel_df['start'])
        flight_df['timestamp'] = pd.to_datetime(flight_df['timestamp'])
        flight_df = pd.merge_asof(flight_df.sort_values('timestamp'), flight_fuel_df[['start', 'fuel_kg']].sort_values('start'), left_on='timestamp', right_on='start', direction='backward')
    else:
        flight_df['fuel_kg'] = np.nan

    print(f"\nProcessing and plotting segments for flight: {flight_id_to_plot}")

    # --- Step 4: Detect Discontinuities and Create Segments ---
    flight_df['timestamp'] = pd.to_datetime(flight_df['timestamp'])
    flight_df = flight_df.sort_values(by='timestamp')

    flight_df['lat_prev'] = flight_df['latitude'].shift(1)
    flight_df['lon_prev'] = flight_df['longitude'].shift(1)
    flight_df['time_prev'] = flight_df['timestamp'].shift(1)

    flight_df['distance_km'] = haversine_distance(
        flight_df['lat_prev'], flight_df['lon_prev'],
        flight_df['latitude'], flight_df['longitude']
    )
    flight_df['time_diff_hours'] = (flight_df['timestamp'] - flight_df['time_prev']) / np.timedelta64(1, 'h')
    flight_df['implied_speed_kmh'] = flight_df['distance_km'] / flight_df['time_diff_hours']

    speed_threshold = 1500
    flight_df['is_discontinuity'] = flight_df['implied_speed_kmh'] > speed_threshold
    flight_df['segment_id'] = flight_df['is_discontinuity'].cumsum()

    num_segments = flight_df['segment_id'].nunique()
    print(f"Detected {num_segments} continuous segment(s) for this flight.")

    # --- Step 5: Plot the Geographic Path ---
    title_map = f'Flight Path for {flight_id_to_plot}'
    if flight_details is not None:
        title_map = f'Flight Path for {flight_id_to_plot}: {flight_details["origin_icao"]} to {flight_details["destination_icao"]}'

    fig_map = go.Figure()

    for segment_id in flight_df['segment_id'].unique():
        segment_df = flight_df[flight_df['segment_id'] == segment_id]
        
        # Create a dynamic hover template
        hover_template = "".join([f"<b>{col}:</b> %{{customdata[{i}]}}<br>" for i, col in enumerate(segment_df.columns)])
        
        fig_map.add_trace(go.Scattergeo(
            lon=segment_df['longitude'], lat=segment_df['latitude'], mode='lines+markers',
            line=dict(width=2), marker=dict(size=4), name=f'Segment {segment_id}',
            hovertemplate=hover_template,
            customdata=segment_df.values
        ))

    if flight_details is not None and not df_apt.empty:
        origin_icao, dest_icao = flight_details["origin_icao"], flight_details["destination_icao"]
        origin_apt, dest_apt = df_apt[df_apt['icao'] == origin_icao], df_apt[df_apt['icao'] == dest_icao]

        if not origin_apt.empty:
            fig_map.add_trace(go.Scattergeo(
                lon=origin_apt['longitude'], lat=origin_apt['latitude'], text=origin_apt['icao'], hoverinfo='text',
                mode='markers+text', textposition="top right", marker=dict(size=10, color='green'), name='Departure'
            ))
        if not dest_apt.empty:
            fig_map.add_trace(go.Scattergeo(
                lon=dest_apt['longitude'], lat=dest_apt['latitude'], text=dest_apt['icao'], hoverinfo='text',
                mode='markers+text', textposition="top right", marker=dict(size=10, color='red'), name='Destination'
            ))

    fig_map.update_layout(
        title_text=title_map, showlegend=True, geo=dict(projection_type='orthographic'),
        margin={"r": 0, "t": 40, "l": 0, "b": 0}, title_x=0.5
    )
    fig_map.show()

    # --- Step 6: Plot the Vertical Profile ---
    title_vertical = f'Vertical Profile for Flight {flight_id_to_plot}'
    fig_vertical = go.Figure()

    for segment_id in flight_df['segment_id'].unique():
        segment_df = flight_df[flight_df['segment_id'] == segment_id]
        
        # Create a dynamic hover template
        hover_template = "".join([f"<b>{col}:</b> %{{customdata[{i}]}}<br>" for i, col in enumerate(segment_df.columns)])

        fig_vertical.add_trace(go.Scatter(
            x=segment_df['timestamp'], y=segment_df['altitude'], mode='lines+markers',
            marker=dict(size=4), name=f'Segment {segment_id}', 
            hovertemplate=hover_template,
            customdata=segment_df.values
        ))

    fig_vertical.update_layout(
        title_text=title_vertical,
        xaxis_title="Time",
        yaxis_title="Altitude (ft)",
        showlegend=True,
        title_x=0.5
    )
    fig_vertical.show()


if __name__ == '__main__':
    plot_flight_segments_on_map()