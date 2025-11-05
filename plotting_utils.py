"""
This module provides flight trajectory plotting using the Plotly library,
-inspired by the project's original plot_flight.py script.
"""
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_flight_profiles(enhanced_df, flight_id, origin_icao, destination_icao, output_filepath):
    """
    Creates and saves a comprehensive plot with map, vertical, and speed profiles.
    """
    df = enhanced_df.copy().sort_values(by='timestamp').reset_index(drop=True)

    # --- 1. Detect Discontinuities ---
    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['implied_speed_kmh'] = (df['dist_from_prev_km'] / (df['time_diff_s'] / 3600.0)).fillna(0)
    speed_threshold = 1500
    df['is_discontinuity'] = df['implied_speed_kmh'] > speed_threshold
    df['segment_id'] = df['is_discontinuity'].cumsum()

    # --- 2. Create Subplots ---
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Flight Path and Dominant Phase", "Vertical Profile", "Ground Speed Profile"),
        specs=[[{"type": "scattergeo"}], [{"type": "scatter"}], [{"type": "scatter"}]],
        vertical_spacing=0.08
    )

    # --- 3. Prepare Categorical Colors for Plotly ---
    phase_categories = df['hard_phase'].astype('category')
    df['phase_code'] = phase_categories.cat.codes
    unique_phases = phase_categories.cat.categories
    
    # Using a built-in Plotly color scale
    colorscale = 'Viridis'

    # --- 4. Plot Continuous Segments ---
    for segment_id in df['segment_id'].unique():
        segment_df = df[df['segment_id'] == segment_id]
        
        # MAP PLOT
        fig.add_trace(go.Scattergeo(
            lon=segment_df['longitude'], lat=segment_df['latitude'],
            mode='lines', line=dict(width=2, color='blue'),
            name=f'Segment {segment_id}', hoverinfo='none'), row=1, col=1)
        
        fig.add_trace(go.Scattergeo(
            lon=segment_df['longitude'], lat=segment_df['latitude'],
            mode='markers', 
            marker=dict(
                size=5, 
                color=segment_df['phase_code'], 
                colorscale=colorscale, 
                cmin=0, 
                cmax=len(unique_phases)-1,
                colorbar=dict(title="Flight Phase", tickvals=list(range(len(unique_phases))), ticktext=unique_phases)
            ),
            hoverinfo='text', text=segment_df['hard_phase'],
            name='Flight Phases'), row=1, col=1)

        # VERTICAL PROFILE
        fig.add_trace(go.Scatter(x=segment_df['timestamp'], y=segment_df['altitude'], mode='lines', name=f'Altitude', line=dict(color='blue')), row=2, col=1)
        
        # GROUND SPEED PROFILE
        fig.add_trace(go.Scatter(x=segment_df['timestamp'], y=segment_df['groundspeed'], mode='lines', name=f'Ground Speed', line=dict(color='green')), row=3, col=1)

    # --- 5. Highlight Discontinuities ---
    discontinuities = df[df['is_discontinuity']]
    for _, gap in discontinuities.iterrows():
        start_point = df.iloc[gap.name - 1]
        end_point = gap

        fig.add_trace(go.Scattergeo(lon=[start_point['longitude'], end_point['longitude']], lat=[start_point['latitude'], end_point['latitude']], mode='lines', line=dict(width=2, color='red', dash='dash'), name='Missing Segment'), row=1, col=1)
        fig.add_vrect(x0=start_point['timestamp'], x1=end_point['timestamp'], fillcolor="red", opacity=0.2, line_width=0, row=2, col=1)
        fig.add_vrect(x0=start_point['timestamp'], x1=end_point['timestamp'], fillcolor="red", opacity=0.2, line_width=0, row=3, col=1)

    # --- 6. Finalize Layout ---
    fig.update_layout(
        title_text=f'Flight {flight_id}: {origin_icao} to {destination_icao}', 
        showlegend=False, 
        height=1400, 
        margin={"r": 20, "t": 80, "l": 20, "b": 20}, 
        title_x=0.5
    )
    fig.update_yaxes(title_text="Altitude (ft)", row=2, col=1)
    fig.update_yaxes(title_text="Ground Speed (kts)", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    fig.update_geos(projection_type='orthographic', row=1, col=1)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    fig.write_image(output_filepath)
    print(f"Plot saved to {output_filepath}")
