"""
This script handles the model training stage of the ML pipeline.

It uses a Gradient Boosting Regressor to train a model on the feature-rich
dataset created by applying the full feature engineering pipeline.
"""
import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import config

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

def classify_flight_phases(traj_df, origin_lat, origin_lon, dest_lat, dest_lon, apt_elevation):
    """Classifies each point in a trajectory into a flight phase based on expert rules."""
    phases = []
    CLIMB_RATE_THRESHOLD, DESCENT_RATE_THRESHOLD = 300, -300
    APPROACH_ALTITUDE_THRESHOLD, TERMINAL_AREA_DISTANCE_THRESHOLD = 10000, 40

    for _, point in traj_df.iterrows():
        alt_agl = point['altitude'] - apt_elevation if apt_elevation else point['altitude']
        dist_to_dest = haversine_distance(point['latitude'], point['longitude'], dest_lat, dest_lon)
        dist_to_origin = haversine_distance(point['latitude'], point['longitude'], origin_lat, origin_lon)

        if dist_to_dest < TERMINAL_AREA_DISTANCE_THRESHOLD and point['vertical_rate'] < 0:
            phases.append('Approach' if alt_agl < APPROACH_ALTITUDE_THRESHOLD else 'Descent')
        elif dist_to_origin < TERMINAL_AREA_DISTANCE_THRESHOLD and point['vertical_rate'] > 0:
            phases.append('Takeoff' if alt_agl < 1500 else 'Climb')
        elif abs(point['vertical_rate']) < CLIMB_RATE_THRESHOLD:
            phases.append('Cruise')
        elif point['vertical_rate'] > CLIMB_RATE_THRESHOLD:
            phases.append('Climb')
        elif point['vertical_rate'] < DESCENT_RATE_THRESHOLD:
            phases.append('Descent')
        else:
            phases.append('Cruise')
    return phases

def engineer_features(df, df_apt):
    """Applies the full feature engineering pipeline to the dataframe."""
    print("Applying full feature engineering pipeline...")
    enhanced_rows = []
    flights_train_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets/flights_train')

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Engineering Features"):
        new_row = row.copy()
        origin_apt = df_apt[df_apt['icao'] == row['origin_icao']]
        dest_apt = df_apt[df_apt['icao'] == row['destination_icao']]
        traj_path = os.path.join(flights_train_dir, f"{row['flight_id']}.parquet")

        if os.path.exists(traj_path) and not origin_apt.empty and not dest_apt.empty:
            origin_lat, origin_lon = origin_apt.iloc[0]['apt_lat'], origin_apt.iloc[0]['apt_lon']
            dest_lat, dest_lon = dest_apt.iloc[0]['apt_lat'], dest_apt.iloc[0]['apt_lon']
            dest_elev = dest_apt.iloc[0]['apt_elev']

            traj_df = pd.read_parquet(traj_path)
            traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
            segment_traj = traj_df[(traj_df['timestamp'] >= new_row['start']) & (traj_df['timestamp'] <= new_row['end'])].copy()

            if len(segment_traj) > 1:
                new_row['segment_duration_s'] = (new_row['end'] - new_row['start']).total_seconds()
                lats, lons = segment_traj['latitude'].values, segment_traj['longitude'].values
                new_row['segment_distance_km'] = np.sum([haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1]) for i in range(len(lats) - 1)])
                new_row['mean_dist_to_origin_km'] = segment_traj.apply(lambda pt: haversine_distance(pt['latitude'], pt['longitude'], origin_lat, origin_lon), axis=1).mean()
                new_row['mean_dist_to_dest_km'] = segment_traj.apply(lambda pt: haversine_distance(pt['latitude'], pt['longitude'], dest_lat, dest_lon), axis=1).mean()

                segment_traj['phase'] = classify_flight_phases(segment_traj, origin_lat, origin_lon, dest_lat, dest_lon, dest_elev)
                phase_fractions = segment_traj['phase'].value_counts(normalize=True)
                for phase in ['Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach']:
                    new_row[f'{phase.lower()}_fraction'] = phase_fractions.get(phase, 0)
            
        enhanced_rows.append(new_row)
    
    return pd.DataFrame(enhanced_rows)

def train():
    """Main training function."""
    print("--- Starting Model Training Stage ---")

    # --- 1. Load Data ---
    processed_data_path = os.path.join(config.PROCESSED_DATA_DIR, f"processed_data{'test' if config.TEST_RUN else ''}.parquet")
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data not found at {processed_data_path}. Please run the data preparation stage first.")
        return

    print(f"Loading processed data from {processed_data_path}...")
    df = pd.read_parquet(processed_data_path)
    df_apt = pd.read_parquet(os.path.join(config.DATA_DIR, 'prc-2025-datasets/apt.parquet'))
    df_apt.rename(columns={'latitude': 'apt_lat', 'longitude': 'apt_lon', 'elevation': 'apt_elev'}, inplace=True)

    # --- 2. Engineer Features ---
    df_featured = engineer_features(df, df_apt)

    base_features = [
        'segment_duration_s', 'segment_distance_km', 'mean_dist_to_origin_km', 'mean_dist_to_dest_km',
        'takeoff_fraction', 'climb_fraction', 'cruise_fraction', 'descent_fraction', 'approach_fraction'
    ]
    engine_features = [col for col in df_featured.columns if col.startswith('eng_')]
    features = base_features + engine_features
    target = 'fuel_kg'

    df_featured.dropna(subset=features + [target], inplace=True)

    X = df_featured[features]
    y = df_featured[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # --- 4. Train Model ---
    print("Training Gradient Boosting Regressor model...")
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='huber')
    gbr.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    print("Evaluating model on validation set...")
    y_pred = gbr.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("\n--- Validation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
    print(f"R-squared (R²): {r2:.4f}")

    # --- 6. Save Model ---
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, f"gbr_model_{run_id}")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    features_path = os.path.join(model_dir, "features.json")

    joblib.dump(gbr, model_path)
    with open(features_path, 'w') as f:
        import json
        json.dump(features, f)

    print(f"\nModel saved to {model_path}")
    print(f"Feature list saved to {features_path}")
    print("--- Model Training Stage Complete ---")

if __name__ == '__main__':
    train()
