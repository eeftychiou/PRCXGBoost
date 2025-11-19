import warnings
import pitot.geodesy
import utils
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

def joincdates(flights, airports, trajectories_dir, rod=-1000, roc=1000):
    ''' Compute corrected dates and fill with original dates if cannot be corrected '''
    cdates = compute_dates(flights, airports, trajectories_dir, rod=rod, roc=roc)
    cdates = fillnadates(flights, cdates)
    
    # Join the corrected dates
    flights_corrected = flights.join(cdates.set_index("flight_id"), on="flight_id", how="inner")
    
    # Rename columns to the desired output format
    flights_corrected = flights_corrected.rename(columns={
        "takeoff": "takeoff_o",
        "landed": "landed_o",
        "t_adep": "takeoff",
        "t_ades": "landed"
    })
    
    return flights_corrected

def compute_dates(flights, airports, trajectories_dir, rod=-1000, roc=1000):
    ''' Compute corrected dates by processing trajectories one by one '''
    airports = airports.rename(columns={"icao": "icao_code", "latitude": "latitude_deg", "longitude": "longitude_deg"})
    
    results = []

    for _, flight_info in tqdm(flights.iterrows(), total=flights.shape[0], desc="Correcting Dates"):
        flight_id = flight_info['flight_id']
        
        traj_path = os.path.join(trajectories_dir, f"{flight_id}.parquet")
        
        if not os.path.exists(traj_path):
            warnings.warn(f"Trajectory file not found for flight_id: {flight_id}")
            continue

        try:
            df_traj = pd.read_parquet(traj_path, columns=['timestamp', 'latitude', 'longitude', 'altitude'])
        except Exception as e:
            warnings.warn(f"Could not read trajectory for {flight_id}: {e}")
            continue

        flight_result = {"flight_id": flight_id}

        for apt_type, icao_code in [("adep", flight_info["origin_icao"]), ("ades", flight_info["destination_icao"])]:
            airport_info = airports[airports['icao_code'] == icao_code]
            if airport_info.empty:
                flight_result[f"t_{apt_type}"] = pd.NaT
                continue

            lat_deg = airport_info.iloc[0]['latitude_deg']
            lon_deg = airport_info.iloc[0]['longitude_deg']

            num_points = len(df_traj)
            lat_array = np.full(num_points, lat_deg)
            lon_array = np.full(num_points, lon_deg)

            df_traj[f"distance_{apt_type}"] = pitot.geodesy.distance(
                df_traj.latitude.to_numpy(),
                df_traj.longitude.to_numpy(),
                lat_array,
                lon_array) / utils.NM2METER
            
            mindist = df_traj.query(f"distance_{apt_type}<10")
            if mindist.empty:
                flight_result[f"t_{apt_type}"] = pd.NaT
                continue

            idxdistmin = mindist["timestamp"].idxmin() if apt_type == "ades" else mindist["timestamp"].idxmax()
            
            xtime = df_traj.loc[idxdistmin, ["timestamp", "altitude"]]
            rocd = rod if apt_type == "ades" else roc
            
            try:
                timedelta_ns = ((xtime.altitude / rocd * 60) * 1e9).round()
                corrected_time = xtime["timestamp"] - pd.to_timedelta(timedelta_ns, unit="ns")
                flight_result[f"t_{apt_type}"] = corrected_time
            except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as e:
                warnings.warn(f"Could not correct date for {flight_id} ({apt_type}) due to: {e}")
                flight_result[f"t_{apt_type}"] = pd.NaT

        results.append(flight_result)

    return pd.DataFrame(results)

def fillnadates(flights, cdates):
    ''' fill with original dates if cannot be corrected '''
    df = flights.rename(columns={"takeoff": "t_adep", "landed": "t_ades"})
    df = df[["flight_id", "t_adep", "t_ades"]]
    
    df = df.set_index("flight_id")
    cdates = cdates.set_index("flight_id")
    
    df['t_adep'] = pd.to_datetime(df['t_adep'])
    df['t_ades'] = pd.to_datetime(df['t_ades'])
    cdates['t_adep'] = pd.to_datetime(cdates['t_adep'])
    cdates['t_ades'] = pd.to_datetime(cdates['t_ades'])

    combined_df = cdates.combine_first(df).reset_index()
    
    return combined_df
