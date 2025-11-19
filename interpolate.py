import pandas as pd
import utils
import os
import matplotlib.pyplot as plt
from traffic.core import Traffic
import numpy as np
import argparse
import readers
import csaps
import scipy
from scipy.ndimage import gaussian_filter1d
import glob
from tqdm import tqdm
import config

MAX_HOLE_SIZE = 20 # seconds


DICO_HOLE_SIZE = {}

def spline(t,v,smooth,derivative=False):
    '''
    Smoothing spline
    '''
    isok = np.logical_not(np.isnan(v))
    if isok.sum()>2:
        cspline = csaps.csaps(xdata=t[isok],ydata=v[isok],smooth=smooth)#.spline
        if derivative:
            return cspline(t),cspline(t,nu=1)
        else:
            return cspline(t),None
    else:
        dv = np.empty_like(v)
        dv[:]=np.nan
        return v,dv


def compute_holes(t,inans):
    '''
    pd.DataFrame.interpolate does not use index values for the 'limit' parameter :-(
    so I implement my own
    '''
    tnan = t.copy()
    tnan[inans] = np.nan
    tf = pd.DataFrame({"tf":tnan}, dtype=np.float64).ffill().values
    tb = pd.DataFrame({"tb":tnan}, dtype=np.float64).bfill().values
    return tb - tf

def interpolate(df,smooth):
    '''
    Smooth the different measurements using splines
    does not interpolate between measurements separated by 20 seconds
    '''
    t = ((df.timestamp - df.timestamp.iloc[0]) / pd.to_timedelta(1, unit="s")).values.astype(np.float64)
    df["t"] = t
    masknan = np.isnan(df["track"].values)
    track = df[["track"]].ffill().bfill().values[:,0]
    unwraped = np.unwrap(track,period=360)
    unwraped[masknan]=np.nan
    df["track_unwrapped"] = unwraped

    assert (t[1:]>t[:-1]).all()
    ddt = {}
    lnanvar = ['latitude', 'longitude', 'altitude', 'groundspeed', 'vertical_rate', 'track_unwrapped', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity','gsx', 'gsy', 'tasx', 'tasy', 'tas', 'wind']
    df = df.drop(columns="track")
    dvar = (
        (smooth,('latitude', 'longitude')),
        (smooth,('altitude',)),
        (smooth*0.1,('groundspeed','gsx', 'gsy', 'tasx', 'tasy', 'tas', 'wind','u_component_of_wind', 'v_component_of_wind')),
        (smooth,('track_unwrapped',)),
        (smooth*0.1,('vertical_rate',)),
        (smooth,('temperature', 'specific_humidity')),
    )
    for v in lnanvar:
        if v in df.columns:
            ddt[v] = compute_holes(df["t"],np.isnan(df[v].values))
    dico  = {}
    for smooth,lvar in dvar:
        for v in lvar:
            if v in df.columns:
                deriv = v == "altitude"
                st,dst = spline(t,df[v].values,smooth,derivative=deriv)
                dico[v]=st
                if deriv:
                    dico["d"+v]=dst * 60
    res = df.assign(**dico)
    for v in lnanvar:
        if v in res.columns:
            res[v] = res[[v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
            res[v] = res[[v]].mask(np.isnan(ddt[v]))
            if ("d"+v) in dico:
                res["d"+v] = res[["d"+v]].mask(ddt[v] > DICO_HOLE_SIZE.get(v,MAX_HOLE_SIZE))
                res["d"+v] = res[["d"+v]].mask(np.isnan(ddt[v]))
    return res.drop(columns="t")


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
        if not os.path.exists(input_dir):
            continue
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
            
            # Rename 'track_unwrapped' back to 'track'
            if 'track_unwrapped' in df_interpolated.columns:
                df_interpolated = df_interpolated.rename(columns={'track_unwrapped': 'track'})

            df_interpolated.to_parquet(output_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate all trajectory files.'
    )
    parser.add_argument("--smooth", type=float, default=0.1, help="Smoothing factor for interpolation.")
    args = parser.parse_args()

    process_all_trajectories(args.smooth)
