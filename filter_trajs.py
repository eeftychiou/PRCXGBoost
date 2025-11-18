import pandas as pd
import numpy as np
import utils
import argparse
import os
import glob
from tqdm import tqdm
import config
from filterclassic import FilterCstPosition, FilterCstSpeed, MyFilterDerivative, FilterCstLatLon,FilterIsolated

# unwrap wrong on 248803487 of 2022-01-03
from traffic.core import Traffic
import matplotlib.pyplot as plt


def nointerpolate(x):
    ''' identity function '''
    return x

def read_trajectories(f, strategy):
    ''' read a trajectory file named @f, and filters points using a @strategy'''
    df = pd.read_parquet(f)
    df.columns = [col.lower() for col in df.columns]
    df = df.drop_duplicates(["flight_id","timestamp"]).sort_values(["flight_id","timestamp"]).reset_index(drop=True)#.head(10_000)
    if strategy == "classic":
        filter = FilterCstLatLon()|FilterCstPosition()|FilterCstSpeed()|MyFilterDerivative()|FilterIsolated()
    else:
        raise Exception(f"strategy '{strategy}' not implemented")
    dftrafficin = Traffic(df).filter(filter=filter,strategy=nointerpolate).eval(max_workers=1).data
    dico_tomask = {
        # "track":["track_unwrapped"],
        "latitude":["u_component_of_wind","v_component_of_wind","temperature"],
        "altitude":["u_component_of_wind","v_component_of_wind","temperature"],
    }
    for k,lvar in dico_tomask.items():
        for v in lvar:
            if v in dftrafficin.columns:
                dftrafficin[v] = dftrafficin[[v]].mask(dftrafficin[k].isna())
    return dftrafficin

def filter_all_trajectories(strategy):
    input_base_dir = os.path.join(config.DATA_DIR, 'prc-2025-datasets')
    output_base_dir = os.path.join(config.DATA_DIR, 'filtered_trajectories')
    sub_dirs = ['flights_train', 'flights_rank', 'flights_final']

    for sub_dir in sub_dirs:
        input_dir = os.path.join(input_base_dir, sub_dir)
        output_dir = os.path.join(output_base_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)

        trajectory_files = glob.glob(os.path.join(input_dir, '*.parquet'))

        for file_path in tqdm(trajectory_files, desc=f"Filtering {sub_dir}", unit="file"):
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(output_dir, file_name)

            if os.path.exists(output_file_path):
                continue

            df = read_trajectories(file_path, strategy)
            df.to_parquet(output_file_path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description='filter out measurements that are likely erroneous',
    )
    parser.add_argument("-strategy", default="classic")
    args = parser.parse_args()
    filter_all_trajectories(args.strategy)



if __name__ == '__main__':
    main()