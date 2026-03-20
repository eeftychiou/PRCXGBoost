"""
Main entry point for the ML pipeline.

This script allows you to run the different stages of the pipeline:
- `profile_data`: Performs a deep analysis of all data sources.
- `interpolate_trajectories`: Interpolates missing values in trajectory files.
- `correct_timestamps`: Corrects takeoff/landing times using trajectory data.
- `prepare_metars`: Pre-processes raw METAR data into a clean dataset.
- `prepare_data`: Preprocesses the raw data and creates introspection files.
- `train`: Trains a model on the processed data.
- `evaluate`: Evaluates a trained model or generates a submission file.
- `setup_ac_perf`: Extracts aircraft types, enriches with OpenAP, and creates behavioral features.
- `filter_trajs`: Filters raw trajectories using the classic strategy.
- `regional_load_factor`: Calculates and saves regional passenger load factors.
"""
import argparse
import subprocess
import json
import re
import os
import config
import extract_aircraft_types
import enrich_aircraft_data
import create_behavioral_features
import trajectory_interpolation
import AugmentationTraining
import AugmentationRank
import AugmentationFinal
import XGBoostTraining_Testing
import XGBoostTraining_Final
import metar_utils
import data_preparation
import data_profiler
import evaluate_model
import filter_trajs
import regionalLoadFactor

def get_best_gpu():
    """Returns the index of the GPU with the lowest memory usage."""
    try:
        # Run nvidia-smi to get memory usage
        # Format: index, memory.used, memory.total
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        
        best_gpu = 0
        min_usage = float('inf')
        
        for line in result.strip().split('\n'):
            parts = line.split(',')
            if len(parts) != 3: continue
            try:
                idx, used, total = map(int, parts)
                usage_ratio = used / total
                if usage_ratio < min_usage:
                    min_usage = usage_ratio
                    best_gpu = idx
            except:
                continue
        
        print(f"--- GPU Selection: Using GPU {best_gpu} (Usage: {min_usage*100:.1f}%) ---")
        return best_gpu
    except Exception:
        # If nvidia-smi fails (e.g., no NVIDIA GPU), return 0 as default
        return 0

def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline.")
    
    # Main pipeline stages
    parser.add_argument("stage", choices=[
        "profile_data", "setup_ac_perf", "filter_trajs", "regional_load_factor",
        "interpolate_trajectories", "correct_timestamps", "prepare_metars", 
        "setup_apt", "prepare_data", "augment", "train_test", "train_final", 
        "train", "evaluate"
    ], help="The pipeline stage to run.")
    
    # Model selection for the training and tuning stages
    parser.add_argument("--model", choices=["xgb"], default="xgb", help="The model to train. Currently only XGBoost is supported in the unified pipeline.")
    
    # Argument for the 'evaluate' stage
    parser.add_argument("--run_type", type=str, default='evaluate', choices=['evaluate', 'rank', 'final'],
                        help="For 'evaluate' stage: 'evaluate' for performance metrics, 'rank' or 'final' for submission generation.")

    # New argument for split selection
    parser.add_argument("--split", choices=["train", "rank", "final"], help="Run the stage only for a specific dataset split.")

    # Caching and Force flags
    parser.add_argument("--force", action="store_true", help="Force rerun of stages, ignoring existing caches (e.g., for augmentation).")
    parser.add_argument("--force-sfs", action="store_true", help="Force rerun of Sequential Feature Selection.")
    parser.add_argument("--force-synthetic", action="store_true", help="Force regeneration of synthetic widebody data.")
    
    # Hyperparameter mode for training
    parser.add_argument("--mode", choices=["legacy", "grid", "optuna"], default="legacy", 
                        help="Optimization mode: 'legacy' (default), 'grid', or 'optuna'.")

    args = parser.parse_args()

    if args.stage == "profile_data":
        data_profiler.profile_data()
    elif args.stage == "setup_ac_perf":
        print("--- Running Aircraft Performance Setup ---")
        extract_aircraft_types.find_unique_aircraft_types()
        enrich_aircraft_data.enrich_aircraft_data()
        create_behavioral_features.create_behavioral_features()
    elif args.stage == "filter_trajs":
        print(f"--- Running Trajectory Filtering {'(' + args.split + ')' if args.split else ''} ---")
        filter_trajs.run(split=args.split)
    elif args.stage == "regional_load_factor":
        print("--- Running Regional Load Factor Calculation ---")
        regionalLoadFactor.run()
    elif args.stage == "interpolate_trajectories":
        print(f"--- Running Trajectory Interpolation {'(' + args.split + ')' if args.split else ''} ---")
        trajectory_interpolation.interpolate_trajectories(test_mode=False, split=args.split)
        print(f"Interpolated trajectories saved to: {config.INTERPOLATED_TRAJECTORIES_DIR}")
    elif args.stage == "correct_timestamps":
        data_preparation.correct_timestamps_for_all(split=args.split)
    elif args.stage == "prepare_metars":
        metar_utils.process_metar_data()
    elif args.stage == "setup_apt":
        import impute_apt
        impute_apt.run()
    elif args.stage == "prepare_data":
        data_preparation.prepare_data(split=args.split)
    elif args.stage == "augment":
        split = args.split
        if split in (None, 'train'):
            print("--- Running Data Augmentation (OpenAP) - Training ---")
            AugmentationTraining.run(dataset_type='train', force=args.force)
        if split in (None, 'rank'):
            print("--- Running Data Augmentation (OpenAP) - Rank ---")
            AugmentationRank.run(force=args.force)
        if split in (None, 'final'):
            print("--- Running Data Augmentation (OpenAP) - Final ---")
            AugmentationFinal.run(force=args.force)
    elif args.stage == "train_test":
        print(f"--- Running XGBoost Training (Testing/Feature Selection) [Mode: {args.mode}] ---")
        best_gpu = get_best_gpu()
        XGBoostTraining_Testing.run(
            gpu_id=best_gpu, 
            force_sfs=args.force_sfs or args.force,
            force_synthetic=args.force_synthetic or args.force,
            opt_mode=args.mode
        )
    elif args.stage == "train_final":
        print(f"--- Running Final XGBoost Training [Mode: {args.mode}] ---")
        best_gpu = get_best_gpu()
        XGBoostTraining_Final.run(
            gpu_id=best_gpu, 
            force_sfs=args.force_sfs or args.force, # In case SFS is needed in final
            force_synthetic=args.force_synthetic or args.force,
            opt_mode=args.mode
        )
    elif args.stage == "train":
        if args.model == "xgb":
            print(f"--- Running XGBoost Training (Final) [Mode: {args.mode}] ---")
            best_gpu = get_best_gpu()
            XGBoostTraining_Final.run(
                gpu_id=best_gpu,
                force_sfs=args.force_sfs or args.force,
                force_synthetic=args.force_synthetic or args.force,
                opt_mode=args.mode
            )
        else:
            print(f"ERROR: Model {args.model} is not yet implemented or supported in the unified pipeline.")
    elif args.stage == "evaluate":
        evaluate_model.main(run_type=args.run_type)

if __name__ == "__main__":
    main()
