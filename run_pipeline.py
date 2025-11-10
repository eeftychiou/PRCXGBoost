"""
Main entry point for the ML pipeline.

This script allows you to run the different stages of the pipeline:
- `profile_data`: Performs a deep analysis of all data sources.
- `interpolate_trajectories`: Interpolates missing values in trajectory files.
- `prepare_data`: Preprocesses the raw data and creates introspection files.
- `train`: Trains a model on the processed data.
- `predict`: Creates the submission file using a trained model.
- `evaluate`: Evaluates a trained model.
- `tune`: Performs hyperparameter tuning for a selected model.
- `inspect_pinn`: A debugging tool to inspect the data fed to the PINN.
"""
import argparse
import data_preparation
import train_gbr
import train_xgb
import train_rf
import create_submission
import data_profiler
import evaluate_model
import tune_model
import trajectory_interpolation # Import the new module
import os # For path manipulation
import config # Import the config module

def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline.")
    
    # Main pipeline stages
    parser.add_argument("stage", choices=["profile_data", "interpolate_trajectories", "prepare_data", "train", "predict", "evaluate", "tune"], help="The pipeline stage to run.")
    
    # Model selection for the training and tuning stages
    parser.add_argument("--model", choices=["gbr", "xgb", "rf"], default="gbr", help="The model to train or tune. Only used for the 'train' and 'tune' stages.")
    
    # Optional path to a JSON file with tuned parameters for the 'train' stage
    parser.add_argument("--params_path", type=str, default=None, help="Path to a JSON file containing hyperparameters for the 'train' stage.")

    # New argument for feature selection method
    parser.add_argument("--feature_selection", type=str, default='sfs_forward', 
                        choices=['sfs_forward', 'sfs_backward', 'none'],
                        help="Feature selection method for the 'train' stage (used by xgb model).")

    # New argument for interpolation test mode
    parser.add_argument("--test_interpolation", action="store_true", help="Run trajectory interpolation in test mode.")

    args = parser.parse_args()

    if args.stage == "profile_data":
        data_profiler.profile_data()
    elif args.stage == "interpolate_trajectories":
        print("--- Running Trajectory Interpolation ---")
        trajectory_interpolation.interpolate_trajectories(test_mode=args.test_interpolation)
        print(f"Interpolated trajectories saved to: {config.INTERPOLATED_TRAJECTORIES_DIR}")
    elif args.stage == "prepare_data":
        data_preparation.prepare_data()
    elif args.stage == "train":
        if args.model == "gbr":
            print("--- Selected model: Gradient Boosting Regressor ---")
            train_gbr.train(params_path=args.params_path)
        elif args.model == "xgb":
            print("--- Selected model: XGBoost Regressor ---")
            train_xgb.train(params_path=args.params_path, feature_selection_method=args.feature_selection)
        elif args.model == "rf":
            print("--- Selected model: RandomForest Regressor ---")
            train_rf.train(params_path=args.params_path)
    elif args.stage == "predict":
        create_submission.main()
    elif args.stage == "evaluate":
        evaluate_model.main()
    elif args.stage == "tune":
        tune_model.tune(args.model)

if __name__ == "__main__":
    main()
