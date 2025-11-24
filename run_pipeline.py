"""
Main entry point for the ML pipeline.

This script allows you to run the different stages of the pipeline:
- `profile_data`: Performs a deep analysis of all data sources.
- `interpolate_trajectories`: Interpolates missing values in trajectory files.
- `correct_timestamps`: Corrects takeoff/landing times using trajectory data.
- `prepare_metars`: Pre-processes raw METAR data into a clean dataset.
- `prepare_data`: Preprocesses the raw data and creates introspection files.
- `select_features`: Performs feature selection and saves the selected features.
- `train`: Trains a model on the processed data.
- `predict`: Creates the submission file using a trained model.
- `evaluate`: Evaluates a trained model or generates a submission file.
- `tune`: Performs hyperparameter tuning for a selected model.
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
import trajectory_interpolation
import feature_selection
import os
import config
import metar_utils

def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline.")
    
    # Main pipeline stages
    parser.add_argument("stage", choices=["profile_data", "interpolate_trajectories", "correct_timestamps", "prepare_metars", "prepare_data", "select_features", "train", "predict", "evaluate", "tune"], help="The pipeline stage to run.")
    
    # Model selection for the training and tuning stages
    parser.add_argument("--model", choices=["gbr", "xgb", "rf"], default="gbr", help="The model to train or tune. Only used for the 'train' and 'tune' stages.")
    
    # Optional path to a JSON file with tuned parameters for the 'train' stage
    parser.add_argument("--params_path", type=str, default=None, help="Path to a JSON file containing hyperparameters for the 'train' stage.")

    # Path to a JSON file with a list of selected features for the 'train' stage
    parser.add_argument("--features", type=str, default=None, help="Path to a JSON file with a list of selected features. Used by 'train' stage.")

    # Arguments for the 'select_features' stage
    parser.add_argument("--fs_model", type=str, default='xgb', choices=["gbr", "xgb", "rf"], 
                        help="Model to use for Sequential Feature Selection (SFS) in 'select_features' stage.")
    parser.add_argument("--fs_method", type=str, default='sfs_forward', 
                        choices=['sfs_forward', 'sfs_backward', 'importance', 'none'],
                        help="Feature selection method for the 'select_features' stage.")

    # Argument for the 'evaluate' stage
    parser.add_argument("--run_type", type=str, default='evaluate', choices=['evaluate', 'rank', 'final'],
                        help="For 'evaluate' stage: 'evaluate' for performance metrics, 'rank' or 'final' for submission generation.")

    # New argument for interpolation test mode
    parser.add_argument("--test_interpolation", action="store_true", help="Run trajectory interpolation in test mode.")

    args = parser.parse_args()

    if args.stage == "profile_data":
        data_profiler.profile_data()
    elif args.stage == "interpolate_trajectories":
        print("--- Running Trajectory Interpolation ---")
        trajectory_interpolation.interpolate_trajectories(test_mode=args.test_interpolation)
        print(f"Interpolated trajectories saved to: {config.INTERPOLATED_TRAJECTORIES_DIR}")
    elif args.stage == "correct_timestamps":
        data_preparation.correct_timestamps_for_all()
    elif args.stage == "prepare_metars":
        metar_utils.process_metar_data()
    elif args.stage == "prepare_data":
        data_preparation.prepare_data()
    elif args.stage == "select_features":
        feature_selection.select_features(model_type=args.fs_model, feature_selection_method=args.fs_method)
    elif args.stage == "train":
        if args.model == "gbr":
            print("--- Selected model: Gradient Boosting Regressor ---")
            train_gbr.train(params_path=args.params_path, selected_features_path=args.features)
        elif args.model == "xgb":
            print("--- Selected model: XGBoost Regressor ---")
            train_xgb.train(params_path=args.params_path, selected_features_path=args.features)
        elif args.model == "rf":
            print("--- Selected model: RandomForest Regressor ---")
            train_rf.train(params_path=args.params_path, selected_features_path=args.features)
    elif args.stage == "predict":
        create_submission.main()
    elif args.stage == "evaluate":
        evaluate_model.main(run_type=args.run_type)
    elif args.stage == "tune":
        tune_model.tune(args.model)

if __name__ == "__main__":
    main()
