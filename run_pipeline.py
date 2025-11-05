"""
Main entry point for the ML pipeline.

This script allows you to run the different stages of the pipeline:
- `profile_data`: Performs a deep analysis of all data sources.
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

def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline.")
    
    # Main pipeline stages
    parser.add_argument("stage", choices=["profile_data", "prepare_data", "train", "predict", "evaluate", "tune"], help="The pipeline stage to run.")
    
    # Model selection for the training and tuning stages
    parser.add_argument("--model", choices=["gbr", "xgb", "rf"], default="gbr", help="The model to train or tune. Only used for the 'train' and 'tune' stages.")
    
    # Optional path to a JSON file with tuned parameters for the 'train' stage
    parser.add_argument("--params_path", type=str, default=None, help="Path to a JSON file containing hyperparameters for the 'train' stage.")

    args = parser.parse_args()

    if args.stage == "profile_data":
        data_profiler.profile_data()
    elif args.stage == "prepare_data":
        data_preparation.prepare_data()
    elif args.stage == "train":
        if args.model == "gbr":
            print("--- Selected model: Gradient Boosting Regressor ---")
            train_gbr.train(params_path=args.params_path)
        elif args.model == "xgb":
            print("--- Selected model: XGBoost Regressor ---")
            train_xgb.train(params_path=args.params_path)
        elif args.model == "rf":
            print("--- Selected model: RandomForest Regressor ---")
            train_rf.train(params_path=args.params_path)
    elif args.stage == "predict":
        # The main logic is now encapsulated in the main() function of create_submission.py
        create_submission.main()
    elif args.stage == "evaluate":
        evaluate_model.main()
    elif args.stage == "tune":
        tune_model.tune(args.model)

if __name__ == "__main__":
    main()
