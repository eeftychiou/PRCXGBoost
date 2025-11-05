# Project Overview

This project aims to build a Machine Learning model to predict fuel consumption for flight segments.

## Project Structure

- **`run_pipeline.py`**: The main entry point for running the ML pipeline.
- **`config.py`**: Contains configuration for the project, including data paths and settings.
- **`data/`**: Contains the datasets.
    - **`acPerf/`**: Raw aircraft performance data.
    - **`prc-2025-datasets/`**: Base datasets for training and ranking.
    - **`processed/`**: Processed data ready for model training.
- **`introspection/`**: Directory for introspection files created during data preparation.
- **`models/`**: Directory to save trained models.
- **`data_preparation.py`**: Script for data preprocessing.
- **`train_*.py`**: Scripts for training different models (e.g., `train_gbr.py`, `train_xgb.py`).
- **`create_submission.py`**: Script to generate the final submission file.
- **`evaluate_model.py`**: Script for model evaluation.
- **`tune_model.py`**: Script for hyperparameter tuning.
- **`data_profiler.py`**: Script for data analysis and profiling.

## Pipeline Stages

The ML pipeline is managed by `run_pipeline.py` and consists of the following stages:

1.  **`profile_data`**: Analyzes the input data to generate a profile report.
2.  **`prepare_data`**: Preprocesses the raw data, handles missing values, and creates features. The processed data is saved in the `data/processed` directory.
3.  **`train`**: Trains a machine learning model. The following models are supported:
    - Gradient Boosting Regressor (`gbr`)
    - XGBoost Regressor (`xgb`)
    - Random Forest Regressor (`rf`)
    The trained model is saved in the `models/` directory.
4.  **`predict`**: Uses a trained model to make predictions and create a submission file.
5.  **`evaluate`**: Evaluates the performance of a trained model.
6.  **`tune`**: Performs hyperparameter tuning for a selected model.

## How to Run

The pipeline can be executed from the command line using `run_pipeline.py`.

**Example:**

To prepare the data:
```bash
python run_pipeline.py prepare_data
```

To train a Gradient Boosting Regressor model:
```bash
python run_pipeline.py train --model gbr
```

## Configuration

The `config.py` file contains important configuration variables:

- **`TEST_RUN`**: Set to `True` to run the pipeline on a small fraction of the data for testing purposes.
- **`TEST_RUN_FRACTION`**: The fraction of data to use when `TEST_RUN` is `True`.
