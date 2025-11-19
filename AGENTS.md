# Project Overview

This project aims to build a Machine Learning model to predict fuel consumption for flight segments.

## Project Structure

- **`run_pipeline.py`**: The main entry point for running the ML pipeline.
- **`config.py`**: Contains configuration for the project, including data paths and settings.
- **`data/`**: Contains the datasets.
    - **`acPerf/`**: Raw aircraft performance data.
    - **`prc-2025-datasets/`**: Base datasets for training and ranking.
    - **`filtered_trajectories/`**: New directory for filtered trajectory data.
    - **`interpolated_trajectories/`**: New directory for interpolated trajectory data.
    - **`processed/`**: Processed data ready for model training.
- **`introspection/`**: Directory for introspection files created during data preparation.
- **`models/`**: Directory to save trained models.
- **`data_preparation.py`**: Script for data preprocessing.
- **`filter_trajs.py`**: Script for filtering erroneous measurements from trajectory files.
- **`interpolate.py`**: Module for interpolating missing values in trajectory files.
- **`correct_date.py`**: Module for correcting takeoff and landing times using trajectory data.
- **`impute_apt.py`**: Enriches airport data by scraping information from SkyVector and calculating true runway headings.
- **`feature_engineering.py`**: Script for creating new features from raw and trajectory data.
- **`train_*.py`**: Scripts for training different models (e.g., `train_gbr.py`, `train_xgb.py`).
- **`create_submission.py`**: Script to generate the final submission file.
- **`evaluate_model.py`**: Script for model evaluation.
- **`tune_model.py`**: Script for hyperparameter tuning.
- **`data_profiler.py`**: Script for data analysis and profiling.

## Pipeline Stages

The ML pipeline is managed by `run_pipeline.py` and consists of the following stages:

1.  **`profile_data`**: Analyzes the input data to generate a profile report.
2.  **`filter_trajectories`**: Filters raw trajectory files to remove erroneous data points and saves the cleaned files to `data/filtered_trajectories`.
3.  **`interpolate_trajectories`**: Processes the *filtered* trajectory files, interpolates missing values (e.g., CAS, TAS, altitude, groundspeed, etc.) per flight, and saves the processed files to `data/interpolated_trajectories`.
4.  **`prepare_data`**: Preprocesses the raw data for the `train`, `rank`, and `final` datasets in a single run. It corrects takeoff and landing times using a memory-efficient, sequential processing of trajectory data, handles missing values, and creates features. The processed data is saved in the `data/processed` directory.
5.  **`train`**: Trains a machine learning model. The following models are supported:
    - Gradient Boosting Regressor (`gbr`)
    - XGBoost Regressor (`xgb`)
    - Random Forest Regressor (`rf`)
    The trained model is saved in the `models/` directory.
6.  **`predict`**: Uses a trained model to make predictions and create a submission file. This stage also uses the *interpolated* trajectory data for feature engineering.
7.  **`evaluate`**: Evaluates the performance of a trained model.
8.  **`tune`**: Performs hyperparameter tuning for a selected model.

## How to Run

The pipeline can be executed from the command line using `run_pipeline.py`.

**Example:**

To filter the raw trajectory files:
```bash
python run_pipeline.py filter_trajectories
```

To interpolate missing values in the filtered trajectory files:
```bash
python run_pipeline.py interpolate_trajectories
```

To prepare the data (after interpolation):
```bash
python run_pipeline.py prepare_data
```

To train an XGBoost Regressor model:
```bash
python run_pipeline.py train --model xgb
```

## Configuration

The `config.py` file contains important configuration variables:

- **`TEST_RUN`**: Set to `True` to run the pipeline on a small fraction of the data for testing purposes. When enabled, the `prepare_data` stage samples the flight list at the beginning to significantly speed up runtime for faster evaluation.
- **`TEST_RUN_FRACTION`**: The fraction of data to use when `TEST_RUN` is `True`.
