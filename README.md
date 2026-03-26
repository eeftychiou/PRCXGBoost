# Flight Fuel Consumption Prediction

## Overview

This project provides a complete machine learning pipeline to predict fuel consumption for individual flight segments. It includes modules for data enrichment, feature engineering, model training, hyperparameter tuning, and submission generation. The pipeline is designed to be modular and is controlled via a single entry point (`run_pipeline.py`), making it easy to execute and reproduce.

### Key Features
- **Modular Pipeline**: Each stage (data preparation, training, evaluation) is independent and can be run separately.
- **Rich Data Integration**: Enriches the core dataset with external sources, including airport data, historical weather (METAR) reports, and regional passenger load factor estimates.
- **Advanced Feature Engineering**: Creates a wide range of features from raw trajectory data, including flight phase detection (taxi, takeoff, climb, cruise, etc.), and physics-informed features using aircraft performance models.
- **Flexible Model Training**: Supports XGBoost with multiple hyperparameter optimization modes.
- **Resource-Optimized Training**: Automatically selects the GPU with the lowest memory usage for training tasks.
- **Progress Tracking**: Real-time `tqdm` progress bars in all long-running augmentation and training loops for clear ETAs.
- **Hyperparameter Tuning**: Includes a standalone Optuna-based script for deep Bayesian hyperparameter optimization.

---

## Project Structure

```
PRCXGBoost/
│
├── data/
│   ├── METARs/                    # Raw METAR files downloaded from the web
│   ├── acPerf/                    # Aircraft performance data (acPerfOpenAP.csv)
│   ├── AugmentedDataFromOPENAP/   # Pre-generated OpenAP augmentation data (provided in repo)
│   ├── prc-2025-datasets/         # Original competition datasets (flight lists, fuel, trajectories)
│   ├── filtered_trajectories/     # Output of the trajectory filtering stage
│   ├── interpolated_trajectories/ # Output of the trajectory interpolation stage
│   ├── htmlfile/                  # HTML files downloaded from SkyVector for airport enrichment
│   └── processed/                 # Output directory for all processed data files
│
├── logs/                          # Contains log files for all pipeline stages
│
├── models/                        # Stores trained model artifacts
│
├── AGENTS.md                      # Detailed developer-focused documentation
├── config.py                      # Main configuration file for paths and settings
├── run_pipeline.py                # Main entry point for executing the pipeline
├── download_metars.py             # Script to download raw METAR data
├── impute_apt.py                  # Script to enrich airport data from SkyVector
├── regionalLoadFactor.py          # Script to estimate passenger load factors
├── metar_utils.py                 # Utility functions for processing METAR data
├── correct_date.py                # Utilities for correcting timestamps
├── data_preparation.py            # Main script for the data preparation stage
├── augment_features.py            # Feature engineering from trajectory data
├── filter_trajs.py                # Trajectory filtering logic
├── trajectory_interpolation.py    # Trajectory interpolation logic
├── AugmentationTraining.py        # OpenAP fuel/mass augmentation for the training dataset
├── AugmentationRank.py            # OpenAP fuel/mass augmentation for the rank dataset
├── AugmentationFinal.py           # OpenAP fuel/mass augmentation for the final dataset
├── XGBoostTraining_Testing.py     # Feature selection, preprocessing, and test-training script
├── XGBoostTraining_Final.py       # Full training script for final submission
├── train_baselines.py             # Trains Ridge, RF, LightGBM, XGBoost baselines for comparison
├── ablation_contributions.py      # Leave-one-out ablation for METAR, load factor, dynamic mass, timestamp correction
├── hyperparameter_tuning_optuna.py# Standalone Optuna hyperparameter search
├── evaluate_model.py              # Model evaluation and submission generation
├── generate_paper_plots.py        # Generates publication-ready figures and tables
├── compare_final_parquets.py      # Compares latest submission against a baseline
├── compare_all_parquets.py        # Plots distributions across multiple models/baselines
└── README.md                      # This file
```

---

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd PRCXGBoost
    ```

2.  **Create a Virtual Environment**:
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    Install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The trajectory filtering stage relies on the `traffic` package, which is incompatible with newer versions of `pandas`. If you encounter a `DatetimeTZBlock` import error, ensure your pandas version is strictly below 2.2.0 (e.g., `pip install "pandas<2.2.0"`).*

---

## How to Reproduce Results: A Unified Step-by-Step Guide

The following stages prepare all the necessary data for model training. The entire workflow is orchestrated through `run_pipeline.py`.

### Step 1: Initial Data Acquisition
These steps only need to be run once to populate the repository with the required data.

1.  **Download the dataset from the PRC website**:
    ```bash
    # Note: Requires mc (MinIO Client) configured with the OpenSky credentials
    mc cp opensky/prc-2025-datasets/ data/prc-2025-datasets/
    ```
    Alternatively, the datasets are available on [Zenodo](https://zenodo.org/records/19184662).

2.  **Generate Aircraft Performance File**:
    Creates `data/acPerf/acPerfOpenAP.csv`, which contains aircraft performance specifications and behavioral signatures. This file is already provided in the repository  only run this if you need to regenerate it.

    ```bash
    python run_pipeline.py setup_ac_perf
    ```

    *This runs `extract_aircraft_types.py`, `enrich_aircraft_data.py`, and `create_behavioral_features.py` sequentially.*

### Step 2: Data Enrichment & Preprocessing

1.  **Enrich Airport Data**:
    Scrapes detailed airport information from SkyVector (runway lengths, headings, elevation) and enriches the `apt.parquet` file.
    ```bash
    python run_pipeline.py setup_apt
    ```

2.  **Filter Trajectories**:
    Removes erroneous data points from the raw trajectory files and saves the cleaned output to `data/filtered_trajectories/`.
    ```bash
    python run_pipeline.py filter_trajs
    ```

3.  **Interpolate Trajectories**:
    Fills missing values in the filtered trajectory files and saves the result to `data/interpolated_trajectories/`.
    ```bash
    python run_pipeline.py interpolate_trajectories
    ```

    > **Tip  Running Specific Splits**: You can run either of the above stages for a single dataset split (`train`, `rank`, or `final`) using the `--split` argument. This is useful when you only need to reprocess one dataset:
    > ```bash
    > python run_pipeline.py filter_trajs --split final
    > python run_pipeline.py interpolate_trajectories --split rank
    > python run_pipeline.py correct_timestamps --split final
    > ```

4.  **Calculate Regional Load Factors**:
    Estimates passenger load factors based on IATA regions and flight routes to enrich the dataset with payload proxies.
    ```bash
    python run_pipeline.py regional_load_factor
    ```

5.  **Correct Timestamps**:
    Adjusts takeoff and landing times in the flight lists using the interpolated trajectory data.
    ```bash
    python run_pipeline.py correct_timestamps
    ```

6.  **Prepare Weather Data**:
    Processes raw METAR files into a single flight-keyed dataset, decoding weather phenomenon codes and mapping them to the nearest airport. This produces `processed/processed_metars.parquet` and only needs to be run once.
    ```bash
    python run_pipeline.py prepare_metars
    ```

### Step 3: Feature Engineering & Augmentation

1.  **Main Data Preparation**:
    Merges all data sources (corrected flight lists, fuel data, aircraft performance, airport data, weather) and creates the core feature set files (`featured_data_{stage}.parquet`).
    ```bash
    python run_pipeline.py prepare_data
    ```

2.  **Data Augmentation (OpenAP Physics Features)**:

    > **Note: Pre-generated augmentation data is already provided in `data/AugmentedDataFromOPENAP/` and is committed to the GitHub repository. This step is extremely time-consuming (several hours per dataset split) and should be skipped unless you need to regenerate the data from scratch.**

    The augmentation computes physics-based fuel predictions using OpenAP FuelFlow models with dynamic mass tracking. Use the `--split` flag to run for a specific dataset, or omit it to run all three sequentially:

    ```bash
    # Run all three splits (train, rank, final) in sequence
    python run_pipeline.py augment

    # Or run a specific split to save time
    python run_pipeline.py augment --split train
    python run_pipeline.py augment --split rank
    python run_pipeline.py augment --split final
    ```

    Each run saves its output to `data/AugmentedDataFromOPENAP/`. **If the pre-generated files already exist, the stage is skipped automatically. Use `--force` to regenerate.**

    ```bash
    python run_pipeline.py augment --force
    ```

### Step 4: Machine Learning Training

The training process follows two sequential steps: a testing/feature-selection run, then a final full-data run. Both scripts support a `--mode` flag to control hyperparameter optimization.

**Available `--mode` options:**
- `legacy` *(default)*: Uses the original high-performance parameters (1455 estimators).
- `grid`: Uses a refined parameter combination (900 estimators).
- `optuna`: Runs a deep Bayesian search using the Optuna framework.

1.  **Feature Selection & Test Training**:
    Generates synthetic widebody samples, performs Sequential Feature Selection (SFS), trains on a validation split, and saves the preprocessors and selected features. Use `--force-sfs` to redo feature selection or `--force-synthetic` to regenerate synthetic samples even if cached versions exist.
    ```bash
    python run_pipeline.py train_test

    # With options:
    python run_pipeline.py train_test --mode optuna --force-sfs --force-synthetic
    ```

2.  **Final Model Training**:
    Retrains the model on 100% of the augmented data (original + synthetic) using the features and preprocessors established in the previous step. Produces the final submission parquets and generates publication-ready plots and tables automatically.
    ```bash
    python run_pipeline.py train_final

    # With options:
    python run_pipeline.py train_final --mode grid
    ```

3.  **Standalone Hyperparameter Tuning (Optuna)**:
    For isolated, high-intensity Bayesian optimization on a dedicated GPU server without running the full training pipeline.
    ```bash
    python hyperparameter_tuning_optuna.py
    ```

### Step 4.5 (Optional): Baseline Model Comparison

To address reproducibility and provide a fair comparison of model architectures, baseline models can be trained on the exact same data, preprocessing pipeline, and train/val split as the XGBoost model. This trains Ridge Regression, Random Forest, LightGBM, and XGBoost (with the legacy reference parameters) on the SFS-selected feature set, then writes per-model metrics (RMSE, MAE, MAPE, R²) to `models/baselines/baseline_comparison.csv`.

```bash
python run_pipeline.py train_baselines
```
### Step 4.6 (Optional): Contribution Ablation Study

Addresses the reviewer requirement for isolated validation of each claimed contribution. Trains one XGBoost model per condition (same hyperparameters, same 80/20 split, same preprocessing) and reports the ΔMAE when each contribution is removed:

| Condition | Contribution tested |
|---|---|
| Full model | Reference (all features) |
| No METAR features | C1  meteorological weather data |
| No load-factor features | C2  OD-level load factor & payload estimation |
| Static MTOW mass | C3  dynamic per-segment mass tracking |
| Raw (uncorrected) timestamps | C4  takeoff/landing timestamp correction |

Results are saved to `processed/ablation_contributions_results.csv`.

```bash
python run_pipeline.py ablate_contributions

# Specify a GPU:
python ablation_contributions.py --gpu 0
```

> **Prerequisite:** `prepare_metars` and `prepare_data` must have been run so that METAR columns (`dep_*` / `arr_*`) are present in `processed/featured_data_train.parquet`. The corrected flightlist (`processed/corrected_flightlist_train.parquet`) must also exist for the C4 timestamp ablation.

### Step 5: Evaluation & Submissions

1.  **Performance Evaluation (Metrics)**:
    Calculates RMSE, MAE, and R² scores by evaluating on a fresh validation split of the training data and saves a detailed `evaluation_details.csv`.
    ```bash
    python run_pipeline.py evaluate --run_type evaluate
    ```

2.  **Rank Stage Submission**:
    Generates the submission file for the `rank` (historical test) dataset.
    ```bash
    python run_pipeline.py evaluate --run_type rank
    ```

3.  **Final Stage Submission**:
    Generates the submission file for the `final` dataset.
    ```bash
    python run_pipeline.py evaluate --run_type final
    ```

  ```

---

## Tips for Remote Execution (`screen`)

For long-running training or Optuna searches on a GPU server, use `screen` to prevent the session from terminating if your SSH connection drops.

1.  **Create a new session**:
    ```bash
    screen -S prc_training
    ```
2.  **Run your script**:
    ```bash
    python run_pipeline.py train_final
    ```
3.  **Detach from the session**: Press `Ctrl + A` followed by `D`.
4.  **Reattach later**:
    ```bash
    screen -r prc_training
    ```
5.  **List active screens**:
    ```bash
    screen -ls
    ```
