# PRCXGBoost: Flight Fuel Consumption Prediction

## Overview

This project provides a complete machine learning pipeline to predict fuel consumption for individual flight segments. It includes modules for data enrichment, feature engineering, model training, and prediction. The pipeline is designed to be modular, allowing for independent execution of its various stages, and includes robust error handling and logging.

Key features include:
- **Modular Pipeline**: Stages for data processing, training, and prediction can be run independently.
- **Rich Data Integration**: Enriches the core dataset with external sources, including:
    - Airport runway and elevation data from SkyVector.
    - Historical weather (METAR) data from Iowa Environmental Mesonet.
    - Regional passenger load factor estimates based on IATA statistics.
- **Advanced Feature Engineering**: Creates a wide range of features from raw trajectory data, including flight phase detection (taxi, takeoff, climb, cruise, etc.).
- **Flexible Model Training**: Supports multiple models, including XGBoost, Gradient Boosting, and Random Forest.
- **Robust Data Handling**: Includes mechanisms for handling web failures (retries) and ensuring idempotent data processing.

---

## Project Structure

```
PRCXGBoost/
│
├── data/
│   ├── METARs/               # Raw METAR files downloaded from the web
│   ├── acPerf/               # Aircraft performance data
│   ├── prc-2025-datasets/    # Original competition datasets (flight lists, fuel, trajectories)
│   ├── interpolated_trajectories/ # Output of the trajectory interpolation stage
│   └── processed/            # Output directory for all processed data files
│
├── logs/                     # Contains log files for all pipeline stages
│
├── models/                   # Stores trained model artifacts
│
├── AGENTS.md                 # Detailed developer-focused documentation
├── config.py                 # Main configuration file for paths and settings
├── download_metars.py        # Script to download raw METAR data
├── impute_apt.py             # Script to enrich airport data from SkyVector
├── regionalLoadFactor.py     # Script to estimate passenger load factors
├── run_pipeline.py           # Main entry point for executing the pipeline
├── metar_utils.py            # Utility functions for processing METAR data
├── correct_date.py           # Utilities for correcting timestamps
├── data_preparation.py       # Main script for the data preparation stage
├── augment_features.py       # Feature engineering from trajectory data
├── train_xgb.py              # Example training script for XGBoost
└── README.md                 # This file
```

---

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd PRCXGBoost
    ```

2.  **Install Dependencies**: It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    Install the required packages. A `requirements.txt` file is not yet present, but the core dependencies are:
    ```bash
    pip install pandas numpy scikit-learn xgboost requests beautifulsoup4 tqdm pygeomag openap fastparquet pyarrow tqdm matplotlib
    ```

3.  **Download WMM Data**: The `impute_apt.py` script requires the World Magnetic Model (WMM) coefficients. Create a `wmm` directory and download the `WMM.COF` file. *Note: The script is currently configured for 2025 data.*

---

## Workflow and Pipeline Execution

The pipeline is designed to be run in a specific order to ensure data dependencies are met.

### Step 1: Initial Data Enrichment (Run Once)

These scripts enrich the raw datasets and should be run before the main pipeline.

1.  **Download Historical Weather Data**:
    This script downloads all necessary METAR reports based on the date ranges in the flight data. It is robust against network failures.
    ```bash
    python download_metars.py
    ```

2.  **Impute Airport Data**:
    This script scrapes SkyVector to enrich `apt.parquet` with runway and elevation data. It is idempotent and can be re-run safely.
    ```bash
    python impute_apt.py
    ```

3.  **Generate Load Factor Estimates**:
    This script creates a CSV of estimated passenger load factors for each airport pair based on IATA statistics.
    ```bash
    python regionalLoadFactor.py
    ```

### Step 2: Main Pipeline Execution

The main pipeline is controlled by `run_pipeline.py`.

1.  **Correct Timestamps**:
    This stage corrects the takeoff and landing times in the raw flight lists using trajectory data. This is a crucial first step.
    ```bash
    python run_pipeline.py correct_timestamps
    ```

2.  **Prepare METAR Data**:
    This stage uses the corrected timestamps from the previous step to process the raw METAR data, creating a clean `processed_metars.parquet` file.
    ```bash
    python run_pipeline.py prepare_metars
    ```

3.  **Prepare Final Dataset**:
    This stage merges all data sources (corrected flight lists, fuel data, airport data, aircraft performance, and processed METARs) and performs the final feature engineering.
    ```bash
    python run_pipeline.py prepare_data
    ```

4.  **Data Augmentation and Imputation**:
    Generate physics-informed features using OpenAP aircraft performance models with dynamic mass tracking and data imputation for ground speed, vertical rate and altitude.
    Using the provided parquets (Adjust the paths for the data)
    ```bash
    python AugmentationTraining.py
    python AugmentationRank.py
    python AugmentationFinal.py
    ```

5.  **Train, Evaluation and Test the XGBoost model for rank and final submission**:
    Train the XGBoost model with feature selection and hyperparameter optimization to predict fuel consumption.
    Caches selected features to JSON
    (Adjust the paths for the data)
    Reuses cache on subsequent runs (unless --force-sfs)

    From step 3 this are needed: 
    
    apt.parquet (airport coordinates)
    featured_data_merged.parquet (pre-computed training features)
    featured_data_rank_merged.parquet (pre-computed rank features)
    featured_data_final_merged.parquet (pre-computed final features)
    ```bash
    python XGBoostTraining_Testing.py
    python XGBoostTraining_Final.py
    ```

6.  **Train a Model 2nd way**:
    After the data is fully prepared, you can train a model.
    ```bash
    python run_pipeline.py train --model xgb
    ```

7.  **Generate Predictions from step 6**:
    Use a trained model to generate a submission file.
    ```bash
    python run_pipeline.py predict
    ```

---

## Pipeline Stages in Detail

The `run_pipeline.py` script accepts the following stages:

-   `correct_timestamps`: Corrects takeoff/landing times using trajectory data.
-   `prepare_metars`: Processes raw METAR files into a clean, analysis-ready format using the corrected timestamps.
-   `prepare_data`: Merges all data sources, including the processed METARs and load factor data, into a final feature-rich dataset for training.
-   `train`: Trains a specified model (`xgb`, `gbr`, `rf`).
-   `predict`: Generates predictions using a trained model.
-   `evaluate`: Evaluates a model's performance.
-   `tune`: Performs hyperparameter tuning.
-   `profile_data`: Generates a detailed profile report of the input data.
-   `interpolate_trajectories`: Interpolates missing values within raw trajectory files.
-   `filter_trajectories`: Filters erroneous data points from trajectories.

---

## Configuration

The `config.py` file allows you to manage key settings for the pipeline:

-   **`TEST_RUN`**: Set to `True` to run the pipeline on a small fraction of the data. This is highly recommended for debugging and faster iterations.
-   **`TEST_RUN_FRACTION`**: Defines the fraction of data (e.g., `0.01` for 1%) to use when `TEST_RUN` is enabled.
-   **Data Paths**: Contains paths to all major data directories (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `METARS_DIR`, etc.), allowing for easy reorganization.
