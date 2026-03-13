# Flight Fuel Consumption Prediction

## Overview

This project provides a complete machine learning pipeline to predict fuel consumption for individual flight segments. It includes modules for data enrichment, feature engineering, model training, hyperparameter tuning, and submission generation. The pipeline is designed to be modular and is controlled via a single entry point (`run_pipeline.py`), making it easy to execute and reproduce.

### Key Features
- **Modular Pipeline**: Each stage (data preparation, training, evaluation) is independent and can be run separately.
- **Rich Data Integration**: Enriches the core dataset with external sources, including airport data, historical weather (METAR) reports, and regional passenger load factor estimates.
- **Advanced Feature Engineering**: Creates a wide range of features from raw trajectory data, including flight phase detection (taxi, takeoff, climb, cruise, etc.), and physics-informed features using aircraft performance models.
- **Flexible Model Training**: Supports multiple models (XGBoost, Gradient Boosting, Random Forest).
- **Resource-Optimized Training**: Automatically selects the GPU with the lowest memory usage for training tasks.
- **Progress Tracking**: Real-time `tqdm` progress bars in all long-running augmentation and training loops for clear ETAs.
- **Hyperparameter Tuning**: Includes a script to optimize model hyperparameters using `RandomizedSearchCV`.

---

## Project Structure

```
PRCXGBoost/
│
├── data/
│   ├── METARs/               # Raw METAR files downloaded from the web
│   ├── acPerf/               # Aircraft performance data
│   ├── AugmentedDataFromOPENAP/# Augmented OpenAP data
│   ├── prc-2025-datasets/    # Original competition datasets (flight lists, fuel, trajectories)
│   ├── filtered_trajectories/ # Output of the trajectory filtering stage
│   ├── interpolated_trajectories/ # Output of the trajectory interpolation stage
│   ├── htmlfile/             # Output directory for all HTML files downloaded from SkyVector
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
├── AugmentationRank.py       # Augmentation and data imputation script, openAP fuel calculation and starting mass using the rank data trajectories
├── AugmentationFinal.py      # Augmentation and data imputation script, openAP fuel calculation and starting mass using the final data trajectories
├── AugmentationTraining.py   # Augmentation and data imputation script, openAP fuel calculation and starting mass using the training data trajectories
├── XGBoostTraining_Testing.py  # Example training script for XGBoost and preparation of preprocessors and selected features
├── XGBoostTraining_Final.py  # Training script for XGBoost for the final submission
└── README.md                 # This file
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

The following stages prepare all the necessary data for model training. The entire workflow is now orchestrated through `run_pipeline.py`.

### Step 1: Initial Data Acquisition
These stages only need to be run once to populate the repository with external data.

1.  **Download the dataset from the PRC website**:
    ```bash
    # Note: Requires mc (MinIO Client) configured
    mc cp opensky/prc-2025-datasets/ data/prc-2025-datasets/
    ```

2.  **Generate Aircraft Performance File**:
    This step creates the `data/acPerf/acPerfOpenAP.csv` file, which contains detailed aircraft performance and behavioral data. While provided in the repository, you can regenerate it if needed.
    
    It scans trajectories to extract aircraft types, enriches them with OpenAP data, and generates a "behavioral signature" for each type.
    
    ```bash
    python run_pipeline.py setup_ac_perf
    ```
    
    *Legacy equivalent:* Runs `extract_aircraft_types.py`, `enrich_aircraft_data.py`, and `create_behavioral_features.py` sequentially.

### Step 2: Data Enrichment & Preprocessing

1.  **Enrich Airport Data**:
    Enriches the `apt.parquet` file by scraping detailed airport data from SkyVector (runway lengths, headings, elevation).
    ```bash
    python run_pipeline.py setup_apt
    ```

2.  **Filter and Interpolate Trajectories**:
    # Stage 1: Filtering (removes erroneous points)
    python run_pipeline.py filter_trajs
    
    # Stage 2: Interpolation (fills missing values)
    python run_pipeline.py interpolate_trajectories

  ### Running Specific Splits
You can now run most stages for a specific dataset split (`train`, `rank`, or `final`) using the `--split` argument:

```bash
# Filter only the final trajectories
python run_pipeline.py filter_trajs --split final

# Interpolate only the final trajectories
python run_pipeline.py interpolate_trajectories --split final

# Correct timestamps only for the final dataset
python run_pipeline.py correct_timestamps --split final
```

This is particularly useful when you have added new data or want to avoid re-processing existing datasets.

3.  **Calculate Regional Load Factors**:
    Estimates passenger load factors based on IATA regions and flight routes to enrich the dataset with payload proxies.
    ```bash
    python run_pipeline.py regional_load_factor
    ```

4.  **Correct Timestamps**:
    Adjusts takeoff and landing times in the flight lists based on aircraft altitude and standard rates of climb/descent.
    ```bash
    python run_pipeline.py correct_timestamps
    ```

4.  **Prepare Weather Data**:
    Processes raw METAR files into a flight-keyed dataset, decoding phenomenon codes and mapping them to the nearest airport.
    ```bash
    python run_pipeline.py prepare_metars
    ```

### Step 3: Feature Engineering & Augmentation

1.  **Main Data Preparation**:
    Merges all data sources (flight lists, fuel, performance, airport, weather) and creates the core feature set (featured_data_{stage}.parquet).
    ```bash
    python run_pipeline.py prepare_data
    ```

2.  **Data Augmentation (Optional)**:
    Computes physics-based fuel predictions using OpenAP FuelFlow models with dynamic mass tracking.
    *Note: Pre-generated data is already provided in `data/AugmentedDataFromOPENAP`.*
    ```bash
    python run_pipeline.py augment
    ```

### Step 4: Machine Learning Training

Follow the two-step training process used in the final submission:

1.  **Feature Selection & Testing**:
    Generates synthetic widebody samples, performs Sequential Feature Selection (SFS), and tunes initial hyperparameters.
    ```bash
    python run_pipeline.py train_test
    ```

2.  **Final Model Training**:
    Retrains the top 10 models on 100% of the augmented data (original + synthetic) and produces the final submission parquets.
    ```bash
    python run_pipeline.py train_final
    ```

### Step 5: Evaluation & Submission
Evaluate the results and generate final leaderboard scores.
```bash
python run_pipeline.py evaluate --run_type final
```

---

## Configuration (`config.py`)
Centralizes all aircraft specifications (`AIRCRAFT_DATA`) and data paths. 
-   **`TEST_RUN`**: Set to `True` for debugging on 5% of data.
