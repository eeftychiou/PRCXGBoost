# Flight Fuel Consumption Prediction

## Overview

This project provides a complete machine learning pipeline to predict fuel consumption for individual flight segments. It includes modules for data enrichment, feature engineering, model training, hyperparameter tuning, and submission generation. The pipeline is designed to be modular and is controlled via a single entry point (`run_pipeline.py`), making it easy to execute and reproduce.

### Key Features
- **Modular Pipeline**: Each stage (data preparation, training, evaluation) is independent and can be run separately.
- **Rich Data Integration**: Enriches the core dataset with external sources, including airport data, historical weather (METAR) reports, and regional passenger load factor estimates.
- **Advanced Feature Engineering**: Creates a wide range of features from raw trajectory data, including flight phase detection (taxi, takeoff, climb, cruise, etc.), and physics-informed features using aircraft performance models.
- **Flexible Model Training**: Supports multiple models (XGBoost, Gradient Boosting, Random Forest).
- **Hyperparameter Tuning**: Includes a script to optimize model hyperparameters using `RandomizedSearchCV`.

---

## Project Structure

```
PRCXGBoost/
│
├── data/
│   ├── METARs/               # Raw METAR files downloaded from the web
│   ├── acPerf/               # Aircraft performance data
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


---

## How to Reproduce Results: A Step-by-Step Guide

These scripts enrich the raw datasets and should be run before the main pipeline.

### Step 1: Data Preparation

These stages prepare all the necessary data for model training. They only need to be run once.

1.  **Download the dataset from the PRC website**:
    ```bash
    mc cp opensky/prc-2025-datasets/ data/prc-2025-datasets/
    ```

2.  **Generate Aircraft Performance File (Optional)**:
    This step creates the `data/acPerf/acPerfOpenAP.csv` file, which contains detailed aircraft performance and behavioral data.
    
    **Note**: This file is already provided in the repository. Running these scripts may produce a different result. This step is for documentation purposes.
    
    First, run `extract_aircraft_types.py` to get all aircraft types used in the dataset. Then fill in the ac performance parameters from OpenAP using `enrich_aircraft_data.py` followed by `create_behavioral_features.py` to analyze flight trajectories and generate a behavioral signature for each aircraft type. Then, run `impute_aircraft_types.py` to fill in the performance parameters of the missing a/c types (A306, MD11 and B77L)..
    
    ```bash
    python extract_aircraft_types.py
    python enrich_aircraft_data.py
    python create_behavioral_features.py
    python impute_aircraft_types.py

    ```
    
    Finally, the wake turbulence category and fuel burn per second for each phase of flight (climb, cruise, descent, etc.) were added manually to the resulting CSV. This data was obtained using Google's Gemini Pro 2.5 with the prompt: *"Fill in a table of the fuel flow consumption in kg/sec of the following aircraft during climb, cruise, descent, level flight and ground taxi:"* followed by the list of aircraft.

3.  **Filter Trajectories**:
    This stage filters the raw trajectory files to remove erroneous data points. It uses the `traffic` library to apply a series of filters that check for inconsistencies in position, speed, and other parameters. This crucial cleaning step ensures the quality of the trajectory data used in downstream tasks. This method is credited to last year's winning submission.
    ```bash
    python filter_trajs.py
    ```

4.  **Interpolate Trajectories**:
    This stage processes the filtered trajectory files. It injects the flight segment start and end timestamps into the data, then uses linear interpolation to fill in any missing values for key fields like `altitude`, `groundspeed`, and `position`. Finally, it uses the `openap` library to detect and label flight phases (e.g., climb, cruise, descent), saving the enriched files to `data/interpolated_trajectories/`.
    ```bash
    python run_pipeline.py interpolate_trajectories
    ```

5.  **Correct Timestamps**:
    This stage uses the interpolated trajectories to correct the takeoff and landing times in the flight lists. For each flight, it identifies the point in the trajectory closest to the origin and destination airports. The timestamp at this point is then adjusted based on the aircraft's altitude and a standard rate of climb/descent to more accurately estimate the true takeoff and landing times. This method is credited to last year's winning submission.
    ```bash
    python run_pipeline.py correct_timestamps
    ```

6.  **Prepare Weather Data**:
    This stage processes the raw METAR files into a clean, flight-keyed dataset. It loads all raw METAR reports and, for each flight, finds the nearest weather station to the origin and destination airports. It then retrieves the most recent weather report for the takeoff and landing times, decodes weather phenomena codes into binary features (e.g., `wx_is_rain`, `wx_is_fog_mist`), and imputes missing values. The final output is a single `processed_metars.parquet` file. While METAR files are included in the repository, they can be re-downloaded by running `python download_metars.py`.
    ```bash
    python run_pipeline.py prepare_metars
    ```
7.  **Enhance apt.parquet**:
    This script enriches the `apt.parquet` file by scraping detailed airport data from SkyVector. For every airport in the flight lists, it fetches runway information (length, true heading, elevation) and the overall airport elevation. It uses the `pygeomag` library with a World Magnetic Model (WMM) file to convert magnetic runway headings to true headings, which requires the `wmm/WMMHR_2025.COF` file to be present.
    ```bash
    python impute_apt.py
    ```

8.  **Final Data Preparation**:
    This is the main data preparation stage. It merges all data sources (flight lists, fuel, aircraft performance, airport data, and weather) and runs the full feature engineering pipeline.
    ```bash
    python run_pipeline.py prepare_data
    ```

### Step 2: Feature Selection 

This stage selects the most relevant features for the model, which can improve performance and reduce training time.

```bash
# Example: Select features using XGBoost's feature importance
python run_pipeline.py select_features --fs_model xgb --fs_method importance
```
This will create a JSON file in `processed/feature_sets/` containing the list of selected features. Note the path to this file for the next step.

### Step 3: Model Training

Train the model using the prepared data and the selected features.

```bash
# Example: Train an XGBoost model with the selected features
python run_pipeline.py train --model xgb --features "path/to/your/selected_features.json"
```
Replace `"path/to/your/selected_features.json"` with the actual path from the previous step. A new model directory will be created in `models/`.

### Step 4: Hyperparameter Tuning

Tune the model you just trained to find the optimal hyperparameters.

1.  **Run the Tuning Script**:
    ```bash
    python tune_model.py
    ```
2.  **Select the Model**: The script will list all available models. Enter the number corresponding to the model you created in Step 3.
3.  **Wait for Tuning to Complete**: The script will run `RandomizedSearchCV` and save the best parameters to a `best_params.json` file inside your model's directory.

### Step 5: Re-train with Best Parameters

Re-train your model one last time using the optimal hyperparameters found during tuning.

```bash
# The tuning script will provide this command at the end
python run_pipeline.py train --model xgb --features "path/to/your/selected_features.json" --params "path/to/your/best_params.json"
```
This will create a new, fully optimized model in the `models/` directory.

### Step 6: Evaluation and Submission

Use your final, tuned model to evaluate its performance and generate submission files.

1.  **Run the Evaluation Script**:
    ```bash
    python run_pipeline.py evaluate
    ```
2.  **Select Your Final Model**: Choose the tuned model you created in the previous step.
3.  **Choose the Run Type**: The script will then prompt you to choose what you want to do:
    -   `evaluate`: Calculates performance metrics (MAE, RMSE, R²) on a validation set.
    -   `rank`: Generates a `fuel_rank.parquet` submission file.
    -   `final`: Generates a `fuel_final.parquet` submission file.

---

## Configuration (`config.py`)

-   **`TEST_RUN`**: Set to `True` to run the pipeline on a small fraction of the data (`5%` by default). This is highly recommended for debugging and faster iterations.
-   **`TEST_RUN_FRACTION`**: Defines the fraction of data to use when `TEST_RUN` is enabled.
-   **Data Paths**: All major data directories are defined here, allowing for easy project reorganization.
