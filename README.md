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
    
    **Note**: This file is already provided in the repository. Running these scripts again will produce a different result. This step is for documentation purposes.
    
    First, run `extract_aircraft_types.py` to get all aircraft types used in the dataset. Then fill in the ac performance parameters from OpenAP using `enrich_aircraft_data.py` followed by `create_behavioral_features.py` to analyze flight trajectories and generate a behavioral signature for each aircraft type. Then, run `impute_aircraft_types.py` to fill in the performance parameters of the missing a/c types (A306, MD11 and B77L).
    
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
    This stage processes the raw METAR files into a clean, flight-keyed dataset. It loads all raw METAR reports and, for each flight, finds the nearest weather station to the origin and destination airports. It then retrieves the most recent weather report for the takeoff and landing times, decodes weather phenomena codes into binary features (e.g., `wx_is_rain`, `wx_is_fog_mist`), and imputes missing values. The final output is a single `processed_metars.parquet` file. While METAR files are included in the repository, they can be re-downloaded by running `python download_metars.py` (download script build upon the last year winners' code).
    ```bash
    python run_pipeline.py prepare_metars
    ```
7.  **Enhance apt.parquet**:
    This script enriches the `apt.parquet` file by scraping detailed airport data from SkyVector. For every airport in the flight lists, it fetches runway information (length, true heading, elevation) and the overall airport elevation. It uses the `pygeomag` library with a World Magnetic Model (WMM) file to convert magnetic runway headings to true headings, which requires the `wmm/WMMHR_2025.COF` file to be present.
    ```bash
    python impute_apt.py
    ```
9.  **Estimate Passenger Load Factors**:
    This script estimates the passenger load factor for each airport based on historical data extracted from the IATA Market reports for the months April to August 2025. It uses the `regionalLoadFactor.py` script to estimate the load factor for each airport based on the reported load factor in the report . The results are saved to `processed/average_load_factor_by_airport_pair_v3.csv`.

10.  **Data Preparation Step 1**:
    This is the main data preparation stage. It merges all data sources (flight lists, fuel, aircraft performance, airport data, and weather) and runs the full feature engineering pipeline via `augment_features.py`. This script creates a wide variety of features, which can be grouped into the following categories:
    - **Basic Flight and Segment Features:** Includes `segment_duration`, `flight_duration_seconds`, `great_circle_distance_km`, and time-based features like day of the week and time of day for flight and segment events.
    - **Trajectory-Based Features:** For each segment, it calculates aggregate statistics (min, max, mean, std) for key trajectory variables like `altitude`, `groundspeed`, and `vertical_rate`. It also computes the total distance traveled during the segment (`segment_distance_km`) and the change in altitude (`alt_diff_rev`).
    - **Flight Phase Features:** The script uses both a custom rule-based method and the `openap` library to classify flight phases. It then calculates the duration and fraction of time spent in each phase (e.g., `phase_duration_cl`, `phase_fraction_cruise`).
    - **Aircraft Performance and Mass Estimation Features:** It calculates an `estimated_takeoff_mass` based on the aircraft's operating empty weight, estimated payload, and estimated fuel load. It also computes a `seg_avg_burn_rate`, which is a weighted average fuel burn rate for the segment based on the time spent in each flight phase.
    - **Advanced Features:** The script generates interaction features (e.g., `duration_x_estimated_takeoff_mass`) and polynomial features (e.g., `segment_duration_sq`) to capture more complex, non-linear relationships.
    This will produce three files featured_data_[stage].parquet in `processed/`.

    python run_pipeline.py prepare_data


10.  **Data Preparation Step 2**:
    To generate the augmented feature set for fuel consumption prediction, run the three Python scripts sequentially. This files process the provided parquet trajectory files. This process took multiple days on an HPC with a Nvidia V100 so we are attaching the csvs in `data/AugmentedDataFromOPENAP`. We used OpenAP FuelFlow models (with use_synonym=True) to compute physics-based fuel predictions for all 36 supported aircraft via true dynamic mass tracking from flight takeoff. If you run the python files after the AugmentationRank and AugmentationFinal finish the process the user will have to merge those two csvs together into the final

    python AugmentationTraining.py, python AugmentationRank.py, python AugmentationFinal.py. 


    

### Machine Learning Training ###
Add the correct paths to the generated data from Step 1 (Data preparation) 
DATA_PATH, APT_PATH, FLIGHTLIST_PATH, FUEL_PATH, TEST_CSV_PATH, FUEL_RANK_PATH, RESULTS_DIR, FLIGHTLIST_RANK_PATH, FEATURED_DATA_TRAIN, FEATURED_DATA_TEST, SYNTHETIC_PATH, SELECTED_FEATURES_PATH
Run 
```bash
python XGBoostTraining_Testing.py
```
**first** to generate preprocessing imputers, encoders, scalers, and selected features (saved as `preprocessors_rank.joblib` and `selected_features_sfs3.json`) using 80/20 train/validation

### Steps of XGBoostTraining_Testing

### Step 1: Synthetic Dataset
Generates 25K synthetic widebody samples (A332/A333/A359/etc.) by perturbing real widebody training data with 5-15% Gaussian noise, emphasizing long segments (top 25% duration) to balance underrepresented widebody classes and reduce RMSE.

### Step 2: Feature Selection 

This stage selects the most relevant features for the model, which can improve performance and reduce training time.
Uses SequentialFeatureSelector (forward selection, 5-fold CV, XGBoost base estimator) on preprocessed training data to automatically select the most predictive features from 100+ candidate (`selected_features_sfs3.json` which is available in the data folder)

### Step 3: Hyperparameter Tuning

Tune the model you just trained to find the optimal hyperparameters.
Applies RandomizedSearchCV (200 iterations, 5-fold CV) over expanded grid (`max_depth` 7-9, `learning_rate` 0.06-0.08, `n_estimators` 850-950, etc.) on selected features, ranking top-10 models by validation RMSE and evaluates overfitting gap (train vs val RMSE) to select robust top-5 configurations.

The randomized search can be inscreased or decreased based on the user

in both `XGBoostTraining_Testing.py` and `XGBoostTraining_Final.py` the param_grid is commented out and we left the best model parameters, feel free to run the hyperparameter tuning by uncommenting the param_grid

### Step 4: Final Model Training

Train the model using the prepared data and the selected features.
Add the correct paths to the generated data from Step 1 (Data preparation) 
DATA_PATH, APT_PATH, FLIGHTLIST_PATH, FUEL_PATH, TEST_CSV_PATH, FUEL_RANK_PATH, RESULTS_DIR, FLIGHTLIST_RANK_PATH, FLIGHTLIST_FINAL_PATH, FEATURED_DATA_TRAIN, FEATURED_DATA_TEST, SYNTHETIC_PATH, SELECTED_FEATURES_PATH
Basically TEST now becomes the FINAL
Run 
```bash
python XGBoostTraining_Final.py 
```
to load XGBoostTraining_Testing.py imputers, retrains top-10 models on 100% augmented data (original + synthetic), produces feature importance analysis and generates hybrid test predictions combining exact XGBoostTraining_Testing.py rank rows with new final submission rows.

### Step 5: Evaluation and Submission

Use your final, tuned model to evaluate its performance and generate submission files.
Top-10 models produce ranked submissions in `Results/` with validation RMSE, MAE, R² metrics and the best model (lowest val RMSE) is recommended for leaderboard submission

## Configuration (`config.py`)

-   **`TEST_RUN`**: Set to `True` to run the pipeline on a small fraction of the data (`5%` by default). This is highly recommended for debugging and faster iterations.
-   **`TEST_RUN_FRACTION`**: Defines the fraction of data to use when `TEST_RUN` is enabled.
-   **Data Paths**: All major data directories are defined here, allowing for easy project reorganization.
