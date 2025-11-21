# Project Overview

This project aims to build a Machine Learning model to predict fuel consumption for individual flight segments. The pipeline is designed to be modular, allowing for independent execution of different data processing and modeling stages.

## Data Structure and Key Files

The project's data is organized into several key directories and files, each serving a specific purpose. Understanding this structure is crucial for extending the project.

-   **`data/prc-2025-datasets/`**: This is the primary source of raw data, divided into `train`, `rank`, and `final` sets.
    -   **`flightlist_[stage].parquet`**: Contains the core flight details for each stage (`train`, `rank`, `final`). This includes `flight_id`, `aircraft_type`, `origin_icao`, `destination_icao`, and initial (uncorrected) `takeoff_timestamp` and `landing_timestamp`.
    -   **`fuel_[stage].parquet`**: Defines the flight **segments** for which fuel consumption is measured or needs to be predicted. Each flight (`flight_id`) can have multiple segments, each defined by a `start` and `end` timestamp. The `fuel_kg` is the target variable and is only present in `fuel_train.parquet`.
    -   **`trajectories/flights_[stage]/`**: Contains the raw, high-frequency trajectory data for each flight, stored in individual `[flight_id].parquet` files. These files contain time-series data like `latitude`, `longitude`, `altitude`, `groundspeed`, etc.

-   **`data/acPerf/`**: Contains aircraft performance specifications in `acPerfOpenAP.csv`, which is joined during data preparation.

-   **`data/METARs/`**: Contains raw weather data in CSV format. These files have a commented header (lines starting with `#`) and provide hourly or half-hourly weather reports for various airport stations.

-   **`data/processed/`**: This is the output directory for all processed data.
    -   **`featured_data_[stage].parquet`**: The main output of the `prepare_data` stage. These files contain the fully merged and feature-engineered dataset, including corrected timestamps.
    -   **`processed_metars.parquet`**: The output of the `prepare_metars` stage. It contains cleaned, encoded, and imputed weather data aligned with the flights from the `featured_data` files.

-   **`data/interpolated_trajectories/`**: Stores the output of the `interpolate_trajectories` stage, where missing values in the raw trajectory data have been filled in.

## Pipeline Stages

The ML pipeline is managed by `run_pipeline.py` and consists of the following stages:

1.  **`profile_data`**: Analyzes the input data to generate a profile report.
2.  **`filter_trajectories`**: Filters raw trajectory files to remove erroneous data points.
3.  **`interpolate_trajectories`**: Processes the *filtered* trajectory files, interpolates missing values, and saves the processed files to `data/interpolated_trajectories`.
4.  **`prepare_data`**: This is a multi-run stage.
    -   **First Run**: Merges `flightlist`, `fuel`, airport, and aircraft data. It corrects takeoff and landing times using trajectory data and saves the result as `featured_data_[stage].parquet`. This initial run is necessary to generate the corrected timestamps required by the `prepare_metars` stage.
    -   **Second Run**: After `prepare_metars` has been run, this stage is executed again. It detects the `processed_metars.parquet` file and merges the weather features into the `featured_data` files.
5.  **`prepare_metars`**: This stage reads the `featured_data` files from the `processed` directory to get the corrected flight details. It then loads the raw METAR data, finds the nearest weather station for each airport (within a 100nm radius), and generates a clean `processed_metars.parquet` file with encoded and imputed weather features.
6.  **`train`**: Trains a machine learning model (`gbr`, `xgb`, `rf`).
7.  **`predict`**: Uses a trained model to make predictions and create a submission file.
8.  **`evaluate`**: Evaluates the performance of a trained model.
9.  **`tune`**: Performs hyperparameter tuning for a selected model.

## How to Run (Updated Workflow)

The recommended workflow to include weather data is a three-step process:

**Step 1: Initial Data Preparation**
Run the `prepare_data` stage to generate the `featured_data` files with corrected timestamps.

```bash
python run_pipeline.py prepare_data
```

**Step 2: Prepare METAR Data**
Run the `prepare_metars` stage. This will use the files from Step 1 to create the `processed_metars.parquet` file.

```bash
python run_pipeline.py prepare_metars
```

**Step 3: Final Data Preparation with Weather**
Run `prepare_data` again. This time, it will detect and merge the weather data, creating the final, feature-rich dataset.

```bash
python run_pipeline.py prepare_data
```

After these steps, you can proceed with training the model:

```bash
python run_pipeline.py train --model xgb
```

## Configuration

-   **`TEST_RUN`**: Set to `True` in `config.py` to run the pipeline on a small fraction of the data for faster testing and debugging.
-   **`TEST_RUN_FRACTION`**: The fraction of data to use when `TEST_RUN` is `True`.
