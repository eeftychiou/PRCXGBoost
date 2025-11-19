"""
Configuration for the ML pipeline.
"""
import os
import logging

# --- Core Paths ---
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "acPerf")
PROCESSED_DATA_DIR = os.path.join("processed")
INTROSPECTION_DIR = "introspection"
MODELS_DIR = "models"

# --- Source Dataset Paths ---
BASE_DATASETS_DIR = os.path.join(DATA_DIR, "prc-2025-datasets")
FLIGHTS_TRAIN_DIR = os.path.join(BASE_DATASETS_DIR, "flights_train")
FLIGHTS_RANK_DIR = os.path.join(BASE_DATASETS_DIR, "flights_rank")
METARS_DIR = os.path.join(DATA_DIR, "metars")

# --- Interpolated Trajectory Paths ---
INTERPOLATED_TRAJECTORIES_DIR = os.path.join(DATA_DIR, "interpolated_trajectories")

# --- Data Preparation ---
TEST_RUN = True
TEST_RUN_FRACTION = 0.05  # Use a fraction of the data for test runs


# Min-Max Scaler Bounds for PINN inputs
SCALER_BOUNDS = {
    'altitude': {'min': 0, 'max': 45000},
    'true_airspeed': {'min': 0, 'max': 600},
    'segment_start_mass': {'min': 20000, 'max': 90000}
}

# --- Logging Configuration ---
LOG_FILE = os.path.join(INTROSPECTION_DIR, "error.log")
LOG_LEVEL = logging.ERROR
