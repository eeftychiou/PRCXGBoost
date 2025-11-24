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
SUBMISSIONS_DIR = "submissions" # Added for evaluate_model.py

# --- Source Dataset Paths ---
BASE_DATASETS_DIR = os.path.join(DATA_DIR, "prc-2025-datasets")
FLIGHTS_TRAIN_DIR = os.path.join(BASE_DATASETS_DIR, "flights_train")
FLIGHTS_RANK_DIR = os.path.join(BASE_DATASETS_DIR, "flights_rank")
METARS_DIR = os.path.join(DATA_DIR, "metars")

# --- Interpolated Trajectory Paths ---
INTERPOLATED_TRAJECTORIES_DIR = os.path.join(DATA_DIR, "interpolated_trajectories")

# --- Data Preparation ---
TEST_RUN = False
TEST_RUN_FRACTION = 0.05  # Use a fraction of the data for test runs

# Feature Engineering Flags
ENABLE_PHASE_DURATION_FEATURES = False # New flag to control phase duration features
USE_OPENAP_PHASE_DETECTION = True # New flag for OpenAP phase detection

# Min-Max Scaler Bounds for PINN inputs
SCALER_BOUNDS = {
    'altitude': {'min': 0, 'max': 45000},
    'true_airspeed': {'min': 0, 'max': 600},
    'segment_start_mass': {'min': 20000, 'max': 90000}
}

# --- Logging Configuration ---
LOG_FILE = os.path.join(INTROSPECTION_DIR, "error.log")
LOG_LEVEL = logging.ERROR
