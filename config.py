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

# Create core directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(INTROSPECTION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# --- Source Dataset Paths ---
BASE_DATASETS_DIR = os.path.join(DATA_DIR, "prc-2025-datasets")
FLIGHTS_TRAIN_DIR = os.path.join(BASE_DATASETS_DIR, "flights_train")
FLIGHTS_RANK_DIR = os.path.join(BASE_DATASETS_DIR, "flights_rank")
METARS_DIR = os.path.join(DATA_DIR, "metars")

# --- Interpolated Trajectory Paths ---
INTERPOLATED_TRAJECTORIES_DIR = os.path.join(DATA_DIR, "interpolated_trajectories")

# --- Specific Data Files ---
APT_PARQUET = os.path.join(BASE_DATASETS_DIR, "apt.parquet")
FLIGHTLIST_TRAIN = os.path.join(BASE_DATASETS_DIR, "flightlist_train.parquet")
FLIGHTLIST_RANK = os.path.join(BASE_DATASETS_DIR, "flightlist_rank.parquet")
FLIGHTLIST_FINAL = os.path.join(BASE_DATASETS_DIR, "flightlist_final.parquet")
FUEL_TRAIN = os.path.join(BASE_DATASETS_DIR, "fuel_train.parquet")
FUEL_RANK = os.path.join(BASE_DATASETS_DIR, "fuel_rank_submission.parquet")
FUEL_FINAL = os.path.join(BASE_DATASETS_DIR, "fuel_final_submission.parquet")

# --- Data Preparation ---
TEST_RUN = False
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

# --- Aircraft Specifications ---
AIRCRAFT_DATA = {
    'A19N': {'mtow_kg': 79200, 'oew_kg': 45100, 'mlw_kg': 67400, 'max_fuel_kg': 27200, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A20N': {'mtow_kg': 79000, 'oew_kg': 45400, 'mlw_kg': 67400, 'max_fuel_kg': 27200, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A21N': {'mtow_kg': 97000, 'oew_kg': 50300, 'mlw_kg': 79200, 'max_fuel_kg': 32840, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.025},
    'A318': {'mtow_kg': 68000, 'oew_kg': 39500, 'mlw_kg': 57500, 'max_fuel_kg': 24210, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.025},
    'A319': {'mtow_kg': 75500, 'oew_kg': 40800, 'mlw_kg': 62500, 'max_fuel_kg': 24210, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A320': {'mtow_kg': 78000, 'oew_kg': 42400, 'mlw_kg': 66000, 'max_fuel_kg': 27200, 'type': 'narrowbody',
             'wing_area_m2': 122.6, 'cd0': 0.024},
    'A321': {'mtow_kg': 93500, 'oew_kg': 48700, 'mlw_kg': 77800, 'max_fuel_kg': 32840, 'type': 'narrowbody',
             'wing_area_m2': 128.0, 'cd0': 0.025},
    'A332': {'mtow_kg': 233000, 'oew_kg': 119500, 'mlw_kg': 182000, 'max_fuel_kg': 139090, 'type': 'widebody',
             'wing_area_m2': 361.6, 'cd0': 0.022},
    'A333': {'mtow_kg': 242000, 'oew_kg': 123400, 'mlw_kg': 187000, 'max_fuel_kg': 139090, 'type': 'widebody',
             'wing_area_m2': 361.6, 'cd0': 0.022},
    'A343': {'mtow_kg': 275000, 'oew_kg': 131000, 'mlw_kg': 192000, 'max_fuel_kg': 155040, 'type': 'widebody',
             'wing_area_m2': 363.1, 'cd0': 0.023},
    'A359': {'mtow_kg': 280000, 'oew_kg': 142400, 'mlw_kg': 205000, 'max_fuel_kg': 138000, 'type': 'widebody',
             'wing_area_m2': 442.0, 'cd0': 0.021},
    'A388': {'mtow_kg': 575000, 'oew_kg': 277000, 'mlw_kg': 427000, 'max_fuel_kg': 320000, 'type': 'widebody',
             'wing_area_m2': 845.0, 'cd0': 0.022},
    'B37M': {'mtow_kg': 82200, 'oew_kg': 45100, 'mlw_kg': 69400, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.023},
    'B38M': {'mtow_kg': 82200, 'oew_kg': 45100, 'mlw_kg': 69400, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.023},
    'B39M': {'mtow_kg': 88300, 'oew_kg': 46550, 'mlw_kg': 72350, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.024},
    'B3XM': {'mtow_kg': 89800, 'oew_kg': 50300, 'mlw_kg': 73940, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 127.4, 'cd0': 0.024},
    'B734': {'mtow_kg': 68000, 'oew_kg': 38300, 'mlw_kg': 56200, 'max_fuel_kg': 26020, 'type': 'narrowbody',
             'wing_area_m2': 105.4, 'cd0': 0.025},
    'B737': {'mtow_kg': 70100, 'oew_kg': 39800, 'mlw_kg': 58000, 'max_fuel_kg': 28600, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.024},
    'B738': {'mtow_kg': 79000, 'oew_kg': 41413, 'mlw_kg': 66360, 'max_fuel_kg': 28600, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.023},
    'B739': {'mtow_kg': 85100, 'oew_kg': 44676, 'mlw_kg': 70300, 'max_fuel_kg': 30190, 'type': 'narrowbody',
             'wing_area_m2': 124.6, 'cd0': 0.024},
    'B744': {'mtow_kg': 412775, 'oew_kg': 178100, 'mlw_kg': 295742, 'max_fuel_kg': 216840, 'type': 'widebody',
             'wing_area_m2': 541.0, 'cd0': 0.023},
    'B748': {'mtow_kg': 447700, 'oew_kg': 197130, 'mlw_kg': 312072, 'max_fuel_kg': 243120, 'type': 'widebody',
             'wing_area_m2': 554.0, 'cd0': 0.022},
    'B752': {'mtow_kg': 115680, 'oew_kg': 58390, 'mlw_kg': 99800, 'max_fuel_kg': 52300, 'type': 'narrowbody',
             'wing_area_m2': 185.3, 'cd0': 0.024},
    'B763': {'mtow_kg': 186880, 'oew_kg': 90010, 'mlw_kg': 145150, 'max_fuel_kg': 91380, 'type': 'widebody',
             'wing_area_m2': 283.3, 'cd0': 0.023},
    'B772': {'mtow_kg': 297560, 'oew_kg': 145150, 'mlw_kg': 213180, 'max_fuel_kg': 171170, 'type': 'widebody',
             'wing_area_m2': 427.8, 'cd0': 0.022},
    'B773': {'mtow_kg': 351530, 'oew_kg': 167830, 'mlw_kg': 251290, 'max_fuel_kg': 181280, 'type': 'widebody',
             'wing_area_m2': 427.8, 'cd0': 0.022},
    'B77W': {'mtow_kg': 351530, 'oew_kg': 167830, 'mlw_kg': 251290, 'max_fuel_kg': 181280, 'type': 'widebody',
             'wing_area_m2': 427.8, 'cd0': 0.022},
    'B788': {'mtow_kg': 227930, 'oew_kg': 119950, 'mlw_kg': 172365, 'max_fuel_kg': 126210, 'type': 'widebody',
             'wing_area_m2': 325.0, 'cd0': 0.021},
    'B789': {'mtow_kg': 254010, 'oew_kg': 128850, 'mlw_kg': 192775, 'max_fuel_kg': 126370, 'type': 'widebody',
             'wing_area_m2': 325.0, 'cd0': 0.021},
    'E145': {'mtow_kg': 22000, 'oew_kg': 12400, 'mlw_kg': 20200, 'max_fuel_kg': 6200, 'type': 'narrowbody',
             'wing_area_m2': 51.2, 'cd0': 0.027},
    'E170': {'mtow_kg': 38600, 'oew_kg': 21620, 'mlw_kg': 35990, 'max_fuel_kg': 11187, 'type': 'narrowbody',
             'wing_area_m2': 72.7, 'cd0': 0.026},
    'E190': {'mtow_kg': 51800, 'oew_kg': 29540, 'mlw_kg': 47790, 'max_fuel_kg': 15200, 'type': 'narrowbody',
             'wing_area_m2': 92.5, 'cd0': 0.025},
    'E195': {'mtow_kg': 52290, 'oew_kg': 29100, 'mlw_kg': 48280, 'max_fuel_kg': 15200, 'type': 'narrowbody',
             'wing_area_m2': 92.5, 'cd0': 0.026},
    'E75L': {'mtow_kg': 39380, 'oew_kg': 22010, 'mlw_kg': 36200, 'max_fuel_kg': 10300, 'type': 'narrowbody',
             'wing_area_m2': 82.0, 'cd0': 0.026},
    'C550': {'mtow_kg': 9072, 'oew_kg': 5125, 'mlw_kg': 8618, 'max_fuel_kg': 3619, 'type': 'narrowbody',
             'wing_area_m2': 31.8, 'cd0': 0.028},
    'GLF6': {'mtow_kg': 45360, 'oew_kg': 24040, 'mlw_kg': 34700, 'max_fuel_kg': 18600, 'type': 'widebody',
             'wing_area_m2': 94.0, 'cd0': 0.022},
}

WIDEBODY_AIRCRAFT = [k for k, v in AIRCRAFT_DATA.items() if v['type'] == 'widebody']

# --- Extended Dataset Paths ---
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "AugmentedDataFromOPENAP")
os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

AUGMENTED_FINAL_CSV = os.path.join(AUGMENTED_DATA_DIR, "augmented_openap_correct_mass_ALL_FLIGHTS_final.csv")
AUGMENTED_FINAL_TEST_CSV = os.path.join(AUGMENTED_DATA_DIR, "augmented_openap_rank_final_ALL_FLIGHTS.csv")
AUGMENTED_RANK_CSV = os.path.join(AUGMENTED_DATA_DIR, "augmented_openap_submission_ALL_FLIGHTSrank.csv")

# Results and Feature Selection
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features_sfs.json")
SYNTHETIC_WIDEBODY_PATH = os.path.join(PROCESSED_DATA_DIR, "synthetic_widebody.parquet")
