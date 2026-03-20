"""
Baseline model comparison for the fuel prediction pipeline.

Trains Ridge Regression, Random Forest, LightGBM, and XGBoost (reference)
on the same data, features, preprocessing, and train/val split as
XGBoostTraining_Testing.py, enabling a fair apples-to-apples comparison.

Results are saved to models/baselines/baseline_comparison.csv.
Individual models are saved as .joblib files in models/baselines/.
"""
import os
import json
import logging
import warnings
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb

import config

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (mirror XGBoostTraining_Testing.py exactly)
# ---------------------------------------------------------------------------
DATA_PATH = config.AUGMENTED_FINAL_CSV
APT_PATH = config.APT_PARQUET
FLIGHTLIST_PATH = config.FLIGHTLIST_TRAIN
FUEL_PATH = config.FUEL_TRAIN
FEATURED_DATA_TRAIN = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
FEATURED_DATA_RANK = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_rank.parquet")
SYNTHETIC_PATH = config.SYNTHETIC_WIDEBODY_PATH
SELECTED_FEATURES_PATH = config.SELECTED_FEATURES_PATH

OUTPUT_DIR = os.path.join(config.MODELS_DIR, "baselines")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extended feature list — identical to XGBoostTraining_Testing.py
EXTENDED_FEATURES_FROM_PARQUET = [
    "origin_icao", "origin_name", "destination_icao", "destination_name", "mfc", "pax_high",
    "fuselage_height", "wing_mac", "wing_t/c", "flaps_type", "flaps_area", "flaps_bf/b",
    "flaps_Sf/S", "cruise_mach", "engine_default", "drag_cd0", "drag_e", "drag_gears",
    "fuel_fuel_coef", "limits_OEW", "origin_longitude", "origin_latitude", "origin_elevation",
    "origin_RWY_1_HEADING_a", "origin_RWY_1_LENGTH", "origin_RWY_2_HEADING_a", "origin_RWY_3_LENGTH",
    "origin_RWY_4_HEADING_b", "origin_RWY_4_LENGTH", "origin_RWY_5_HEADING_b", "origin_RWY_5_LENGTH",
    "origin_RWY_8_HEADING_b", "destination_longitude", "destination_latitude", "destination_elevation",
    "destination_RWY_1_LENGTH", "destination_RWY_2_HEADING_a", "destination_RWY_2_HEADING_b",
    "destination_RWY_2_LENGTH", "destination_RWY_3_HEADING_a", "destination_RWY_3_HEADING_b",
    "destination_RWY_3_LENGTH", "destination_RWY_4_HEADING_a", "destination_RWY_4_HEADING_b",
    "destination_RWY_5_HEADING_b", "destination_RWY_5_LENGTH", "destination_RWY_6_LENGTH",
    "segment_duration", "seg_latitude_mean", "seg_latitude_std",
    "seg_longitude_min", "seg_longitude_mean", "seg_longitude_std", "seg_altitude_min",
    "seg_altitude_max", "seg_altitude_mean", "seg_groundspeed_mean", "seg_track_max",
    "seg_track_mean", "seg_track_std", "seg_vertical_rate_max", "seg_vertical_rate_std",
    "seg_mach_min", "seg_mach_max", "seg_mach_mean", "seg_mach_std", "seg_TAS_min",
    "seg_TAS_max", "seg_TAS_mean", "seg_TAS_std", "seg_calculated_speed_min",
    "seg_calculated_speed_mean", "seg_calculated_speed_std", "seg_vertical_rate_change_min",
    "seg_vertical_rate_change_max", "seg_vertical_rate_change_mean", "seg_vertical_rate_change_std",
    "seg_dist_to_origin_km_min", "seg_dist_to_origin_km_max", "seg_dist_to_origin_km_mean",
    "seg_dist_to_origin_km_std", "seg_dist_to_dest_km_min", "seg_dist_to_dest_km_max",
    "seg_dist_to_dest_km_mean", "seg_dist_to_dest_km_std", "phase_fraction_climb",
    "phase_fraction_cruise", "phase_fraction_descent", "phase_fraction_approach",
    "start_alt_rev", "end_alt_rev", "departure_rwy_length", "segment_distance_km",
    "alt_diff_rev", "alt_diff_rev_std", "seg_latitude_delta", "seg_longitude_delta",
    "seg_altitude_delta", "seg_vertical_rate_delta", "seg_mach_delta", "seg_TAS_delta",
    "seg_CAS_delta", "seg_calculated_speed_delta", "seg_dist_to_origin_km_delta",
    "seg_dist_to_dest_km_delta", "takeoff_delta", "landing_delta", "mean_time_in_air",
    "aircraft_encoded", "start_time_encoded", "end_time_encoded",
    "fuel_consumption_gnd", "fuel_consumption_cl",
    "fuel_consumption_de", "fuel_consumption_lvl", "fuel_consumption_cr",
    "fuel_consumption_na", "fuel_consumption", "seg_avg_burn_rate",
    "average_load_factor", "estimated_payload_kg", "trip_fuel_kg",
    "contingency_fuel_kg", "final_reserve_fuel_kg", "estimated_total_fuel_kg",
    "estimated_takeoff_mass",
]

BASE_FEATURES = [
    "starting_mass_kg", "alt_end_ft", "alt_avg_ft", "gs_avg_kts", "vs_avg_fpm",
    "interval_duration_sec", "altitude_change_rate", "great_circle_distance",
    "aircraft_type", "end_hour", "interval_elapsed_from_flight_start",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_file = os.path.join(OUTPUT_DIR, f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    if pd.isna([lon1, lat1, lon2, lat2]).any():
        return np.nan
    try:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * asin(sqrt(a)) * 6371
    except Exception:
        return np.nan


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def evaluate(model, X_train, y_train, X_val, y_val, transform_log):
    """Return dict with train/val metrics. y values are in original kg scale."""
    if transform_log:
        y_train_pred = np.expm1(model.predict(X_train))
        y_val_pred = np.expm1(model.predict(X_val))
    else:
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

    return {
        "train_rmse": _rmse(y_train, y_train_pred),
        "train_mae": _mae(y_train, y_train_pred),
        "train_mape": _mape(y_train, y_train_pred),
        "train_r2": _r2(y_train, y_train_pred),
        "val_rmse": _rmse(y_val, y_val_pred),
        "val_mae": _mae(y_val, y_val_pred),
        "val_mape": _mape(y_val, y_val_pred),
        "val_r2": _r2(y_val, y_val_pred),
    }


# ---------------------------------------------------------------------------
# Data loading — identical merge pipeline as XGBoostTraining_Testing.py
# ---------------------------------------------------------------------------
def load_data():
    logger.info("Loading airport data …")
    apt = pd.read_parquet(APT_PATH)[["icao", "longitude", "latitude"]]

    logger.info("Loading flightlist …")
    flightlist = pd.read_parquet(FLIGHTLIST_PATH)
    flightlist = flightlist.merge(apt, left_on="origin_icao", right_on="icao", how="left")
    flightlist = flightlist.rename(columns={"longitude": "origin_lon", "latitude": "origin_lat"})
    flightlist = flightlist.drop(columns=["icao"], errors="ignore")
    flightlist = flightlist.merge(apt, left_on="destination_icao", right_on="icao", how="left")
    flightlist = flightlist.rename(columns={"longitude": "dest_lon", "latitude": "dest_lat"})
    flightlist = flightlist.drop(columns=["icao"], errors="ignore")
    flightlist["great_circle_distance"] = flightlist.apply(
        lambda r: haversine(r.get("origin_lon"), r.get("origin_lat"), r.get("dest_lon"), r.get("dest_lat")),
        axis=1,
    )

    logger.info("Loading fuel data …")
    fuel = pd.read_parquet(FUEL_PATH)
    fuel_intervals = (
        fuel[["flight_id", "idx", "fuel_kg", "start", "end"]]
        .copy()
        .rename(columns={"idx": "interval_idx"})
    )

    logger.info("Loading extended feature data …")
    featured_train = pd.read_parquet(FEATURED_DATA_TRAIN)
    featured_rank = pd.read_parquet(FEATURED_DATA_RANK)

    common_cols = set(featured_train.columns).intersection(featured_rank.columns)
    available_ext = ["flight_id", "idx"] + [
        c for c in EXTENDED_FEATURES_FROM_PARQUET if c in common_cols
    ]
    featured_sel = featured_train[available_ext].rename(columns={"idx": "interval_idx"})

    logger.info(f"Loading augmented CSV: {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH, delimiter=";", low_memory=False)
    logger.info(f"Augmented CSV rows: {len(df):,}")

    fl_cols = [
        "flight_id", "takeoff", "landed", "great_circle_distance",
        "origin_icao", "destination_icao", "aircraft_type",
        "origin_lon", "origin_lat", "dest_lon", "dest_lat",
    ]
    df = df.merge(flightlist[fl_cols], on="flight_id", how="left")
    df = df.merge(fuel_intervals, on=["flight_id", "interval_idx"], how="left")
    df = df.merge(featured_sel, on=["flight_id", "interval_idx"], how="left")
    logger.info(f"Total columns after merges: {len(df.columns)}")

    # Computed columns
    if "alt_avg_ft" not in df.columns:
        df["alt_avg_ft"] = (df.get("alt_start_ft", 0) + df.get("alt_end_ft", 0)) / 2
    if "altitude_change_rate" not in df.columns:
        df["altitude_change_rate"] = df.get("alt_change_ft", 0) / (df.get("interval_duration_sec", 60) + 1e-6)
    if "end_hour" not in df.columns:
        df["end_hour"] = pd.to_datetime(df.get("end"), errors="coerce").dt.hour.fillna(-1).astype(int)
    if "interval_elapsed_from_flight_start" not in df.columns:
        df["interval_elapsed_from_flight_start"] = 0

    ext_available = [c for c in available_ext[2:] if c not in BASE_FEATURES]
    feature_cols = BASE_FEATURES + ext_available
    feature_cols = [c for c in feature_cols if c in df.columns]

    target_col = "actual_fuel_kg"
    df_feat = df[feature_cols + [target_col]].copy()
    df_feat = df_feat.dropna(subset=[target_col])
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    logger.info(f"Dataset before synthetic augmentation: {len(df_feat):,} rows")

    # Append cached synthetic widebody samples if they exist
    if os.path.exists(SYNTHETIC_PATH):
        df_syn = pd.read_parquet(SYNTHETIC_PATH)
        # Keep only columns that are present in df_feat
        syn_cols = [c for c in feature_cols + [target_col] if c in df_syn.columns]
        df_syn = df_syn[syn_cols]
        df_feat = pd.concat([df_feat, df_syn], ignore_index=True)
        logger.info(f"Synthetic samples added: {len(df_syn):,}. Total: {len(df_feat):,}")
    else:
        logger.warning("No cached synthetic widebody data found — using real data only.")

    return df_feat, feature_cols, target_col


# ---------------------------------------------------------------------------
# Preprocessing — same as XGBoostTraining_Testing.py Phase 3
# ---------------------------------------------------------------------------
def preprocess(X_train, X_val, feature_cols):
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Drop 100 % NaN columns
    nan_cols = [c for c in feature_cols if X_train[c].isna().all()]
    if nan_cols:
        logger.info(f"Dropping {len(nan_cols)} fully-NaN column(s): {nan_cols}")
        feature_cols = [c for c in feature_cols if c not in nan_cols]
        X_train = X_train.drop(columns=nan_cols)
        X_val = X_val.drop(columns=nan_cols)

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    logger.info(f"Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")

    if num_cols:
        num_imp = SimpleImputer(strategy="mean")
        X_train[num_cols] = num_imp.fit_transform(X_train[num_cols])
        X_val[num_cols] = num_imp.transform(X_val[num_cols])

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imp.fit_transform(X_train[cat_cols])
        X_val[cat_cols] = cat_imp.transform(X_val[cat_cols])

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_val[cat_cols] = enc.transform(X_val[cat_cols])
    else:
        enc = None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    preprocessor = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imp if num_cols else None,
        "cat_imputer": cat_imp if cat_cols else None,
        "encoder": enc,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    return X_train_s, X_val_s, preprocessor


# ---------------------------------------------------------------------------
# Feature masking from SFS JSON (same logic as XGBoostTraining_Testing.py)
# ---------------------------------------------------------------------------
def apply_sfs_mask(X_train_s, X_val_s, feature_cols):
    if not os.path.exists(SELECTED_FEATURES_PATH):
        logger.info("No SFS feature file found — using all features.")
        return X_train_s, X_val_s, feature_cols

    with open(SELECTED_FEATURES_PATH, "r") as f:
        sfs_data = json.load(f)

    selected = sfs_data.get("selected_features", [])
    mask = np.array([f in selected for f in feature_cols])
    missing = [f for f in selected if f not in feature_cols]
    if missing:
        logger.warning(f"{len(missing)} SFS feature(s) not present in current data: {missing}")

    used = [f for f, m in zip(feature_cols, mask) if m]
    logger.info(f"SFS mask applied: {sum(mask)}/{len(feature_cols)} features retained")
    return X_train_s[:, mask], X_val_s[:, mask], used


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------
def run():
    logger.info("=" * 70)
    logger.info("BASELINE MODEL COMPARISON")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df, feature_cols, target_col = load_data()

    X_full = df[feature_cols]
    y_full = df[target_col].values.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Train / val split — identical to XGBoostTraining_Testing.py
    # ------------------------------------------------------------------
    all_idx = np.arange(len(X_full))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42, shuffle=True)

    X_train_raw = X_full.iloc[train_idx]
    X_val_raw = X_full.iloc[val_idx]
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]

    logger.info(f"Train: {len(X_train_raw):,}  |  Val: {len(X_val_raw):,}")

    # ------------------------------------------------------------------
    # 3. Preprocessing
    # ------------------------------------------------------------------
    logger.info("\nPhase 3: Preprocessing …")
    X_train_s, X_val_s, preprocessor = preprocess(X_train_raw, X_val_raw, feature_cols)

    # ------------------------------------------------------------------
    # 4. Apply SFS feature mask
    # ------------------------------------------------------------------
    X_train_s, X_val_s, used_features = apply_sfs_mask(X_train_s, X_val_s, preprocessor["feature_cols"])
    n_feat = X_train_s.shape[1]
    logger.info(f"Feature count for baselines: {n_feat}")

    # log1p targets for all models
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # ------------------------------------------------------------------
    # 5. Define baseline models
    # ------------------------------------------------------------------
    LEGACY_XGB_PARAMS = dict(
        n_estimators=1455,
        learning_rate=0.02885922756814833,
        max_depth=9,
        min_child_weight=4,
        gamma=6.24155979490078e-08,
        subsample=0.9991625118585123,
        colsample_bytree=0.6701135673048045,
        reg_alpha=0.004878930563988692,
        reg_lambda=2.3991563444540384e-08,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    models = {
        "ridge": {
            "model": Ridge(alpha=1.0),
            "log_transform": True,
            "description": "Ridge Regression (L2, alpha=1)",
        },
        "random_forest": {
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            "log_transform": True,
            "description": "Random Forest (n=300, max_depth=20)",
        },
        "lightgbm": {
            "model": lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "log_transform": True,
            "description": "LightGBM (n=500, lr=0.05, num_leaves=63)",
        },
        "xgboost_reference": {
            "model": XGBRegressor(**LEGACY_XGB_PARAMS),
            "log_transform": True,
            "description": "XGBoost (legacy params — main model reference)",
        },
    }

    # ------------------------------------------------------------------
    # 6. Train, evaluate, and save each model
    # ------------------------------------------------------------------
    results = []

    for name, cfg in models.items():
        model = cfg["model"]
        use_log = cfg["log_transform"]
        desc = cfg["description"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {name}  ({desc})")
        logger.info(f"{'='*60}")

        y_tr = y_train_log if use_log else y_train
        model.fit(X_train_s, y_tr)

        metrics = evaluate(model, X_train_s, y_train, X_val_s, y_val, transform_log=use_log)

        logger.info(
            f"  Val  RMSE={metrics['val_rmse']:.2f} kg  "
            f"MAE={metrics['val_mae']:.2f} kg  "
            f"MAPE={metrics['val_mape']:.2f}%  "
            f"R²={metrics['val_r2']:.4f}"
        )

        # Save model
        save_path = os.path.join(OUTPUT_DIR, f"{name}.joblib")
        joblib.dump(model, save_path)
        logger.info(f"  Saved: {save_path}")

        results.append({"model": name, "description": desc, **metrics})

    # ------------------------------------------------------------------
    # 7. Print and save comparison table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("val_rmse")

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON SUMMARY (sorted by val RMSE)")
    logger.info("=" * 70)
    display_cols = ["model", "val_rmse", "val_mae", "val_mape", "val_r2",
                    "train_rmse", "train_mae", "train_r2"]
    logger.info("\n" + results_df[display_cols].to_string(index=False))

    csv_path = os.path.join(OUTPUT_DIR, "baseline_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nComparison table saved to: {csv_path}")

    # Also save the preprocessor and feature list for reproducibility
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
    with open(os.path.join(OUTPUT_DIR, "features_used.json"), "w") as f:
        json.dump(used_features, f, indent=2)
    logger.info(f"Preprocessor and feature list saved to {OUTPUT_DIR}/")

    logger.info(f"\nDone. All baseline results saved to: {OUTPUT_DIR}")
    return results_df


if __name__ == "__main__":
    run()
