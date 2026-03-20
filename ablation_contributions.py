"""
ablation_contributions.py
=========================
Leave-one-out ablation study for the four claimed design contributions:

  C1 — METAR meteorological features (dep_*/arr_* weather columns added from
        metar_utils via data_preparation; NOT in the hardcoded EXTENDED list,
        so this study quantifies the value of *including* vs. *excluding* them)
  C2 — Load factor & payload estimation features (average_load_factor,
        estimated_payload_kg, trip_fuel_kg, contingency_fuel_kg, …)
  C3 — Dynamic per-segment mass tracking (starting_mass_kg from OpenAP) vs.
        a static MTOW-based mass estimate per aircraft type
  C4 — Timestamp correction step: corrected_flightlist_train.parquet takeoff
        vs. raw flightlist_train.parquet takeoff when computing
        interval_elapsed_from_flight_start

All five conditions use:
  • identical XGBoost hyperparameters (production legacy params)
  • identical 80/20 train/val split (random_state=42)
  • identical preprocessing (SimpleImputer + OrdinalEncoder + StandardScaler)
  • evaluation on the same real-data-only validation set

"Full model" is defined here as all features including METAR (dep_/arr_*
columns from featured_data_train.parquet), which is a superset of what the
production code currently uses.  This gives each contribution a fair stage.

Output: processed/ablation_contributions_results.csv
        (one row per condition per split: Overall / Narrowbody / Widebody)
"""
import os
import json
import logging
import argparse
import time
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import config

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'ablation_contributions.log'), mode='w'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH           = config.AUGMENTED_FINAL_CSV
APT_PATH            = config.APT_PARQUET
FLIGHTLIST_RAW_PATH = config.FLIGHTLIST_TRAIN                        # uncorrected
FLIGHTLIST_COR_PATH = os.path.join(config.PROCESSED_DATA_DIR,
                                   'corrected_flightlist_train.parquet')
FUEL_PATH           = config.FUEL_TRAIN
FEATURED_TRAIN_PATH = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
FEATURED_RANK_PATH  = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_rank.parquet')
SYNTHETIC_PATH      = config.SYNTHETIC_WIDEBODY_PATH

OUTPUT_CSV = os.path.join(config.PROCESSED_DATA_DIR, 'ablation_contributions_results.csv')

WIDEBODY_AIRCRAFT = config.WIDEBODY_AIRCRAFT
AIRCRAFT_DATA     = config.AIRCRAFT_DATA     # {icao_code: {mtow_kg: ..., oew_kg: ...}}

# ── Production-identical hyperparameters ──────────────────────────────────────
MODEL_PARAMS = {
    'n_estimators':     1455,
    'learning_rate':    0.02885922756814833,
    'max_depth':        9,
    'min_child_weight': 4,
    'gamma':            6.24155979490078e-08,
    'subsample':        0.9991625118585123,
    'colsample_bytree': 0.6701135673048045,
    'reg_alpha':        0.004878930563988692,
    'reg_lambda':       2.3991563444540384e-08,
    'objective':        'reg:squarederror',
    'tree_method':      'hist',
    'random_state':     42,
    'n_jobs':           -1,
    'verbosity':        0,
}

# ── Feature groups ────────────────────────────────────────────────────────────
#  (columns that are definitely load-factor-derived)
LOAD_FACTOR_FEATURES = [
    'average_load_factor', 'estimated_payload_kg',
    'trip_fuel_kg', 'contingency_fuel_kg', 'final_reserve_fuel_kg',
    'estimated_total_fuel_kg', 'estimated_takeoff_mass',
]

BASE_FEATURES = [
    'starting_mass_kg', 'alt_end_ft', 'alt_avg_ft', 'gs_avg_kts', 'vs_avg_fpm',
    'interval_duration_sec', 'altitude_change_rate', 'great_circle_distance',
    'aircraft_type', 'end_hour', 'interval_elapsed_from_flight_start',
]


# ── Utilities ─────────────────────────────────────────────────────────────────
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


def _metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    r2   = float(1 - np.sum((y_true - y_pred) ** 2) /
                 np.sum((y_true - np.mean(y_true)) ** 2))
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2)


def collect_metrics(condition_name, y_true, y_pred, is_wb):
    rows = []
    rows.append({'condition': condition_name, 'split': 'Overall',
                 'n_segments': len(y_true), **_metrics(y_true, y_pred)})
    nb_mask = ~is_wb
    if nb_mask.sum() > 0:
        rows.append({'condition': condition_name, 'split': 'Narrowbody',
                     'n_segments': int(nb_mask.sum()),
                     **_metrics(y_true[nb_mask], y_pred[nb_mask])})
    if is_wb.sum() > 0:
        rows.append({'condition': condition_name, 'split': 'Widebody',
                     'n_segments': int(is_wb.sum()),
                     **_metrics(y_true[is_wb], y_pred[is_wb])})
    return rows


def preprocess(X_train_df, X_val_df, feature_cols):
    """Fit on train; transform train and val. Returns scaled numpy arrays."""
    X_tr = X_train_df[feature_cols].copy()
    X_vl = X_val_df[feature_cols].copy()

    nan_cols = [c for c in feature_cols if X_tr[c].isna().all()]
    if nan_cols:
        feature_cols = [c for c in feature_cols if c not in nan_cols]
        X_tr = X_tr.drop(columns=nan_cols)
        X_vl = X_vl.drop(columns=nan_cols)

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_tr[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    if num_cols:
        num_imp = SimpleImputer(strategy='mean')
        X_tr[num_cols] = num_imp.fit_transform(X_tr[num_cols])
        X_vl[num_cols] = num_imp.transform(X_vl[num_cols])
    if cat_cols:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X_tr[cat_cols] = cat_imp.fit_transform(X_tr[cat_cols])
        X_vl[cat_cols] = cat_imp.transform(X_vl[cat_cols])
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
        X_vl[cat_cols] = enc.transform(X_vl[cat_cols])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    return X_tr_s, X_vl_s, feature_cols


def train_and_eval(X_tr_s, X_vl_s, y_train, y_val, is_wb_val, condition_name, gpu_id):
    params = {**MODEL_PARAMS, 'device': f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'}
    log.info(f"  Training [{condition_name}] on {X_tr_s.shape[0]:,} samples, "
             f"{X_tr_s.shape[1]} features …")
    t0 = time.time()
    model = XGBRegressor(**params)
    model.fit(X_tr_s, np.log1p(y_train))
    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s")

    y_pred = np.maximum(np.expm1(model.predict(X_vl_s)), 0)
    rows = collect_metrics(condition_name, y_val, y_pred, is_wb_val)

    for r in rows:
        if r['split'] == 'Overall':
            log.info(f"  {condition_name:<35s}  MAE={r['mae']:.1f} kg  "
                     f"RMSE={r['rmse']:.1f} kg  MAPE={r['mape']:.2f}%  R²={r['r2']:.4f}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def run(gpu_id=None):
    log.info("=" * 72)
    log.info("ABLATION: Claimed Design Contributions (C1–C4)")
    log.info("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("\n[1] Loading data …")

    apt = pd.read_parquet(APT_PATH)[['icao', 'longitude', 'latitude']]

    # Raw and corrected flightlists (for C4)
    fl_raw = pd.read_parquet(FLIGHTLIST_RAW_PATH)
    fl_raw = fl_raw.merge(apt, left_on='origin_icao', right_on='icao', how='left')
    fl_raw = fl_raw.rename(columns={'longitude': 'origin_lon', 'latitude': 'origin_lat'})
    fl_raw = fl_raw.drop(columns=['icao'], errors='ignore')
    fl_raw = fl_raw.merge(apt, left_on='destination_icao', right_on='icao', how='left')
    fl_raw = fl_raw.rename(columns={'longitude': 'dest_lon', 'latitude': 'dest_lat'})
    fl_raw = fl_raw.drop(columns=['icao'], errors='ignore')
    fl_raw['great_circle_distance'] = fl_raw.apply(
        lambda r: haversine(r.get('origin_lon'), r.get('origin_lat'),
                            r.get('dest_lon'), r.get('dest_lat')), axis=1)

    # Timestamp correction lookup (C4)
    ts_correction_available = os.path.exists(FLIGHTLIST_COR_PATH)
    if ts_correction_available:
        fl_cor = pd.read_parquet(FLIGHTLIST_COR_PATH)[['flight_id', 'takeoff']].copy()
        fl_cor = fl_cor.rename(columns={'takeoff': 'corrected_takeoff'})
        fl_raw_ts = fl_raw[['flight_id', 'takeoff']].copy().rename(
            columns={'takeoff': 'raw_takeoff'})
        ts_lookup = fl_cor.merge(fl_raw_ts, on='flight_id', how='inner')
        ts_lookup['corrected_takeoff'] = pd.to_datetime(
            ts_lookup['corrected_takeoff'], errors='coerce', utc=True)
        ts_lookup['raw_takeoff'] = pd.to_datetime(
            ts_lookup['raw_takeoff'], errors='coerce', utc=True)
        ts_lookup['correction_sec'] = (
            ts_lookup['corrected_takeoff'] - ts_lookup['raw_takeoff']
        ).dt.total_seconds().fillna(0)
        ts_lookup = ts_lookup[['flight_id', 'correction_sec']]
        log.info(f"  Timestamp correction lookup: {len(ts_lookup):,} flights  "
                 f"(median offset = {ts_lookup['correction_sec'].median():.1f}s, "
                 f"max |offset| = {ts_lookup['correction_sec'].abs().max():.1f}s)")
    else:
        ts_correction_available = False
        log.warning(f"  corrected_flightlist_train.parquet not found (C4 will be skipped): "
                    f"{FLIGHTLIST_COR_PATH}")

    fuel = pd.read_parquet(FUEL_PATH)
    fuel_intervals = (fuel[['flight_id', 'idx', 'fuel_kg', 'start', 'end']]
                      .copy()
                      .rename(columns={'idx': 'interval_idx'}))

    # Featured-data parquet — load ALL columns (not filtered to EXTENDED list)
    featured_train = pd.read_parquet(FEATURED_TRAIN_PATH)
    featured_rank  = pd.read_parquet(FEATURED_RANK_PATH)
    common_parquet_cols = set(featured_train.columns) & set(featured_rank.columns)
    # All columns present in both train and rank parquets (minus id cols)
    extended_cols = [c for c in featured_train.columns
                     if c in common_parquet_cols and c not in ('flight_id', 'idx')]
    feat_slim = featured_train[['flight_id', 'idx'] + extended_cols].copy()
    feat_slim = feat_slim.rename(columns={'idx': 'interval_idx'})
    log.info(f"  Featured parquet: {len(extended_cols)} common extended columns "
             f"(incl. METAR dep_/arr_* if present)")

    # Identify METAR columns
    metar_cols = [c for c in extended_cols if c.startswith('dep_') or c.startswith('arr_')]
    log.info(f"  METAR columns detected: {len(metar_cols)}")
    if metar_cols:
        log.info(f"    Sample: {metar_cols[:8]}")

    # Load augmented CSV (source of starting_mass_kg — dynamic mass tracking)
    log.info(f"  Loading augmented CSV: {DATA_PATH} …")
    df_raw = pd.read_csv(DATA_PATH, delimiter=';', low_memory=False)
    log.info(f"  Augmented CSV: {len(df_raw):,} rows")

    fl_cols = ['flight_id', 'takeoff', 'landed', 'great_circle_distance',
               'origin_icao', 'destination_icao', 'aircraft_type',
               'origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']
    # Only bring in columns that don't already exist in df_raw to avoid
    # pandas adding _x/_y suffixes which would drop the bare column name.
    fl_merge_cols = ['flight_id'] + [c for c in fl_cols
                                     if c != 'flight_id'
                                     and c in fl_raw.columns
                                     and c not in df_raw.columns]
    df_raw = df_raw.merge(fl_raw[fl_merge_cols], on='flight_id', how='left')
    df_raw = df_raw.merge(fuel_intervals, on=['flight_id', 'interval_idx'], how='left')
    df_raw = df_raw.merge(feat_slim, on=['flight_id', 'interval_idx'], how='left')

    # Computed columns (same guard as XGBoostTraining_Testing.py)
    if 'alt_avg_ft' not in df_raw.columns:
        df_raw['alt_avg_ft'] = (df_raw.get('alt_start_ft', 0) + df_raw.get('alt_end_ft', 0)) / 2
    if 'altitude_change_rate' not in df_raw.columns:
        df_raw['altitude_change_rate'] = (df_raw.get('alt_change_ft', 0) /
                                          (df_raw.get('interval_duration_sec', 60) + 1e-6))
    if 'end_hour' not in df_raw.columns:
        if 'end' in df_raw.columns:
            df_raw['end_hour'] = (pd.to_datetime(df_raw['end'], errors='coerce')
                                  .dt.hour.fillna(-1).astype(int))
        else:
            df_raw['end_hour'] = -1
    if 'interval_elapsed_from_flight_start' not in df_raw.columns:
        df_raw['interval_elapsed_from_flight_start'] = 0

    # ── C4: pre-compute the "no-correction" elapsed time ─────────────────────
    # raw_elapsed = corrected_elapsed - correction_sec
    # (correction_sec = corrected_takeoff - raw_takeoff; subtract it to undo)
    if ts_correction_available:
        df_raw = df_raw.merge(ts_lookup, on='flight_id', how='left')
        df_raw['interval_elapsed_raw'] = (
            df_raw['interval_elapsed_from_flight_start']
            - df_raw['correction_sec'].fillna(0)
        )
        df_raw = df_raw.drop(columns=['correction_sec'])

    # ── C3: pre-compute static MTOW mass per row ──────────────────────────────
    default_mtow = np.median([v['mtow_kg'] for v in AIRCRAFT_DATA.values()])
    if 'aircraft_type' not in df_raw.columns:
        # Fall back to fl_raw lookup if the column was lost in merges
        df_raw = df_raw.merge(
            fl_raw[['flight_id', 'aircraft_type']].drop_duplicates('flight_id'),
            on='flight_id', how='left')
    df_raw['static_mass_kg'] = df_raw['aircraft_type'].map(
        {k: v['mtow_kg'] for k, v in AIRCRAFT_DATA.items()}
    ).fillna(default_mtow)

    # ── Build full feature column list ────────────────────────────────────────
    ext_available = [c for c in extended_cols if c not in BASE_FEATURES]
    full_feature_cols = BASE_FEATURES + ext_available
    full_feature_cols = [c for c in full_feature_cols if c in df_raw.columns]

    target_col = 'actual_fuel_kg'
    df_all = df_raw[full_feature_cols + [target_col, 'flight_id']].copy()
    df_all = df_all.dropna(subset=[target_col])
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    log.info(f"  Usable intervals: {len(df_all):,}  |  Total features: {len(full_feature_cols)}")

    # ── 2. Append cached synthetic widebody data (same as production) ─────────
    if os.path.exists(SYNTHETIC_PATH):
        df_syn = pd.read_parquet(SYNTHETIC_PATH)
        syn_cols_use = [c for c in full_feature_cols + [target_col] if c in df_syn.columns]
        for missing in [c for c in full_feature_cols if c not in df_syn.columns]:
            df_syn[missing] = np.nan
        df_syn['flight_id'] = -1  # placeholder
        df_syn_aligned = df_syn.reindex(columns=full_feature_cols + [target_col, 'flight_id'],
                                        fill_value=np.nan)
        # For synthetic rows: static_mass_kg already in df_all-only cols — fill if needed
        if 'static_mass_kg' in full_feature_cols and 'static_mass_kg' not in df_syn.columns:
            df_syn_aligned['static_mass_kg'] = default_mtow
        if 'interval_elapsed_raw' in full_feature_cols and 'interval_elapsed_raw' not in df_syn.columns:
            df_syn_aligned['interval_elapsed_raw'] = df_syn_aligned.get(
                'interval_elapsed_from_flight_start', pd.Series(0, index=df_syn_aligned.index))
        df_all = pd.concat([df_all, df_syn_aligned], ignore_index=True)
        log.info(f"  +{len(df_syn):,} synthetic rows → total {len(df_all):,}")
    else:
        log.warning("  Synthetic widebody cache not found — using real data only.")

    # ── 3. 80/20 split (real data only for val evaluation) ───────────────────
    log.info("\n[2] Splitting data (80/20, random_state=42) …")
    n_real = (df_all['flight_id'] != -1).sum()  # synthetic have flight_id=-1

    all_idx = np.arange(len(df_all))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42, shuffle=True)

    # Validation = real rows only from val_idx
    val_is_real = df_all['flight_id'].iloc[val_idx].values != -1
    val_idx_real = val_idx[val_is_real]

    at_val = df_all['aircraft_type'].iloc[val_idx_real].values
    is_wb_val = np.isin(at_val, WIDEBODY_AIRCRAFT)
    y_val = df_all[target_col].iloc[val_idx_real].values.astype(np.float32)

    log.info(f"  Train (incl. synthetic): {len(train_idx):,}  "
             f"| Real-only val: {len(val_idx_real):,}  "
             f"(widebody segments: {is_wb_val.sum():,})")

    X_df = df_all[full_feature_cols].reset_index(drop=True)
    y_arr = df_all[target_col].values.astype(np.float32)

    X_train_df  = X_df.iloc[train_idx]
    X_val_df    = X_df.iloc[val_idx_real]
    y_train     = y_arr[train_idx]

    # ── 4. Define ablation conditions ─────────────────────────────────────────
    # Each condition is a (label, feature_cols, X_train_df_variant, X_val_df_variant)
    # We rebuild variants by column-masking + optional value substitution.

    lf_present  = [c for c in LOAD_FACTOR_FEATURES if c in full_feature_cols]
    met_present = [c for c in full_feature_cols if c.startswith('dep_') or c.startswith('arr_')]

    conditions = []

    # ── A: Full model ─────────────────────────────────────────────────────────
    conditions.append(('Full model (all contributions)', full_feature_cols,
                       X_train_df.copy(), X_val_df.copy()))

    # ── B: No METAR (C1) ─────────────────────────────────────────────────────
    if met_present:
        fc_no_metar = [c for c in full_feature_cols if c not in met_present]
        conditions.append(('No METAR features (–C1)', fc_no_metar,
                           X_train_df.copy(), X_val_df.copy()))
        log.info(f"  C1: removing {len(met_present)} METAR columns")
    else:
        log.warning("  C1 (METAR): no dep_*/arr_* columns found in featured parquet — "
                    "ensure prepare_metars + prepare_data have been run.  Skipping C1 ablation.")

    # ── C: No load factor (C2) ────────────────────────────────────────────────
    if lf_present:
        fc_no_lf = [c for c in full_feature_cols if c not in lf_present]
        conditions.append(('No load-factor features (–C2)', fc_no_lf,
                           X_train_df.copy(), X_val_df.copy()))
        log.info(f"  C2: removing {len(lf_present)} load-factor columns")
    else:
        log.warning("  C2 (load factor): none of the load-factor columns found in data. "
                    "Skipping C2 ablation.")

    # ── D: Static mass / no dynamic tracking (C3) ─────────────────────────────
    if 'starting_mass_kg' in full_feature_cols and 'static_mass_kg' in df_all.columns:
        X_train_static = X_train_df.copy()
        X_val_static   = X_val_df.copy()
        X_train_static['starting_mass_kg'] = (
            df_all['static_mass_kg'].iloc[train_idx].values)
        X_val_static['starting_mass_kg'] = (
            df_all['static_mass_kg'].iloc[val_idx_real].values)
        conditions.append(('Static MTOW mass (–C3 dynamic tracking)',
                           full_feature_cols,
                           X_train_static, X_val_static))
        log.info("  C3: starting_mass_kg replaced with MTOW per aircraft type")
    else:
        log.warning("  C3 (dynamic mass): 'starting_mass_kg' not in features.  Skipping.")

    # ── E: No timestamp correction (C4) ───────────────────────────────────────
    if ts_correction_available and 'interval_elapsed_raw' in df_all.columns:
        X_train_rawts = X_train_df.copy()
        X_val_rawts   = X_val_df.copy()
        X_train_rawts['interval_elapsed_from_flight_start'] = (
            df_all['interval_elapsed_raw'].iloc[train_idx].values)
        X_val_rawts['interval_elapsed_from_flight_start'] = (
            df_all['interval_elapsed_raw'].iloc[val_idx_real].values)
        conditions.append(('Raw timestamps, no correction (–C4)',
                           full_feature_cols,
                           X_train_rawts, X_val_rawts))
        log.info("  C4: interval_elapsed_from_flight_start reverted to raw-flightlist timestamps")
    else:
        log.warning("  C4 (timestamp correction): corrected flightlist or 'interval_elapsed_raw' "
                    "not available. Skipping.")

    # ── 5. Train & evaluate all conditions ────────────────────────────────────
    log.info(f"\n[3] Running {len(conditions)} ablation condition(s)…")
    all_rows = []
    for label, feat_cols, X_tr_variant, X_vl_variant in conditions:
        log.info(f"\n{'─'*60}")
        log.info(f"  Condition: {label}")
        log.info(f"  Features:  {len(feat_cols)}")
        X_tr_s, X_vl_s, used_cols = preprocess(X_tr_variant, X_vl_variant, feat_cols)
        rows = train_and_eval(X_tr_s, X_vl_s, y_train, y_val, is_wb_val, label, gpu_id)
        all_rows.extend(rows)

    # ── 6. Print summary ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)

    log.info("\n" + "=" * 90)
    log.info(f"{'Condition':<40} {'Split':<14} {'MAE':>10} {'RMSE':>10} "
             f"{'MAPE':>8} {'R²':>8} {'N':>8}")
    log.info("-" * 90)
    for _, r in results_df.iterrows():
        log.info(
            f"{r['condition']:<40} {r['split']:<14} {r['mae']:>10.1f} "
            f"{r['rmse']:>10.1f} {r['mape']:>7.2f}% {r['r2']:>8.4f} "
            f"{int(r['n_segments']):>8,}"
        )
    log.info("=" * 90)

    # Δ-MAE relative to full model
    log.info("\nMAE delta vs. Full model (Overall split):")
    full_mae = results_df[
        (results_df['condition'].str.startswith('Full')) &
        (results_df['split'] == 'Overall')
    ]['mae'].values
    if len(full_mae) > 0:
        full_mae = full_mae[0]
        for _, r in results_df[results_df['split'] == 'Overall'].iterrows():
            if not r['condition'].startswith('Full'):
                delta = r['mae'] - full_mae
                pct   = delta / full_mae * 100
                direction = 'degradation' if delta > 0 else 'improvement'
                log.info(f"  {r['condition']:<40} ΔMAE={delta:+.1f} kg "
                         f"({pct:+.2f}%)  ← {direction}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\n[+] Results saved to: {OUTPUT_CSV}")
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ablation study: claimed design contributions (C1–C4)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device id (e.g. 0). Omit for CPU.')
    args = parser.parse_args()
    run(gpu_id=args.gpu)
