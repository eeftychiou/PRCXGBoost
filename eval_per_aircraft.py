"""
Quick evaluation: per-aircraft and per-phase XGBoost metrics on the val set.
Uses 500 estimators (fast) and the same 80/20 split as the ablation study.
"""
import os, sys, json, time
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import config

DATA_PATH            = config.AUGMENTED_FINAL_CSV
APT_PATH             = config.APT_PARQUET
FLIGHTLIST_PATH      = config.FLIGHTLIST_TRAIN
FUEL_PATH            = config.FUEL_TRAIN
FEATURED_DATA_TRAIN  = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_train.parquet')
FEATURED_DATA_TEST   = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_final.parquet')
SYNTHETIC_PATH       = config.SYNTHETIC_WIDEBODY_PATH
SELECTED_FEATURES_PATH = config.SELECTED_FEATURES_PATH
WIDEBODY_AIRCRAFT    = config.WIDEBODY_AIRCRAFT

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

MODEL_PARAMS = {
    'n_estimators': 1455,
    'learning_rate': 0.02885922756814833,
    'max_depth': 9,
    'min_child_weight': 4,
    'gamma': 6.24155979490078e-08,
    'subsample': 0.9991625118585123,
    'colsample_bytree': 0.6701135673048045,
    'reg_alpha': 0.004878930563988692,
    'reg_lambda': 2.3991563444540384e-08,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}


def haversine(lon1, lat1, lon2, lat2):
    if pd.isna([lon1, lat1, lon2, lat2]).any():
        return np.nan
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371


def rmse(y, yp): return float(np.sqrt(np.mean((np.array(y) - np.array(yp))**2)))
def mae(y, yp):  return float(np.mean(np.abs(np.array(y) - np.array(yp))))
def mape(y, yp): return float(np.mean(np.abs((np.array(y)-np.array(yp))/(np.array(y)+1e-8)))*100)
def r2(y, yp):
    y, yp = np.array(y), np.array(yp)
    return float(1 - np.sum((y-yp)**2)/np.sum((y-y.mean())**2))


print("Loading data...")
apt = pd.read_parquet(APT_PATH)[['icao','longitude','latitude']]
flightlist = pd.read_parquet(FLIGHTLIST_PATH)
flightlist = flightlist.merge(apt, left_on='origin_icao', right_on='icao', how='left')
flightlist = flightlist.rename(columns={'longitude':'origin_lon','latitude':'origin_lat'}).drop(columns=['icao'],errors='ignore')
flightlist = flightlist.merge(apt, left_on='destination_icao', right_on='icao', how='left', suffixes=('','_dest'))
flightlist = flightlist.rename(columns={'longitude':'dest_lon','latitude':'dest_lat'}).drop(columns=['icao','icao_dest'],errors='ignore')
flightlist['great_circle_distance'] = flightlist.apply(
    lambda r: haversine(r.get('origin_lon'), r.get('origin_lat'), r.get('dest_lon'), r.get('dest_lat')), axis=1)

fuel = pd.read_parquet(FUEL_PATH)
featured_data = pd.read_parquet(FEATURED_DATA_TRAIN)
featured_data_final = pd.read_parquet(FEATURED_DATA_TEST)
common_cols = set(featured_data.columns) & set(featured_data_final.columns)
available_features = ['flight_id','idx'] + [c for c in EXTENDED_FEATURES_FROM_PARQUET if c in common_cols]
feat_slim = featured_data[available_features].copy().rename(columns={'idx':'interval_idx'})

df_raw = pd.read_csv(DATA_PATH, delimiter=';', low_memory=False)
print(f"Raw rows: {len(df_raw)}")

flightlist_cols = ['flight_id','takeoff','landed','great_circle_distance','origin_icao','destination_icao','aircraft_type','origin_lon','origin_lat','dest_lon','dest_lat']
df_raw = df_raw.merge(flightlist[[c for c in flightlist_cols if c in flightlist.columns]], on='flight_id', how='left')
fuel_intervals = fuel[['flight_id','idx','fuel_kg','start','end']].copy().rename(columns={'idx':'interval_idx'})
df_raw = df_raw.merge(fuel_intervals, on=['flight_id','interval_idx'], how='left')
df_raw = df_raw.merge(feat_slim, on=['flight_id','interval_idx'], how='left')

base_features = [
    'starting_mass_kg', 'alt_end_ft', 'alt_avg_ft', 'gs_avg_kts', 'vs_avg_fpm',
    'interval_duration_sec', 'altitude_change_rate', 'great_circle_distance',
    'aircraft_type', 'end_hour', 'interval_elapsed_from_flight_start',
]
extended_features_available = [c for c in available_features[2:] if c not in base_features]
feature_cols = base_features + extended_features_available

if 'alt_avg_ft' not in df_raw.columns:
    df_raw['alt_avg_ft'] = (df_raw.get('alt_start_ft', 0) + df_raw.get('alt_end_ft', 0)) / 2
if 'altitude_change_rate' not in df_raw.columns:
    df_raw['altitude_change_rate'] = df_raw.get('alt_change_ft', 0) / (df_raw.get('interval_duration_sec', 60) + 1e-6)
if 'end_hour' not in df_raw.columns:
    df_raw['end_hour'] = pd.to_datetime(df_raw.get('end'), errors='coerce').dt.hour.fillna(-1).astype(int)
if 'interval_elapsed_from_flight_start' not in df_raw.columns:
    df_raw['interval_elapsed_from_flight_start'] = 0

target_col = 'actual_fuel_kg'
feature_cols = [c for c in feature_cols if c in df_raw.columns]

if os.path.exists(SELECTED_FEATURES_PATH):
    with open(SELECTED_FEATURES_PATH) as f:
        feat_data = json.load(f)
    selected = feat_data['selected_features'] if isinstance(feat_data, dict) else feat_data
    sfs_set = set(selected)
    feature_cols = [c for c in feature_cols if c in sfs_set or c == 'aircraft_type']
    print(f"SFS: {len(feature_cols)} features")

# Keep phase from augmented CSV for val breakdown
phase_series = df_raw['phase'].copy() if 'phase' in df_raw.columns else pd.Series(['UNK']*len(df_raw))
openap_series = df_raw['openap_fuel_kg'].copy() if 'openap_fuel_kg' in df_raw.columns else pd.Series([np.nan]*len(df_raw))

df_features = df_raw[feature_cols + [target_col]].copy()
df_features = df_features.dropna(subset=[target_col]).replace([np.inf, -np.inf], np.nan)
valid_idx = df_features.index  # original indices in df_raw for phase/openap lookup

phase_aligned        = phase_series.loc[valid_idx].reset_index(drop=True)
openap_aligned       = openap_series.loc[valid_idx].reset_index(drop=True)
aircraft_aligned     = df_features['aircraft_type'].reset_index(drop=True) if 'aircraft_type' in df_features.columns else pd.Series(['UNK']*len(df_features))
flight_id_aligned    = df_raw['flight_id'].loc[valid_idx].reset_index(drop=True) if 'flight_id' in df_raw.columns else pd.Series([np.nan]*len(df_features))
interval_idx_aligned = df_raw['interval_idx'].loc[valid_idx].reset_index(drop=True) if 'interval_idx' in df_raw.columns else pd.Series([np.nan]*len(df_features))
fuel_kg_aligned      = df_raw['fuel_kg'].loc[valid_idx].reset_index(drop=True) if 'fuel_kg' in df_raw.columns else pd.Series([np.nan]*len(df_features))
df_features = df_features.reset_index(drop=True)

print(f"Real intervals: {len(df_features)}")

# --- Mirror XGBoostTraining exactly: combine real + synthetic first, then split ---
print("Loading synthetic...")
df_synthetic = pd.read_parquet(SYNTHETIC_PATH)
X_synthetic = df_synthetic.reindex(columns=feature_cols, fill_value=np.nan)
y_synthetic = (df_synthetic['actual_fuel_kg'] if 'actual_fuel_kg' in df_synthetic.columns else df_synthetic['fuel_kg']).values.astype(np.float32)

n_real = len(df_features)
n_synth = len(X_synthetic)

X_real = df_features[feature_cols].reset_index(drop=True)
y_real = df_features[target_col].values.astype(np.float32)

# Build combined dataset with an is_real marker to recover real rows after split
X_full = pd.concat([X_real, X_synthetic], ignore_index=True)
y_full = np.concatenate([y_real, y_synthetic]).astype(np.float32)
is_real_mask = np.array([True]*n_real + [False]*n_synth)

# Pad metadata arrays to combined length (synthetic rows get NaN/placeholder)
phase_full        = pd.concat([phase_aligned,        pd.Series(['UNK']*n_synth)], ignore_index=True)
openap_full       = pd.concat([openap_aligned,       pd.Series([np.nan]*n_synth)], ignore_index=True)
aircraft_full     = pd.concat([aircraft_aligned,     pd.Series(['UNK']*n_synth)], ignore_index=True)
flight_id_full    = pd.concat([flight_id_aligned,    pd.Series([np.nan]*n_synth)], ignore_index=True)
interval_idx_full = pd.concat([interval_idx_aligned, pd.Series([np.nan]*n_synth)], ignore_index=True)
fuel_kg_full      = pd.concat([fuel_kg_aligned,      pd.Series([np.nan]*n_synth)], ignore_index=True)

# Same 80/20 split as XGBoostTraining (on combined data, random_state=42)
indices = np.arange(len(X_full))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

X_aug_train = X_full.iloc[train_idx].reset_index(drop=True)
y_aug_train = y_full[train_idx]
X_test_full = X_full.iloc[test_idx].reset_index(drop=True)
y_test_full = y_full[test_idx]

# For evaluation, restrict to real-only rows in the test split
real_in_test = is_real_mask[test_idx]
y_real_val   = y_test_full[real_in_test]

at_val           = aircraft_full.iloc[test_idx].reset_index(drop=True)[real_in_test].reset_index(drop=True)
phase_val        = phase_full.iloc[test_idx].reset_index(drop=True)[real_in_test].reset_index(drop=True)
openap_val       = openap_full.iloc[test_idx].reset_index(drop=True)[real_in_test].reset_index(drop=True)
flight_id_val    = flight_id_full.iloc[test_idx].reset_index(drop=True)[real_in_test].reset_index(drop=True)
interval_idx_val = interval_idx_full.iloc[test_idx].reset_index(drop=True)[real_in_test].reset_index(drop=True)
fuel_kg_val      = fuel_kg_full.iloc[test_idx].reset_index(drop=True)[real_in_test].reset_index(drop=True)

print(f"Combined train: {len(X_aug_train)}  Test split total: {len(X_test_full)}  Real test rows: {real_in_test.sum()}")

# Preprocess
print("Preprocessing...")
numerical_features   = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_aug_train[c])]
categorical_features = [c for c in feature_cols if c not in numerical_features]
nan_cols = [c for c in numerical_features if X_aug_train[c].isna().all()]
numerical_features = [c for c in numerical_features if c not in nan_cols]

num_imp = SimpleImputer(strategy='mean')
cat_imp = SimpleImputer(strategy='most_frequent')
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()

# X_test_real: the real-only rows from the test split (for evaluation)
X_test_real = X_test_full[real_in_test].reset_index(drop=True)

def proc(Xtr, Xte):
    Xtr, Xte = [x.drop(columns=nan_cols, errors='ignore') for x in [Xtr, Xte]]
    if numerical_features:
        Xtr[numerical_features] = num_imp.fit_transform(Xtr[numerical_features])
        Xte[numerical_features] = num_imp.transform(Xte[numerical_features])
    if categorical_features:
        cat_cols = [c for c in categorical_features if c not in nan_cols]
        Xtr[cat_cols] = cat_imp.fit_transform(Xtr[cat_cols])
        Xte[cat_cols] = cat_imp.transform(Xte[cat_cols])
        Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols])
        Xte[cat_cols] = enc.transform(Xte[cat_cols])
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s

X_tr_s, X_te_s = proc(X_aug_train.copy(), X_test_real.copy())
y_au_log = np.log1p(y_aug_train)

print("Training (matches XGBoostTraining split)...")
t0 = time.time()
model = XGBRegressor(**MODEL_PARAMS)
model.fit(X_tr_s, y_au_log)
print(f"Training done in {time.time()-t0:.1f}s")

pred_log = model.predict(X_te_s)
pred  = np.maximum(np.expm1(pred_log), 0)
y_val = y_real_val

print("\n=== OVERALL ===")
print(f"MAE={mae(y_val,pred):.2f}  RMSE={rmse(y_val,pred):.2f}  MAPE={mape(y_val,pred):.2f}%  R2={r2(y_val,pred):.4f}  N={len(y_val)}")

print("\n=== XGBoost by PHASE ===")
for ph in sorted(phase_val.unique()):
    mask = phase_val == ph
    if mask.sum() < 5: continue
    yt, yp = y_val[mask], pred[mask]
    yo = openap_val[mask].values
    valid_oa = ~np.isnan(yo)
    oa_rmse = float(np.sqrt(np.mean((yt[valid_oa]-yo[valid_oa])**2))) if valid_oa.sum()>0 else np.nan
    print(f"  {ph:8s} [N={mask.sum():5d}] XGB: MAE={mae(yt,yp):.1f}  RMSE={rmse(yt,yp):.1f}  MAPE={mape(yt,yp):.2f}%  R2={r2(yt,yp):.4f}  |  OpenAP RMSE={oa_rmse:.1f}")

print("\n=== XGBoost by AIRCRAFT ===")
rows = []
for ac in sorted(at_val.unique()):
    mask = at_val == ac
    if mask.sum() < 5: continue
    yt, yp = y_val[mask], pred[mask]
    yo = openap_val[mask].values
    valid_oa = ~np.isnan(yo)
    oa_rmse = float(np.sqrt(np.mean((yt[valid_oa]-yo[valid_oa])**2))) if valid_oa.sum()>0 else np.nan
    rows.append({
        'Aircraft': ac,
        'N': int(mask.sum()),
        'Mean Actual (kg)': float(np.mean(yt)),
        'XGB MAE': float(mae(yt,yp)),
        'XGB RMSE': float(rmse(yt,yp)),
        'XGB MAPE%': float(mape(yt,yp)),
        'XGB R2': float(r2(yt,yp)),
        'OpenAP RMSE': oa_rmse,
    })
tbl = pd.DataFrame(rows).sort_values('N', ascending=False)
print(tbl.to_string(index=False, float_format='%.1f'))

tbl.to_csv('processed/per_aircraft_xgb_metrics.csv', index=False)
print("\nSaved to processed/per_aircraft_xgb_metrics.csv")

# Row-level predictions CSV for visualization
val_pred_df = pd.DataFrame({
    'flight_id':      flight_id_val.values,
    'interval_idx':   interval_idx_val.values,
    'aircraft_type':  at_val.values,
    'phase':          phase_val.values,
    'fuel_kg':        fuel_kg_val.values,       # from fuel intervals parquet
    'actual_fuel_kg': y_val,                    # from augmented CSV (same value)
    'xgb_pred_kg':    pred,
    'openap_fuel_kg': openap_val.values,
})
val_pred_df.to_csv('processed/val_predictions.csv', index=False)
print("Saved to processed/val_predictions.csv")
