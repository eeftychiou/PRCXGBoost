import os
import pandas as pd
import numpy as np
import json
import joblib
from tqdm import tqdm
import config
import logging
import augment_features
import data_preparation  # reuse drop list and imputation policy
import train_xgb  # reuse engineer_features and training-time encoders

# --- Setup Logging ---
log_file = 'create_submission.log'
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def find_latest_model_dir(models_base_dir):
    """Finds the most recently created model directory."""
    all_model_dirs = [os.path.join(models_base_dir, d) for d in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, d))]
    if not all_model_dirs:
        return None
    latest_dir = max(all_model_dirs, key=os.path.getmtime)
    return latest_dir

def main():
    """Fills the fuel_rank_submission.parquet file with predictions."""
    try:
        # --- 1. Load and Prepare Data (mirroring data_preparation.py) ---
        logging.info("--- Starting Submission Creation ---")
        
        # Try to load cached processed ranking features
        cache_path = os.path.join(config.PROCESSED_DATA_DIR, 'featured_data_rank.parquet')
        df_featured = None
        built_now = False
        if os.path.exists(cache_path):
            try:
                logging.info(f"Found cached processed ranking features at {cache_path}. Loading...")
                df_featured = pd.read_parquet(cache_path)
                logging.info(f"Loaded cached features with shape {df_featured.shape}.")
            except Exception as e:
                logging.warning(f"Failed to load cached features due to: {e}. Will rebuild.")

        if df_featured is None:
            # Load raw data
            logging.info("Loading raw data for ranking...")
            df_fuel_rank = pd.read_parquet(os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet'))
            df_flightlist_rank = pd.read_parquet(os.path.join(config.BASE_DATASETS_DIR, 'flightlist_rank.parquet'))
            df_ac_perf = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'acPerfOpenAP.csv'))
            df_apt = pd.read_parquet(os.path.join(config.BASE_DATASETS_DIR, 'apt.parquet'))

            # Preprocess and merge
            logging.info("Preprocessing and merging datasets...")
            df_merged = pd.merge(df_flightlist_rank, df_ac_perf, left_on='aircraft_type', right_on='ICAO_TYPE_CODE', how='left')
            df_merged = pd.merge(df_fuel_rank, df_merged, on='flight_id', how='left')
            df_merged = pd.merge(df_merged, df_apt.add_prefix('origin_'), left_on='origin_icao', right_on='origin_icao', how='left')
            df_merged = pd.merge(df_merged, df_apt.add_prefix('destination_'), left_on='destination_icao', right_on='destination_icao', how='left')
            
            df_merged['start'] = pd.to_datetime(df_merged['start'])
            df_merged['end'] = pd.to_datetime(df_merged['end'])
            df_merged.reset_index(drop=True, inplace=True)

            # Augment features
            logging.info("Augmenting features for ranking data...")
            interpolated_flights_rank_dir = os.path.join(config.INTERPOLATED_TRAJECTORIES_DIR, 'flights_rank')
            df_featured = augment_features.augment_features(df_merged, interpolated_flights_rank_dir)
            built_now = True



        # Align with data_preparation.py: drop high-missing columns and impute using same policy
        logging.info("Dropping columns with high missing values (aligned with data_preparation)...")
        cols_to_drop = data_preparation._get_columns_to_drop()
        cols_to_drop_existing = [c for c in cols_to_drop if c in df_featured.columns]
        if cols_to_drop_existing:
            df_featured.drop(columns=cols_to_drop_existing, inplace=True)
            logging.info(f"Dropped {len(cols_to_drop_existing)} columns.")

        # Impute trajectory-related numeric columns with median (same as data_preparation)
        logging.info("Imputing missing values (aligned with data_preparation)...")
        trajectory_cols = [
            'std_vertical_rate', 'ending_altitude', 'altitude_difference', 'mean_track', 'std_track',
            'starting_altitude', 'mean_vertical_rate', 'mean_dist_to_origin_km', 'mean_dist_to_dest_km'
        ]
        for col in trajectory_cols:
            if col in df_featured.columns and df_featured[col].isnull().any():
                median_val = df_featured[col].median()
                df_featured[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed missing values in '{col}' with median value: {median_val}")

        # Impute all numeric seg_* columns with median
        seg_cols = [c for c in df_featured.columns if c.startswith('seg_')]
        for col in seg_cols:
            if pd.api.types.is_numeric_dtype(df_featured[col]) and df_featured[col].isnull().any():
                median_val = df_featured[col].median()
                df_featured[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed missing values in segment feature '{col}' with median value: {median_val}")

        # Impute aircraft categorical columns with mode
        aircraft_cols = ['engine_default', 'flaps_type', 'engine_mount']
        for col in aircraft_cols:
            if col in df_featured.columns and df_featured[col].isnull().any():
                try:
                    mode_val = df_featured[col].mode(dropna=True)[0]
                except Exception:
                    mode_val = None
                if mode_val is not None:
                    df_featured[col].fillna(mode_val, inplace=True)
                    logging.info(f"Imputed missing values in '{col}' with mode value: {mode_val}")

        # Save processed ranking features to cache (overwrite if we rebuilt them now)
        try:
            if built_now:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df_featured.to_parquet(cache_path, index=False)
                logging.info(f"Saved processed ranking features to cache at {cache_path}")
        except Exception as e:
            logging.warning(f"Could not save processed ranking features to cache due to: {e}")


        # --- 2. Load Model and Selected Features ---
        logging.info("Finding the latest trained model...")
        latest_model_dir = find_latest_model_dir(config.MODELS_DIR)
        
        if not latest_model_dir:
            logging.error(f"No trained models found in '{config.MODELS_DIR}'. Please train a model first.")
            return
        
        logging.info(f"Using model from: {os.path.basename(latest_model_dir)}")

        # Load the feature list (either 'selected_features.json' or 'features.json')
        features_path = os.path.join(latest_model_dir, "selected_features.json")
        if not os.path.exists(features_path):
            features_path = os.path.join(latest_model_dir, "features.json")
        
        if not os.path.exists(features_path):
            logging.error(f"Feature list ('selected_features.json' or 'features.json') not found in {latest_model_dir}.")
            return

        logging.info(f"Loading feature list from {features_path}...")
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
        
        # Load the model
        model_path = os.path.join(latest_model_dir, "model.joblib")
        if not os.path.exists(model_path):
            logging.error(f"Model file 'model.joblib' not found in {latest_model_dir}.")
            return
            
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        # Try to load training-time preprocessors and spec
        preproc_path = os.path.join(latest_model_dir, "preprocessors.joblib")
        spec_path = os.path.join(latest_model_dir, "preprocessing_spec.json")
        use_artifacts = False
        label_encoders = None
        if os.path.exists(preproc_path) and os.path.exists(spec_path):
            try:
                logging.info("Loading preprocessing artifacts and spec from training run...")
                preprocessors = joblib.load(preproc_path)
                with open(spec_path, 'r') as f:
                    spec = json.load(f)
                num_imputer = preprocessors.get('num_imputer')
                cat_imputer = preprocessors.get('cat_imputer')
                cat_encoder = preprocessors.get('cat_encoder')
                scaler = preprocessors.get('scaler')
                label_encoders = preprocessors.get('label_encoders', {})
                # Basic sanity check
                if scaler is not None:
                    use_artifacts = True
            except Exception as e:
                logging.warning(f"Failed to load preprocessing artifacts/spec: {e}. Will fall back to safe alignment.")

        if use_artifacts:
            # Engineer features with the same label encoders used in training
            df_eng, _ = train_xgb.engineer_features(df_featured.copy(), label_encoders=label_encoders, fit_encoders=True)

            # Ensure all spec features exist in df_eng
            numerical_features = spec.get('numerical_features', [])
            categorical_features = spec.get('categorical_features', [])
            imputed_feature_order = spec.get('imputed_feature_order', numerical_features + categorical_features)
            scaled_feature_order = spec.get('scaled_feature_order', imputed_feature_order)

            for col in set(imputed_feature_order):
                if col not in df_eng.columns:
                    df_eng[col] = np.nan

            # Apply imputers
            if numerical_features:
                X_num = num_imputer.transform(df_eng[numerical_features])
                df_eng.loc[:, numerical_features] = X_num
            if categorical_features:
                X_cat_imp = cat_imputer.transform(df_eng[categorical_features])
                X_cat_enc = cat_encoder.transform(X_cat_imp)
                df_eng.loc[:, categorical_features] = X_cat_enc

            # Reconstruct matrix in exact order expected by scaler
            X_imputed = df_eng[imputed_feature_order]
            X_scaled_arr = scaler.transform(X_imputed[scaled_feature_order])
            X_scaled = pd.DataFrame(X_scaled_arr, columns=scaled_feature_order, index=df_eng.index)

            # Finally, select only the features the model was trained on
            missing_for_selected = [c for c in selected_features if c not in X_scaled.columns]
            if missing_for_selected:
                logging.warning("Missing columns after preprocessing: " + ", ".join(missing_for_selected) + ". Filling with NaN.")
                for c in missing_for_selected:
                    X_scaled[c] = np.nan
            X_submission = X_scaled[selected_features]
            logging.info("Successfully rebuilt submission feature matrix using training-time preprocessors.")
        else:
            # Fallback path: engineer minimal features and align columns; attempt scaling via X_train stats
            logging.warning("Preprocessing artifacts not found. Falling back to basic alignment path; predictions may be less accurate.")
            df_eng, _ = train_xgb.engineer_features(df_featured.copy(), fit_encoders=True)
            # Replace inf with NaN to avoid issues
            df_eng = df_eng.replace([np.inf, -np.inf], np.nan)

            # Attempt to approximate scaling using training X statistics if available
            train_stats_means = None
            train_stats_stds = None
            x_train_path = os.path.join(latest_model_dir, "X_train.parquet")
            if os.path.exists(x_train_path):
                try:
                    X_train_ref = pd.read_parquet(x_train_path)
                    train_stats_means = X_train_ref.mean(numeric_only=True)
                    train_stats_stds = X_train_ref.std(numeric_only=True).replace(0, 1.0)
                    logging.info("Loaded X_train statistics for approximate scaling in fallback path.")
                except Exception as e:
                    logging.warning(f"Could not load X_train stats for fallback scaling: {e}")

            # Ensure all selected features exist
            missing_cols = [c for c in selected_features if c not in df_eng.columns]
            if missing_cols:
                logging.warning("Missing columns after basic engineering: " + ", ".join(missing_cols) + ". Filling with NaN.")
                for c in missing_cols:
                    df_eng[c] = np.nan

            X_submission = df_eng[selected_features].copy()

            # Ensure all features are numeric for model consumption
            non_numeric_cols = X_submission.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                for c in non_numeric_cols:
                    try:
                        # Use categorical codes to provide a stable numeric representation
                        X_submission[c] = pd.Categorical(X_submission[c].astype(str)).codes
                    except Exception:
                        # Fallback: coerce to numeric, leave non-convertible as NaN
                        X_submission[c] = pd.to_numeric(X_submission[c], errors='coerce')
                logging.info(f"Converted {len(non_numeric_cols)} non-numeric feature(s) to categorical codes for fallback path.")

            # Replace infinities that may have arisen during conversions
            X_submission.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Apply approximate scaling if stats are available (only on numeric columns present in training stats)
            if train_stats_means is not None and train_stats_stds is not None:
                numeric_cols = X_submission.select_dtypes(include=[np.number]).columns
                common_cols = [c for c in numeric_cols if c in train_stats_means.index]
                if common_cols:
                    X_submission.loc[:, common_cols] = (
                        X_submission[common_cols].astype(float)
                        - train_stats_means[common_cols]
                    ) / train_stats_stds[common_cols]
                    logging.info(f"Applied approximate scaling to {len(common_cols)} numeric feature(s) using training stats.")



        # --- 4. Generate Predictions ---
        logging.info("Generating predictions...")
        predictions = model.predict(X_submission)

        # The model is trained on log1p(target); inverse-transform to get fuel_kg
        try:
            predictions_linear = np.expm1(predictions.astype(float))
        except Exception:
            # Fallback in case dtype casting fails unexpectedly
            predictions_linear = np.expm1(pd.to_numeric(pd.Series(predictions), errors='coerce')).values

        # Replace non-finite values and clip to non-negative (fuel cannot be negative)
        finite_mask = np.isfinite(predictions_linear)
        if not finite_mask.all():
            n_bad = (~finite_mask).sum()
            logging.warning(f"Found {n_bad} non-finite prediction(s) after inverse-transform; imputing with median of finite values.")
            finite_vals = predictions_linear[finite_mask]
            if finite_vals.size > 0:
                median_val = np.median(finite_vals)
            else:
                median_val = 0.0
            predictions_linear[~finite_mask] = median_val

        predictions_linear = np.maximum(predictions_linear, 0.0)
        logging.info("Applied inverse log1p transform and non-negativity clipping to predictions.")

        # --- 5. Create and Save Submission File ---
        submission_template_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')
        df_submission = pd.read_parquet(submission_template_path)
        
        # Ensure the prediction order matches the submission template order
        # This assumes df_featured has the same row order as the original submission template
        df_submission['fuel_kg'] = predictions_linear.astype(float)

        output_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission_final.parquet')
        df_submission.to_parquet(output_path, index=False)

        logging.info(f"\nSuccessfully generated {len(df_submission)} predictions.")
        logging.info(f"Submission file saved to: {output_path}")
        logging.info("\nSubmission Head:")
        logging.info(df_submission.head())

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
