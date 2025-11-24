import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import config
import json
import logging
import datetime

# Setup logging for this module
logger = logging.getLogger(__name__)
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', f'feature_selection_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                         logging.FileHandler(log_file, mode='w'),
                         logging.StreamHandler()
                     ])

def load_data_for_feature_selection(stage='train'):
    """
    Load data for a given stage, handling test/full run mode.
    This function is specific to feature selection and ensures consistency.
    """
    if config.TEST_RUN:
        logger.info("TEST RUN active: loading a fraction of the data for feature selection.")
        file_name = f"featured_data_{stage}_test.parquet"
    else:
        file_name = f"featured_data_{stage}.parquet"
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for stage '{stage}' not found at {data_path}. Please run the data preparation pipeline first.")

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    return df

def select_features(model_type='xgb', feature_selection_method='sfs_forward'):
    """
    Performs feature selection and saves the selected features.
    
    Args:
        model_type (str): The type of model to use for selection ('xgb', 'gbr', 'rf').
        feature_selection_method (str): 'sfs_forward', 'sfs_backward', 'importance', or 'none'.
    """
    print(f"--- Starting Feature Selection Stage ({model_type.upper()} with {feature_selection_method.replace('_', ' ').upper()}) ---")

    # 1. Load Data
    df_train = load_data_for_feature_selection('train')
    
    target_col = 'fuel_kg'
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in the training data.")

    df_train.dropna(subset=[target_col], inplace=True)
    
    # Select all features except the target
    feature_cols = [col for col in df_train.columns if col != target_col]
    X = df_train[feature_cols]
    y = df_train[target_col]
    y_log = np.log1p(y)

    # 2. Preprocessing (Aligned with training scripts)
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"One-hot encoding categorical features: {categorical_cols}")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)
    
    # Handle all-NaN columns after encoding
    all_nan_cols = X_encoded.columns[X_encoded.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"The following columns are entirely NaN and will be dropped: {all_nan_cols}")
        X_encoded = X_encoded.drop(columns=all_nan_cols)

    # Re-identify numeric columns after encoding and dropping
    processed_numeric_cols = X_encoded.select_dtypes(include=np.number).columns.tolist()

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_processed = X_encoded.copy()
    X_processed[processed_numeric_cols] = imputer.fit_transform(X_processed[processed_numeric_cols])
    X_processed[processed_numeric_cols] = scaler.fit_transform(X_processed[processed_numeric_cols])
    
    logger.info(f"Data preprocessed. Shape after encoding and scaling: {X_processed.shape}")

    # 3. Perform Feature Selection
    if model_type == 'xgb':
        estimator = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
    elif model_type == 'gbr':
        estimator = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    elif model_type == 'rf':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model type for feature selection: {model_type}")

    selected_feature_names = []
    if feature_selection_method in ['sfs_forward', 'sfs_backward']:
        direction = 'forward' if feature_selection_method == 'sfs_forward' else 'backward'
        print(f"\n--- Performing Sequential Feature Selection ({direction.capitalize()}) ---")
        
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select='auto',
            direction=direction,
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1
        )
        
        print(f"Fitting SFS ({direction}) to find optimal features (this may take a while)...")
        sfs.fit(X_processed, y_log)
        
        selected_feature_names = list(X_processed.columns[sfs.get_support()])
        print(f"SFS selected {len(selected_feature_names)} features out of {len(X_processed.columns)}.")

    elif feature_selection_method == 'importance':
        print("\n--- Performing Feature Selection based on Model Importance ---")
        print("Training model once to get feature importances...")
        estimator.fit(X_processed, y_log)
        
        importances = estimator.feature_importances_
        # Select features with importance greater than a small threshold
        selected_feature_names = list(X_processed.columns[importances > 1e-5])
        print(f"Selected {len(selected_feature_names)} features out of {len(X_processed.columns)} based on importance > 1e-5.")

    elif feature_selection_method == 'none':
        print("\n--- Skipping feature selection. Using all available features. ---")
        selected_feature_names = X_processed.columns.tolist()
    
    else:
        raise ValueError(f"Invalid feature_selection_method: '{feature_selection_method}'.")

    # 4. CRITICAL STEP: Map one-hot encoded names back to original feature names
    # The training script expects the original names to start its own preprocessing.
    original_features_selected = set()
    for feature_name in selected_feature_names:
        # Check if the feature is an encoded categorical column (e.g., 'aircraft_type_A320')
        is_categorical = False
        for cat_col in categorical_cols:
            if feature_name.startswith(f"{cat_col}_"):
                original_features_selected.add(cat_col)
                is_categorical = True
                break
        # If it's not a categorical column, it must be a numeric one
        if not is_categorical:
            original_features_selected.add(feature_name)
            
    final_feature_list = sorted(list(original_features_selected))
    logger.info(f"Mapped {len(selected_feature_names)} selected features back to {len(final_feature_list)} original feature names.")

    # 5. Save Selected Original Features
    feature_sets_dir = os.path.join(config.PROCESSED_DATA_DIR, 'feature_sets')
    os.makedirs(feature_sets_dir, exist_ok=True)
    
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"selected_features_{model_type}_{feature_selection_method}_{run_id}.json"
    output_path = os.path.join(feature_sets_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(final_feature_list, f, indent=4)
    
    print(f"Selected original features saved to {output_path}")
    print("--- Feature Selection Stage Complete ---")

if __name__ == '__main__':
    pass
