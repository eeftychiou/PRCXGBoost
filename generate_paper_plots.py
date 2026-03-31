import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import argparse
import config

# Requires `shap` for SHAP functionality.
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: 'shap' package not found. SHAP plots will be skipped.")

try:
    import plotly.express as px
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("WARNING: 'plotly' package not found. Spatial maps will use matplotlib fallback.")

# Global Matplotlib styling
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 26,
    'axes.titlesize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 28,
    'font.family': 'serif', # Often preferred in academic papers (e.g., LaTeX default)
    'lines.linewidth': 2.5,
    'figure.autolayout': True
})

PLOTS_DIR = "paper_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
PAPER_PLOTS_DIR = PLOTS_DIR # Resolve dependency

def ensure_derived_columns(df):
    """Ensures that derived columns exist and provide aliases for model compatibility."""
    # Parquet to model aliases
    aliases = {
        'estimated_takeoff_mass': 'starting_mass_kg',
        'seg_groundspeed_mean': 'gs_avg_kts',
        'segment_duration': 'interval_duration_sec',
        'great_circle_distance_km': 'great_circle_distance',
        'seg_vertical_rate_mean': 'vs_avg_fpm',
        'seg_altitude_mean': 'alt_avg_ft'
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # 2. Derived Columns
    if 'alt_avg_ft' not in df.columns:
        if 'start_alt_rev' in df.columns and 'end_alt_rev' in df.columns:
            df['alt_avg_ft'] = (df['start_alt_rev'] + df['end_alt_rev']) / 2
            
    if 'altitude_change_rate' not in df.columns:
        # Use altitude delta metrics
        diff = df.get('alt_diff_rev', df.get('alt_change_ft', 0))
        dur = df.get('interval_duration_sec', 60)
        df['altitude_change_rate'] = diff / (dur + 1e-6)
        
    if 'interval_elapsed_from_flight_start' not in df.columns:
        # Default initialization
        df['interval_elapsed_from_flight_start'] = 0.0
            
    return df

def select_model():
    """Helper to select a trained model from the models directory."""
    if not os.path.exists(config.MODELS_DIR):
        print(f"Models directory not found at {config.MODELS_DIR}")
        return None
        
    model_dirs = sorted([d for d in os.listdir(config.MODELS_DIR) if os.path.isdir(os.path.join(config.MODELS_DIR, d))])
    if not model_dirs:
        print("No models found.")
        return None
        
    print("Available Models:")
    for i, md in enumerate(model_dirs):
        print(f"[{i+1}] {md}")
        
    # Priority models
    priority_models = ['final_xgb_model_rank1', 'test_xgb_model_rank1', 'xgb_model_rank1']
    chosen_path = None
    
    # Search root directories
    for p in priority_models:
        if p in model_dirs:
            chosen_path = os.path.join(config.MODELS_DIR, p)
            break
            
    # Search optimization subdirectories
    if not chosen_path:
        for subdir in ['grid', 'optuna', 'legacy']:
            if subdir in model_dirs:
                subpath = os.path.join(config.MODELS_DIR, subdir)
                for p in priority_models:
                    if os.path.exists(os.path.join(subpath, p)):
                        chosen_path = os.path.join(subpath, p)
                        break
                if chosen_path: break
                
    if not chosen_path:
        rank1_dirs = [d for d in model_dirs if 'rank1' in d and 'rank10' not in d]
        if rank1_dirs:
            chosen_path = os.path.join(config.MODELS_DIR, rank1_dirs[0])
        else:
            chosen_path = os.path.join(config.MODELS_DIR, model_dirs[-1])

    print(f"Automatically selected best model: {os.path.basename(chosen_path)}")
    return chosen_path


def load_training_data(drop_na_target='fuel_kg'):
    """
    Loads training data:
      1. Augmented CSV (AUGMENTED_FINAL_CSV) prioritized.
      2. Merges featured_data_train.parquet features.
      3. Computes runtime derived columns.
      4. Fallback to base parquet with aliasing.
    """
    aug_path = config.AUGMENTED_FINAL_CSV
    parquet_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")

    if os.path.exists(aug_path):
        df = pd.read_csv(aug_path, delimiter=';', low_memory=False)
        # The augmented CSV may use 'actual_fuel_kg'
        if 'actual_fuel_kg' in df.columns and 'fuel_kg' not in df.columns:
            df = df.rename(columns={'actual_fuel_kg': 'fuel_kg'})
        # Merge extended features
        if os.path.exists(parquet_path):
            featured = pd.read_parquet(parquet_path)
            if 'idx' in featured.columns and 'interval_idx' not in featured.columns:
                featured = featured.rename(columns={'idx': 'interval_idx'})
            # Only pull in columns not already in df
            extra_cols = ['flight_id', 'interval_idx'] + [
                c for c in featured.columns
                if c not in df.columns and c not in ('idx',)
            ]
            featured_slim = featured[[c for c in extra_cols if c in featured.columns]]
            df = df.merge(featured_slim, on=['flight_id', 'interval_idx'], how='left')

        # Runtime derived columns
        if 'alt_avg_ft' not in df.columns:
            if 'alt_start_ft' in df.columns and 'alt_end_ft' in df.columns:
                df['alt_avg_ft'] = (df['alt_start_ft'] + df['alt_end_ft']) / 2
            elif 'alt_end_ft' in df.columns:
                df['alt_avg_ft'] = df['alt_end_ft']

        if 'altitude_change_rate' not in df.columns:
            if 'alt_change_ft' in df.columns and 'interval_duration_sec' in df.columns:
                df['altitude_change_rate'] = df['alt_change_ft'] / (df['interval_duration_sec'] + 1e-6)
            else:
                df['altitude_change_rate'] = 0.0

        if 'great_circle_distance' not in df.columns:
            try:
                from math import radians, cos, sin, asin, sqrt
                def _haversine(lon1, lat1, lon2, lat2):
                    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [lon1, lat1, lon2, lat2]):
                        return np.nan
                    R = 6371
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                    dlon, dlat = lon2 - lon1, lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    return 2 * R * asin(sqrt(a))
                apt = pd.read_parquet(config.APT_PARQUET)[['icao', 'longitude', 'latitude']]
                fl = pd.read_parquet(config.FLIGHTLIST_TRAIN)[['flight_id', 'origin_icao', 'destination_icao']]
                fl = fl.merge(apt.rename(columns={'icao': 'origin_icao', 'longitude': 'origin_lon', 'latitude': 'origin_lat'}), on='origin_icao', how='left')
                fl = fl.merge(apt.rename(columns={'icao': 'destination_icao', 'longitude': 'dest_lon', 'latitude': 'dest_lat'}), on='destination_icao', how='left')
                fl['great_circle_distance'] = fl.apply(lambda r: _haversine(r.get('origin_lon'), r.get('origin_lat'), r.get('dest_lon'), r.get('dest_lat')), axis=1)
                df = df.merge(fl[['flight_id', 'great_circle_distance']], on='flight_id', how='left')
            except Exception:
                df['great_circle_distance'] = 0.0

        if 'interval_elapsed_from_flight_start' not in df.columns:
            if 'start' in df.columns and 'end' in df.columns:
                try:
                    df['_fs'] = pd.to_datetime(df['start'], errors='coerce')
                    df['_fe'] = pd.to_datetime(df['end'], errors='coerce')
                    fs_min = df.groupby('flight_id')['_fs'].min().rename('_flight_start')
                    df = df.merge(fs_min, on='flight_id', how='left')
                    # Compute elapsed time in hours
                    df.drop(columns=['_fs', '_fe', '_flight_start'], inplace=True)
                except Exception:
                    df['interval_elapsed_from_flight_start'] = 0.0
            else:
                df['interval_elapsed_from_flight_start'] = 0.0

        if 'end_hour' not in df.columns and 'end' in df.columns:
            df['end_hour'] = pd.to_datetime(df['end'], errors='coerce').dt.hour.fillna(-1).astype(int)

    else:
        # Fallback: load base parquet and alias columns
        if not os.path.exists(parquet_path):
            print(f"No training data found (checked {aug_path} and {parquet_path}).")
            return None
        df = pd.read_parquet(parquet_path)
        # Column aliasing
        aliases = {
            'estimated_takeoff_mass': 'starting_mass_kg',
            'seg_groundspeed_mean':   'gs_avg_kts',
            'segment_duration':       'interval_duration_sec',
            'great_circle_distance_km': 'great_circle_distance',
            'seg_vertical_rate_mean': 'vs_avg_fpm',
            'seg_altitude_mean':      'alt_avg_ft',
        }
        for src, dst in aliases.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]
        # Derived columns (fallback parquet path)
        if 'alt_avg_ft' not in df.columns and 'start_alt_rev' in df.columns:
            df['alt_avg_ft'] = (df['start_alt_rev'] + df.get('end_alt_rev', df['start_alt_rev'])) / 2
        if 'altitude_change_rate' not in df.columns:
            diff = df.get('alt_diff_rev', df.get('alt_change_ft', 0))
            dur  = df.get('interval_duration_sec', 60)
            df['altitude_change_rate'] = diff / (dur + 1e-6)
        if 'interval_elapsed_from_flight_start' not in df.columns:
            df['interval_elapsed_from_flight_start'] = 0.0

    df = ensure_derived_columns(df)
    if drop_na_target and drop_na_target in df.columns:
        df = df.dropna(subset=[drop_na_target])
    return df


def preprocess_features(df, model_dir, feature_cols):
    """
    Applies the model's preprocessor to the input dataframe.
    Uses reindex with fill_value=0 so missing columns don't cause a KeyError.
    Returns a numpy array matching the model's training format.
    """
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        # No preprocessor: reindex gracefully
        return df.reindex(columns=feature_cols, fill_value=0).values

    preprocessor = joblib.load(preprocessor_path)

    if isinstance(preprocessor, dict):
        num_feas = preprocessor.get('numerical_features', [])
        cat_feas = preprocessor.get('categorical_features', [])
        all_feas = num_feas + cat_feas

        # Fill missing features with 0
        X_df = df.reindex(columns=all_feas, fill_value=0).copy()

        num_imputer = preprocessor.get('num_imputer_full')
        cat_imputer = preprocessor.get('cat_imputer_full')
        cat_encoder = preprocessor.get('cat_encoder_full')
        scaler = preprocessor.get('scaler_full')

        if num_feas and num_imputer:
            # Bypass sklearn feature name validation
            X_df[num_feas] = num_imputer.transform(X_df[num_feas].values)
        if cat_feas and cat_imputer:
            X_df[cat_feas] = cat_imputer.transform(X_df[cat_feas].values)
            if cat_encoder:
                X_df[cat_feas] = cat_encoder.transform(X_df[cat_feas].values)

        # Standardize features
        X_scaled = scaler.transform(X_df.values)

        # Apply SFS mask if available
        mask = preprocessor.get('selected_mask')
        if mask is not None:
            X_scaled = X_scaled[:, mask]

        # Explicitly return as numpy array to avoid feature name mismatch errors in XGBoost
        return np.asarray(X_scaled)
    else:
        # Simple scaler/transformer — reindex gracefully and return as numpy array
        # Use .values to bypass name validation
        X_numpy_input = df.reindex(columns=feature_cols, fill_value=0).values
        X_processed = preprocessor.transform(X_numpy_input)
        return np.asarray(X_processed)

def plot_shap_summary(model_dir):
    """Generates the SHAP summary plot for feature importance."""
    if not HAS_SHAP:
        return
        
    print("Generating SHAP Summary Plot...")
    model_path = os.path.join(model_dir, "model.joblib")
    features_path = os.path.join(model_dir, "selected_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model or features list not found.")
        return
        
    try:
        import traceback
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
            if isinstance(feature_cols, dict) and 'selected_features' in feature_cols:
                feature_cols = feature_cols['selected_features']
            
        # Load training data
        df = load_training_data()
        if df is None:
            print("Could not load training data for SHAP.")
            return
        
        # Sample for performance
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        X_sample_processed = preprocess_features(df_sample, model_dir, feature_cols)
        
        # Convert to numpy array
        X_numpy = np.asarray(X_sample_processed)
        
        # Reset booster feature names
        try:
            booster = model.get_booster()
            booster.feature_names = None
            explainer = shap.TreeExplainer(booster)
        except Exception as be:
            print(f"Warning: Could not clear booster names: {be}")
            explainer = shap.TreeExplainer(model)
            
        shap_values = explainer.shap_values(X_numpy)
    
        fig = plt.figure(figsize=(14, 10))
        # Generate the standard SHAP summary plot
        display_feature_names = [f.replace('_', ' ').title() for f in feature_cols]
        # Plot SHAP summary
        shap.summary_plot(shap_values, X_numpy, feature_names=display_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP plot saved.")
    except Exception as e:
        print(f"Failed to generate SHAP summary plot: {e}")
        plt.close()

def plot_parity(model_dir):
    """Generates a high-quality parity plot (Predicted vs. Actual Fuel)."""
    print("Generating Parity Plot...")
    eval_csv = os.path.join(model_dir, "evaluation_details.csv")
    
    if not os.path.exists(eval_csv):
        print(f"Evaluation results not found at {eval_csv}. Run 'evaluate_model.py' first.")
        return
        
    df = pd.read_csv(eval_csv)
    
    # Load metadata for categorization
    try:
        flightlist = pd.read_parquet(config.FLIGHTLIST_TRAIN)
        # Apply category mapping
        df = df.merge(flightlist[['flight_id', 'aircraft_type']], on='flight_id', how='left')
        df['Aircraft Category'] = df['aircraft_type'].apply(
            lambda x: 'Widebody' if str(x) in config.WIDEBODY_AIRCRAFT else 'Narrowbody'
        )
    except Exception as e:
        print(f"Could not load aircraft types for color coding: {e}")
        df['Aircraft Category'] = 'All Aircraft'

    y_true = df['actual_fuel_kg']
    y_pred = df['predicted_fuel_kg']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create scatterplot with hue
    if 'Aircraft Category' in df.columns and len(df['Aircraft Category'].unique()) > 1:
        sns.scatterplot(
            data=df, x='actual_fuel_kg', y='predicted_fuel_kg', 
            hue='Aircraft Category', alpha=0.4, s=60, ax=ax,
            palette={'Narrowbody': '#1f77b4', 'Widebody': '#ff7f0e'}
        )
        ax.legend(handles=handles, labels=labels, title="Category", title_fontsize=24)
    else:
        ax.scatter(y_true, y_pred, alpha=0.3, s=50, color='#1f77b4')
    
    # Plot perfect agreement line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, label='Perfect Prediction')
    
    ax.set_xlabel("Actual Fuel Consumption (kg)")
    ax.set_ylabel("Predicted Fuel Consumption (kg)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "parity_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Parity plot saved.")

def plot_synthetic_distributions():
    """Generates distribution plots comparing original widebody data vs. synthetic."""
    print("Generating Synthetic Distribution Plot...")
    
    orig_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    synth_path = config.SYNTHETIC_WIDEBODY_PATH
    
    if not os.path.exists(orig_path) or not os.path.exists(synth_path):
        print("Missing required data for synthetic distribution plots.")
        return
        
    df_orig = pd.read_parquet(orig_path)
    df_synth = pd.read_parquet(synth_path)
    
    # Widebody subset comparison
    df_orig_wb = df_orig[df_orig['aircraft_type'].isin(config.WIDEBODY_AIRCRAFT)]
    
    features_to_plot = []
    
    # Identify available metric columns
    for col in ['segment_duration', 'interval_duration_sec', 'duration']:
        if col in df_orig_wb.columns and col in df_synth.columns:
            features_to_plot.append(col)
            break
            
    # Try to find a mass column
    for col in ['estimated_takeoff_mass', 'starting_mass_kg', 'takeoff_mass_kg']:
        if col in df_orig_wb.columns and col in df_synth.columns:
            features_to_plot.append(col)
            break
    
    for feat in features_to_plot:
        if feat in df_orig_wb.columns and feat in df_synth.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sns.kdeplot(df_orig_wb[feat].dropna(), fill=True, label='Original Widebody', ax=ax, alpha=0.5, color='#1f77b4')
            sns.kdeplot(df_synth[feat].dropna(), fill=True, label='Synthetic Widebody', ax=ax, alpha=0.5, color='#ff7f0e')
            
            ax.set_xlabel(feat.replace('_', ' ').title())
            ax.set_ylabel("Density")
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"dist_{feat}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Distribution plot for {feat} saved.")

def plot_dynamic_mass():
    """Plots the dynamically decreasing mass and altitude profile over time for a single flight."""
    print("Generating Dynamic Mass Profile Plot...")
    
    # For this plot, we need augmented data which contains starting_mass_kg for every interval
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    df = ensure_derived_columns(df)
    
    # Select flight with high interval count
    flight_counts = df['flight_id'].value_counts()
    if flight_counts.empty:
        return
        
    # Pick a flight that has at least 10 intervals
    suitable_flights = flight_counts[flight_counts > 10].index
    if len(suitable_flights) == 0:
        target_flight = flight_counts.index[0]
    else:
        target_flight = suitable_flights[0]
        
    flight_data = df[df['flight_id'] == target_flight].copy()
    
    # Sort by interval index or time
    if 'idx' in flight_data.columns:
        flight_data = flight_data.sort_values('idx')
    elif 'start' in flight_data.columns:
        flight_data = flight_data.sort_values('start')
    
    # Check altitude column
    alt_col = next((c for c in ['seg_altitude_mean', 'alt_avg_ft', 'start_alt_rev'] if c in flight_data.columns), None)
        
    # Create the dual-axis plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # X axis can just be the interval index or elapsed time
    x = range(len(flight_data))
    
    # Plot Altitude on Ax1
    color1 = '#1f77b4' # Blue
    ax1.set_xlabel("Flight Interval Index")
    
    if alt_col:
        ax1.set_ylabel(f"Average Altitude ({alt_col.replace('seg_','').replace('_',' ').title()})", color=color1)
        ax1.plot(x, flight_data[alt_col], color=color1, marker='o', linewidth=3, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color1)
    else:
        ax1.set_ylabel("Altitude (Unknown Column)", color=color1)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    
    # Dynamic mass profile
    mass_col = next((c for c in ['estimated_takeoff_mass', 'starting_mass_kg', 'takeoff_mass_kg'] if c in flight_data.columns), None)
            
    if mass_col:
        color2 = '#d62728' # Red
        ax2.set_ylabel(f"Dynamic Mass ({mass_col.replace('_',' ').title()})", color=color2)
        # The mass drops at each interval entry point
        ax2.plot(x, flight_data[mass_col], color=color2, marker='s', linewidth=3, markersize=8, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
    else:
        print("Warning: No mass column found. Only plotting altitude.")
        ax2.set_yticks([]) # Hide the second axis ticks
    
    
    # Manual legend construction for dual axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dynamic_mass_profile.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Dynamic mass profile saved.")

def plot_class_imbalance():
    """Generates a bar chart showing class imbalance before and after synthetic augmentation."""
    print("Generating Class Imbalance Plot...")
    
    orig_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    synth_path = config.SYNTHETIC_WIDEBODY_PATH
    
    if not os.path.exists(orig_path):
        print("Missing original data for class imbalance plot.")
        return
        
    df_orig = pd.read_parquet(orig_path)
    
    # Calculate original counts
    orig_wb = df_orig['aircraft_type'].isin(config.WIDEBODY_AIRCRAFT).sum()
    orig_nb = len(df_orig) - orig_wb
    
    # Calculate synthetic counts
    synth_wb = 0
    if os.path.exists(synth_path):
        df_synth = pd.read_parquet(synth_path)
        synth_wb = len(df_synth)
        
    # Data for plotting
    categories = ['Original Data', 'Augmented Data']
    narrowbody_counts = [orig_nb, orig_nb]  # Narrowbody stays the same
    widebody_counts = [orig_wb, orig_wb + synth_wb]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, narrowbody_counts, width, label='Narrowbody', color='#1f77b4')
    ax.bar(x + width/2, widebody_counts, width, label='Widebody', color='#ff7f0e')
    
    ax.set_ylabel('Number of Interval Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add text labels on top of bars
    for i, v in enumerate(narrowbody_counts):
        ax.text(i - width/2, v + (max(narrowbody_counts)*0.01), f'{v:,}', ha='center', va='bottom', fontsize=20)
    for i, v in enumerate(widebody_counts):
        ax.text(i + width/2, v + (max(narrowbody_counts)*0.01), f'{v:,}', ha='center', va='bottom', fontsize=20)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_imbalance_after.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate the "Before" plot explicitly
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(['Narrowbody', 'Widebody'], [orig_nb, orig_wb], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Number of Interval Samples')
    for i, v in enumerate([orig_nb, orig_wb]):
        ax.text(i, v + (orig_nb*0.01), f'{v:,}', ha='center', va='bottom', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_imbalance_before.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Class imbalance plots (before and after) saved.")

def plot_feature_correlation():
    """Generates a correlation heatmap for key features and the target variable."""
    print("Generating Feature Correlation Heatmap...")
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        print("Data not found for correlation plot.")
        return
        
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=['fuel_kg'])
    df = ensure_derived_columns(df)
    
    # Feature correlation matrix
    sfs_features = []
    if os.path.exists(config.SELECTED_FEATURES_PATH):
        try:
            with open(config.SELECTED_FEATURES_PATH, 'r') as f:
                feature_data = json.load(f)
                if isinstance(feature_data, dict) and 'selected_features' in feature_data:
                    sfs_features = feature_data['selected_features']
                elif isinstance(feature_data, list):
                    sfs_features = feature_data
        except Exception as e:
            pass
            
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if sfs_features:
        # Use top SFS features
        top_features = [f for f in sfs_features if f in numeric_cols and f != 'fuel_kg'][:15]
        
        if not top_features:
             print("No numeric SFS features found for correlation.")
             return
             
        # Calculate correlation of top features strictly with fuel_kg (Nx1 matrix/rectangle)
        corr_with_target = df[top_features + ['fuel_kg']].corr()[['fuel_kg']].drop('fuel_kg')
        
        # Sort by absolute correlation
        corr_with_target['abs_fuel_kg'] = corr_with_target['fuel_kg'].abs()
        corr_with_target = corr_with_target.sort_values(by='abs_fuel_kg', ascending=False)
        corr_with_target = corr_with_target[['fuel_kg']] # drop the temp column
        
        fig, ax = plt.subplots(figsize=(8, 10))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        formatted_labels = [c.replace('_', ' ').title().replace('Kg', '(kg)').replace('Sec', '(s)').replace('Km', '(km)').replace('Ft', '(ft)').replace('Kts', '(kts)').replace('Fpm', '(fpm)') for c in corr_with_target.index]
        
        sns.heatmap(corr_with_target, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=False, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=True, fmt=".2f", annot_kws={"size": 16}, ax=ax,
                    yticklabels=formatted_labels, xticklabels=['Fuel Consumption (kg)'])
                    
        plt.yticks(rotation=0)
        
    else:
        key_features = [
            'fuel_kg', 'estimated_takeoff_mass', 'takeoff_mass_kg', 'starting_mass_kg',
            'segment_duration', 'segment_distance_km', 'great_circle_distance_km',
            'seg_altitude_mean', 'alt_avg_ft', 'seg_groundspeed_mean', 'gs_avg_kts',
            'seg_vertical_rate_mean', 'vs_avg_fpm', 'mean_time_in_air', 'dep_tmpf', 'temperature'
        ]
        
        plot_cols = [c for c in key_features if c in numeric_cols]
        
        if len(plot_cols) < 2:
            print("Not enough key features found for correlation plot.")
            return
            
        # Calculate correlation matrix
        corr = df[plot_cols].corr()
        
        # Larger figure to avoid label clipping/warnings
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Nice divergent colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # plt.figure(figsize=(16, 14))
        
        # Feature name formatting
        formatted_labels = [c.replace('_', ' ').title().replace('Kg', '(kg)').replace('Sec', '(s)').replace('Km', '(km)').replace('Ft', '(ft)').replace('Kts', '(kts)').replace('Fpm', '(fpm)') for c in plot_cols]
        
        sns.heatmap(corr, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    square=False, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=True, fmt=".2f", annot_kws={"size": 14}, ax=ax,
                    xticklabels=formatted_labels, yticklabels=formatted_labels)
                    
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    # Adjust layout manually if tight_layout fails
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(os.path.join(PLOTS_DIR, "feature_correlation.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "feature_correlation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation heatmap saved.")

def plot_optuna_hyperparameters():
    """Generates plots showing the effect of hyperparameters from Optuna trials."""
    print("Generating Optuna Hyperparameter Impact Plots...")
    
    trials_path = os.path.join(config.MODELS_DIR, 'optuna_trials_history.csv')
    if not os.path.exists(trials_path):
        trials_path = os.path.join(config.MODELS_DIR, 'test_optuna_trials_history.csv')
        if not os.path.exists(trials_path):
            print("Optuna trials history not found. Skipping hyperparameter plots.")
            return

    df = pd.read_csv(trials_path)
    if 'state' in df.columns:
        df = df[df['state'] == 'COMPLETE']
        
    if df.empty:
        print("No complete Optuna trials found.")
        return
        
    def plot_param_vs_rmse(param_col, title, filename, log_x=False):
        if param_col not in df.columns: return
        try:
            plt.figure(figsize=(12, 7))
            
            # If learning rate, add IQR bands by binning
            if 'learning_rate' in param_col:
                # Log-scale binning
                if log_x:
                    bins = np.logspace(np.log10(df[param_col].min()), np.log10(df[param_col].max()), 10)
                else:
                    bins = np.linspace(df[param_col].min(), df[param_col].max(), 10)
                
                df['bin'] = pd.cut(df[param_col], bins=bins)
                bin_stats = df.groupby('bin', observed=False)['value'].agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
                bin_centers = bins[:-1] + np.diff(bins)/2
                
                plt.fill_between(bin_centers, bin_stats.iloc[:, 2], bin_stats.iloc[:, 3], color='gray', alpha=0.2, label='IQR Band')
                plt.plot(bin_centers, bin_stats['median'], 'r--', linewidth=2, label='Median Trend')

            sns.scatterplot(x=df[param_col], y=df['value'], alpha=0.5, s=60, color='#1f77b4', edgecolor='k', label='Trials')
            
            if log_x: plt.xscale('log')
            
            clean_param = param_col.replace('params_', '').replace('_', ' ').title()
            plt.xlabel(clean_param)
            plt.ylabel('Validation RMSE')
            plt.legend()
            plt_sns.despine()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not plot {param_col}: {e}")
            plt.close()

    plot_param_vs_rmse('params_learning_rate', 'Learning Rate vs Performance', 'optuna_learning_rate_effect.png', log_x=True)
    plot_param_vs_rmse('params_n_estimators', 'Number of Estimators vs Performance', 'optuna_n_estimators_effect.png')
    plot_param_vs_rmse('params_reg_alpha', 'L1 Regularization (Alpha) Impact', 'optuna_l1_alpha_impact.png', log_x=True)
    plot_param_vs_rmse('params_reg_lambda', 'L2 Regularization (Lambda) Impact', 'optuna_l2_lambda_impact.png', log_x=True)
    plot_param_vs_rmse('params_max_depth', 'Max Depth vs Performance', 'optuna_max_depth_impact.png')
    plot_param_vs_rmse('params_subsample', 'Subsampling Ratio Effect', 'optuna_subsample_effect.png')
    plot_param_vs_rmse('params_colsample_bytree', 'Column Sample by Tree Effect', 'optuna_colsample_effect.png')
    print("Optuna plots saved.")

def plot_advanced_feature_importance(model_dir):
    """Generates explicit feature importance plots for Weight, Gain, and Cover."""
    print("Generating Advanced Feature Importance Plots (Gain, Weight, Cover)...")
    
    model_path = os.path.join(model_dir, "model.joblib")
    features_path = os.path.join(model_dir, "selected_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Missing model or features path for advanced importance.")
        return
        
    try:
        model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
            if isinstance(feature_cols, dict) and 'selected_features' in feature_cols:
                feature_cols = feature_cols['selected_features']
                
        booster = model.get_booster()
        fmap = {f'f{i}': name for i, name in enumerate(feature_cols)}
        
        for imp_type in ['gain', 'weight', 'cover']:
            scores = booster.get_score(importance_type=imp_type)
            if not scores: continue
            
            # Standard to real feature mapping
            scores_mapped = {fmap.get(k, k): v for k, v in scores.items()}
            df = pd.DataFrame(list(scores_mapped.items()), columns=['Feature', 'Score'])
            
            # Normalize to 100%
            df['Score'] = (df['Score'] / df['Score'].sum()) * 100
            
            df = df.sort_values('Score', ascending=False).head(20)
            
            # Format feature names
            df['Feature'] = df['Feature'].apply(lambda x: str(x).replace('_', ' ').title() if len(str(x)) < 25 else str(x)[:22] + '...')
            
            fig, ax = plt.subplots(figsize=(8, 10))
            sns.barplot(x='Score', y='Feature', data=df, hue='Feature', palette='viridis', ax=ax, legend=False)
            ax.set_xlabel(f'Relative Importance ({imp_type.capitalize()}) [%]')
            ax.set_ylabel('')
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"feature_importance_{imp_type}_rank1.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        print("Advanced feature importance plots saved.")
    except Exception as e:
        print(f"Failed to generate advanced feature importance: {e}")

def plot_basic_feature_importance(model_dir):
    """Generates standard XGBoost feature importance (weight/gain) if SHAP is unavailable."""
    print("Generating Basic Feature Importance Plot...")
    
    model_path = os.path.join(model_dir, "model.joblib")
    features_path = os.path.join(model_dir, "selected_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return
        
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)
        
    # Extraction of feature importance
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("Model does not expose feature_importances_")
        return
        # Create dataframe and normalize
    imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    
    # Normalize to 100%
    imp_df['Importance'] = (imp_df['Importance'] / imp_df['Importance'].sum()) * 100
    
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20) # Top 20
    
    # Format feature names
    imp_df['Feature'] = imp_df['Feature'].apply(lambda x: x.replace('_', ' ').title() if len(x) < 25 else x[:22] + '...')
    # Plot basic importance
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=imp_df, hue='Feature', palette='magma', ax=ax, legend=False)
    ax.set_xlabel('Feature Importance (%)')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance_basic.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Basic feature importance plot saved.")

def plot_learning_curves(model_dir):
    """Extracts and plots learning curves if eval_results are available in the model."""
    print("Attempting to generate Learning Curves plot...")
    
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        return
        
    model = joblib.load(model_path)
    
    # Validation metric tracking
    try:
        results = model.evals_result_
    except AttributeError:
        print("Learning curves unavailable: model was not trained with eval_set tracking.")
        return
        
    if not results:
        print("Learning curves unavailable: evals_result is empty.")
        return
        
    epochs = len(results['validation_0']['rmse'])
    # Plot RMSE curves
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    
    if 'validation_1' in results:
        ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
        
    ax.set_xlabel('Boosting Round (Epoch)')
    ax.set_ylabel('RMSE Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Learning curves plot saved.")

def plot_data_size_learning_curve(model_dir):
    """Plots training and validation error against training set size."""
    print("Generating Data-Size Learning Curve (Train size vs Error)...")
    
    features_path = os.path.join(model_dir, "selected_features.json")

    try:
        import traceback
        df = load_training_data()
        if df is None:
            print("Could not load training data for learning curve.")
            return
        
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
            if isinstance(feature_cols, dict) and 'selected_features' in feature_cols:
                feature_cols = feature_cols['selected_features']

        X = preprocess_features(df, model_dir, feature_cols)
        # Numpy conversion
        X_numpy = np.asarray(X, dtype=np.float32)
        y = np.log1p(df['fuel_kg'].values).astype(np.float32)
        
        from sklearn.model_selection import learning_curve
        from xgboost import XGBRegressor
        
        train_sizes = np.linspace(0.1, 1.0, 5)
        # Learning curve model configuration
        lc_model = XGBRegressor(
            n_estimators=50, 
            learning_rate=0.1, 
            max_depth=5, 
            tree_method='hist', 
            device='cpu', 
            n_jobs=1,
            enable_categorical=False
        )
        
        # Execute learning curve computation
        train_sizes, train_scores, test_scores = learning_curve(
            lc_model, X_numpy, y, 
            cv=3, 
            scoring='neg_mean_squared_error', 
            train_sizes=train_sizes, 
            n_jobs=1
        )
        
        train_rmse = np.sqrt(-train_scores)
        test_rmse  = np.sqrt(-test_scores)

        train_mean = train_rmse.mean(axis=1)
        test_mean  = test_rmse.mean(axis=1)
        train_q25  = np.percentile(train_rmse, 25, axis=1)
        train_q75  = np.percentile(train_rmse, 75, axis=1)
        test_q25   = np.percentile(test_rmse,  25, axis=1)
        test_q75   = np.percentile(test_rmse,  75, axis=1)

        plt.figure(figsize=(10, 7))
        plt.plot(train_sizes, train_mean, 'o-', color='tab:red',   label='Training Score')
        plt.fill_between(train_sizes, train_q25, train_q75, alpha=0.2, color='tab:red')
        plt.plot(train_sizes, test_mean,  'o-', color='tab:green', label='Cross-validation Score')
        plt.fill_between(train_sizes, test_q25,  test_q75,  alpha=0.2, color='tab:green')
        plt.xlabel("Training Examples")
        plt.ylabel("RMSE (log scale)")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "data_size_learning_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Data-size learning curve saved.")
    except Exception as e:
        print(f"Failed to generate data-size learning curve: {e}")
        import traceback
        traceback.print_exc()
        plt.close()

def plot_phase_wise_fuel_flow():
    """Plots the distribution of fuel flow rate across different flight phases."""
    print("Generating Phase-wise Fuel Flow Analysis Plot...")
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        return
        
    df = pd.read_parquet(data_path).dropna(subset=['fuel_kg'])
    
    # Phase classification
    phase_cols = ['phase_fraction_climb', 'phase_fraction_cruise', 'phase_fraction_descent', 'phase_fraction_approach']
    available_phases = [c for c in phase_cols if c in df.columns]
    
    if not available_phases:
        print("No phase columns found for phase-wise analysis.")
        return
        
    # Get duration column
    dur_col = next((c for c in ['segment_duration', 'interval_duration_sec', 'duration'] if c in df.columns), None)
    if not dur_col:
        return
        
    df['primary_phase'] = df[available_phases].idxmax(axis=1).apply(lambda x: x.replace('phase_fraction_', '').title())
    df['fuel_flow_kg_sec'] = df['fuel_kg'] / (df[dur_col] + 1e-6)
    
    # Outlier rejection
    q_high = df['fuel_flow_kg_sec'].quantile(0.99)
    df_plot = df[df['fuel_flow_kg_sec'] < q_high].copy()
    
    # Add category for NB/WB
    df_plot['Category'] = df_plot['aircraft_type'].apply(
        lambda x: 'Widebody' if str(x) in config.WIDEBODY_AIRCRAFT else 'Narrowbody'
    )
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='primary_phase', y='fuel_flow_kg_sec', hue='Category', data=df_plot, 
                       split=True, inner="quart",
                       palette={'Narrowbody': '#1f77b4', 'Widebody': '#ff7f0e'})
    
    plt.xlabel("Flight Phase")
    plt.ylabel("Fuel Flow Rate (kg/s)")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "phase_wise_fuel_flow.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Phase-wise fuel flow plot saved (Violin plot).")

def plot_spatial_coverage(stage='train'):
    """Creates a spatial coverage map showing the global footprint of the dataset."""
    print(f"Generating Spatial Coverage Map ({stage})...")
    data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data_{stage}.parquet")
    if not os.path.exists(data_path):
        print(f"  Data not found for stage '{stage}': {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    
    # Lat/Lon selection
    lat_cols = ['origin_latitude', 'dest_latitude', 'origin_lat', 'dest_lat']
    lon_cols = ['origin_longitude', 'dest_longitude', 'origin_lon', 'dest_lon']
    
    lats = [c for c in lat_cols if c in df.columns]
    lons = [c for c in lon_cols if c in df.columns]
    
    if not lats or not lons:
        print("No latitude/longitude columns found for spatial coverage.")
        return
        
    # Origin coordinate extraction
    plot_df = df[[lats[0], lons[0]]].copy().rename(columns={lats[0]: "latitude", lons[0]: "longitude"})
    
    suffix = f"_{stage}" if stage != 'train' else ''
    
    if HAS_PLOTLY:
        # Subsampling for visualization
        sample_size = min(15000, len(plot_df))
        df_geo = plot_df.sample(n=sample_size, random_state=42)
        
        fig = px.scatter_geo(
            df_geo, 
            lat="latitude", 
            lon="longitude",
            projection="natural earth",
            # No title as per user rules
            opacity=0.4,
            width=1200, height=800
        )
        
        # Use a more academic/clean template
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        # Save as PNG
        map_path = os.path.join(PLOTS_DIR, f"spatial_coverage{suffix}.png")
        try:
            fig.write_image(map_path, scale=2)
            print(f"Spatial coverage map saved as PNG ({stage}).")
        except Exception as e:
            # Fallback to HTML if kaleido/image export fails
            html_path = os.path.join(PLOTS_DIR, f"spatial_coverage{suffix}.html")
            fig.write_html(html_path)
            print(f"Warning: Failed to save map as PNG (likely missing kaleido): {e}")
            print(f"Saved interactive map as HTML instead: {html_path}")
            
            # Matplotlib fallback
            plt.figure(figsize=(16, 9))
            plt.scatter(plot_df["longitude"], plot_df["latitude"], s=1, alpha=0.1, color='#1f77b4')
            plt.xlim(-180, 180)
            plt.ylim(-90, 90)
            plt.tight_layout()
            plt.savefig(map_path, dpi=300)
            plt.close()
            
    else:
        print(f"Plotly not found. Using matplotlib fallback for spatial coverage ({stage}).")
        plt.figure(figsize=(16, 9))
        plt.scatter(plot_df["longitude"], plot_df["latitude"], s=1, alpha=0.1, color='#1f77b4')
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"spatial_coverage{suffix}.png"), dpi=300)
def plot_trajectory_network_map():
    """
    Creates a high-density trajectory flow map using actual trajectory files.
    Plots lines with high transparency to show network density (Image 1 style).
    """
    print("Generating High-Density Trajectory Network Map...")
    # Trajectory directory search
    potential_paths = [
        os.path.join(config.BASE_DATASETS_DIR, "flights_train"),
        os.path.join(config.DATA_DIR, "prc-2025-datasets", "flights_train"),
        # Added direct absolute path check for server robustness
        "/home/ygrigo01/fileserverdata/PRCXGBoost/data/prc-2025-datasets/flights_train"
    ]
    
    traj_dir = None
    for p in potential_paths:
        if os.path.exists(p):
            traj_dir = p
            print(f"Using trajectory directory: {traj_dir}")
            break
            
    if not traj_dir:
        print(f"Trajectory directory 'flights_train' not found in any expected location.")
        return

    all_files = [f for f in os.listdir(traj_dir) if f.endswith('.parquet')]
    if not all_files:
        print("No trajectory files found.")
        return

    # Sample trajectories
    sample_files = np.random.choice(all_files, size=min(800, len(all_files)), replace=False)

    plt.figure(figsize=(16, 12))
    
    # Increase alpha for the "network density" feel
    for f in sample_files:
        try:
            t_df = pd.read_parquet(os.path.join(traj_dir, f))
            if 'latitude' in t_df.columns and 'longitude' in t_df.columns:
                plt.plot(t_df['longitude'], t_df['latitude'], color='black', alpha=0.03, linewidth=0.5)
        except:
            continue

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle=':', alpha=0.3)
    
    # Zoom into Europe context by default
    plt.xlim(-25, 45)
    plt.ylim(30, 70)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trajectory_network_web.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Trajectory network map saved.")

def plot_od_pair_density_map(stage='train'):
    """Generates an Origin-Destination pair density map with colorbar legend."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not found. Skipping O-D pair density map (requires plotly).")
        return
        
    print(f"Generating O-D Pair Density Map ({stage})...")
    
    if stage == 'all':
        dfs = []
        for s in ['train', 'rank', 'final']:
            p = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data_{s}.parquet")
            if os.path.exists(p):
                dfs.append(pd.read_parquet(p))
            else:
                print(f"  Warning: {p} not found for combined O-D map.")
        if not dfs:
            print("  No data found for combined O-D map.")
            return
        df = pd.concat(dfs, ignore_index=True)
    else:
        data_path = os.path.join(config.PROCESSED_DATA_DIR, f"featured_data_{stage}.parquet")
        if not os.path.exists(data_path):
            print(f"  Data not found for stage '{stage}': {data_path}")
            return
        df = pd.read_parquet(data_path)
    
    # O-D pair frequency distribution
    od_counts = df.groupby(['origin_icao', 'destination_icao']).size().reset_index(name='nflights')
    
    # Airport coordinate join
    apt = pd.read_parquet(config.APT_PARQUET)[['icao', 'longitude', 'latitude']]
    od_counts = od_counts.merge(apt.rename(columns={'icao': 'origin_icao', 'longitude': 'lon1', 'latitude': 'lat1'}), on='origin_icao')
    od_counts = od_counts.merge(apt.rename(columns={'icao': 'destination_icao', 'longitude': 'lon2', 'latitude': 'lat2'}), on='destination_icao')
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Density-based sorting
    od_counts = od_counts.sort_values('nflights')
    COLOR_MIN = 1
    COLOR_MAX = 1000  # Colorscale ceiling
    LOG_MIN = np.log10(COLOR_MIN)
    LOG_MAX = np.log10(COLOR_MAX)

    # Dummy trace for colorbar
    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            showscale=True,
            cmin=LOG_MIN,
            cmax=LOG_MAX,
            colorbar=dict(
                title=dict(
                    text="N flights (log10)",
                    side="top",
                    font=dict(size=18)
                ),
                thickness=25,
                outlinecolor="rgba(0,0,0,0)",
                ticks="outside",
                ticklen=8,
                tickfont=dict(size=16),
                tickvals=[0, 1, 2, 3],
                ticktext=['1', '10', '100', '1000+'],
                x=0.88,
                y=0.25,
                len=0.55
            ),
        ),
        showlegend=False
    ))

    # Sample pairs
    od_sample = od_counts.tail(1200) 

    # Flow visualization
    for i, row in od_sample.iterrows():
        n = max(COLOR_MIN, row['nflights'])
        val = (np.log10(min(n, COLOR_MAX)) - LOG_MIN) / (LOG_MAX - LOG_MIN)
        # Simple color mapping for lines (mimicking Viridis)
        # 0.0: dark blue -> 0.5: green -> 1.0: yellow
        r = int(255 * (val if val > 0.5 else 0))
        g = int(255 * (val if val <= 0.8 else 1))
        b = int(255 * (1 - val if val < 0.5 else 0))
        opacity = 0.2 + 0.6 * val
        
        fig.add_trace(
            go.Scattergeo(
                lon = [row['lon1'], row['lon2']],
                lat = [row['lat1'], row['lat2']],
                mode = 'lines',
                line = dict(width = 1.2, color = f'rgba({r}, {g}, {b}, {opacity})'),
                showlegend = False
            )
        )

    fig.update_layout(
        geo = dict(
            projection_type = 'natural earth',
            showland = True,
            landcolor = 'rgb(240, 240, 240)',
            countrycolor = 'rgb(215, 215, 215)',
            coastlinecolor = 'rgb(200, 200, 200)',
            showcoastlines = True,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    suffix = f"_{stage}" if stage != 'train' else ''
    try:
        import plotly.io as pio
        fig.write_image(os.path.join(PLOTS_DIR, f"od_flow_density{suffix}.png"), scale=2)
        print(f"O-D flow density map saved ({stage}).")
    except Exception as e:
        print(f"Could not save Plotly PNG: {e}. Saving HTML.")
        fig.write_html(os.path.join(PLOTS_DIR, f"od_flow_density{suffix}.html"))

def plot_categorical_impact():
    """Plots the distribution of fuel consumption across top aircraft types."""
    print("Generating Categorical Feature Impact Plot...")
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        return
        
    df = pd.read_parquet(data_path).dropna(subset=['fuel_kg'])
    
    # Top aircraft types
    top_types = df['aircraft_type'].value_counts().head(10).index
    df_plot = df[df['aircraft_type'].isin(top_types)].copy()
    
    plt.figure(figsize=(14, 10))
    sns.boxenplot(x='fuel_kg', y='aircraft_type', data=df_plot, order=top_types, palette='viridis', hue='aircraft_type', legend=False)
    
    plt.xlabel("Fuel Consumption (kg)")
    plt.ylabel("Aircraft Type")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "categorical_impact_aircraft.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Categorical impact plot saved.")

def generate_png_tables(model_dir):
    """Generates PNG versions of the results and hyperparameters tables."""
    print("Generating PNG Tables for Results...")
    
    # Hyperparameters Table
    params_path = os.path.join(model_dir[:model_dir.rfind('xb_model')-1], f'PARAMETERS_rank1.txt')
    if not os.path.exists(params_path):
        # try simple search
        params_path = os.path.join(os.path.dirname(model_dir), 'PARAMETERS_rank1.txt')
        if not os.path.exists(params_path):
            params_path = os.path.join(os.path.dirname(model_dir), 'test_parameters_rank1.txt')
            
    if os.path.exists(params_path):
        params_dict = {}
        with open(params_path, 'r') as f:
            lines = f.readlines()
            parsing_params = False
            for line in lines:
                if line.startswith('Hyperparameters:'):
                    parsing_params = True
                    continue
                if parsing_params and ':' in line:
                    key, val = line.strip().split(':', 1)
                    params_dict[key.strip()] = val.strip()
                    
        if params_dict:
            df_params = pd.DataFrame(list(params_dict.items()), columns=['Hyperparameter', 'Value'])
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df_params.values, colLabels=df_params.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            plt.tight_layout()
            plt.savefig(os.path.join(PAPER_PLOTS_DIR, 'hyperparams_table.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Performance Metrics Table
    # We can calculate this from evaluate_model if available, or just read the rank1 stats
    eval_csv = os.path.join(os.path.dirname(model_dir), 'random_search_top10_validation_results.csv')
    if not os.path.exists(eval_csv):
        eval_csv = os.path.join(os.path.dirname(model_dir), 'evaluate_results.csv')
        
    if os.path.exists(eval_csv):
        df_results = pd.read_csv(eval_csv)
        if 'final_rank' in df_results.columns:
            rank1 = df_results[df_results['final_rank'] == 1].iloc[0]
            metrics = {
                'RMSE (kg)': f"{rank1.get('val_rmse', rank1.get('rmse', 0)):.2f}",
                'MAE (kg)': f"{rank1.get('val_mae', rank1.get('mae', 0)):.2f}",
                'MAPE (%)': f"{rank1.get('val_mape', rank1.get('mape', 0)):.2f}",
                'R² Score': f"{rank1.get('val_r2', rank1.get('r2', 0)):.4f}"
            }
            
            df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Validation Score'])
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            plt.tight_layout()
            plt.savefig(os.path.join(PAPER_PLOTS_DIR, 'results_table.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("PNG tables saved.")

def plot_augmentation_ablation():
    """
    Grouped bar chart comparing validation MAE with and without synthetic widebody
    augmentation, broken down by Narrowbody / Widebody / Overall.

    Requires: data/processed/ablation_augmentation_results.csv
    (produced by running: python ablation_augmentation.py [--gpu 0])
    """
    print("Generating Augmentation Ablation Plot...")

    results_path = os.path.join(config.PROCESSED_DATA_DIR, 'ablation_augmentation_results.csv')
    if not os.path.exists(results_path):
        print(f"  Ablation results not found at {results_path}.")
        print("  Run: python ablation_augmentation.py [--gpu 0]  then re-generate plots.")
        return

    df = pd.read_csv(results_path)

    splits   = ['Narrowbody', 'Widebody', 'Overall']
    models   = ['No Augmentation', 'With Augmentation']
    colors   = {'No Augmentation': '#1f77b4', 'With Augmentation': '#ff7f0e'}

    x      = np.arange(len(splits))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, model_label in enumerate(models):
        mae_vals = []
        for split in splits:
            row = df[(df['model'] == model_label) & (df['split'] == split)]
            mae_vals.append(row['mae'].values[0] if len(row) else np.nan)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, mae_vals, width, label=model_label,
                      color=colors[model_label], edgecolor='black', linewidth=0.6)

        for bar, val in zip(bars, mae_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(mae_vals) * 0.01,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('Validation MAE (kg)', fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Widebody improvement annotation
    wb_base = df[(df['model'] == 'No Augmentation') & (df['split'] == 'Widebody')]
    wb_aug  = df[(df['model'] == 'With Augmentation') & (df['split'] == 'Widebody')]
    if len(wb_base) and len(wb_aug):
        delta = wb_aug.iloc[0]['mae'] - wb_base.iloc[0]['mae']
        pct   = delta / wb_base.iloc[0]['mae'] * 100
        ax.annotate(
            f'WB: {pct:+.1f}%',
            xy=(x[splits.index('Widebody')] + 0.5 * width, wb_aug.iloc[0]['mae']),
            xytext=(0, 30), textcoords='offset points',
            ha='center', fontsize=13, color='#ff7f0e',
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5)
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'augmentation_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Augmentation ablation plot saved.")

def plot_dataset_shift():
    """Compares raw aircraft types and distances across the three provided datasets (as-is)."""
    print("Generating Dataset Shift Comparison (Raw provided data)...")
    
    stages = ['train', 'rank', 'final']
    # GCD comparison
    paths = [
        os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet"),
        os.path.join(config.PROCESSED_DATA_DIR, "featured_data_rank.parquet"),
        os.path.join(config.PROCESSED_DATA_DIR, "featured_data_final.parquet"),
    ]
    
    wb_pcts = []
    mean_dists = []
    
    for stage, path in zip(stages, paths):
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Widebody composition
            wb_pct = df['aircraft_type'].isin(config.WIDEBODY_AIRCRAFT).mean() * 100
            wb_pcts.append(wb_pct)
            
            # Distance distribution
            if 'great_circle_distance_km' in df.columns:
                mean_dists.append(df['great_circle_distance_km'].mean())
            elif 'great_circle_distance' in df.columns:
                mean_dists.append(df['great_circle_distance'].mean())
            else:
                mean_dists.append(0)
        else:
            wb_pcts.append(0)
            mean_dists.append(0)
            print(f"Warning: {path} not found for dataset shift plot.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Widebody Composition
    bars1 = ax1.bar(stages, wb_pcts, color=['royalblue', 'orange', 'forestgreen'])
    ax1.set_ylabel('Widebody Aircraft Percentage (%)', fontsize=16)
    ax1.set_ylim(0, max(wb_pcts) * 1.3)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=14)
    
    # Mean Distance
    if any(d > 0 for d in mean_dists):
        bars2 = ax2.bar(stages, mean_dists, color=['royalblue', 'orange', 'forestgreen'])
        ax2.set_ylabel('Average Segment Distance (km)', fontsize=16)
        ax2.set_ylim(0, max(mean_dists) * 1.3)
        
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{int(yval)} km', ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_shift_comparison_raw.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Raw dataset shift plot saved.")

def plot_target_distribution():
    """Generates a histogram/KDE of actual_fuel_kg to show the target range."""
    print("Generating Target Variable Distribution...")
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        print("Data not found for target distribution plot.")
        return
        
    df = pd.read_parquet(data_path)
    target = 'fuel_kg' # Changed from actual_fuel_kg
    if target not in df.columns:
        print(f"Column {target} not found among: {df.columns.tolist()[:5]}")
        return
        
    # Log-scaling
    valid_data = df[df[target] > 0][target]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    sns.histplot(valid_data, bins=50, kde=True, ax=ax1, color='teal')
    ax1.set_xlabel('Actual Fuel Consumption (kg)', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    sns.histplot(np.log10(valid_data), bins=50, kde=True, ax=ax2, color='darkred')
    ax2.set_xlabel('Log10(Actual Fuel Consumption)', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "target_distribution_fuel.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Target distribution plot saved.")

def plot_physical_sensitivity():
    """Scatter plot of Starting Mass vs Actual Fuel Burn to validate physical consistency."""
    print("Generating Physical Sensitivity Plot (Mass vs Fuel)...")
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        return
        
    df = pd.read_parquet(data_path)
    # Feature extraction
    m_col = 'estimated_takeoff_mass'
    f_col = 'fuel_kg'
    
    if m_col not in df.columns or f_col not in df.columns:
        print(f"Required columns {m_col} or {f_col} not found.")
        return
        
    # Points subsampling
    df_sample = df.sample(min(2000, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 8))
    # Category mapping
    df_sample['is_widebody'] = df_sample['aircraft_type'].isin(config.WIDEBODY_AIRCRAFT).map({True: 'Widebody', False: 'Narrowbody'})
    
    sns.scatterplot(data=df_sample, x=m_col, y=f_col, hue='is_widebody', alpha=0.5, edgecolor=None)
    plt.xlabel('Starting Mass (kg)', fontsize=14)
    plt.ylabel('Segment Fuel Consumption (kg)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Aircraft Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "physical_sensitivity_mass_fuel.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Physical sensitivity plot saved.")

def plot_metar_impact():
    """Visualizes the correlation between environmental factors (wind/temp) and fuel flow rates."""
    print("Generating METAR Impact Visualization...")
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        return
        
    df = pd.read_parquet(data_path)
    
    # Fuel flow vs wind speed
    # Using fuel_kg, segment_duration, and dep_sknt
    f_col = 'fuel_kg'
    d_col = 'segment_duration'
    w_col = 'dep_sknt'
    
    if f_col not in df.columns or d_col not in df.columns or w_col not in df.columns:
        print(f"Required METAR impact columns ({f_col}, {d_col}, {w_col}) not found.")
        return
        
    df['fuel_flow_kgs'] = df[f_col] / (df[d_col] + 1e-6)
    
    # Climb phase segmentation
    climb_df = df.copy()
    
    plt.figure(figsize=(10, 8))
    # Wind speed discretization
    climb_df['wind_bin'] = pd.cut(climb_df[w_col], bins=[0, 5, 10, 15, 20, 25, 30, 200], labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+'])
    
    sns.boxplot(data=climb_df, x='wind_bin', y='fuel_flow_kgs', color='skyblue', showfliers=False)
    plt.xlabel('Average Wind Speed (kts)', fontsize=14)
    plt.ylabel('Fuel Flow Rate (kg/s)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metar_impact_wind_fuel.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("METAR impact plot saved.")

def plot_aircraft_type_frequency():
    # Aircraft frequency distribution
    print("Generating Aircraft Type Frequency Distribution...")
    
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(data_path):
        return
        
    df = pd.read_parquet(data_path)
    if 'aircraft_type' not in df.columns:
        return
        
    type_counts = df['aircraft_type'].value_counts()
    
    plt.figure(figsize=(12, 6))
    colors = ['orange' if t in config.WIDEBODY_AIRCRAFT else 'royalblue' for t in type_counts.index]
    
    bars = plt.bar(type_counts.index, type_counts.values, color=colors)
    plt.ylabel('Number of Interval Samples', fontsize=14)
    plt.xticks(rotation=45)
    
    # Category legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='royalblue', lw=4, label='Narrowbody'),
                      Line2D([0], [0], color='orange', lw=4, label='Widebody')]
    plt.legend(handles=legend_elements)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "aircraft_type_frequency.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Aircraft type frequency plot saved.")

def plot_error_vs_fuel_magnitude():
    """Scatter of absolute prediction error vs actual fuel_kg (heteroscedasticity check)."""
    print("Generating Error vs Fuel Magnitude plot...")
    val_path = os.path.join(config.PROCESSED_DATA_DIR, "val_predictions.csv")
    if not os.path.exists(val_path):
        print("val_predictions.csv not found. Skipping.")
        return

    df = pd.read_csv(val_path).dropna(subset=['actual_fuel_kg', 'xgb_pred_kg'])
    df['abs_error'] = (df['xgb_pred_kg'] - df['actual_fuel_kg']).abs()
    df['mape'] = df['abs_error'] / (df['actual_fuel_kg'].abs() + 1e-8) * 100

    phase_colors = {'CLIMB': '#e15759', 'CRUISE': '#4e79a7', 'DESCENT': '#f28e2b', 'LEVEL': '#76b7b2'}
    phases = df['phase'].unique() if 'phase' in df.columns else []

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Absolute error vs fuel_kg
    ax = axes[0]
    if 'phase' in df.columns:
        for ph, grp in df.groupby('phase'):
            ax.scatter(grp['actual_fuel_kg'], grp['abs_error'],
                       alpha=0.15, s=6, color=phase_colors.get(ph, 'grey'), label=ph, rasterized=True)
    else:
        ax.scatter(df['actual_fuel_kg'], df['abs_error'], alpha=0.15, s=6, rasterized=True)

    # Rollmedian trend
    df_sorted = df.sort_values('actual_fuel_kg')
    bin_edges = np.percentile(df_sorted['actual_fuel_kg'], np.linspace(0, 100, 30))
    bin_mids, medians = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (df_sorted['actual_fuel_kg'] >= lo) & (df_sorted['actual_fuel_kg'] < hi)
        if mask.sum() > 5:
            bin_mids.append((lo + hi) / 2)
            medians.append(df_sorted.loc[mask, 'abs_error'].median())
    ax.plot(bin_mids, medians, 'k-', lw=2.5, label='Median error')
    ax.set_xlabel('Actual Fuel Consumption (kg)')
    ax.set_ylabel('Absolute Error (kg)')
    ax.legend(markerscale=3, framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Panel 2: MAPE vs fuel_kg
    ax2 = axes[1]
    if 'phase' in df.columns:
        for ph, grp in df.groupby('phase'):
            ax2.scatter(grp['actual_fuel_kg'], grp['mape'].clip(upper=200),
                        alpha=0.15, s=6, color=phase_colors.get(ph, 'grey'), label=ph, rasterized=True)
    else:
        ax2.scatter(df['actual_fuel_kg'], df['mape'].clip(upper=200), alpha=0.15, s=6, rasterized=True)
    mape_sorted = df.sort_values('actual_fuel_kg')
    mape_mids, mape_meds = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (mape_sorted['actual_fuel_kg'] >= lo) & (mape_sorted['actual_fuel_kg'] < hi)
        if mask.sum() > 5:
            mape_mids.append((lo + hi) / 2)
            mape_meds.append(mape_sorted.loc[mask, 'mape'].clip(upper=200).median())
    ax2.plot(mape_mids, mape_meds, 'k-', lw=2.5, label='Median MAPE')
    ax2.set_xlabel('Actual Fuel Consumption (kg)')
    ax2.set_ylabel('MAPE (%)')
    ax2.legend(markerscale=3, framealpha=0.8)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "error_vs_fuel_magnitude.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Error vs fuel magnitude plot saved.")


def plot_per_aircraft_overview():
    """Bar chart: per-aircraft mean predicted vs actual fuel burn rate (kg/s), with mean distance overlay."""
    print("Generating Per-Aircraft Overview plot...")
    val_path  = os.path.join(config.PROCESSED_DATA_DIR, "val_predictions.csv")
    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    if not os.path.exists(val_path):
        print("val_predictions.csv not found. Skipping.")
        return

    val = pd.read_csv(val_path).dropna(subset=['actual_fuel_kg', 'xgb_pred_kg'])

    # Join segment metadata
    if os.path.exists(feat_path):
        try:
            feat = pd.read_parquet(feat_path,
                                   columns=['segment_duration'])
            # interval_idx in val_predictions is the original DataFrame row index
            feat = feat.reset_index().rename(columns={'index': 'interval_idx'})
            val['interval_idx'] = val['interval_idx'].astype(int)
            val = val.merge(feat, on='interval_idx', how='left')
        except Exception as e:
            print(f"  Warning: could not merge featured data: {e}")

    # Fuel flow rate (kg/s)
    if 'segment_duration' in val.columns:
        dur = val['segment_duration'].replace(0, np.nan)
        val['actual_kgs']    = val['actual_fuel_kg'] / dur
        val['predicted_kgs'] = val['xgb_pred_kg']    / dur
    else:
        print("  segment_duration not available; falling back to total kg.")
        val['actual_kgs']    = val['actual_fuel_kg']
        val['predicted_kgs'] = val['xgb_pred_kg']

    ac = val.groupby('aircraft_type').agg(
        mean_actual_kgs  =('actual_kgs',    'mean'),
        mean_pred_kgs    =('predicted_kgs', 'mean'),
        N                =('actual_kgs',    'count'),
    ).reset_index().rename(columns={'aircraft_type': 'Aircraft'})
    ac = ac.sort_values('N', ascending=False)

    x = np.arange(len(ac))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(max(16, len(ac) * 1.1), 10))
    ax1.bar(x - w/2, ac['mean_actual_kgs'], width=w, label='Mean Actual (kg/s)',    color='#4e79a7')
    ax1.bar(x + w/2, ac['mean_pred_kgs'],   width=w, label='Mean Predicted (kg/s)', color='#59a14f', alpha=0.85)
    ax1.set_ylabel('Fuel Burn Rate (kg/s)', fontsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{row['Aircraft']} (N={row['N']:,})" for _, row in ac.iterrows()],
        rotation=45, ha='right', fontsize=24
    )
    ax1.legend(loc='upper right', fontsize=24)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(PLOTS_DIR, "per_aircraft_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Per-aircraft overview plot saved.")


def plot_metar_ablation():
    """Ablation: val RMSE with all features vs without METAR (dep_*/arr_*) features."""
    print("Generating METAR Ablation plot...")
    from sklearn.model_selection import cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from xgboost import XGBRegressor

    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_data_train.parquet")
    sel_path  = getattr(config, 'SELECTED_FEATURES_PATH', None)
    if not os.path.exists(feat_path):
        print("featured_data_train.parquet not found. Skipping METAR ablation.")
        return

    df = pd.read_parquet(feat_path).dropna(subset=['fuel_kg'])
    y  = np.log1p(df['fuel_kg'].values)

    # Feature selection
    feature_cols = None
    if sel_path and os.path.exists(sel_path):
        with open(sel_path) as f:
            raw = json.load(f)
        feature_cols = raw['selected_features'] if isinstance(raw, dict) else raw
        feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols if c != 'fuel_kg']

    metar_cols = [c for c in feature_cols if c.startswith('dep_') or c.startswith('arr_')]
    non_metar  = [c for c in feature_cols if c not in metar_cols]

    if not metar_cols:
        print("No METAR columns found among selected features. Skipping.")
        return

    def _cv_rmse(cols):
        X = df[cols].copy()
        cat = X.select_dtypes(include='object').columns.tolist()
        num = X.select_dtypes(include=[np.number]).columns.tolist()
        imp_n = SimpleImputer(strategy='mean')
        imp_c = SimpleImputer(strategy='most_frequent')
        if num: X[num] = imp_n.fit_transform(X[num])
        if cat:
            X[cat] = imp_c.fit_transform(X[cat])
            X[cat] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[cat])
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             tree_method='hist', device='cpu', n_jobs=-1,
                             random_state=42, verbosity=0)
        scores = cross_val_score(model, X.values, y, cv=5,
                                 scoring='neg_root_mean_squared_error', n_jobs=1)
        return -scores  # shape (5,)

    print(f"  Running CV with all {len(feature_cols)} features...")
    scores_all = _cv_rmse(feature_cols)
    print(f"  Running CV without {len(metar_cols)} METAR features...")
    scores_no  = _cv_rmse(non_metar)

    labels = ['With METAR', 'Without METAR']
    means  = [scores_all.mean(), scores_no.mean()]
    q25    = [np.percentile(scores_all, 25), np.percentile(scores_no, 25)]
    q75    = [np.percentile(scores_all, 75), np.percentile(scores_no, 75)]
    colors = ['#4e79a7', '#e15759']

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(labels, means, color=colors, width=0.45, zorder=3)
    for i, (m, lo, hi) in enumerate(zip(means, q25, q75)):
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt='none',
                    color='black', capsize=10, linewidth=2.5, zorder=4)
        ax.text(i, hi + 1, f'{m:.1f} kg', ha='center', va='bottom', fontsize=20, fontweight='bold')

    delta = scores_no.mean() - scores_all.mean()
    ax.set_ylabel('CV RMSE (kg, log-space back-transformed)')
    ax.set_title(f'METAR Feature Ablation  ($\\Delta$RMSE = +{delta:.1f} kg without METAR)')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y', zorder=0)
    ax.set_ylim(0, max(q75) * 1.2)

    # Feature count annotation
    ax.text(0.98, 0.97, f'{len(metar_cols)} METAR features\n({len(feature_cols)} total)',
            transform=ax.transAxes, ha='right', va='top', fontsize=18,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metar_ablation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("METAR ablation plot saved.")


def run_all(model_dir=None):
    """
    Runs the entire suite of plots and tables for the paper.
    If model_dir is provided, it uses that for model-specific plots.
    Otherwise, it attempts to select the best model automatically.
    """
    print("--- Generating Academic Paper Plots ---")
    
    # Data and distribution plots
    plot_synthetic_distributions()
    plot_dynamic_mass()
    plot_class_imbalance()
    plot_feature_correlation()
    plot_phase_wise_fuel_flow()
    
    for stage in ['train', 'rank', 'final']:
        plot_spatial_coverage(stage)
        plot_od_pair_density_map(stage)
        
    # Combined O-D Map
    plot_od_pair_density_map('all')
        
    plot_trajectory_network_map()
    plot_categorical_impact()
    
    # NEW PLOTS
    plot_augmentation_ablation()
    plot_dataset_shift()
    plot_target_distribution()
    plot_physical_sensitivity()
    plot_metar_impact()
    plot_aircraft_type_frequency()
    plot_error_vs_fuel_magnitude()
    plot_per_aircraft_overview()
    plot_metar_ablation()
    
    if model_dir is None:
        model_dir = select_model()
        
    if model_dir:
        print(f"Using model directory: {model_dir}")
        plot_parity(model_dir)
        plot_basic_feature_importance(model_dir)
        plot_advanced_feature_importance(model_dir)
        plot_learning_curves(model_dir)
        plot_data_size_learning_curve(model_dir)
        plot_optuna_hyperparameters()        
        generate_png_tables(model_dir)
        
        # Only try SHAP if available
        if HAS_SHAP:
            plot_shap_summary(model_dir)
    else:
        print("No model directory provided or found. Skipping model-specific plots.")
        
    print(f"\nAll plots and PNG tables have been saved to the '{PLOTS_DIR}' and '{PAPER_PLOTS_DIR}' directories.")
    if not HAS_SHAP:
        print("Note: SHAP plots were skipped. Run 'pip install shap' on your server and re-run this script to generate them.")


if __name__ == "__main__":
    run_all()

