import os
import pandas as pd
import numpy as np
import argparse
import config
import datetime
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_dir):
    """Sets up a dedicated logger for the error analysis report."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'error_analysis_report.log')
    
    logger = logging.getLogger('error_analysis')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file, mode='w')
    stream_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def analyze_error_distribution(df_eval, output_dir, logger):
    """Analyzes and visualizes the overall distribution of prediction errors."""
    logger.info("\n" + "="*50)
    logger.info("1. Overall Error Distribution Analysis")
    logger.info("="*50)

    error_stats = df_eval['prediction_error_kg'].describe()
    logger.info("Descriptive Statistics for Prediction Error (kg):\n" + error_stats.to_string())

    mean_error = df_eval['prediction_error_kg'].mean()
    if mean_error > 1:
        logger.info(f"\nModel shows a tendency to OVERPREDICT (average error: {mean_error:.2f} kg).")
    elif mean_error < -1:
        logger.info(f"\nModel shows a tendency to UNDERPREDICT (average error: {mean_error:.2f} kg).")
    else:
        logger.info("\nModel appears to be well-calibrated with no significant overall bias.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_eval['prediction_error_kg'], kde=True, ax=ax)
    ax.set_title('Distribution of Prediction Errors')
    ax.set_xlabel('Prediction Error (Predicted - Actual) in kg')
    ax.set_ylabel('Frequency')
    ax.axvline(mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.2f}')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "error_distribution_histogram.png")
    plt.savefig(plot_path)
    logger.info(f"Saved error distribution histogram to {plot_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='actual_fuel_kg', y='predicted_fuel_kg', data=df_eval, alpha=0.5, ax=ax)
    ax.plot([df_eval['actual_fuel_kg'].min(), df_eval['actual_fuel_kg'].max()], 
            [df_eval['actual_fuel_kg'].min(), df_eval['actual_fuel_kg'].max()], 
            'r--', label='Perfect Prediction')
    ax.set_title('Actual vs. Predicted Fuel Consumption')
    ax.set_xlabel('Actual Fuel (kg)')
    ax.set_ylabel('Predicted Fuel (kg)')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "actual_vs_predicted_scatter.png")
    plt.savefig(plot_path)
    logger.info(f"Saved actual vs. predicted scatter plot to {plot_path}")
    plt.close(fig)

def analyze_worst_predictions(df_merged, output_dir, logger, num_worst=20):
    """Identifies and reports on the top N worst predictions."""
    logger.info("\n" + "="*50)
    logger.info(f"2. Analysis of Top {num_worst} Worst Predictions")
    logger.info("="*50)

    worst_predictions = df_merged.nlargest(num_worst, 'absolute_error_kg')
    
    key_cols = [
        'flight_id', 'aircraft_type', 'origin_icao', 'destination_icao',
        'segment_duration', 'actual_fuel_kg', 'predicted_fuel_kg', 'prediction_error_kg',
        'great_circle_distance_km', 'flight_duration_seconds', 'seg_altitude_mean', 'seg_groundspeed_mean'
    ]
    existing_key_cols = [col for col in key_cols if col in worst_predictions.columns]

    logger.info(f"Top {num_worst} Worst Predictions (by absolute error):\n" + worst_predictions[existing_key_cols].to_string())
    
    csv_path = os.path.join(output_dir, "worst_predictions_details.csv")
    worst_predictions.to_csv(csv_path)
    logger.info(f"\nFull details of worst predictions saved to: {csv_path}")

def analyze_aircraft_type_distribution(df_merged, output_dir, logger):
    """Analyzes and visualizes the distribution of aircraft types."""
    logger.info("\n" + "="*50)
    logger.info("3. Aircraft Type Distribution Analysis")
    logger.info("="*50)
    
    # We need to count unique flights, not segments, for a true distribution
    aircraft_distribution = df_merged.drop_duplicates(subset='flight_id')['aircraft_type'].value_counts()
    
    logger.info("Flight Count by Aircraft Type:\n" + aircraft_distribution.to_string())
    
    # Plot the distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    aircraft_distribution.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Number of Unique Flights per Aircraft Type')
    ax.set_xlabel('Number of Flights')
    ax.set_ylabel('Aircraft Type')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "aircraft_type_distribution.png")
    plt.savefig(plot_path)
    logger.info(f"Saved aircraft type distribution plot to {plot_path}")
    plt.close(fig)

def analyze_errors_by_feature(df_merged, feature, output_dir, logger, is_numeric=True, bins=10):
    """Analyzes prediction errors across different values of a given feature."""
    if feature not in df_merged.columns:
        logger.warning(f"Feature '{feature}' not found in data. Skipping analysis for this feature.")
        return

    logger.info("\n" + "-"*50)
    logger.info(f"Error Analysis by Feature: '{feature}'")
    logger.info("-"*50)

    grouping_feature = feature
    if is_numeric:
        if df_merged[feature].nunique() > bins:
            try:
                df_merged[f'{feature}_bin'] = pd.cut(df_merged[feature], bins=bins, duplicates='drop')
                grouping_feature = f'{feature}_bin'
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not create bins for '{feature}'. Analyzing by unique value instead. Error: {e}")
        else:
            logger.info(f"Feature '{feature}' has few unique values, analyzing by unique value.")

    error_by_feature = df_merged.groupby(grouping_feature).agg(
        mean_abs_error=('absolute_error_kg', 'mean'),
        mean_error=('prediction_error_kg', 'mean'),
        count=('flight_id', 'count')
    ).sort_values(by='mean_abs_error', ascending=False)

    logger.info(f"Average Error Metrics by '{feature}':\n" + error_by_feature.to_string())

    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        error_by_feature.index.astype(str)
        error_by_feature['mean_abs_error'].plot(kind='bar', ax=ax)
        ax.set_title(f'Mean Absolute Error by {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Mean Absolute Error (kg)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"error_by_{feature}.png")
        plt.savefig(plot_path)
        logger.info(f"Saved error analysis plot for '{feature}' to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Could not generate plot for feature '{feature}'. Error: {e}")

def analyze_errors_by_flight_phase(df_merged, output_dir, logger):
    """Analyzes prediction errors by detected flight phase."""
    logger.info("\n" + "="*50)
    logger.info("4. Error Analysis by Flight Phase")
    logger.info("="*50)

    if 'phase' not in df_merged.columns:
        logger.warning("Flight phase information ('phase' column) not found. Skipping this analysis.")
        return

    error_by_phase = df_merged.groupby('phase').agg(
        mean_abs_error=('absolute_error_kg', 'mean'),
        mean_error=('prediction_error_kg', 'mean'),
        count=('flight_id', 'size')
    ).sort_values(by='mean_abs_error', ascending=False)

    logger.info("Average Error Metrics by Flight Phase:\n" + error_by_phase.to_string())

    fig, ax = plt.subplots(figsize=(12, 7))
    error_by_phase['mean_abs_error'].plot(kind='bar', ax=ax)
    ax.set_title('Mean Absolute Error by Flight Phase')
    ax.set_xlabel('Flight Phase')
    ax.set_ylabel('Mean Absolute Error (kg)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "error_by_flight_phase.png")
    plt.savefig(plot_path)
    logger.info(f"Saved error analysis plot for flight phase to {plot_path}")
    plt.close(fig)

def plot_residual_analysis(df_eval, output_dir, logger):
    """Plots residuals to diagnose model issues like bias and heteroscedasticity."""
    logger.info("\n" + "="*50)
    logger.info("5. Residual Analysis")
    logger.info("="*50)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='predicted_fuel_kg', y='prediction_error_kg', data=df_eval, alpha=0.4, ax=ax)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title('Residual Plot (Error vs. Predicted Value)')
    ax.set_xlabel('Predicted Fuel (kg)')
    ax.set_ylabel('Prediction Error (kg)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "residual_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Saved residual plot to {plot_path}")
    plt.close(fig)

def analyze_errors_by_route(df_merged, output_dir, logger, num_routes=20):
    """Analyzes prediction errors on a per-route basis."""
    logger.info("\n" + "="*50)
    logger.info(f"6. Error Analysis by Route (Top {num_routes})")
    logger.info("="*50)

    df_merged['route'] = df_merged['origin_icao'] + '-' + df_merged['destination_icao']
    
    error_by_route = df_merged.groupby('route').agg(
        mean_abs_error=('absolute_error_kg', 'mean'),
        mean_error=('prediction_error_kg', 'mean'),
        count=('flight_id', 'size')
    ).sort_values(by='mean_abs_error', ascending=False)

    logger.info(f"Top {num_routes} Worst Routes by Mean Absolute Error:\n" + error_by_route.head(num_routes).to_string())

    fig, ax = plt.subplots(figsize=(12, 8))
    error_by_route.head(num_routes).sort_values(by='mean_abs_error').plot(kind='barh', y='mean_abs_error', ax=ax)
    ax.set_title(f'Top {num_routes} Worst Routes by Mean Absolute Error')
    ax.set_xlabel('Mean Absolute Error (kg)')
    ax.set_ylabel('Route')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "error_by_route.png")
    plt.savefig(plot_path)
    logger.info(f"Saved error analysis plot for routes to {plot_path}")
    plt.close(fig)

def analyze_error_concentration(df_eval, output_dir, logger):
    """Analyzes the concentration of errors in the worst predictions."""
    logger.info("\n" + "="*50)
    logger.info("7. Error Concentration Analysis")
    logger.info("="*50)

    df_eval_sorted = df_eval.sort_values(by='absolute_error_kg', ascending=False)
    total_abs_error = df_eval_sorted['absolute_error_kg'].sum()
    
    percentages = [0.01, 0.05, 0.10, 0.25]
    for p in percentages:
        num_top = int(len(df_eval_sorted) * p)
        top_errors_sum = df_eval_sorted.head(num_top)['absolute_error_kg'].sum()
        error_concentration = (top_errors_sum / total_abs_error) * 100
        logger.info(f"The top {p*100:.0f}% of predictions account for {error_concentration:.2f}% of the total absolute error.")

def analyze_error_correlation_with_numeric_features(df_merged, output_dir, logger, top_n=10):
    """
    Analyzes the correlation between absolute error and key numerical features.
    Logs the top N features with the highest absolute correlation.
    """
    logger.info("\n" + "="*50)
    logger.info(f"8. Correlation of Absolute Error with Numerical Features (Top {top_n})")
    logger.info("="*50)

    numeric_cols = df_merged.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude target and error-related columns
    exclude_cols = ['actual_fuel_kg', 'predicted_fuel_kg', 'prediction_error_kg', 'absolute_error_kg', 'squared_error_kg2']
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

    correlations = []
    for col in analysis_cols:
        if df_merged[col].nunique() > 1: # Ensure there's variance to correlate
            try:
                # Using Spearman for non-linear relationships and robustness to outliers
                corr = df_merged['absolute_error_kg'].corr(df_merged[col], method='spearman')
                correlations.append({'feature': col, 'spearman_correlation': corr})
            except Exception as e:
                logger.warning(f"Could not calculate correlation for '{col}': {e}")

    df_correlations = pd.DataFrame(correlations)
    if not df_correlations.empty:
        df_correlations['abs_correlation'] = df_correlations['spearman_correlation'].abs()
        df_correlations = df_correlations.sort_values(by='abs_correlation', ascending=False)
        
        logger.info(f"Top {top_n} Numerical Features by Absolute Spearman Correlation with Absolute Error:\n" + 
                    df_correlations.head(top_n).to_string(index=False))
    else:
        logger.info("No numerical features found for correlation analysis.")

def analyze_errors_by_time_of_flight(df_merged, output_dir, logger):
    """Analyzes prediction errors by time of day and month."""
    logger.info("\n" + "="*50)
    logger.info("9. Error Analysis by Time of Flight (Hour and Month)")
    logger.info("="*50)

    df_temp = df_merged.copy() # Work on a copy to avoid modifying original df_merged in place

    # Analyze by Month
    if 'flight_date' in df_temp.columns:
        try:
            df_temp['flight_date_dt'] = pd.to_datetime(df_temp['flight_date'])
            df_temp['month'] = df_temp['flight_date_dt'].dt.month
            analyze_errors_by_feature(df_temp, 'month', output_dir, logger, is_numeric=False)
        except Exception as e:
            logger.warning(f"Could not analyze errors by month: {e}")
    else:
        logger.warning("Feature 'flight_date' not found. Skipping month analysis.")

    # Analyze by Hour of Day
    if 'seg_start_time_decimal' in df_temp.columns:
        try:
            df_temp['start_hour'] = df_temp['seg_start_time_decimal'].astype(int)
            analyze_errors_by_feature(df_temp, 'start_hour', output_dir, logger, is_numeric=False)
        except Exception as e:
            logger.warning(f"Could not analyze errors by hour: {e}")
    else:
        logger.warning("Feature 'seg_start_time_decimal' not found. Skipping hour analysis.")

def analyze_errors_by_aircraft_characteristics(df_merged, output_dir, logger):
    """Analyzes prediction errors by aircraft weight and size characteristics."""
    logger.info("\n" + "="*50)
    logger.info("10. Error Analysis by Aircraft Characteristics (Weight/Size)")
    logger.info("="*50)

    aircraft_numeric_features = ['mtow', 'mlw', 'oew', 'mfc', 'fuselage_length', 'wing_span']
    
    for feature in aircraft_numeric_features:
        if feature in df_merged.columns:
            # Use fewer bins for these to get meaningful groups
            analyze_errors_by_feature(df_merged, feature, output_dir, logger, is_numeric=True, bins=5) 
        else:
            logger.warning(f"Aircraft characteristic feature '{feature}' not found. Skipping analysis.")

def analyze_errors_by_weather_severity(df_merged, output_dir, logger):
    """Analyzes prediction errors by weather severity indicators."""
    logger.info("\n" + "="*50)
    logger.info("11. Error Analysis by Weather Severity")
    logger.info("="*50)

    weather_categorical_features = [
        'dep_wx_is_thunderstorm', 'dep_wx_is_freezing', 'dep_wx_is_shower', 'dep_wx_is_rain', 'dep_wx_is_snow', 'dep_wx_is_fog_mist', 'dep_wx_is_haze_smoke',
        'arr_wx_is_thunderstorm', 'arr_wx_is_freezing', 'arr_wx_is_shower', 'arr_wx_is_rain', 'arr_wx_is_snow', 'arr_wx_is_fog_mist', 'arr_wx_is_haze_smoke'
    ]
    weather_numeric_features = ['dep_wx_intensity', 'arr_wx_intensity']

    for feature in weather_categorical_features:
        if feature in df_merged.columns:
            # These are typically boolean (0/1) or very few categories
            analyze_errors_by_feature(df_merged, feature, output_dir, logger, is_numeric=False)
        else:
            logger.warning(f"Weather severity feature '{feature}' not found. Skipping analysis.")
            
    for feature in weather_numeric_features:
        if feature in df_merged.columns:
            # Weather intensity might have a range, so binning is useful
            analyze_errors_by_feature(df_merged, feature, output_dir, logger, is_numeric=True, bins=5)
        else:
            logger.warning(f"Weather severity feature '{feature}' not found. Skipping analysis.")


def main(model_dir_path):
    """Main function to run the comprehensive error analysis."""
    
    logger = setup_logging(model_dir_path)
    logger.info(f"--- Starting Comprehensive Error Analysis for model in {model_dir_path} ---")

    eval_details_path = os.path.join(model_dir_path, "evaluation_details.csv")
    if not os.path.exists(eval_details_path):
        logger.error(f"Evaluation details not found at {eval_details_path}. Please run the 'evaluate' stage first.")
        return
    # Load the evaluation data, using the first column as the index.
    df_eval = pd.read_csv(eval_details_path, index_col=0)
    
    # The index name is set during saving, but we can be explicit.
    df_eval.index.name = 'original_index'

    is_test_run = "X_val.parquet" in os.listdir(model_dir_path)
    data_filename = 'featured_data_train_test.parquet' if is_test_run else 'featured_data_train.parquet'
    logger.info(f"Detected a {'TEST' if is_test_run else 'FULL'} run. Loading '{data_filename}'.")
        
    train_data_path = os.path.join(config.PROCESSED_DATA_DIR, data_filename)
    if not os.path.exists(train_data_path):
        logger.error(f"Full training data not found at {train_data_path}. Cannot perform detailed analysis.")
        return
    df_train_full = pd.read_parquet(train_data_path)

    # Drop the redundant flight_id from the evaluation frame before joining
    if 'flight_id' in df_eval.columns:
        df_eval = df_eval.drop(columns=['flight_id'])

    # The join should now work correctly as both dataframes share the same index.
    df_merged = df_train_full.join(df_eval, how='inner')
    if df_merged.empty:
        logger.error("Failed to merge evaluation data with training features. Index mismatch is likely.")
        return
    logger.info(f"Successfully merged evaluation data with features, resulting in {len(df_merged)} rows.")

    # --- Run Analysis Modules ---
    analyze_error_distribution(df_eval, model_dir_path, logger)
    analyze_worst_predictions(df_merged, model_dir_path, logger)
    analyze_aircraft_type_distribution(df_merged, model_dir_path, logger)
    analyze_errors_by_flight_phase(df_merged, model_dir_path, logger)
    plot_residual_analysis(df_eval, model_dir_path, logger)
    analyze_errors_by_route(df_merged, model_dir_path, logger)
    analyze_error_concentration(df_eval, model_dir_path, logger)

    analyze_error_correlation_with_numeric_features(df_merged, model_dir_path, logger)
    
    # New analysis modules
    analyze_errors_by_time_of_flight(df_merged, model_dir_path, logger)
    analyze_errors_by_aircraft_characteristics(df_merged, model_dir_path, logger)
    analyze_errors_by_weather_severity(df_merged, model_dir_path, logger)

    # Define features to analyze (existing list, can be extended further if needed)
    phase_duration_features = [col for col in df_merged.columns if 'phase_duration_' in col]
    numeric_features_to_analyze = [
        'segment_duration', 'great_circle_distance_km', 'flight_duration_seconds',
        'seg_altitude_mean', 'seg_groundspeed_mean', 'start_alt_rev', 'alt_diff_rev',
        'dep_sknt', 'arr_sknt', 'dep_tmpf', 'arr_tmpf', 'dep_vsby', 'arr_vsby' # Added key weather features
    ] + phase_duration_features
    
    categorical_features_to_analyze = [
        'aircraft_type', 'origin_icao', 'destination_icao', 'seg_start_day_of_week'
    ]

    for feature in numeric_features_to_analyze:
        analyze_errors_by_feature(df_merged, feature, model_dir_path, logger, is_numeric=True)
    
    for feature in categorical_features_to_analyze:
        analyze_errors_by_feature(df_merged, feature, model_dir_path, logger, is_numeric=False)

    logger.info("\n--- Comprehensive Error Analysis Complete ---")
    logger.info(f"All reports and plots saved in: {model_dir_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform comprehensive error analysis on a trained model.")
    parser.add_argument("model_dir", type=str, nargs='?', default=None, help="Optional: The full path to the model directory.")
    
    args = parser.parse_args()
    
    model_to_analyze = args.model_dir
    
    if model_to_analyze is None or not os.path.isdir(model_to_analyze):
        if model_to_analyze is not None:
            print(f"'{model_to_analyze}' is not a valid directory. Please choose from the list below:")
        else:
            print("No model directory provided. Please choose from the list below:")

        saved_model_dirs = sorted([
            d for d in os.listdir(config.MODELS_DIR) 
            if os.path.isdir(os.path.join(config.MODELS_DIR, d))
        ])
        if not saved_model_dirs:
            print("Error: No trained models found in the models directory.")
        else:
            for i, model_dir_name in enumerate(saved_model_dirs):
                print(f"[{i+1}] {model_dir_name}")
            try:
                choice = int(input("\nPlease select a model: ").strip()) - 1
                if 0 <= choice < len(saved_model_dirs):
                    model_to_analyze = os.path.join(config.MODELS_DIR, saved_model_dirs[choice])
                    main(model_to_analyze)
                else:
                    print("Invalid selection.")
            except (ValueError, IndexError):
                print("Invalid input.")
    else:
        main(model_to_analyze)
