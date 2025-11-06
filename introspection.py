import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

def generate_introspection_files(df: pd.DataFrame, phase: str, run_id: str, target_variable: str = None):
    """
    Generates and saves introspection files for a given phase of the pipeline.

    These files provide insights into the data at different stages, which can be
    used for debugging, analysis, and improving the model.

    For each phase, the following files are generated:
    - {phase}_output.csv: The main DataFrame of the phase.
    - {phase}_summary.txt: A text file with a summary of the DataFrame.
    - {phase}_missing_values.csv: A report on missing values.
    - {phase}_descriptive_stats.csv: Descriptive statistics for numerical columns.
    - {phase}_correlation_matrix.csv: The full correlation matrix.
    - {phase}_correlation_heatmap.png: A heatmap of the correlation matrix.
    - {phase}_target_correlation.csv: Correlation of features with the target variable.

    Args:
        df (pd.DataFrame): The DataFrame to be analyzed.
        phase (str): The name of the pipeline phase.
        run_id (str): The unique identifier for the current pipeline run.
        target_variable (str, optional): The name of the target variable for correlation analysis.
    """
    introspection_phase_dir = os.path.join(config.INTROSPECTION_DIR, run_id, phase)
    os.makedirs(introspection_phase_dir, exist_ok=True)
    print(f"Generating introspection files for '{phase}' phase in {introspection_phase_dir}")

    # 1. Save the main DataFrame
    df.to_csv(os.path.join(introspection_phase_dir, f"{phase}_output.csv"), index=False)

    # 2. Generate a summary file
    summary_path = os.path.join(introspection_phase_dir, f"{phase}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Introspection Summary for '{phase}' phase\n")
        f.write("="*40 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape of the DataFrame: {df.shape}\n\n")
        f.write("DataFrame Info:\n")
        df.info(buf=f)

    # 3. Generate a missing values report
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['column', 'missing_count']
    missing_values['missing_percentage'] = (missing_values['missing_count'] / len(df)) * 100
    missing_values = missing_values[missing_values['missing_count'] > 0].sort_values(by='missing_percentage', ascending=False)
    missing_values.to_csv(os.path.join(introspection_phase_dir, f"{phase}_missing_values.csv"), index=False)

    # 4. Generate descriptive statistics for numerical columns
    descriptive_stats = df.describe().transpose()
    descriptive_stats.to_csv(os.path.join(introspection_phase_dir, f"{phase}_descriptive_stats.csv"))

    # 5. Generate Correlation Analysis Files
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    if not numerical_df.empty:
        # Full correlation matrix
        corr_matrix = numerical_df.corr()
        corr_matrix.to_csv(os.path.join(introspection_phase_dir, f"{phase}_correlation_matrix.csv"))

        # Correlation heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title(f"Correlation Matrix for {phase} phase")
        plt.savefig(os.path.join(introspection_phase_dir, f"{phase}_correlation_heatmap.png"))
        plt.close()

        # Correlation with target variable
        if target_variable and target_variable in numerical_df.columns:
            target_corr = corr_matrix[[target_variable]].sort_values(by=target_variable, ascending=False)
            target_corr.to_csv(os.path.join(introspection_phase_dir, f"{phase}_target_correlation.csv"))

    print(f"Successfully generated introspection files for '{phase}' phase.")
