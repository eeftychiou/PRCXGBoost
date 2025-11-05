import os
import pandas as pd
import config

def fix_submission_columns():
    """Loads the submission file, renames a column, and saves it back."""
    submission_path = os.path.join(config.BASE_DATASETS_DIR, 'fuel_rank_submission.parquet')

    if not os.path.exists(submission_path):
        print(f"Error: Submission file not found at {submission_path}")
        return

    print(f"Loading submission file from {submission_path}...")
    df_submission = pd.read_parquet(submission_path)

    if 'fuel_consumption' in df_submission.columns:
        print("Renaming column 'fuel_consumption' to 'fuel_kg'...")
        df_submission.rename(columns={'fuel_consumption': 'fuel_kg'}, inplace=True)
        
        print("Saving updated submission file...")
        df_submission.to_parquet(submission_path, index=False)
        print(f"File saved successfully to {submission_path}")
        print("\nUpdated Submission Head:")
        print(df_submission.head())
    else:
        print("Column 'fuel_consumption' not found. No changes made.")

if __name__ == '__main__':
    fix_submission_columns()
