import pandas as pd
import sys

def main():
    """
    Interactively loads, processes (samples or filters), and saves a Parquet file.
    """
    print("--- Parquet File Processor ---")

    # 1. Ask for the input file name
    try:
        input_path = input("Enter the path to the input Parquet file: ")
        df = pd.read_parquet(input_path)
        print(f"Successfully loaded '{input_path}'.")
        print(f"DataFrame shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    processed_df = None

    # 2. Ask whether to sample or filter
    while True:
        choice = input("Do you want to 'sample' or 'filter' the data? ").lower().strip()
        if choice in ['sample', 'filter']:
            break
        print("Invalid choice. Please enter 'sample' or 'filter'.")

    # 3. Sample the data
    if choice == 'sample':
        while True:
            try:
                n_samples = int(input("Enter the number of samples (rows) to select: "))
                if n_samples > len(df):
                    print(f"Warning: Number of samples ({n_samples}) is larger than the DataFrame size ({len(df)}). Using all rows.")
                    n_samples = len(df)
                processed_df = df.sample(n=n_samples, random_state=42)
                print(f"Sampled {len(processed_df)} rows from the DataFrame.")
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    # 4. Filter the data by flight_id
    elif choice == 'filter':
        if 'flight_id' not in df.columns:
            print("Error: 'flight_id' column not found in the DataFrame. Cannot filter.", file=sys.stderr)
            sys.exit(1)
            
        while True:
            flight_ids_str = input("Enter the flight_ids to keep (comma-separated): ")
            flight_ids = [fid.strip() for fid in flight_ids_str.split(',') if fid.strip()]
            
            if not flight_ids:
                print("No flight_ids provided. Please enter at least one.")
                continue

            # Convert flight_ids to the correct type if necessary
            try:
                # Attempt to convert to the same type as the DataFrame column
                flight_id_dtype = df['flight_id'].dtype
                flight_ids = [flight_id_dtype.type(fid) for fid in flight_ids]
            except (ValueError, TypeError):
                print(f"Warning: Could not convert input to the same type as 'flight_id' column ({flight_id_dtype}). Proceeding with string comparison.")

            mask = df['flight_id'].isin(flight_ids)
            processed_df = df[mask]
            
            if processed_df.empty:
                print("Warning: No matching flight_ids found. The resulting DataFrame is empty.")
            else:
                print(f"Filtered the DataFrame. Kept {len(processed_df)} rows for {len(flight_ids)} flight_id(s).")
            break

    # 5. Ask for a name and save the parquet file
    if processed_df is not None:
        while True:
            output_path = input("Enter the path to save the new Parquet file (e.g., 'processed_data.parquet'): ")
            if not output_path.lower().endswith('.parquet'):
                print("Warning: The file name does not end with '.parquet'. It's recommended to use this extension.")
            
            try:
                processed_df.to_parquet(output_path, index=False)
                print(f"Successfully saved the processed data to '{output_path}'.")
                break
            except Exception as e:
                print(f"An error occurred while saving the file: {e}", file=sys.stderr)
                retry = input("Do you want to try saving again with a different name? (yes/no): ").lower()
                if retry != 'yes':
                    break
    else:
        print("No data was processed, so no file will be saved.")

if __name__ == '__main__':
    main()
