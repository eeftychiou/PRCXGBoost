"""
This script enriches a list of aircraft types with detailed performance data
by querying the openap library. It ensures all original aircraft types are present
in the final output, even if they are not found in the database.
"""
import os
import pandas as pd
from tqdm import tqdm
import json

# Suppress the informational message from openap
import logging
logging.getLogger("openap").setLevel(logging.WARNING)

from openap import prop

def flatten_dict(d, parent_key='', sep='_'):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v))) # Convert lists to JSON strings
        else:
            items.append((new_key, v))
    return dict(items)

def enrich_aircraft_data():
    """Loads aircraft types, queries openap, and saves enriched data."""
    print("--- Starting Aircraft Data Enrichment ---")

    input_csv = "unique_aircraft_types.csv"
    output_csv = "enriched_aircraft_details.csv"

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at '{input_csv}'. Please run extract_aircraft_types.py first.")
        return

    df_types = pd.read_csv(input_csv)
    print(f"Loaded {len(df_types)} unique aircraft types from {input_csv}.")

    enriched_data = []
    not_found_count = 0

    for ac_type in tqdm(df_types['aircraft_type'], desc="Querying openap"):
        try:
            aircraft_details = prop.aircraft(ac_type)
            flat_details = flatten_dict(aircraft_details)
            flat_details['query_aircraft_type'] = ac_type
            enriched_data.append(flat_details)

        except:
            # If aircraft not found, create a placeholder record to ensure it's in the output
            not_found_count += 1
            missing_record = {'query_aircraft_type': ac_type}
            enriched_data.append(missing_record)
            continue

    if not enriched_data:
        print("Could not retrieve or create records for any of the aircraft types.")
        return

    print(f"\nSuccessfully retrieved data for {len(enriched_data) - not_found_count} aircraft types.")
    if not_found_count > 0:
        print(f"Could not find {not_found_count} aircraft types in the openap database. These will be included as empty rows.")

    # --- Convert to DataFrame and Save ---
    df_enriched = pd.DataFrame(enriched_data)
    
    cols = ['query_aircraft_type'] + [col for col in df_enriched.columns if col != 'query_aircraft_type']
    df_enriched = df_enriched[cols]

    df_enriched.to_csv(output_csv, index=False)

    print(f"--- Enrichment Complete ---")
    print(f"Enriched aircraft data saved to: {output_csv}")

if __name__ == '__main__':
    enrich_aircraft_data()
