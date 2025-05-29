"""
Alex's "Data Wrangling" section from notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def process_claims_data(dataset_from_database):
    """Alex's data wrangling logic."""

    print(f"Processing dataset: {dataset_from_database.shape}")

    # The model won't be useful.
    if dataset_from_database.empty:
        raise ValueError("Dataset is empty")


    total = dataset_from_database.isnull().sum()
    percent = (dataset_from_database.isnull().sum() / dataset_from_database.isnull().count() * 100)
    missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in dataset_from_database.columns:
        dtype = str(dataset_from_database[col].dtype)
        types.append(dtype)
    missing_df['Types'] = types

    print("Missing values analysis:")
    print(missing_df[missing_df['Total'] > 0])


    # Safe column dropping with assignment - lack will lead to crashing the pipeline
    columns_to_drop = []
    if 'family_history_3' in dataset_from_database.columns:
        columns_to_drop.append('family_history_3')
    if 'employment_type' in dataset_from_database.columns:
        columns_to_drop.append('employment_type')

    if columns_to_drop:
        dataset_from_database = dataset_from_database.drop(columns=columns_to_drop)
        print(f"Dropped columns with missing values: {columns_to_drop}")

    # Gracefully handles missing columns and converts string categorical values to pandas
    # categorical type for efficient XGBoost processing
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected',
                     'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2',
                     'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 'product_var_3',
                     'health_status', 'driving_record', 'previous_claim_rate', 'education_level', 'income_level',
                     'n_dependents']

    converted_count = 0
    for column in non_numerical:
        if column in dataset_from_database.columns:
            try:
                dataset_from_database[column] = dataset_from_database[column].astype('category')
                converted_count += 1
            except Exception as e:
                print(f"Warning: Could not convert {column} to categorical: {e}")

    print(f"Converted {converted_count} columns to categorical")
    print(f"Final processed shape: {dataset_from_database.shape}")

    return dataset_from_database


if __name__ == "__main__":
    # Load and process data when run directly
    data_path = Path(__file__).parent.parent / "data" / "sample_applications.csv"
    dataset_from_database = pd.read_csv(data_path)

    processed_data = process_claims_data(dataset_from_database)

    # Save processed data
    processed_path = Path(__file__).parent.parent / "data" / "processed_applications.csv"
    processed_data.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")