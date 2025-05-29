"""
Alex's "Data Wrangling" section from notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def process_claims_data(dataset_from_database):
    """Alex's data wrangling logic."""

    print(f"Processing dataset: {dataset_from_database.shape}")

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


    dataset_from_database_no_missing_values = pd.DataFrame()
    dataset_from_database_no_missing_values = dataset_from_database.drop(
        columns=['family_history_3', 'employment_type'])
    dataset_from_database.drop(columns=['family_history_3', 'employment_type'], inplace=True)

    print("Dropped columns with missing values")

    # Alex's column inspection
    for i in dataset_from_database.columns:
        print(f"Column: {i}, dtype: {dataset_from_database[i].dtype}")

    # Alex's categorical conversion
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected',
                     'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2',
                     'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 'product_var_3',
                     'health_status', 'driving_record', 'previous_claim_rate', 'education_level', 'income_level',
                     'n_dependents']

    for column in non_numerical:
        dataset_from_database[column] = dataset_from_database[column].astype('category')

    print(f"Converted {len(non_numerical)} columns to categorical")
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