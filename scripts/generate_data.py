import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification



def collect_from_database(query: str) -> pd.DataFrame:
    """
    Alex's original data generation function - extracted as-is.
    This simulates pulling from a SQL database (Alex's comment).
    """
    # Basic input validation
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    print(f"Executing: {query}")

    # Alex's original parameters
    n_rows = 10_000
    n_features = 16

    # Step 1: Generate base features using sklearn
    features, labels = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=7,
        n_redundant=4,
        n_repeated=3,
        n_classes=2,
        class_sep=1.2,
        flip_y=0.035,  # Alex's comment: Randomly invert y for added noise
        weights=[0.85, 0.15],
        random_state=1889,  # Alex's seed
    )

    # Step 2: Create DataFrame with Alex's structure
    df = pd.DataFrame(features, columns=[f'numeric_{i + 1}' for i in range(n_features)])
    df.insert(value=labels, loc=0, column='claim_status')

    # Step 3: Rename to business features (Alex's mapping)
    df = df.rename(columns={
        'numeric_1': 'age',
        'numeric_2': 'height_cm',
        'numeric_3': 'weight_kg',
        'numeric_4': 'income',
        'numeric_5': 'financial_hist_1',
        'numeric_6': 'financial_hist_2',
        'numeric_7': 'financial_hist_3',
        'numeric_8': 'financial_hist_4',
        'numeric_9': 'credit_score_1',
        'numeric_10': 'credit_score_2',
        'numeric_11': 'credit_score_3',
        'numeric_12': 'insurance_hist_1',
        'numeric_13': 'insurance_hist_2',
        'numeric_14': 'insurance_hist_3',
        'numeric_15': 'insurance_hist_4',
        'numeric_16': 'insurance_hist_5',
    })

    print(f"Generated {len(df)} rows with {len(df.columns)} initial features")
    return df


def save_sample_data():
    """Generate and save sample data as JSON for demo purposes."""

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Generate the dataset using Alex's logic
    print("Generating sample dataset...")
    df = collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")

    # Save as JSON (preserves data types and handles None values)
    output_file = data_dir / "sample_applications.json"
    df.to_json(output_file, orient='records', indent=2)

    print(f"Saved {len(df)} applications to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return df


if __name__ == "__main__":
    # Test Step 1
    print("Testing Step 1: Basic data generation")
    df = collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Claim status distribution:\n{df['claim_status'].value_counts()}")
    print("\nFirst few rows:")
    print(df.head())

    # Save the data
    print("\n" + "=" * 50)
    save_sample_data()

    print("\nStep 1 completed! Data ready for processing.")