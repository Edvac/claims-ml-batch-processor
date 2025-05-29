"""
MVP Claims Prediction Pipeline

Minimal viable pipeline that demonstrates:
- Load data
- Process with Alex's logic
- Generate predictions
- Save results

Start simple, then enhance.
"""
import argparse

import pandas as pd
from pathlib import Path
from datetime import datetime

# Import Alex's modules
from claims.data_processing import process_claims_data
from claims.model import load_model, predict_claims
from claims.training import prepare_training_data, train_basic_model


def train_model():
    """
    MVP model training - just the basics to get a working model.
    """
    print("=" * 50)
    print("TRAINING MODEL (MVP)")
    print("=" * 50)

    try:
        # Load training data
        print("Loading training data...")
        data_path = Path(__file__).parent / "data" / "sample_applications.csv"
        training_data = pd.read_csv(data_path)
        print(f"Loaded {len(training_data)} records for training")

        # Process training data
        print("Processing training data...")
        processed_data = process_claims_data(training_data)

        # Prepare train/test splits
        print("Preparing train/test splits...")
        X_train, X_test, y_train, y_test = prepare_training_data(processed_data)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Train basic model (skip the complex CV for MVP)
        print("Training basic XGBoost model...")
        model = train_basic_model(X_train, y_train, X_test, y_test)
        print("Model training completed!")

        print("=" * 50)
        print("MODEL TRAINING COMPLETED - Ready for predictions!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"CRITICAL ERROR: Model training failed: {e}")
        return False

def process_single_file(file_path, model, output_dir):
    """
    CRITICAL: Process single file for batch operations.
    """
    try:
        print(f"Processing: {file_path}")

        # Load and process data
        raw_data = pd.read_csv(file_path)
        processed_data = process_claims_data(raw_data)

        # Prepare features
        if 'claim_status' in processed_data.columns:
            X_data = processed_data.drop('claim_status', axis=1)
        else:
            X_data = processed_data

        # Generate predictions
        class_predictions, prob_predictions = predict_claims(model, X_data)

        # Create results
        results_df = pd.DataFrame({
            'application_id': range(1, len(X_data) + 1),
            'claim_likelihood': prob_predictions,
            'claim_prediction': class_predictions,
            'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low'
                           for p in prob_predictions]
        })

        # Save results
        file_stem = Path(file_path).stem
        output_file = output_dir / f"{file_stem}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)

        print(f"  -> Processed {len(X_data)} records, saved to: {output_file}")
        return True

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to process {file_path}: {e}")
        return False

def run_simple_pipeline():
    """
    Minimal viable pipeline - just the core flow.
    """
    print("=" * 50)
    print("STARTING MVP CLAIMS PREDICTION PIPELINE")
    print("=" * 50)

    try:
        # Step 1: Load data (simulate daily ETL)
        print("Step 1: Loading data...")
        data_path = Path(__file__).parent / "data" / "sample_applications.csv"
        raw_data = pd.read_csv(data_path)
        print(f"Loaded {len(raw_data)} applications")

        # Step 2: Process data (Alex's logic)
        print("Step 2: Processing data...")
        processed_data = process_claims_data(raw_data)
        print(f"Processed data shape: {processed_data.shape}")

        # Step 3: Load model and predict
        print("Step 3: Loading model and generating predictions...")
        model = load_model()

        if model is None:
            print(f"No trained model found. Please train a model first.")
            return False

        # Prepare features (remove target if present)
        if 'claim_status' in processed_data.columns:
            X_data = processed_data.drop('claim_status', axis=1)
        else:
            X_data = processed_data

        # Generate predictions
        class_predictions, prob_predictions = predict_claims(model, X_data)
        print(f"Generated predictions for {len(X_data)} applications")

        # Step 4: Create simple results
        print("Step 4: Creating results...")
        results_df = pd.DataFrame({
            'application_id': range(1, len(X_data) + 1),
            'claim_likelihood': prob_predictions,
            'claim_prediction': class_predictions,
            'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low'
                           for p in prob_predictions]
        })

        # Quick summary
        risk_summary = results_df['risk_level'].value_counts()
        print(f"Risk Distribution: {risk_summary.to_dict()}")

        # Step 5: Save results
        print("Step 5: Saving results...")
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        print("=" * 50)
        print("MVP PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVP Claims Pipeline")
    parser.add_argument("--train", action="store_true", help="Train model first")

    args = parser.parse_args()

    if args.train:
        success = train_model()  # FIXED: was train_model_mvp()
    else:
        success = run_simple_pipeline()

    if not success:
        exit(1)
