"""
Alex's model operations from notebook.

Alex used direct XGBoost operations throughout his notebook for model loading,
prediction, and saving. 
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path


def load_model(model_path=None):
    """
    Alex's model loading pattern.
    He checked for optimized model first, then basic model.
    """
    if model_path is None:
        models_dir = Path(__file__).parent.parent / "models"

        # Alex's optimized model filename
        optimized_model_path = models_dir / "xgboost_model_optimised_with_cross_validation.json"
        if optimized_model_path.exists():
            model_path = optimized_model_path
        else:
            # Alex's basic model filename
            model_path = models_dir / "xgboost_model.json"

    # Better error handling with added checks to prevent silent failures and batch job crashes.
    if Path(model_path).exists():
        try:
            # Alex's loading approach
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            return model
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model from {model_path}: {e}")
            return None
    else:
        print(f"CRITICAL ERROR: Model file not found at {model_path}")
        return None


def save_model(model, filepath):
    """
    Alex's model saving pattern.
    Direct XGBoost save_model call.
    """
    model.save_model(filepath)


def predict_claims(model, X_data):
    """
    Alex's prediction patterns.
    He always used [:, 1] for probabilities to get positive class only.
    """
    class_predictions = model.predict(X_data)
    prob_predictions = model.predict_proba(X_data)[:, 1]
    return class_predictions, prob_predictions


def get_booster(model):
    """
    Alex's booster access for SHAP analysis.
    Used as: native_model = model.get_booster()
    """
    return model.get_booster()