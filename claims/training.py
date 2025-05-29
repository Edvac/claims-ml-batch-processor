"""
Alex's model training from notebook.

Alex used sequential training steps with XGBoost, including basic training
and cross-validation hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy import stats
from pathlib import Path


def prepare_training_data(dataset_from_database):
    """
    Alex's data preparation for training.
    """
    # Alex's feature/target separation
    X, y = dataset_from_database.drop('claim_status', axis=1), dataset_from_database[['claim_status']]

    # Alex's train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1889)

    return X_train, X_test, y_train, y_test


def train_basic_model(X_train, y_train, X_test, y_test):
    """
    Alex's initial model training approach.
    """
    # Alex's evaluation set and metrics
    eval_set = [(X_train, y_train)]
    eval_metrics = ['auc', 'rmse', 'logloss']

    # Alex's basic model configuration
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=eval_metrics,
        enable_categorical=True
    )

    # Alex's training approach
    model.fit(X_test, y_test, eval_set=eval_set, verbose=10)

    # Alex's model saving
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model.save_model(str(models_dir / "xgboost_model.json"))

    return model


def train_cv_model(X_train, y_train, X_test, y_test):
    """
    Alex's cross-validation model training.
    """
    # Alex's evaluation set and metrics
    eval_set = [(X_train, y_train)]
    eval_metrics = ['auc', 'rmse', 'logloss']

    # Alex's hyperparameter search space
    param_distributions = {
        'n_estimators': stats.randint(50, 500),
        'learning_rate': stats.uniform(0.01, 0.75),
        'subsample': stats.uniform(0.25, 0.75),
        'max_depth': stats.randint(1, 8),
        'colsample_bytree': stats.uniform(0.1, 0.75),
        'min_child_weight': [1, 3, 5, 7, 9],
    }

    # Alex's RandomizedSearchCV setup
    parameter_gridSearch = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=eval_metrics,
            early_stopping_rounds=15,
            enable_categorical=True,
        ),
        param_distributions=param_distributions,
        cv=5,
        n_iter=100,
        verbose=False,
        scoring='roc_auc',
    )

    # Alex's hyperparameter search
    parameter_gridSearch.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    print("Best parameters are: ", parameter_gridSearch.best_params_)

    # Alex's final model with best parameters
    model3 = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=eval_metrics,
        early_stopping_rounds=15,
        enable_categorical=True,
        **parameter_gridSearch.best_params_  # Alex's comment: Not sure what this does, from StackOverflow
    )

    # Alex's final model training
    model3.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Alex's optimized model saving
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model3.save_model(str(models_dir / "xgboost_model_optimised_with_cross_validation.json"))

    return model3