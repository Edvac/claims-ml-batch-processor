"""
Alex's model evaluation from notebook.

Alex used comprehensive evaluation metrics including Cohen Kappa, accuracy,
confusion matrices, ROC curves, and feature importance plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix, roc_auc_score, log_loss, roc_curve


def get_model_predictions(model, X_train, X_test):
    """
    Alex's prediction pattern for evaluation.
    He always got both class and probability predictions for train and test.
    """
    # Model validation
    if model is None:
        raise ValueError("CRITICAL ERROR: Model is None, cannot generate predictions. Check model loading.")

    train_class_preds = model.predict(X_train)
    test_class_preds = model.predict(X_test)
    train_prob_preds = model.predict_proba(X_train)[:, 1]
    test_prob_preds = model.predict_proba(X_test)[:, 1]

    return train_class_preds, test_class_preds, train_prob_preds, test_prob_preds


def calculate_kappa_scores(y_train, y_test, train_class_preds, test_class_preds):
    """
    Alex's Cohen Kappa calculation with his specific array handling.
    """
    # Empty data validation
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("CRITICAL ERROR: Empty dataset provided for kappa calculation")

    # Alex's training data kappa calculation
    y = np.array(y_train)
    y = y.astype(int)
    yhat = np.array(train_class_preds)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    training_data_kappa_score = round(cohen_kappa_score(yhat, y, weights='quadratic'), 2)

    # Alex's test data kappa calculation
    y = np.array(y_test)
    y = y.astype(int)
    yhat = np.array(test_class_preds)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    test_data_kappa_score = round(cohen_kappa_score(yhat, y, weights='quadratic'), 2)

    return training_data_kappa_score, test_data_kappa_score


def print_evaluation_metrics(y_train, y_test, train_class_preds, test_class_preds, train_prob_preds, test_prob_preds):
    """
    Alex's comprehensive evaluation output exactly as he formatted it.
    """
    # CRITICAL FIX: Array length validation - to avoid incorrect metrics and silent failures
    if len(y_train) != len(train_class_preds) or len(y_test) != len(test_class_preds):
        raise ValueError("CRITICAL ERROR: Mismatched array lengths between predictions and targets")


    # Alex's kappa scores
    training_kappa, test_kappa = calculate_kappa_scores(y_train, y_test, train_class_preds, test_class_preds)
    print(f"The Cohen Kappa score on the training data is: {training_kappa}")
    print(f"The Cohen Kappa score on the test data is: {test_kappa}")

    print()
    # Alex's accuracy scores
    print("The accuracy on train dataset is: ", accuracy_score(y_train, train_class_preds))
    print("The accuracy on test dataset is: ", accuracy_score(y_test, test_class_preds))

    print()
    # Alex's confusion matrices
    print("Train confusion matrix: ", confusion_matrix(y_train, train_class_preds))

    print()
    print("Test confusion matrix: ", confusion_matrix(y_test, test_class_preds))

    print()
    # Alex's ROC scores
    print("ROC on train data: ", roc_auc_score(y_train, train_prob_preds))
    print("ROC on test data: ", roc_auc_score(y_test, test_prob_preds))

    print()
    # Alex's log loss
    print("Train log loss: ", log_loss(y_train, train_prob_preds))
    print("Test log loss: ", log_loss(y_test, test_prob_preds))

    print()
    # Alex's additional metrics
    print("F1 score is: ", f1_score(y_test, test_class_preds))
    print("Precision is: ", precision_score(y_test, test_class_preds))
    print("Recall is: ", recall_score(y_test, test_class_preds))


def plot_roc_curve(y_test, test_prob_preds):
    """
    Alex's ROC curve plotting exactly as he implemented it.
    """
    # Plot error handling
    try:
        print()
        fpr, tpr, _ = roc_curve(y_test, test_prob_preds)
        random_fpr, random_tpr, _ = roc_curve(y_test, [0 for _ in range(len(y_test))])
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='.', label='XGBoost')
        plt.plot(random_fpr, random_tpr, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Receiver Operating Curve")
        plt.show()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not create ROC curve plot: {e}")
        return




def plot_feature_importance(model):
    """
    Alex's feature importance plotting.
    """
    # Plot error handling
    try:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        xgb.plot_importance(model, ax=ax1)
        plt.show()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to plot feature importance: {e}")



def plot_model_tree(model):
    """
    Alex's model tree plotting.
    """
    try:
        fig2, ax2 = plt.subplots(figsize=(16, 16))
        xgb.plot_tree(model, rankdir='LR', ax=ax2)
        plt.show()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to plot model tree: {e}")


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Alex's complete evaluation workflow.
    """
    print("Starting model evaluation...")

    # Get predictions
    train_class_preds, test_class_preds, train_prob_preds, test_prob_preds = get_model_predictions(model, X_train,
                                                                                                   X_test)

    # Print all metrics
    print_evaluation_metrics(y_train, y_test, train_class_preds, test_class_preds, train_prob_preds, test_prob_preds)

    # Plot ROC curve
    plot_roc_curve(y_test, test_prob_preds)

    # Plot feature importance
    plot_feature_importance(model)