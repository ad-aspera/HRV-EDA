from statsmodels.discrete.discrete_model import Logit
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def logistic_classifier(data, features):
    """
    Train a logistic regression model using the given data and features.
    """
    X = data[features]
    Y = data['DPN']
    X_scaled = sm.add_constant(StandardScaler().fit_transform(X))
    model = Logit(Y, X_scaled)
    result = model.fit(method='newton', maxiter=1000)
    return result, X_scaled, Y


def eval_model(result, X_scaled, y):
    """
    Evaluate the logistic regression model and return evaluation metrics.
    """
    print(result.summary())
    y_pred_proba = result.predict(X_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    return {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "roc_auc_score": roc_auc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred
    }


def run_train_test(data, features):
    """
    Split the data into train and test sets, train the model, and evaluate it.
    """
    train = data.groupby('DPN', group_keys=False).apply(lambda x: x.sample(frac=0.75))
    test = data.drop(train.index)

    # Train the model
    result, X_train_scaled, y_train = logistic_classifier(train, features)

    # Evaluate on train data
    train_eval = eval_model(result, X_train_scaled, y_train)

    # Evaluate on test data
    X_test = test[features]
    X_test_scaled = sm.add_constant(StandardScaler().transform(X_test))
    y_test = test['DPN']
    test_eval = eval_model(result, X_test_scaled, y_test)

    return train_eval, test_eval


def plot_evaluation_metrics(res, y_true, title="Evaluation Metrics", figsize=(12, 6)):
    """
    Plot confusion matrix and ROC curve for the model evaluation.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[1, 1.5])

    # Plot confusion matrix
    ConfusionMatrixDisplay(res['confusion_matrix'], display_labels=['False', 'True']).plot(ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    #colorbar = axes[0].collections[0].colorbar
    #colorbar.ax.set_box_aspect(0.5)  # Set the colorbar height to half of default

    # Plot ROC curve
    RocCurveDisplay.from_predictions(y_true, res['y_pred_proba'], ax=axes[1])

    axes[1].set_title("ROC Curve")

    # Adjust the colorbar height for the confusion matrix


    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

