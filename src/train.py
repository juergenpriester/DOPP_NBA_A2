from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging as log

from constants import DATA_DIR, PLOTS_DIR
from utils import check_create_dir

EVAL_DIR = PLOTS_DIR + '/evaluation'
check_create_dir(EVAL_DIR)
SEED = 42
np.random.seed(SEED)


log.basicConfig(level=log.INFO)


def calc_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    log.info(f"Metrics: {metrics}")
    return metrics


def plot_proba_hist(y_prob):
    # Plot histogram of probabilities
    plt.hist(y_prob[:, 1], bins=20)
    plt.xlabel('Probability of win')
    plt.ylabel('Frequency')
    plt.savefig(EVAL_DIR+"/probability_histogram.png")
    plt.close()


def plot_feature_importance(model, X):
    # Plot feature importance
    feature_importances = model.feature_importances_
    features = X.columns
    indices = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(EVAL_DIR+"/feature_importance.png")
    plt.close()


def plot_prediction_hist(y_test, y_pred):
    # Plot histogram of predictions
    width = 0.25
    bins = np.arange(len(np.unique(y_test))) - width / 2

    plt.bar(bins - width, np.bincount(y_test), width=width, alpha=0.5, label='True')
    plt.bar(bins + width, np.bincount(y_pred), width=width, alpha=0.5, label='Predictions')
    plt.xticks(bins, np.unique(y_test))
    plt.legend(loc='upper right')
    plt.savefig(EVAL_DIR+"/prediction_histogram.png")
    plt.close()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(EVAL_DIR+"/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(EVAL_DIR+"/roc_curve.png")
    plt.close()


def plot_precision_recall_curve(y_test, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:, 1])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(EVAL_DIR+"/precision_recall_curve.png")
    plt.close()


def perform_grid_search(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    return best_clf


def train_model(clf, data: pd.DataFrame, param_grid=None):

    # Split the data into features and target
    X = data.drop(columns=['WL', 'SEASON_YEAR', 'GAME_ID', 'GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'], inplace=False)
    y = data['WL']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    if param_grid is not None:
        best_clf = perform_grid_search(clf, param_grid, X_train, y_train)
    else:
        best_clf = clf

    best_clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)

    metrics = calc_metrics(y_test, y_pred)

    plot_feature_importance(best_clf, X)
    plot_prediction_hist(y_test, y_pred)
    plot_proba_hist(y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)


if __name__ == '__main__':
    data = pd.read_csv('data/nba_data_combined.csv')
    log.info(f"Shape of data: {data.shape}")

    # clf = RandomForestClassifier(random_state=SEED)
    clf = GradientBoostingClassifier(random_state=SEED)
    # Define the parameter grid
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10],
        'min_samples_split': [5],
        'min_samples_leaf': [1]
    }

    train_model(clf, data)
