from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def train_model(data, seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Create a random forest classifier
    clf = RandomForestClassifier(random_state=seed)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Split the data into features and target
    X = data.drop(columns=['WL', 'SEASON_YEAR', 'GAME_ID', 'GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'], inplace=False)
    y = data['WL']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_clf = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # Fit the model on the training data
    best_clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = best_clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Calculate the f1 score of the model
    f1 = f1_score(y_test, y_pred, average='binary')
    print(f"F1 Score: {f1}")

    # Calculate the precision of the model
    precision = precision_score(y_test, y_pred, average='binary')
    print(f"Precision: {precision}")

    # Calculate the recall of the model
    recall = recall_score(y_test, y_pred, average='binary')
    print(f"Recall: {recall}")

    # Plot feature importance
    feature_importances = best_clf.feature_importances_
    features = X.columns
    indices = feature_importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('data/nba_data_combined.csv')
    train_model(data)
