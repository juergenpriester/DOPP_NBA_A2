## TODO

- [x] Loaded GameLogs from nba_api

- [ ] Data Exploration
    - [x] Wins over teams
    - [x] Label Distribution
    - [x] Scatter Magtrix
    - [x] Correlation Matrix
    - [x] Correlation to target variable
    - [ ] Additional Plots

- [ ] Preprocessing
    - [x] Converted Datatypes
    - [x] Aggregated Game Statistics with rolling means
    - [x] Combined Home and Away statistics
    - [ ] Maybe add additional data
        - [ ] What data?
            - [ ] Injured/missing players impact score (advanced statstic maybe Box Plus Minus or Winshares (from stahead))
            - [ ] Is a game the second one in a back to back (2 games in 2 days)?
        - [ ] Feature Engineering

- [ ] Model Training and Evaluation 
    - [ ] Added simple CV pipeline with hyperparameter tuning
        - [x] Logistic Regression
        - [x] Random Forest
        - [x] Gradient Boosting
    - [ ] Feature Selection
    - [ ] Scaling of Data

    - Evaluation Plots
        - [x] Confusion Matrix
        - [x] ROC Curve
        - [x] Precision-Recall Curve
        - [x] Feature Importance (if given by model)
        - [x] Histogram of ground truth and predictions
        - 