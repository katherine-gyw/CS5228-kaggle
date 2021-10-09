# Random Forest Hyper Parameters
rf_n_estimators = [200, 500, 1000]
rf_max_features = ['auto', 'sqrt']
rf_criterion = ['squared_error']
rf_min_samples_split = [2, 4]

# Create the grid
rf_grid = {'n_estimators': rf_n_estimators,
           'max_features': rf_max_features,
           'criterion': rf_criterion,
           'min_samples_split': rf_min_samples_split}


xgb_grid = {'n_estimators': [300, 500, 1000],
            'max_depth': [2, 4, 6],
            'min_samples_split': [2, 5],
            'learning_rate': [0.01, 0.05]}
