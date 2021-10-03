# Random Forest Hyper Parameters
rf_n_estimators = [200, 500, 1000, 1500]
rf_max_depth = [10, 20, 30, None]
rf_max_features = ['auto', 'sqrt', 'log2']
rf_criterion = ['mse']
rf_min_samples_split = [2, 5, 10]
rf_min_impurity_decrease = [0.0, 0.05, 0.1]
rf_bootstrap = [True, False]

# Create the grid
rf_grid = {'n_estimators': rf_n_estimators,
           'max_depth': rf_max_depth,
           'max_features': rf_max_features,
           'criterion': rf_criterion,
           'min_samples_split': rf_min_samples_split,
           'min_impurity_decrease': rf_min_impurity_decrease,
           'bootstrap': rf_bootstrap}



