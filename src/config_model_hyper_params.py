import numpy as np
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll import scope

rf_n_estimators = [200, 500, 1000]
rf_max_features = ['auto', 'sqrt']
rf_criterion = ['mse']
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

param_hyperopt_gb = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 7, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 30, 1)),
    'max_features': hp.uniform('max_features', 0.8, 1),
    'subsample': hp.uniform('subsample', 0.8, 1)
}