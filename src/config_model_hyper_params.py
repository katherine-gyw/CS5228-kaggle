import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope


param_hyperopt_gb = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 7, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 30, 1)),
    'max_features': hp.uniform('max_features', 0.8, 1),
    'subsample': hp.uniform('subsample', 0.8, 1)
}

param_hyperopt_rf = {
    'max_depth': scope.int(hp.quniform('max_depth', 20, 100, 10)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
}
