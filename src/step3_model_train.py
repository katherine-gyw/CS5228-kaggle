import pandas as pd
import numpy as np
import joblib
import os
import time
import pickle as pkl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, Trials
from src.config import target_feature, model_xgb_pth, model_rf_pth, selected_features_save_pth, random_state,\
    train_clean_data_pth
from src.config_model_hyper_params import param_hyperopt_gb, param_hyperopt_rf


# load cleaned train data
with open(selected_features_save_pth, 'rb') as f:
    feature_ls = pkl.load(f)
    print('Features used for model training: {}'.format(feature_ls))
train_data = pd.read_csv(train_clean_data_pth, index_col=0)

# split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(train_data[feature_ls],
                                                                            train_data[target_feature],
                                                                            test_size=0.2,
                                                                            random_state=random_state)


def bayesian_hyper_tuning(param_hyperopt, regressor):
    start = time.time()

    def _fn(params):
        model = regressor(random_state=random_state, **params)
        best_score = cross_val_score(model,
                                     train_features,
                                     train_labels,
                                     cv=5,
                                     scoring='neg_root_mean_squared_error').mean()
        return -best_score

    trials = Trials()
    best_param = fmin(fn=_fn,
                      space=param_hyperopt,
                      max_evals=75,
                      trials=trials,
                      rstate=np.random.RandomState(random_state),
                      algo=tpe.suggest)

    print('It takes %s minutes' % ((time.time() - start)/60))
    return best_param


# XGB
gbm_best_param = bayesian_hyper_tuning(param_hyperopt_gb, GradientBoostingRegressor)
xgb_best = GradientBoostingRegressor(learning_rate=gbm_best_param['learning_rate'],
                                     max_depth=int(gbm_best_param['max_depth']),
                                     max_features=gbm_best_param['max_features'],
                                     n_estimators=int(gbm_best_param['n_estimators']),
                                     subsample=gbm_best_param['subsample'])
xgb_best.fit(train_features, train_labels)
xgb_test_rmse = np.sqrt(mean_squared_error(xgb_best.predict(test_features), test_labels))
print('XGB test RMSE: {}'.format(xgb_test_rmse))

# RF
rf_best_param = bayesian_hyper_tuning(param_hyperopt_rf, RandomForestRegressor)
rf_best = RandomForestRegressor(min_samples_split=rf_best_param['min_samples_split'],
                                max_depth=int(rf_best_param['max_depth']),
                                n_estimators=int(rf_best_param['n_estimators']))
rf_best.fit(train_features, train_labels)
rf_test_rmse = np.sqrt(mean_squared_error(rf_best.predict(test_features), test_labels))
print('XGB test RMSE: {}'.format(rf_test_rmse))

# save model
if not os.path.exists('./models'):
    os.mkdir('./models')

joblib.dump(rf_best, model_rf_pth, compress=3)
joblib.dump(xgb_best, model_xgb_pth, compress=3)
