import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from model_config import rf_grid, xgb_grid

# load cleaned train data
train_data = pd.read_csv('./data/train_clean.csv', index_col=0)

target_feature = 'price'
feature_ls = [item for item in train_data.columns if item not in target_feature]
print('Features used for model training: {}'.format(feature_ls))

target_array = np.array(train_data[target_feature])
# split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(train_data[feature_ls],
                                                                            target_array,
                                                                            test_size=0.25,
                                                                            random_state=42)
# # Linear Regression
# lr_model = LinearRegression().fit(train_features, train_labels)
# lr_predictions = lr_model.predict(test_features)
# lr_errors = mean_squared_error(test_labels, lr_predictions)
# print('LR MSE: {}'.format(lr_errors))


# Random Forest Model
rf_base = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf_base,
                               param_distributions=rf_grid,
                               cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(train_features, train_labels)
rf_predictions = rf_random.predict(test_features)
rf_errors = mean_squared_error(test_labels, rf_predictions)
print('Random Forest MSE: {}'.format(np.sqrt(rf_errors)))

# XGB
xgb_base = GradientBoostingRegressor()
xgb_random = RandomizedSearchCV(estimator=xgb_base,
                                param_distributions=xgb_grid,
                                cv=5,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)
xgb_random.fit(train_features, train_labels)
xgb_predictions = xgb_random.predict(test_features)
xgb_errors = mean_squared_error(test_labels, xgb_predictions)
print('XGB MSE: {}'.format(np.sqrt(xgb_errors)))

# save model
if not os.path.exists('./models'):
    os.mkdir('./models')
joblib.dump(rf_random, './models/random_forest.joblib', compress=3)
joblib.dump(xgb_random, './models/xgb.joblib', compress=3)
