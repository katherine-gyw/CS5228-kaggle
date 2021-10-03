import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.model_config import rf_grid

target_feature = ['price']

# load cleaned train data
train_data = pd.read_csv('./data/train_clean.csv', index_col=0)
feature_ls = [item for item in train_data.columns if item not in target_feature]
print('Features used for model training: {}'.format(feature_ls))

# split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(train_data[feature_ls],
                                                                            train_data[target_feature],
                                                                            test_size=0.25,
                                                                            random_state=42)

# Create the model to be tuned
rf_base = RandomForestRegressor()

# Create the random search Random Forest
rf_random = RandomizedSearchCV(estimator=rf_base,
                               param_distributions=rf_grid,
                               n_iter=200,
                               cv=4,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)

# Train the model on training data
rf_random.fit(train_features, train_labels)
# Use the forest's predict method on the test data
predictions = rf_random.predict(test_features)
# Calculate the absolute errors
errors = mean_squared_error(test_labels, predictions)
# Print out the mean absolute error (mae)
print(np.sqrt(errors))

# save model
if not os.path.exist('./models'):
    os.mkdir('./models')
joblib.dump(rf_random, './models/random_forest.joblib', compress=3)
