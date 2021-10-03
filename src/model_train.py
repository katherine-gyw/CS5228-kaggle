import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

rf = RandomForestRegressor(n_estimators=1000,
                           random_state=42)
# Train the model on training data
rf.fit(train_features, train_labels)
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = mean_squared_error(test_labels, predictions)
# Print out the mean absolute error (mae)
print(np.sqrt(errors))

# save model
joblib.dump(rf, './models/random_forest.joblib', compress=3)
