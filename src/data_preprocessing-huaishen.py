import pandas as pd
import numpy as np
from help import *
from sklearn import preprocessing

# load raw data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

feature_ls = train_data.columns
print('Original feature list: {}'.format(feature_ls))


# step1: remove features with >95% null values: [original_reg_date, opc_scheme, indicative_price]
null_percentage = train_data.isnull().sum()/train_data.shape[0]
print(null_percentage)
null_feature_ls = list(null_percentage[null_percentage > 0.95].index)
feature_ls = [item for item in feature_ls if item not in null_feature_ls]
print("STEP1: Remove {} features with more than 95% null values, there are {} features left.".format(null_feature_ls,
                                                                                              len(feature_ls)))


# step2: remove categorical features with only 1 or too many unique value
# eco_category: only has 1 unique value, remove it.
# listing_id: 99.66% of the values are unique, remove it.
unique_value_percentage = train_data.nunique()/train_data.shape[0]
unique_feature_ls = list(unique_value_percentage[(unique_value_percentage == 1/train_data.shape[0]) |
                                                 (unique_value_percentage > 0.99)].index)
feature_ls = [item for item in feature_ls if item not in unique_feature_ls]
print("STEP2: Remove {} feature, there are {} features left.".format(unique_feature_ls, len(feature_ls)))


# step3: features with description words, and need additional processing
# todo: process these features, then add them back
description_features = ["description", 'features', 'accessories']
feature_ls = [item for item in feature_ls if item not in description_features]
print('STEP3: Remove {} features, there are {} features left.'.format(description_features, len(feature_ls)))

# todo: add categorical features
# step4: one-hot encoding for categorical features
process_categorical_features(train_data)
process_categorical_features(test_data)
categorical_features = ["make", "type_of_vehicle", "transmission"]
categorical_features_ls = []
for feature in categorical_features:
    encoder = preprocessing.OneHotEncoder(handle_unknown="ignore").fit(train_data[[feature]])
    new_columns = encoder.get_feature_names_out()
    train_data = generate_one_hot_data(encoder, train_data, feature, new_columns)
    test_data = generate_one_hot_data(encoder, test_data, feature, new_columns)
    categorical_features_ls += list(new_columns)

"""
Summary so far: 
- feature_ls: 21 features + 1 target
- 11 numerical features: ['manufactured', 'curb_weight', 'power', 'engine_cap', 'no_of_owners', 'depreciation', 'coe', 'road_tax', 'dereg_value', 'mileage', 'omv']
- 2 date features: ['reg_date', 'lifespan']
    * reg_date: seperate it to year, month, date
    * lifespan: convert it to the time difference with reg_date
- 8 categorical features: 
    * category: convert it
"""

# step4: date feature transformation
date_ls = ['reg_date', 'lifespan']
train_data, date_added_feature_ls = add_date_features(train_data, date_ls)
test_data, _ = add_date_features(test_data, date_ls)
print('STEP4: Add {} features.'.format(date_added_feature_ls))

# step 5: numerical features:
numerical_features_ls = ['manufactured', 'curb_weight', 'power', 'engine_cap', 'no_of_owners', 'depreciation', 'coe', 'road_tax', 'dereg_value', 'mileage', 'omv']
std_scaler = preprocessing.MinMaxScaler().fit(train_data[numerical_features_ls])
train_data[numerical_features_ls] = std_scaler.transform(train_data[numerical_features_ls])
test_data[numerical_features_ls] = std_scaler.transform(test_data[numerical_features_ls])
print('STEP5: Standardize {} features'.format(numerical_features_ls))

# step 6: processing the nan values in train and test dataset
# todo: change this
train_mean = train_data[numerical_features_ls].mean()
train_data[numerical_features_ls].fillna(train_mean, inplace=True)
test_data[numerical_features_ls].fillna(train_mean, inplace=True)
train_data = train_data.replace(np.nan, -1)
test_data = test_data.replace(np.nan, -1)

other_feature_ls = []
target_ls = ['price']
final_feature_ls = numerical_features_ls + date_added_feature_ls + categorical_features_ls + other_feature_ls
train_data[final_feature_ls+target_ls].to_csv('./data/train_clean.csv')
test_data[final_feature_ls].to_csv('./data/test_clean.csv')