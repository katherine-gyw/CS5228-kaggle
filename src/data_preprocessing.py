import pandas as pd
import numpy as np
import os
from help import add_date_features, add_desc_features

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#os.chdir(ROOT_DIR)

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
description_features_ls = ["description", 'features', 'accessories', 'title']
train_data, description_added_feature_ls = add_desc_features(train_data)
test_data, _ = add_desc_features(test_data)
print('STEP3: Add {} features.'.format(description_added_feature_ls))

# todo: add categorical features
#

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

# step 6: processing the nan values in train and test dataset
# todo: change this
train_data = train_data.replace(np.nan, -1)
test_data = test_data.replace(np.nan, -1)

target_ls = ['price']
final_feature_ls = numerical_features_ls + date_added_feature_ls + description_added_feature_ls
train_data[final_feature_ls+target_ls].to_csv('./data/train_clean.csv')
test_data[final_feature_ls].to_csv('./data/test_clean.csv')
