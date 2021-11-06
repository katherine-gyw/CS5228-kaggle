import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.help import add_date_features, add_desc_features, generate_one_hot_data, process_categorical_features, remove_outliers
from src.config import train_data_pth, test_data_pth, train_clean_data_pth, validation_clean_data_pth, test_clean_data_pth, target_feature


# load data
train_data = pd.read_csv(train_data_pth)
test_data = pd.read_csv(test_data_pth)
feature_ls = [item for item in train_data.columns if item != target_feature]
print('Original feature list: {}'.format(feature_ls))


# step1: remove features with >70% null values: [original_reg_date, opc_scheme, indicative_price]
null_percentage = train_data.isnull().sum()/train_data.shape[0]
null_feature_ls = list(null_percentage[null_percentage > 0.7].index)
feature_ls = [item for item in feature_ls if item not in null_feature_ls]
print("STEP1: Remove {} features with more than 70% null values, there are {} features left.".format(null_feature_ls, len(feature_ls)))


# step2: remove categorical features with only 1 or too many unique value
# eco_category: only has 1 unique value, remove it.
# listing_id: 99.66% of the values are unique, remove it.
unique_value_percentage = train_data.nunique()/train_data.shape[0]
unique_feature_ls = list(unique_value_percentage[(unique_value_percentage == 1/train_data.shape[0]) |
                                                 (unique_value_percentage > 0.99)].index)
feature_ls = [item for item in feature_ls if item not in unique_feature_ls]
print("STEP2: Remove {} feature, there are {} features left.".format(unique_feature_ls, len(feature_ls)))


# step3: description features
description_features_ls = ["description", 'features', 'accessories', 'title']
train_data, description_added_feature_ls = add_desc_features(train_data)
test_data, _ = add_desc_features(test_data)
feature_ls = [item for item in feature_ls if item not in description_features_ls]
feature_ls = feature_ls + description_added_feature_ls
print('STEP3: Add {} features.'.format(description_added_feature_ls))


# step4: categorical features
process_categorical_features(train_data)
process_categorical_features(test_data)
categorical_features = ["type_of_vehicle", "transmission", 'make']
categorical_features_ls = []
for feature in categorical_features:
    encoder = preprocessing.OneHotEncoder(handle_unknown="ignore").fit(train_data[[feature]])
    new_columns = encoder.get_feature_names_out()
    train_data = generate_one_hot_data(encoder, train_data, feature, new_columns)
    test_data = generate_one_hot_data(encoder, test_data, feature, new_columns)
    categorical_features_ls += list(new_columns)
# feature make as too many categories, we only keep the categories with large amount
make_count = train_data['make'].value_counts()
make_count_ls = list(make_count[make_count > train_data.shape[0]*0.01].index)
categorical_remove_feature_ls = [item for item in categorical_features_ls if item.startswith('make_') and item[5:] not in make_count_ls]
categorical_features_ls = [item for item in categorical_features_ls if item not in categorical_remove_feature_ls] + ['make_others']
train_data['make_others'] = np.sum(train_data[categorical_remove_feature_ls], axis=1)
test_data['make_others'] = np.sum(test_data[categorical_remove_feature_ls], axis=1)
print('STEP4: Add {} features.'.format(categorical_features_ls))
feature_ls = [item for item in feature_ls if item not in categorical_features]
feature_ls = feature_ls + categorical_features_ls
feature_ls.remove('model')
feature_ls.remove('category')


# step5: date feature transformation
date_ls = ['reg_date', 'lifespan']
train_data, date_added_feature_ls = add_date_features(train_data, date_ls)
test_data, _ = add_date_features(test_data, date_ls)
print('STEP5: Add {} features.'.format(date_added_feature_ls))
feature_ls = [item for item in feature_ls if item not in date_ls]
feature_ls = feature_ls + date_added_feature_ls


# step6: luxury car list
price90q = np.quantile(train_data[target_feature], 0.9)
luxury_car = train_data[train_data[target_feature] >= price90q]
luxury_car_dict = dict((luxury_car['make']+luxury_car['model']).value_counts())
total_cnt_dict = dict((train_data['make']+train_data['model']).value_counts())
luxury_car_per = [(item[0], item[1]/total_cnt_dict[item[0]]) for item in luxury_car_dict.items()]
luxury_car_ls = [item[0] for item in luxury_car_per if item[1] > 0.8]
print('Our defined luxury car list: {}'.format(luxury_car_ls))
train_luxury = train_data['make']+train_data['model']
test_luxury = test_data['make']+test_data['model']
train_data['luxury_car'] = train_luxury.apply(lambda x: x in luxury_car_ls)
test_data['luxury_car'] = test_luxury.apply(lambda x: x in luxury_car_ls)
print('STEP6: Add feature luxury_car.')
feature_ls = feature_ls + ['luxury_car']


# step7: remove outliers
print('STEP7: Remove outliers.')
train_data = remove_outliers(train_data, mode='train')


# take 20% train data as validation data
N = train_data.shape[0]
validation_data_idx = np.random.choice(range(N), int(N*0.2), replace=False)
train_data_idx = list(set(range(N)) - set(validation_data_idx))
train_data.reset_index(drop=True, inplace=True)
validation_data = train_data.take(validation_data_idx)


# step8: processing the nan values in train and test dataset
print('STEP8: Replace nan values with column mean.')
train_data = train_data.take(train_data_idx)
train_data = train_data.fillna(train_data.mean())
validation_data = validation_data.fillna(validation_data.mean())
test_data = test_data.fillna(test_data.mean())


print('Finish data preprocessing')
print('There are {} rows, {} features in train dataset.'.format(train_data.shape[0], len(feature_ls)))
print('There are {} rows, {} features in validation dataset.'.format(validation_data.shape[0], len(feature_ls)))
print(feature_ls)
train_data[feature_ls+[target_feature]].to_csv(train_clean_data_pth)
validation_data[feature_ls+[target_feature]].to_csv(validation_clean_data_pth)
test_data[feature_ls].to_csv(test_clean_data_pth)
