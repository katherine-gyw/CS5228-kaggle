import os
project_root_dir = '/Users/yuewen/Desktop/NUS-2021-sem1/CS5228/CS5228-kaggle/'

train_data_pth = os.path.join(project_root_dir, './data/train.csv')
test_data_pth = os.path.join(project_root_dir, './data/test.csv')
train_clean_data_pth = os.path.join(project_root_dir, './data/train_clean.csv')
test_clean_data_pth = os.path.join(project_root_dir, './data/test_clean.csv')
selected_features_save_pth = os.path.join(project_root_dir, './models/selected_features.pkl')
model_rf_pth = os.path.join(project_root_dir, './models/random_forest.joblib')
model_xgb_pth = os.path.join(project_root_dir, './models/xgb.joblib')


target_feature = 'price'
random_state = 2021
