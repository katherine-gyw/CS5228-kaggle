from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import pickle as pkl
from src.config import selected_features_save_pth, train_clean_data_pth, target_feature


def select_features_with_boruta(df):
    X = df[feature_ls]
    y = df[target_feature]
    forest = RandomForestRegressor(n_jobs=-1, max_depth=5)
    boruta = BorutaPy(estimator=forest, n_estimators='auto', max_iter=100)
    boruta.fit(np.array(X), np.array(y))
    green_area = X.columns[boruta.support_].to_list()
    blue_area = X.columns[boruta.support_weak_].to_list()
    used_features_ls = green_area + blue_area
    return used_features_ls


def select_features_with_correlation(df, corr_thres=0.05):
    feature_corr = df[feature_ls+[target_feature]].corr()[target_feature]
    used_features_ls = list(feature_corr[feature_corr >= corr_thres].index)
    used_features_ls.remove(target_feature)
    return used_features_ls


if __name__ == '__main__':
    train_data = pd.read_csv(train_clean_data_pth, index_col=0)
    feature_ls = [item for item in train_data.columns if item != target_feature]

    boruta_features_ls = select_features_with_boruta(train_data)
    corr_features_ls = select_features_with_correlation(train_data, 0.05)
    used_features_ls = list(set(boruta_features_ls+corr_features_ls))
    print("{} features are picked by boruta method.".format(len(boruta_features_ls)))
    print(boruta_features_ls)
    print("{} features are picked by correlation method.".format(len(corr_features_ls)))
    print(corr_features_ls)
    print("Combine these two methods, {} features are selected.".format(len(used_features_ls)))
    print(used_features_ls)

    with open(selected_features_save_pth, 'wb') as f:
        pkl.dump(used_features_ls, f)
