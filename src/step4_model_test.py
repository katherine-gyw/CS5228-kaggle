import pandas as pd
import numpy as np
import joblib
import pickle as pkl
from src.config import test_clean_data_pth, model_xgb_pth, model_rf_pth, xgb_selected_features_save_pth,\
    rf_selected_features_save_pth

xgb_feature_ls = pkl.load(open(xgb_selected_features_save_pth, 'rb'))
print('Features used for xgb model: {}'.format(xgb_feature_ls))
rf_feature_ls = pkl.load(open(rf_selected_features_save_pth, 'rb'))
print('Features used for rf model: {}'.format(rf_feature_ls))
test_data = pd.read_csv(test_clean_data_pth, index_col=0)

xgb_model = joblib.load(model_xgb_pth)
rf_model = joblib.load(model_rf_pth)

xgb_submission_arr = xgb_model.predict(test_data[xgb_feature_ls])
rf_submission_arr = rf_model.predict(test_data[rf_feature_ls])
submission_arr = (xgb_submission_arr + rf_submission_arr)/2

submission_df = pd.DataFrame()
submission_df['Id'] = np.arange(0, 5000)
submission_df['Predicted'] = submission_arr
submission_df.to_csv('./submissions/submission.csv', index=False)
