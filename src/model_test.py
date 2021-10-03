import pandas as pd
import numpy as np
import joblib
import os

# load cleaned train data
test_data = pd.read_csv('./data/test_clean.csv', index_col=0)

# load trained model
model_ls = os.listdir('./models/')
submission_arr = None
model_cnt = 0
for model_pth in model_ls:
    if not model_pth.endswith('.joblib'):
        continue
    model_cnt += 1
    model = joblib.load(os.path.join('./models/', model_pth))
    if submission_arr is None:
        submission_arr = model.predict(test_data)
    else:
        submission_arr += model.predict(test_data)
submission_arr /= model_cnt

submission_df = pd.DataFrame()
submission_df['Id'] = np.arange(0, 5000)
submission_df['Predicted'] = submission_arr

submission_df.to_csv('./submissions/submission2.csv', index=False)
