import pandas as pd
import numpy as np
import joblib

# load cleaned train data
test_data = pd.read_csv('./data/test_clean.csv', index_col=0)

# load trained model
model = joblib.load('./models/random_forest.joblib')

submission_arr = model.predict(test_data)
submission_df = pd.DataFrame()
submission_df['Id'] = np.arange(0, 5000)
submission_df['Predicted'] = submission_arr

submission_df.to_csv('./submissions/submission.csv', index=False)
