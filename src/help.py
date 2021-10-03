from datetime import datetime
import numpy as np


def add_date_features(data, date_ls):
    def _lifespan(x):
        try:
            return (x['lifespan']-x['reg_date']).days
        except:
            return -1

    for feature in date_ls:
        data[feature] = data[feature].replace(np.nan, '')
        data[feature] = data[feature].apply(lambda x: datetime.strptime(x, '%d-%b-%Y') if x != '' else '')

    data['reg_date_year'] = data['reg_date'].apply(lambda x: x.year)
    data['reg_date_mon'] = data['reg_date'].apply(lambda x: x.month)
    data['reg_date_day'] = data['reg_date'].apply(lambda x: x.day)
    data['lifespan_days'] = data.apply(_lifespan, axis=1)
    date_added_features_ls = ['reg_date_year', 'reg_date_mon', 'reg_date_day', 'lifespan_days']
    return data, date_added_features_ls
