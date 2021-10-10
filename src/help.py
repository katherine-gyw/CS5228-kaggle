from datetime import datetime
import numpy as np
import pandas as pd

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = pickle.load(open('./models/vectorizer', 'rb'))
word_price_map = pickle.load(open('./models/word_price_map.dict', 'rb'))

def add_date_features(data, date_ls):
    def _lifespan(x):
        try:
            return (x['lifespan'] - x['reg_date']).days
        except:
            return None

    for feature in date_ls:
        data[feature] = data[feature].replace(np.nan, '')
        data[feature] = data[feature].apply(lambda x: datetime.strptime(x, '%d-%b-%Y') if x != '' else '')

    data['reg_date_year'] = data['reg_date'].apply(lambda x: x.year)
    data['reg_date_mon'] = data['reg_date'].apply(lambda x: x.month)
    data['reg_date_day'] = data['reg_date'].apply(lambda x: x.day)
    data['lifespan_days'] = data.apply(_lifespan, axis=1) #lifespan dropped because missing ratio high
    date_added_features_ls = ['reg_date_year', 'reg_date_mon', 'reg_date_day', 'lifespan_days']
    return data, date_added_features_ls

def process_categorical_features(data):
    data.loc[data['make'].isnull(), 'make'] = data['title'].map(lambda x: x.split(" ")[0])
    data['make'] = data['make'].str.lower()


def generate_one_hot_data(encoder, data, feature, new_columns):
    train_one_hot_fields = encoder.transform(data[[feature]]).toarray()
    train_one_hot_df = pd.DataFrame(train_one_hot_fields)
    train_one_hot_df.columns = new_columns
    return pd.concat([data, train_one_hot_df], axis=1)

def process_date_features(df):
    # reg_date -> reg_age
    df['reg_date'] = df['reg_date'].apply(lambda x: datetime.strptime(x, '%d-%b-%Y') if x is not np.nan else x)
    print("fillna reg_date = ", df.shape[0]-df['reg_date'].dropna().shape[0])
    df['reg_date'] = df['reg_date'].fillna(df['reg_date'].mean())
        
    df['reg_age'] = df['reg_date'].apply(lambda x: (datetime.now() - x).days) 

    # manufactured -> manufactured_age
    print("fillna manufactured = ", df.shape[0]-df['manufactured'].dropna().shape[0])
    df['manufactured'] = df['manufactured'].fillna(df['manufactured'].mean())

    df['manufactured_age'] = df['manufactured'].apply(lambda x: (2021 - x)) 
    
    return df, ['reg_age', 'manufactured_age']

def remove_outliers(df, mode):
    '''
    https://www.motortrend.com/features/20-of-the-lightest-cars-sold-in-the-u-s/
    '''
    if mode == 'train':
        print("remove curb_weight below normal range for training data: ", df[df['curb_weight']<=800].shape[0])
        df = df[df['curb_weight']>800]
        
        print("remove reg_date beyond today for training data: ", df[df['reg_date']>datetime.now()].shape[0])
        df = df[df['reg_date']<=datetime.now()]
        
        print("remove manufactured date beyond today for training data: ", df[df['manufactured']>2021].shape[0])
        df = df[df['manufactured']<=2021]
        
    return df

def add_cross_features(df):
    # mileage per day
    df['mileage_per_day'] = df.apply(lambda row: row['mileage']/row['reg_age'] if row['mileage'] is not None else row['mileage'], axis=1)

    # mileage per year
    df['mileage_per_year'] = df.apply(lambda row: row['mileage']/row['manufactured_age'] if (row['mileage'] is not None and row['manufactured_age'] != 0) else row['mileage'], axis=1)

    # mileage per owner
    df['mileage_per_owner'] = df.apply(lambda row: row['mileage']/row['no_of_owners'] if (row['mileage'] is not None and row['no_of_owners'] is not None) else row['mileage'], axis=1)

    # power_ratio
    df['power_ratio'] = df.apply(lambda row: row['curb_weight']/row['power'] if (row['power'] is not None and row['curb_weight'] is not None) else None, axis=1)
    
    return df, ['mileage_per_day','mileage_per_year','mileage_per_owner','power_ratio']

def add_desc_features(df, N=2):
    
    title_ls = df.title.tolist()
    description_ls = df.description.tolist()
    features_ls = df.features.tolist()
    accessories_ls = df.accessories.tolist()
    
    description_ls = [str(title_ls[i])+" "+str(description_ls[i])+" "+str(features_ls[i])+" "+str(accessories_ls[i])
                      for i in range(len(title_ls))]
    
    # df = pd.concat((df, pd.DataFrame(description_ls, columns=['desc_concat'])), axis=1)
    
    vectors = vectorizer.transform(description_ls)
    dense = vectors.todense()
    denselist = dense.tolist()
    denselist = np.array(denselist)

    topNidx = np.argsort(denselist)[:, -N:][:,::-1]
#     df_word = pd.DataFrame(topNidx, columns=['tfidf_1_price', 'tfidf_2_price', 'tfidf_3_price', 'tfidf_4_price', 'tfidf_5_price'])
    df_word = pd.DataFrame(topNidx, columns=['tfidf_1_price', 'tfidf_2_price'])
    
    df = pd.concat((df, df_word), axis=1)
    
    df['tfidf_1_price'] = df['tfidf_1_price'].map(word_price_map)
    df['tfidf_2_price'] = df['tfidf_2_price'].map(word_price_map)
#     df['tfidf_3_price'] = df['tfidf_3_price'].map(word_price_map)
#     df['tfidf_4_price'] = df['tfidf_4_price'].map(word_price_map)
#     df['tfidf_5_price'] = df['tfidf_5_price'].map(word_price_map)
    
#     df['desc_mean_price'] = df.apply(lambda row: np.mean([row['tfidf_1_price'],row['tfidf_2_price'],row['tfidf_3_price'],row['tfidf_4_price'],row['tfidf_5_price']]), axis=1)
    df['desc_mean_price'] = df.apply(lambda row: np.mean([row['tfidf_1_price'],row['tfidf_2_price']]), axis=1)
    
    df = df.drop(columns=['tfidf_1_price', 'tfidf_2_price'])
    
    return df, ['desc_mean_price']
