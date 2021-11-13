from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import vectorizer_pth, word_dict_pth


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


def generate_word_price_map(df, description_ls):
    if os.path.exists(word_dict_pth) and os.path.exists(vectorizer_pth):
        word_price_map = pickle.load(open(word_dict_pth, 'rb'))
        vectorizer = pickle.load(open(vectorizer_pth, 'rb'))
    else:
        print('It will take veryy long time to generate word_price_map.')
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(description_ls)
        feature_names = vectorizer.get_feature_names()
        word_price_map = {k: [] for k in range(len(feature_names) + 1)}
        analyzer = vectorizer.build_analyzer()

        def tokenize(s, vectorizer, analyzer):
            # OOV word will be assigned to the last index
            r = list(map(lambda x: vectorizer.vocabulary_.get(x, len(vectorizer.get_feature_names())), analyzer(s)))
            return sorted(set(r))

        for price, desc in zip(df['price'], df['desc_concat']):
            tl = tokenize(desc, vectorizer, analyzer)
            for i in tl:
                word_price_map[i].append(price)
        pickle.dump(word_price_map, open(word_dict_pth, 'wb'))
        pickle.dump(vectorizer, open(vectorizer_pth, 'wb'))
    return word_price_map, vectorizer


def add_desc_features(df):
    N = 5
    title_ls = df.title.tolist()
    description_ls = df.description.tolist()
    features_ls = df.features.tolist()
    accessories_ls = df.accessories.tolist()

    description_ls = [
        str(title_ls[i]) + " " + str(description_ls[i]) + " " + str(features_ls[i]) + " " + str(accessories_ls[i])
        for i in range(len(title_ls))]

    df = pd.concat((df, pd.DataFrame(description_ls, columns=['desc_concat'])), axis=1)
    word_price_map, vectorizer = generate_word_price_map(df, description_ls)

    vectors = vectorizer.transform(description_ls)
    dense = vectors.todense()
    denselist = dense.tolist()
    denselist = np.array(denselist)

    topNidx = np.argsort(denselist)[:, -N:][:, ::-1]
    df_word = pd.DataFrame(topNidx, columns=['tfidf_1_price', 'tfidf_2_price', 'tfidf_3_price', 'tfidf_4_price',
                                             'tfidf_5_price'])

    df = pd.concat((df, df_word), axis=1)

    df['tfidf_1_price'] = df['tfidf_1_price'].map(word_price_map)
    df['tfidf_2_price'] = df['tfidf_2_price'].map(word_price_map)
    df['tfidf_3_price'] = df['tfidf_3_price'].map(word_price_map)
    df['tfidf_4_price'] = df['tfidf_4_price'].map(word_price_map)
    df['tfidf_5_price'] = df['tfidf_5_price'].map(word_price_map)

    df['desc_mean_price'] = df.apply(lambda row: np.mean(
        [row['tfidf_1_price'], row['tfidf_2_price'], row['tfidf_3_price'], row['tfidf_4_price'], row['tfidf_5_price']]),
                                     axis=1)
    df = df.drop(columns=['tfidf_1_price', 'tfidf_2_price', 'tfidf_3_price', 'tfidf_4_price', 'tfidf_5_price'])
    return df, ['desc_mean_price']


def process_categorical_features(data):
    data.loc[data['make'].isnull(), 'make'] = data['title'].map(lambda x: x.split(" ")[0])
    data['make'] = data['make'].str.lower()


def generate_one_hot_data(encoder, data, feature, new_columns):
    train_one_hot_fields = encoder.transform(data[[feature]]).toarray()
    train_one_hot_df = pd.DataFrame(train_one_hot_fields)
    train_one_hot_df.columns = new_columns
    return pd.concat([data, train_one_hot_df], axis=1)


def remove_outliers(df, mode):
    '''
    https://www.motortrend.com/features/20-of-the-lightest-cars-sold-in-the-u-s/
    '''
    if mode == 'train':
        print("remove curb_weight below normal range for training data: ", df[df['curb_weight'] <= 800].shape[0])
        df = df[df['curb_weight'] > 800]

        print("remove reg_date beyond today for training data: ", df[df['reg_date'] > datetime.now()].shape[0])
        df = df[df['reg_date'] <= datetime.now()]

        print("remove manufactured date beyond today for training data: ", df[df['manufactured'] > 2021].shape[0])
        df = df[df['manufactured'] <= 2021]

    return df
