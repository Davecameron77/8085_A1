import numpy as np
import pandas as pd
import pickle
import re

import xgboost as xgb
from datetime import datetime

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from textblob import Word
from tqdm import tqdm


class Dave:
    @staticmethod
    def clean_text(text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))
        text = text.lower()  # lowercase text
        text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
        return text

    @staticmethod
    def generate_tokens(text):
        return pos_tag(word_tokenize(text))

    @staticmethod
    def create_training_data(tokens):
        return ["{}_{}".format(word, tag) for word, tag in tokens]

    @staticmethod
    def create_training_string(tokens):
        return [" ".join(tokens)]

    @staticmethod
    def train_models(filename):
        # Open + Setup
        starttime = datetime.now()
        func_start = datetime.now()
        print('Loading data...')
        df = pd.read_json(filename, lines=True)
        df = df[['text', 'stars', 'useful', 'funny', 'cool']]
        print(f'Data loaded in {datetime.now() - func_start} seconds\n')

        # Clean Text, drop invalid
        func_start = datetime.now()
        print('Cleaning data...')
        df['text'] = df['text'].apply(Dave.clean_text)
        df = df.dropna(subset=['text', 'stars', 'funny', 'cool', 'useful'])
        df = df.drop_duplicates()
        print(f'Data cleaned in {datetime.now() - func_start} seconds\n')

        # Additional stopwords
        func_start = datetime.now()
        print('Removing additional stopwords...')
        other_stop_words = ['get', 'would', 'got', 'us', 'also', 'even', 'ive', 'im']
        df['text'] = df['text'].apply(lambda x: " ".join(word for word in x.split() if word not in other_stop_words))
        print(f'Other stops removed in {datetime.now() - func_start} seconds\n')

        # Lemmatize
        func_start = datetime.now()
        print('Lemmatizing...')
        df['lemmatized'] = df['text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
        print(f'Lemmatized in {datetime.now() - func_start} seconds\n')

        # 0 index stars
        df['stars'] = df['stars'].apply(lambda x: x - 1)

        # Vectorize
        func_start = datetime.now()
        print(f'Vectorizing...')
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(df['lemmatized'])
        y = df['stars']
        print(f'Vectorized in {datetime.now() - func_start} seconds\n')

        # region Classifier

        # Train Classifier Model
        func_start = datetime.now()
        print(f'Training LogisticRegression...')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LogisticRegression(n_jobs=-1, max_iter=2000, solver='sag', random_state=42)
        print(f'Trained LogisticRegression in {datetime.now() - func_start} seconds\n')

        # Predict Stars
        func_start = datetime.now()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        print(classification_report(y_test, y_pred))
        print(f'Made classification prediction in {datetime.now() - func_start} seconds\n')

        # Dump
        with open('classifier.pkl', 'wb') as file:
            pickle.dump(model, file)

        # endregion

        # region Regression

        # Train regression Model
        func_start = datetime.now()
        print('Training regression model...')
        y1 = df['useful']
        y2 = df['funny']
        y3 = df['cool']

        x_train, x_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x, y1, y2, y3,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'gamma': 0.85,
            'n_jobs': -1
        }

        regressor = xgb.XGBRegressor(**params)
        regressor.fit(x_train, np.column_stack((y1_train, y2_train, y3_train)))
        print(f'Trained regression model in {datetime.now() - func_start} seconds\n')

        # Predict regression values
        func_start = datetime.now()
        preds = regressor.predict(x_test)
        rmse_y1 = np.sqrt(mean_squared_error(y1_test, preds[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(y2_test, preds[:, 1]))
        rmse_y3 = np.sqrt(mean_squared_error(y3_test, preds[:, 2]))

        print("RMSE for Target Useful:", rmse_y1)
        print("RMSE for Target Funny:", rmse_y2)
        print("RMSE for Target Cool:", rmse_y3)
        print(f'Regression prediction completed in {datetime.now() - func_start} seconds\n')

        regressor.save_model('regression.json')
        print(f'Script complete in {datetime.now() - starttime} seconds\n')

        #endregion

        # region POS tagging

        # Open and preprocess training data
        df2 = pd.read_json('yelp_reviews.json', lines=True, nrows=500_000)
        df2 = df2.fillna(0)
        df2 = df2.dropna(subset=['text', 'stars', 'funny', 'cool', 'useful'])
        df2 = df2.drop_duplicates()
        df2 = df2[['text', 'stars', 'useful', 'funny', 'cool']]
        df2['text'] = df2['text'].apply(Dave.clean_text)
        df2['text'] = df2['text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

        # Generate tags
        df2['tags'] = df2['text'].apply(Dave.generate_tokens)
        df2['trainable'] = df2['tags'].apply(Dave.create_training_data)
        df2['training_ready'] = df2['trainable'].apply(Dave.create_training_string)
        df2['training_ready'] = df2['training_ready'].astype(str)

        # Vectorize
        x2 = vectorizer.fit_transform(df2['training_ready'])
        y2 = df2['stars']

        # Train model
        x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=42)
        model = LogisticRegression(n_jobs=-1, max_iter=2000, solver='sag', random_state=42)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Validate
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        print(classification_report(y_test, y_pred))

        with open('classifier_pos.pkl', 'wb') as file:
            pickle.dump(model, file)

        # endregion

        # region Adjustment

        # Work on a copy of data
        df3 = df

        # Perform adjustments
        sid = SentimentIntensityAnalyzer()

        down = 0
        up = 0
        for i, row in tqdm(df3.iterrows(), total=len(df)):
            if row['stars'] == 0 or row['stars'] == 4:
                continue
            else:
                scores = sid.polarity_scores(row['text'])
                # Reduce 2-4 star reviews with negative scores
                if scores['neg'] > 0.1 and 0.0 < row['stars'] < 4.0:
                    row['stars'] -= 1
                    down += 1
                # Reduce 2-4 star reviews with low compound sentiment
                if scores['compound'] < 0.25 and 0.0 < row['stars'] < 4.0:
                    row['stars'] -= 1
                    down += 1
                # Increase 2-4 star reviews with low negativity
                if scores['neg'] < 0.01 and 4.0 > row['stars'] > 0.0:
                    row['stars'] += 1
                    up += 1

        print(f'Altered {down} rows down and {up} rows up')

        # Vectorize
        x = vectorizer.fit_transform(df3['text'])
        y = df['stars']

        # Train model
        x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=42)
        model = LogisticRegression(n_jobs=-1, max_iter=2000, solver='sag', random_state=42)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Validate
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        print(classification_report(y_test, y_pred))

        with open('classifier_adjusted.pkl', 'wb') as file:
            pickle.dump(model, file)

        # endregion

    @staticmethod
    def validate_models(filename):
        starttime = datetime.now()
        df = pd.read_json(filename, lines=True, nrows=5_000)
        df = df[['text', 'stars', 'funny', 'useful', 'cool']]

        # region Classification validation

        with open('classifier.pkl', 'rb') as file:
            model = pickle.load(file)

        vectorizer = TfidfVectorizer()
        x_test = vectorizer.fit_transform(df['text'])
        y_test = df['stars']

        model.fit(x_test, y_test)
        y_pred = model.predict(x_test)

        print('\n')
        print('Classifier Accuracy')
        print('************************************************************\n')

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        print(classification_report(y_test, y_pred))

        # endregion

        # region Regression Validation

        print('Regression Accuracy')
        print('************************************************************\n')

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'gamma': 0.85,
            'n_jobs': -1
        }

        model = xgb.XGBRegressor(**params)
        model.load_model('regression.json')

        x = vectorizer.fit_transform(df['text'])
        y1 = df['useful'].astype(int)
        y2 = df['funny'].astype(int)
        y3 = df['cool'].astype(int)
        model.fit(x, np.column_stack((y1, y2, y3)))

        preds = model.predict(x)

        rmse_y1 = np.sqrt(mean_squared_error(y1, preds[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(y2, preds[:, 1]))
        rmse_y3 = np.sqrt(mean_squared_error(y3, preds[:, 2]))

        print("Root mean squared error for Useful:", rmse_y1)
        print("Root mean squared error for Funny:", rmse_y2)
        print("Root mean squared error for Cool:", rmse_y3)

        # endregion

        # region POS tagged validation
        #TODO
        with open('classifier_pos.pkl', 'rb') as file:
            model = pickle.load(file)

        df2 = pd.read_csv('pos_df.csv')
        x_test = vectorizer.fit_transform(df2['training_ready'])
        y_test = df2['stars']

        model.fit(x_test, y_test)
        y_pred = model.predict(x_test)

        print('\n')
        print('Vader Adjusted Classifier Accuracy')
        print('************************************************************\n')

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        print(classification_report(y_test, y_pred))

        # endregion

        # region Sentiment Adjusted

        with open('classifier_adjusted.pkl', 'rb') as file:
            model = pickle.load(file)

        x_test = vectorizer.fit_transform(df['text'])
        y_test = df['stars']

        model.fit(x_test, y_test)
        y_pred = model.predict(x_test)

        print('\n')
        print('Vader Adjusted Classifier Accuracy')
        print('************************************************************\n')

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        print(classification_report(y_test, y_pred))

        # endregion

        print(f'\nPredictions complete in {datetime.now() - starttime} seconds')
