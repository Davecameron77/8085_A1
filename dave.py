import numpy as np
import pandas as pd
import pickle
import re

import xgboost as xgb
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from textblob import Word


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

        # Train regression Model
        func_start = datetime.now()
        print('Training regression model...')
        y1 = df['useful']
        y2 = df['funny']
        y3 = df['cool']

        x_train, x_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x, y1, y2, y3,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)
        dtrain = xgb.DMatrix(x_train, label=np.column_stack((y1_train, y2_train, y3_train)))
        dtest = xgb.DMatrix(x_test, label=np.column_stack((y1_test, y2_test, y3_test)))

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'gamma': 0.85,
            'n_jobs': -1
        }

        num_rounds = 500
        bst = xgb.train(params, dtrain, num_rounds)
        print(f'Trained regression model in {datetime.now() - func_start} seconds\n')

        # Predict regression values
        func_start = datetime.now()
        preds = bst.predict(dtest)
        rmse_y1 = np.sqrt(mean_squared_error(y1_test, preds[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(y2_test, preds[:, 1]))
        rmse_y3 = np.sqrt(mean_squared_error(y3_test, preds[:, 2]))

        print("RMSE for Target Useful:", rmse_y1)
        print("RMSE for Target Funny:", rmse_y2)
        print("RMSE for Target Cool:", rmse_y3)
        print(f'Regression prediction completed in {datetime.now() - func_start} seconds\n')

        bst.save_model('regression.json')
        print(f'Script complete in {datetime.now() - starttime} seconds\n')

    @staticmethod
    def validate_models(filename):
        starttime = datetime.now()
        df = pd.read_json(filename, lines=True, nrows=5_000)
        df = df[['text', 'stars', 'funny', 'useful', 'cool']]

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

        print('Regression Accuracy')
        print('************************************************************\n')

        model = xgb.Booster()
        model.load_model('regression.json')

        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(df['text'])
        y1 = df['useful'].astype(int)
        y2 = df['funny'].astype(int)
        y3 = df['cool'].astype(int)

        dtest = xgb.DMatrix(x, label=np.column_stack((y1, y2, y3)))

        preds = model.predict(dtest)

        rmse_y1 = np.sqrt(mean_squared_error(y1, preds[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(y2, preds[:, 1]))
        rmse_y3 = np.sqrt(mean_squared_error(y3, preds[:, 2]))

        print("Root mean squared error for Useful:", rmse_y1)
        print("Root mean squared error for Funny:", rmse_y2)
        print("Root mean squared error for Cool:", rmse_y3)
        print(f'\nPredictions complete in {datetime.now() - starttime} seconds')
