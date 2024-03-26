import argparse
import nltk
import numpy as np
import pandas as pd
import pickle
import re
from textblob import Word
import xgboost as xgb
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('heldout_filename')
    parser.add_argument('classification_method')
    parser.add_argument('training')
    args, unknown = parser.parse_known_args()
    heldout_filename = args.heldout_filename
    classification_method = args.classification_method
    training = True if str.lower(args.training) == 'True' else False

    dataframe = pd.read_json(heldout_filename, lines=True)

    # Execute
    if classification_method == 'neural_network':
        neural_network(dataframe, training)
    elif classification_method == 'naive_bayes':
        naive_bayes(dataframe, training)
    else:
        logistic_regression(dataframe, training)


def neural_network(df, training=False):
    # TODO Raymond
    return


def naive_bayes(df, training=False):
    # TODO Nate
    return


def logistic_regression(df, training=False):
    df = df[['text', 'stars', 'useful', 'funny', 'cool']]
    if training:
        # Setup frame
        df['text'] = df['text'].apply(clean_text)
        df = df.dropna(subset=['text', 'stars', 'funny', 'cool', 'useful'])
        df = df.drop_duplicates()

        # Stopwords
        stop_words = stopwords.words('english')
        df['text'] = df['text'].apply(
            lambda x: " ".join(word for word in x.split() if word not in stop_words))
        other_stop_words = ['get', 'would', 'got', 'us', 'also', 'even', 'ive', 'im']
        df['text'] = df['text'].apply(lambda x: " ".join(word for word in x.split() if word not in other_stop_words))

        # Lemmatize
        df['lemmatized'] = df['clean_reviews'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

        # 0 index stars
        df['stars'] = df['stars'].apply(lambda x: x - 1)

        # Vectorize
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(df['lemmatized'])
        y = df['stars']

        # Train Classifier Model
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LogisticRegression(n_jobs=-1, max_iter=2000, solver='sag', random_state=42)

        # Predict Stars
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        print(classification_report(y_test, y_pred))

        # Dump
        with open('classifier', 'wb') as file:
            pickle.dump(model, file)

        # Train Linear Model
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
        pickle.dump(bst, open('linear_model.pkl', 'wb'))

        # Predict Linear Values
        preds = bst.predict(dtest)
        rmse_y1 = np.sqrt(mean_squared_error(y1_test, preds[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(y2_test, preds[:, 1]))
        rmse_y3 = np.sqrt(mean_squared_error(y3_test, preds[:, 2]))

        print("RMSE for Target Useful:", rmse_y1)
        print("RMSE for Target Funny:", rmse_y2)
        print("RMSE for Target Cool:", rmse_y3)

        # Dump
        with open('linear.pkl', 'wb') as file:
            pickle.dump(model, file)

    with open('classifier.pkl', 'rb') as file:
        model = pickle.load(file)

        x_test = vectorizer.fit_transform(df['text'])
        y_test = df['stars']

        model.fit(x_test, y_test)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)

        print(classification_report(y_test, y_pred))

    with open('linear.pkl') as file:
        model = pickle.load(file)

        preds = bst.predict(dtest)
        rmse_y1 = np.sqrt(mean_squared_error(y1_test, preds[:, 0]))
        rmse_y2 = np.sqrt(mean_squared_error(y2_test, preds[:, 1]))
        rmse_y3 = np.sqrt(mean_squared_error(y3_test, preds[:, 2]))

        print("RMSE for Target Useful:", rmse_y1)
        print("RMSE for Target Funny:", rmse_y2)
        print("RMSE for Target Cool:", rmse_y3)


if __name__ == "__main__":
    main()
