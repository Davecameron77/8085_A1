import argparse
import numpy as np
import pandas as pd
import pickle
from dave import Dave
from NaiveBayes import NBClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
import deeplearning 
import torch
import yelp

def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('heldout_filename')
    parser.add_argument('classification_method')
    parser.add_argument('training')
    
    args, unknown = parser.parse_known_args()
    heldout_filename = args.heldout_filename
    classification_method = args.classification_method
    training = True if str.lower(args.training) == 'true' else False
    if classification_method != 'neural_network': 
        dataframe = pd.read_json(heldout_filename, lines=True)
    else:
        dataframe = pd.read_json(heldout_filename)
    # Execute
    if classification_method == 'neural_network':
        target = 'stars'
        # target = 'funny' 
        # target = 'cool'
        # target = 'useful'
        neural_network(dataframe, training, target)
    elif classification_method == 'naive_bayes':
        naive_bayes(dataframe, training)
    else:
        if training:
            Dave.train_models(heldout_filename)
        else:
            Dave.validate_models(heldout_filename)


def neural_network(df, training=False, target='stars'):
    if training:
        train_dataset = yelp.YelpDataset(df)
        train_loader = yelp.DataLoader(train_dataset, shuffle=True)
        if target == 'stars':
            deeplearning.training_TransformerRNNClassifier(train_loader)    
        else: 
           deeplearning.training(train_loader, target)
    else:
        # model = 'TransformerRNNClassifier'
        # model = 'TransformerRNNRegressor_cool'
        # model = 'TransformerRNNRegressor_useful'
        model = 'TransformerRNNRegressor_funny'
        model = torch.load(model)
        test = yelp.YelpDataset(df)
        test_loader = yelp.DataLoader(test, shuffle=True)
        if target == 'stars':
            deeplearning.validation(model, test_loader)
        else: 
            deeplearning.validation_for_regression(model, test_loader)
    return


def naive_bayes(df, training=False):
    if training:
        X = df['text']
        y = df[['stars', 'cool', 'useful', 'funny']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = NBClassifier(alpha_s=0.07, alpha_cuf=0.00000000000001, ngram=1)

        clf.train(X_train, y_train)

    else:
        clf = pickle.load(open('NBmodel', 'rb'))

        X_test = df['text']
        y_test = df[['stars', 'cool', 'useful', 'funny']]

    pred = clf.predict(X_test)

    s_true = y_test['stars'].values
    c_true = y_test['cool'].values
    u_true = y_test['useful'].values
    f_true = y_test['funny'].values

    accuracy = accuracy_score(s_true, pred[0])
    print("Accuracy: ", accuracy)

    print(classification_report(s_true, pred[0]))

    c_error = np.sqrt(mean_squared_error(c_true, pred[1]))
    u_error = np.sqrt(mean_squared_error(u_true, pred[2]))
    f_error = np.sqrt(mean_squared_error(f_true, pred[3]))

    print("RMSE for Target Useful:", u_error)
    print("RMSE for Target Funny:", f_error)
    print("RMSE for Target Cool:", c_error)
    return


if __name__ == "__main__":
    main()
