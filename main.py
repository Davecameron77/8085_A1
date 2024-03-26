import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('heldout_filename')
    parser.add_argument('classification_method')
    parser.add_argument('training')
    args, unknown = parser.parse_known_args()
    heldout_filename = args.heldout_filename
    classification_method = args.classification_method
    training = True if str.lower(args.training) == 'True' else False

    dataframe = pd.read_csv(heldout_filename, lines=True)

    if classification_method == 'neural_network':
        neural_network(dataframe, training)
    elif classification_method == 'naive_bayes':
        naive_bayes(dataframe, training)
    else:
        logistic_regression(dataframe, training)


def neural_network(df, training=False):
    #TODO Raymond
    return


def naive_bayes(df, training=False):
    #TODO Nate
    return


def logistic_regression(df, training=False):
    #TODO Dave
    return


if __name__ == "__main__":
    main()