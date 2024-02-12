#!/usr/bin/env python3
import pandas as pd
import numpy as np


# Import data
pd.set_option("max_colwidth", 50)
pd.set_option("display.max_columns", Nones)
dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', header=0, low_memory=False)
print("Dataset loaded\n")

# Transform object columns
# sport / dsport are really categorical, not numeric
# Booleans are treated as categorical
dataset['srcip'], _ = pd.factorize(dataset['srcip'])
dataset['sport'], _ = pd.factorize(dataset['sport'])
dataset['dstip'], _ = pd.factorize(dataset['dstip'])
dataset['dsport'], _ = pd.factorize(dataset['dsport'])
dataset['proto'], _ = pd.factorize(dataset['proto'])
dataset['state'], _ = pd.factorize(dataset['state'])
dataset['service'], _ = pd.factorize(dataset['service'])
dataset['ct_flw_http_mthd'], _ = pd.factorize(dataset['ct_flw_http_mthd'])
dataset['is_ftp_login'], _ = pd.factorize(dataset['is_ftp_login'])
dataset['ct_ftp_cmd'], _ = pd.factorize(dataset['ct_ftp_cmd'])
dataset['attack_cat'], _ = pd.factorize(dataset['attack_cat'])

# Dependant variable is both classification and prediction for now
X = dataset.iloc[:, :-2].values


print("Correlation\n**********************************************************")
correlation = dataset.corr()
with pd.option_context('expand_frame_repr', True):
    print(correlation)

print("\nCovariance\n**********************************************************")
covariance = dataset.cov()
with pd.option_context('expand_frame_repr', True):
    print(covariance)
