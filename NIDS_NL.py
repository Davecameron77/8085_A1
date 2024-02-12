import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



dataset = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)

#print("Unencoded:")
#print(dataset.head(10))

o = (dataset.dtypes == 'object')
object_cols = list(o[o].index)

#print("categorical variables:")
#print (object_cols)

le = LabelEncoder()
for label in object_cols:
    dataset[label] = le.fit_transform(dataset[label])

dataset.fillna(0, inplace=True)

#print("Encoded:")
#print (dataset.head(10))

print("Dataset read")

X = dataset.iloc[:, 0:46].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

clf = RandomForestClassifier()
sel = RFE(clf, n_features_to_select=10 , step=5)
sel = sel.fit(X_train, y_train)

res = sel.get_support()

print("Selected features:")
for b, label in zip(res,dataset.columns):
    if b:
        print(label)



"""
print("Dataset split")

clf = DecisionTreeClassifier(criterion="entropy", max_depth=500)
clf.fit(X_train, y_train)

print("Training Done")

y_pred = clf.predict(X_test)
ac = accuracy_score(y_test, y_pred) * 100
print("Accuracy is ", ac)
"""