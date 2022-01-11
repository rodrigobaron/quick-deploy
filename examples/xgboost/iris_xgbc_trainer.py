import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import pickle

# load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# split the training data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train the model
pipe = Pipeline([('scaler', StandardScaler()), ('lgbm', XGBClassifier(n_estimators=3))])
pipe.fit(X_train, y_train)

# save the model
with open("iris_xgbc.bin", "wb") as p_file:
    pickle.dump(pipe, p_file)
