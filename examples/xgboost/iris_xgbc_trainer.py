import os
import numpy
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from xgboost import XGBClassifier

data = load_iris()
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('lgbm', XGBClassifier(n_estimators=3))])
pipe.fit(X_train, y_train)

with open("iris_xgbc.bin", "wb") as p_file:
    pickle.dump(pipe, p_file)
