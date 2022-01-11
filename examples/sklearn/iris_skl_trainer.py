from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pickle

# load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# split the training data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train the model
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# save the model
with open("iris_cls.bin", "wb") as p_file:
    pickle.dump(clr, p_file)
