from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)
clr.predict_proba = None

with open("iris_cls.bin", "wb") as p_file:
    pickle.dump(clr, p_file)
import pdb; pdb.set_trace()

import onnxruntime as rt
import numpy
sess = rt.InferenceSession("models/iris_cls/iris_cls/1/model.bin")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
import pdb; pdb.set_trace()