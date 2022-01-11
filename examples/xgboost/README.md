# XGBoost Iris Classifier

In this example we'll cover the deployment of an `XGBClassifier` for the classical `Iris` dataset to demonstrate how easy is move machine learning model to a production ready soluction using **Quick-Deploy**.

## Install

Instal the dependencies:
```bash
$ pip install quick-deploy[xgboost] tritonclient geventhttpclient
```

## Model Training

First of all we need train the model ([iris_xgbc_trainer.py](iris_xgbc_trainer.py)):

```python
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

```

## Model Definition

Before move to production we need supply the IO definition ([iris_xgbc.yaml](iris_xgbc.yaml)), we know the dataset contains 4 input columns and 3 classes:

```yaml
kind: IOSchema
inputs:
  - name: input
    dtype: float32
    shape: [4]
outputs:
  - name: label
    shape: [-1]
    dtype: int64
  - name: probabilities
    shape: [-1, 3]
    dtype: float32
```

## Deploy

Now with the model trained `iris_xgbc.bin` and with the IO definition `iris_xgbc.yaml` we can use **Quick-Deploy**:

```bash
$ quick-deploy xgboost \
    -n iris_xgbc \
    -m iris_xgbc.bin \
    -o ./models \
    -f iris_xgbc.yaml
```

The arguments is prety straightforward. Now we run the `triton inference sever` ([run_inference_server.sh](run_inference_server.sh)), in this example using docker:

## Spin Up the Server 

```bash
$ docker run -it --rm \
    --shm-size 256m \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:21.11-py3 \
    tritonserver --model-repository=/models
```

With the server running we can consume our model using `tritonclient` ([iris_xgbc_triton.py](iris_xgbc_triton.py)):

## Consume the Model

```python
import numpy as np
import tritonclient.http

from sklearn.datasets import load_iris
import random


# load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# pick a random example
example_idx = random.randint(0, len(X))
example_X = X[example_idx]
example_y = y[example_idx]

# setup the server endpoint and models
model_name = "iris_xgbc"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

# verify if the model is ready to consume
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

# Define the input/output
model_input = tritonclient.http.InferInput(name="input", shape=(batch_size, 4), datatype="FP32")
model_label = tritonclient.http.InferRequestedOutput(name="label", binary_data=False)
model_proba = tritonclient.http.InferRequestedOutput(name="probabilities", binary_data=False)

# set the data and call the model
model_input.set_data_from_numpy(np.array([example_X]).astype(np.float32))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_label, model_proba]
)

# get model results
label_pred = response.as_numpy("label")[0]
probs = response.as_numpy("probabilities")[0]
label_prob = probs[label_pred]

print(f"Model predicted {label_pred}: with score of {label_prob}, the truth is {example_y}!")

```

Example display:
```
Model predicted 2: with score of 0.6975727677345276, the truth is 2!
```
