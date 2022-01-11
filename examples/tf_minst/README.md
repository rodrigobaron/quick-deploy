# Tensorflow MNIST example

In this example we'll cover the deployment of an `Tensorflow Dense Neural Network` for the classical `MNIST` dataset to demonstrate how easy is move machine learning model to a production ready soluction using **Quick-Deploy**.

## Install

Instal the dependencies:
```bash
$ pip install quick-deploy[tf] tritonclient geventhttpclient
```

## Model Training

First of all we need train the model ([mnist_train.py](mnist_train.py)):

```python
import os

import tensorflow as tf
from tensorflow import keras

# load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# take a subset for fast demostration
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# reshape to fit a dense layer
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential(
        [
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    return model


# Train the model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model.
model.save('mnist_model')

```

## Model Definition

Before move to production we need supply the IO definition ([mnist.yaml](mnist.yaml)), we know the dataset contains a 784 pixel input and 10 classes:

```yaml
kind: IOSchema
inputs:
  - name: dense_input
    dtype: float32
    shape: [784]
outputs:
  - name: dense_1
    shape: [10]
    dtype: float32
```

## Deploy

Now with the model trained `mnist_model` and with the IO definition `mnist.yaml` we can use **Quick-Deploy**:

```bash
$ quick-deploy tf \
    --name mnist \
    --model mnist_model \
    --output ./models \
    --file mnist.yaml
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

With the server running we can consume our model using `tritonclient` ([mnist_triton.py](mnist_triton.py)):

## Consume the Model

```python
import numpy as np
import tensorflow as tf
import tritonclient.http
from scipy.special import softmax


# setup the server endpoint and models
model_name = "mnist"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

# load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# take a example
label_example = test_labels[1]
image_example = test_images[1].reshape(-1, 28 * 28) / 255.0

# verify if the model is ready to consume
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

# Define the input/output
model_input = tritonclient.http.InferInput(name="dense_input", shape=(batch_size, 784), datatype="FP32")
model_output = tritonclient.http.InferRequestedOutput(name="dense_1", binary_data=False)

# set the data and call the model
model_input.set_data_from_numpy(image_example.astype(np.float32))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_output]
)

output = response.as_numpy("dense_1")
class_pred = np.argmax(softmax(output))

print(f"Model predicted {class_pred}, truth is {label_example}!")

```

Example display:
```
Model predicted 2, truth is 2!
```