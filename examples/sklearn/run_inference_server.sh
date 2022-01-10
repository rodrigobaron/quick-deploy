#!/bin/bash

docker run -it --rm \
    --shm-size 256m \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:21.11-py3 \
    tritonserver --model-repository=/models
