# Fast Deploy

$ docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/triton_models:/models nvcr.io/nvidia/tritonserver:21.11-py3 bash -c "pip install transformers && tritonserver --model-repository=/models"
