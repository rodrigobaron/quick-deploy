#!/bin/bash

quick-deploy xgboost \
    -n iris_xgbc \
    -m iris_xgbc.bin \
    -o ./models/iris_xgbc \
    -f iris_xgbc.yaml