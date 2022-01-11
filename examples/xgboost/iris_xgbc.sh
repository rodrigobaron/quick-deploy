#!/bin/bash


quick-deploy xgboost \
    -n iris_xgbc \
    -m iris_xgbc.bin \
    -o ./models \
    -f iris_xgbc.yaml
