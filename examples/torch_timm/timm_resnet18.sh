#!/bin/bash


quick-deploy torch \
    -n resnet18 \
    -m resnet18.pt \
    -o ./models \
    -f resnet18.yaml \
    --no-quant
