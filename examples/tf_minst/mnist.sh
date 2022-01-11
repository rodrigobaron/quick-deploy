#!/bin/bash


quick-deploy tf \
    -n mnist \
    -m mnist_model \
    -o ./models \
    -f mnist.yaml
