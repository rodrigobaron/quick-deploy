#!/bin/bash

fast-deploy sklearn \
    -n iris_cls \
    -m iris_cls.bin \
    -o ./models/iris_cls \
    -f iris_cls.yaml