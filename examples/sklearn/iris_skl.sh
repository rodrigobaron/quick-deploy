#!/bin/bash

fast-deploy sklearn \
    -n iris_cls \
    -m iris_cls.bin \
    -o ./models \
    -f iris_cls.yaml