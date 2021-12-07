#!/bin/bash

fast-deploy torch \
    -n mobilenetv3 \
    -m mobilenetv3_large_100.pt \
    -o ./models/mobilenetv3 \
    --input-shape "3, 224, 224"
