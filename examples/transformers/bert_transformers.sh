#!/bin/bash

fast-deploy transformers \
    -n my-bert-base \
    -p fill-mask \
    -m bert-base-uncased \
    -o ../models \
    --model-type bert \
    --seq-len 128 \
    --cuda
