#!/bin/bash

fast-deploy transformers \
    -n my-bert-base \
    -p text-classification \
    -m bert-base-uncased \
    -o ./models/bert \
    --model-type bert \
    --seq-len 128