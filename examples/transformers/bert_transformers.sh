#!/bin/bash
    #-m "textattack/distilbert-base-uncased-imdb" \

fast-deploy transformers \
    -n my-bert-base \
    -p text-classification \
    -m bert-base-uncased \
    -o ./models/bert-imdb \
    --model-type bert \
    --seq-len 128
