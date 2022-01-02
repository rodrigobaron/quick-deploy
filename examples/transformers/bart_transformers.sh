#!/bin/bash

quick-deploy transformers \
    -n my-bart-base \
    -p summarization \
    -m "facebook/bart-base" \
    -o ./models/bart \
    --model-type bart \
    --num-heads 12 \
    --hidden-size 768 \
    --seq-len 512
