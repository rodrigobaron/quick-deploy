#!/bin/bash

python -c "import timm; import torch; torch.save(timm.create_model('resnet18', pretrained=True), 'resnet18.pt')"

quick-deploy torch \
    -n resnet18 \
    -m resnet18.pt \
    -o ./models/resnet18 \
    -f resnet18.yaml \
    --no-quant
