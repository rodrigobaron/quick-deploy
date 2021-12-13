from fast_deploy.templates.transformer_triton import TransformersConfiguration
from fast_deploy.templates.generic_triton import transformers_configuration

from transformers import pipeline

from fast_deploy.utils import parse_transformer_torch_input
from fast_deploy.backend.transformers_ort import (
    transformers_convert_pytorch
)
from pathlib import Path
import torch
import numpy as np

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

pipe = pipeline("fill-mask", "bert-base-uncased")
# pipe = pipeline("text-classification", "bert-base-uncased")
pipe_tokenizer = pipe.tokenizer
pipe_model = pipe.model
dataset = [{"text": "The goal of life is [MASK]."}]
text = dataset[0]["text"]

input_ids = pipe_tokenizer.encode(text, return_tensors="pt")
mask_token_index = torch.where(input_ids == pipe_tokenizer.mask_token_id)[1]
print('mask_token_index', mask_token_index)

with torch.inference_mode():
    token_logits = pipe_model(input_ids)[0]
    token_logits = token_logits.numpy()

# mask_token_logits = token_logits[0, mask_token_index, :]
# mask_token_logits = torch.softmax(mask_token_logits, dim=1)
# top_5 = torch.topk(mask_token_logits, 5, dim=1)
import pdb; pdb.set_trace()

mask_token_index = 6
mask_token_logits = token_logits[0, mask_token_index, :]
mask_token_logits = softmax(mask_token_logits)

top_5_indices = np.argpartition(mask_token_logits, -5)[:5]


top_5_tokens = zip(top_5.indices[0].tolist(), top_5.values[0].tolist())

for token, score in top_5_tokens:
    print(text.replace(pipe_tokenizer.mask_token, pipe_tokenizer.decode([token])), f"(score: {score})")