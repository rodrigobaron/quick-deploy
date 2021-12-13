import numpy as np
import tritonclient.http
from transformers import BertTokenizer, BertModel, TensorType
from scipy.special import softmax


def topK(x, k, axis=0):
    idx = np.argpartition(x, -k)[:,-k:]
    indices = idx[:, np.argsort((-x)[:, idx][0])][0]
    return indices


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


model_name = f"test_model"
model_name = f"test_model_tokenizer"

url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

text = "The goal of life is [MASK]."
tokens = tokenizer(text=text, return_tensors=TensorType.NUMPY)

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

# input_ids = tritonclient.http.InferInput(name="input_ids", shape=(batch_size, 9), datatype="INT64")
# token_type_ids = tritonclient.http.InferInput(name="token_type_ids", shape=(batch_size, 9), datatype="INT64")
# attention = tritonclient.http.InferInput(name="attention_mask", shape=(batch_size, 9), datatype="INT64")
# model_output = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

# input_ids.set_data_from_numpy(tokens['input_ids'] * batch_size)
# token_type_ids.set_data_from_numpy(tokens['token_type_ids'] * batch_size)
# attention.set_data_from_numpy(tokens['attention_mask'] * batch_size)

query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
text_output = tritonclient.http.InferRequestedOutput(name="TEXT_OUTPUT", binary_data=True)

query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[query], outputs=[text_output]
)
import pdb; pdb.set_trace()
token_logits = response.as_numpy("TEXT_OUTPUT")
mask_token_index = np.where(tokens['input_ids']  == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
mask_token_logits = softmax(mask_token_logits, axis=1)

top_5_indices = topK(mask_token_logits, 5, axis=1)
top_5_values = mask_token_logits[:,top_5_indices][0]

top_5_tokens = zip(top_5_indices[0].tolist(), top_5_values[0].tolist())

for token, score in top_5_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])), f"(score: {score})")
