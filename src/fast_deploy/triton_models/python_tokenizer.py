import os
from typing import Dict

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType
from scipy.special import softmax


def topK(x, k, axis=0):
    idx = np.argpartition(x, -k)[:,-k:]
    indices = idx[:, np.argsort((-x)[:, idx][0])][0]
    return indices


class TritonPythonModel:
    """Custom Triton model handler.

    This is used to get more than one input, in our case 'input_ids' and 'attention_mask'.

    Attributes
    ----------
    tokenizer: PreTrainedTokenizer
        The transformers tokenize
    """

    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]):
        """Initialize attributes from Triton configuration.

        This is used to initialize the transformer tokenizer from model repository config.

        Parameters
        ----------
        args: Dict[str, str]
            Triton parsed configuration
        """
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def execute(self, requests):
        """Handle custom Triton inference execution

        Parameters
        ----------
        requests: Iterable
            The request instances.

        Returns
        ----------
        pb_utils.InferenceResponse
            The Triton response
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            tokens: Dict[str, np.ndarray] = self.tokenizer(text=query, return_tensors=TensorType.NUMPY)
            # communicate the tokenization results to Triton server
            input_ids = pb_utils.Tensor("input_ids", tokens["input_ids"])
            attention = pb_utils.Tensor("attention_mask", tokens["attention_mask"])
            outputs = [input_ids, attention]
            if "token_type_ids" in tokens.keys():
                token_type_ids = pb_utils.Tensor("token_type_ids", tokens["token_type_ids"])
                outputs.append(token_type_ids)

            inference_request = pb_utils.InferenceRequest(
                model_name='test_model',
                requested_output_names=['output'],
                inputs=outputs
            )

            inference_response = inference_request.exec()

            # Check if the inference response has an error
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                # Extract the output tensors from the inference response.
                token_logits = pb_utils.get_output_tensor_by_name(inference_response, 'output').as_numpy()
                mask_token_index = np.where(tokens['input_ids']  == self.tokenizer.mask_token_id)[1]
                mask_token_logits = token_logits[0, mask_token_index, :]
                mask_token_logits = softmax(mask_token_logits, axis=1)

                top_5_indices = topK(mask_token_logits, 5, axis=1)
                top_5_values = mask_token_logits[:,top_5_indices][0]

                top_5_tokens = zip(top_5_indices[0].tolist(), top_5_values[0].tolist())
                # text_output = pb_utils.Tensor("TEXT_OUTPUT", np.array([t.encode() for t in self.tokenizer.decode([[k] for k in top_5_indices[0].tolist()])]))

                text_output_bytes = []
                for token, score in top_5_tokens:
                    text_output_bytes.append(self.tokenizer.decode([token]).encode())
                #     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])), f"(score: {score})")

                text_output = pb_utils.Tensor("TEXT_OUTPUT", np.array(text_output_bytes))
                inference_response = pb_utils.InferenceResponse(output_tensors=[text_output])
                responses.append(inference_response)

        return responses
