import os
from typing import Dict

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


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
            # mask_token_index = pb_utils.get_input_tensor_by_name(request, "mask_token_index").as_numpy().tolist()
            mask_token_index = 6
            token_logits = pb_utils.get_input_tensor_by_name(request, "output").as_numpy()[0]
            
            mask_token_logits = token_logits[mask_token_index, :]
            mask_token_logits = softmax(mask_token_logits)

            top_5_indices = np.argpartition(mask_token_logits, -5)[:5]
            top_5_tokens_t = pb_utils.Tensor("mask_token", top_5_indices)
            inference_response = pb_utils.InferenceResponse(output_tensors=[top_5_tokens_t])
            responses.append(inference_response)

        return responses
