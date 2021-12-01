import os
from typing import Dict

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType


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
        model_name: str = args["model_name"]
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
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query, return_tensors=TensorType.NUMPY
            )
            # communicate the tokenization results to Triton server
            input_ids = pb_utils.Tensor("input_ids", tokens["input_ids"])
            attention = pb_utils.Tensor("attention_mask", tokens["attention_mask"])
            outputs = [input_ids, attention]
            if "token_type_ids" in tokens.keys():
                token_type_ids = pb_utils.Tensor(
                    "token_type_ids", tokens["token_type_ids"]
                )
                outputs.append(token_type_ids)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses
