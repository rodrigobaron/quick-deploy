import numpy as np
from scipy.special import softmax


class OnnxPipeline:
    """Transformers pipeline compatible with onnx model

    This is used to create an standalone execution.

    Parameters
    ----------
    model: OnnxModel
        The onnx model to get the logits.
    tokenizer: Tokenizer
        Transformers tokenizer.
    intents:
        Convert tokens to words.
    """

    def __init__(self, model, tokenizer, intents=None):
        self.model = model
        self.tokenizer = tokenizer
        self.intents = intents

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [{"label": self.intents.int2str(pred_idx), "score": probs[pred_idx]}]
