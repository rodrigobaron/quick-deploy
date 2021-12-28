# import tensorflow as tf

# from keras2onnx import convert_keras, save_model


# def tf_convert_onnx(
#     model: tf.keras.Model,
#     model_name: str,
#     output_path: str,
#     opset_version: int = 12,
#     verbose: bool = False,
# ) -> None:
#     """Convert an tensorflow tansformer model to onnx.

#     This model conversion is specific for transformers.

#     Parameters
#     ----------
#     model: torch.nn.Module
#         the tensorflow model to convert to.
#     output_path: str
#         the onnx output filepath
#     inputs_tf: OrderedDict[str, torch.Tensor]
#         the model inputs
#     opset_version: int
#         the onnx op version to use. Default is 12.
#     verbose: bool
#         show detailed logging. Defaul is False.
#     """
#     onnx_model = convert_keras(model, model_name, target_opset=opset_version)
#     save_model(onnx_model, output_path)
