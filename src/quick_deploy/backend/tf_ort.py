import tensorflow as tf
from tf2onnx.convert import from_keras


def tf_convert_onnx(
    model: tf.keras.Model,
    output_path: str,
    opset_version: int = 12,
    verbose: bool = False,
):
    """Convert an tensorflow model to onnx.

    This model conversion is specific for tensorflow.

    Parameters
    ----------
    model: tf.keras.Model
        the pytorch model to convert to.
    output_path: str
        the onnx output filepath
    opset_version: int
        the onnx op version to use. Default is 12.
    verbose: bool
        show detailed logging. Defaul is False.
    """
    from_keras(model, opset=opset_version, output_path=output_path)
