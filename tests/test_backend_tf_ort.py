import mock
import tensorflow as tf

from quick_deploy.backend.tf_ort import tf_convert_onnx
from quick_deploy.utils import parse_tf_input


@mock.patch("quick_deploy.backend.tf_ort.from_keras")
def test_tf_convert_onnx(m):
    inputs_tf, inputs_onnx = parse_tf_input(shape=(1,), batch_size=1)
    model = tf.keras.Model()
    tf_convert_onnx(model=model, output_path='/tmp/tf', verbose=False)

    name, args, kwargs = m.mock_calls[0]

    assert (model,) == args
    assert '/tmp/tf' == kwargs['output_path']
    assert 12 == kwargs['opset']
