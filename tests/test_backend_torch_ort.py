from quick_deploy.backend.torch_ort import torch_convert_onnx
from quick_deploy.utils import parse_torch_input
import mock


@mock.patch("quick_deploy.backend.torch_ort.torch.onnx.export")
def test_torch_convert_onnx(m):
    inputs_pytorch, inputs_onnx = parse_torch_input(shape=(1,), batch_size=1, use_cuda=False)

    torch_convert_onnx(
        model='test', output_path='/tmp/torch', inputs_pytorch=inputs_pytorch, verbose=False
    )

    name, args, kwargs = m.mock_calls[0]

    assert ('test',) == args
    assert (1, 1) == kwargs['args'][0].shape
    assert '/tmp/torch' == kwargs['f']
    assert 12 == kwargs['opset_version']
    assert kwargs['do_constant_folding']
    assert ['input'] == kwargs['input_names']
    assert ['output'] == kwargs['output_names']
    assert {
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    } ==  kwargs['dynamic_axes']
