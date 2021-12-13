import os
import shutil
from pathlib import Path
from abc import ABC
from typing import List
from enum import Enum
from fast_deploy.utils import slugify


class TritonIOTypeConf(Enum):
    STRING = 'TYPE_STRING'
    FP32 = 'TYPE_FP32'
    INT64 = 'TYPE_INT64'


class TritonIOConf:
    name: str
    data_type: str
    dims: List[int]

    def __init__(self, name: str, data_type: TritonIOTypeConf, dims: List[int]):
        self.name = name
        self.data_type = data_type.value
        self.dims = dims


class TritonModelConf:
    def __init__(
        self,
        workind_directory: str,
        model_name: str,
        batch_size: int,
        nb_instance: int,
        use_cuda: bool,
        model_inputs: List[TritonIOConf],
        model_outputs: List[TritonIOConf]
    ):
        self.model_name = model_name
        self.folder_name = slugify(model_name)
        self.batch_size = batch_size
        assert nb_instance > 0, f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        self.include_token_type = True
        self.workind_directory = workind_directory
        self.input_type = "TYPE_INT64"
        self.inference_platform = "onnxruntime_onnx"
        self.kind = "KIND_GPU" if use_cuda else "KIND_CPU"
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs

    def _get_inputs(self):
        inputs_conf = []
        for input_conf in self.model_inputs:
            input_conf = f"""{{
        name: "{input_conf.name}"
        data_type: {input_conf.data_type}
        dims: {input_conf.dims}
    }}"""
            inputs_conf.append(input_conf)
        return ",\n    ".join(inputs_conf)
    
    def _get_outputs(self):
        outputs_conf = []
        for output_conf in self.model_outputs:
            output_conf = f"""{{
        name: "{output_conf.name}"
        data_type: {output_conf.data_type}
        dims: {output_conf.dims}
    }}"""
            outputs_conf.append(output_conf)
        return ",\n    ".join(outputs_conf)

    def _instance_group(self):
        return f"""
instance_group [
    {{
      count: {self.nb_instance}
      kind: {self.kind}
    }}
]
""".strip()
    
    def get_conf(self) -> str:
        return f"""
name: "{self.folder_name}"
max_batch_size: {self.batch_size}
platform: "{self.inference_platform}"
default_model_filename: "model.bin"

input [
    {self._get_inputs()}
]

output [
    {self._get_outputs()}
]

{self._instance_group()}
""".strip()
    
    def write(self, model_path: str):
        """Create the models files for Triton.

        This create model structure for Triton create the models endpoints
        and setup the ensemble of tokenizer+model using an custom python
        request handler.

        Parameters
        ----------
        model_path: str
            Model full path
        """
        wd_path = Path(self.workind_directory)
        wd_path.mkdir(parents=True, exist_ok=True)

        current_folder = wd_path.joinpath(self.folder_name)
        current_folder.mkdir(exist_ok=True)

        conf = self.get_conf()
        current_folder.joinpath("config.pbtxt").write_text(conf)
        version_folder = current_folder.joinpath("1")
        version_folder.mkdir(exist_ok=True)

        model_folder_path = wd_path.joinpath(self.folder_name).joinpath("1")
        shutil.copy(model_path, os.path.join(model_folder_path, "model.bin"))


class TritonTokenizeConf(TritonModelConf):
    def __init__(
        self,
        workind_directory: str,
        model_name: str,
        batch_size: int,
        nb_instance: int,
        use_cuda: bool,
        model_inputs: List[TritonIOConf],
        model_outputs: List[TritonIOConf]
    ):
        super().__init__(
            workind_directory=workind_directory,
            model_name=model_name,
            batch_size=batch_size,
            nb_instance=nb_instance,
            use_cuda=use_cuda,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
        )

    def get_conf(self) -> str:
        return f"""
name: "{self.folder_name}"
max_batch_size: {self.batch_size}
backend: "python"
input [
    {self._get_inputs()}
]
output [
    {self._get_outputs()}
]
{self._instance_group()}
""".strip()

    def write(self, model_path: str):
        wd_path = Path(self.workind_directory)
        wd_path.mkdir(parents=True, exist_ok=True)

        current_folder = wd_path.joinpath(self.folder_name)
        current_folder.mkdir(exist_ok=True)

        conf = self.get_conf()
        current_folder.joinpath("config.pbtxt").write_text(conf)
        version_folder = current_folder.joinpath("1")
        version_folder.mkdir(exist_ok=True)

        tokenizer_folder_path = wd_path.joinpath(current_folder).joinpath("1")
        tokenizer_model_path = (
            Path(__file__).absolute().parent.joinpath("triton_models").joinpath("python_tokenizer.py")
        )

        for tokenizer_file in os.listdir(model_path):
            tokenizer_file_path = os.path.join(model_path, tokenizer_file)
            shutil.copy(
                str(tokenizer_file_path),
                str(Path(tokenizer_folder_path).joinpath(tokenizer_file)),
            )
        
        shutil.copy(
            str(tokenizer_model_path),
            str(Path(tokenizer_folder_path).joinpath("model.py")),
        )



def transformers_configuration(
    model_name="test",
    batch_size=0,
    nb_output_shape=[-1, -1],
    nb_instance=1,
    include_token_type=True,
    workind_directory="/home/rodrigo/triton_test",
    use_cuda=False,
    model_path=None,
    tokenize_path=None
):
    text = TritonIOConf(
        name='TEXT',
        data_type=TritonIOTypeConf.STRING,
        dims=[-1]
    )

    text_output = TritonIOConf(
        name='TEXT_OUTPUT',
        data_type=TritonIOTypeConf.STRING,
        dims=[5, -1]
    )

    input_ids = TritonIOConf(
        name='input_ids',
        data_type=TritonIOTypeConf.INT64,
        dims=[-1, -1]
    )

    token_type_ids= TritonIOConf(
        name='token_type_ids',
        data_type=TritonIOTypeConf.INT64,
        dims=[-1, -1]
    )

    attention_mask= TritonIOConf(
        name='attention_mask',
        data_type=TritonIOTypeConf.INT64,
        dims=[-1, -1]
    )

    output= TritonIOConf(
        name='output',
        data_type=TritonIOTypeConf.FP32,
        dims=[-1] + nb_output_shape
    )

    model_input = [input_ids, token_type_ids, attention_mask]
    model_output = [output]
    tokenize_input = [text]
    tokenize_output = [text_output]

    model_conf = TritonModelConf(
        workind_directory= workind_directory,
        model_name= model_name,
        batch_size= batch_size,
        nb_instance= nb_instance,
        use_cuda= use_cuda,
        model_inputs= model_input,
        model_outputs=model_output
    )
    model_conf.write(model_path)

    tokenizer_conf = TritonTokenizeConf(
        workind_directory= workind_directory,
        model_name= f"{model_name}_tokenizer",
        batch_size= batch_size,
        nb_instance= nb_instance,
        use_cuda= use_cuda,
        model_inputs= tokenize_input,
        model_outputs=tokenize_output
    )
    tokenizer_conf.write(tokenize_path)
