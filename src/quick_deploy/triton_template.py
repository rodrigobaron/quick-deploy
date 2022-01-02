import os
import shutil
from enum import Enum
from pathlib import Path
from typing import List

from quick_deploy.utils import slugify


class TritonIOTypeConf(Enum):
    STRING = 'TYPE_STRING'
    FP32 = 'TYPE_FP32'
    INT64 = 'TYPE_INT64'

    @classmethod
    def from_str(cls, content):
        content = content.lower().strip()
        if 'string' == content:
            return TritonIOTypeConf.STRING
        if 'float32' == content:
            return TritonIOTypeConf.FP32
        if 'int64' == content:
            return TritonIOTypeConf.INT64

        raise ValueError


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
        model_outputs: List[TritonIOConf],
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
        for _input in self.model_inputs:
            input_conf = f"""{{
        name: "{_input.name}"
        data_type: {_input.data_type}
        dims: {_input.dims}
    }}"""
            inputs_conf.append(input_conf)
        return ",\n    ".join(inputs_conf)

    def _get_outputs(self):
        outputs_conf: List[str] = []
        for output in self.model_outputs:
            output_conf = f"""{{
        name: "{output.name}"
        data_type: {output.data_type}
        dims: {output.dims}
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
