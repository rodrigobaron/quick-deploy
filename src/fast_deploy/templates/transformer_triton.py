import os
import shutil
from pathlib import Path

from transformers import PreTrainedTokenizer


class TransformersConfiguration:
    """Transformer Triton configuration

    This create the triton api for the models, in case of an transformer
    model it create three services: model, tokenizer and ensemble of
    tokenizer + model.

    Parameters
    ----------
    workind_directory: str
        The directory to output the models and configuration.
    model_name: str
        The model to registry in Triton.
    batch_size: int
        The optimized batch size.
    nb_output: int
        The output dimension.
    nb_instance: int
        Number of model instances Triton should run.
    include_token_type: bool
        Handle the token input.
    use_cuda: bool
        Handle GPU model allocation.
    """

    def __init__(
        self,
        workind_directory: str,
        model_name: str,
        batch_size: int,
        nb_output: int,
        nb_instance: int,
        include_token_type: bool,
        use_cuda: bool,
    ):
        self.model_name = f"{model_name}_onnx"
        self.model_folder_name = f"{self.model_name}_model"
        self.tokenizer_folder_name = f"{self.model_name}_tokenize"
        self.inference_folder_name = f"{self.model_name}_inference"
        self.batch_size = batch_size
        self.nb_model_output = nb_output
        assert (
            nb_instance > 0
        ), f"nb_instance=={nb_instance}: nb model instances should be positive"
        self.nb_instance = nb_instance
        self.include_token_type = include_token_type
        self.workind_directory = workind_directory
        self.input_type = "TYPE_INT64"
        self.inference_platform = "onnxruntime_onnx"
        self.kind = "KIND_GPU" if use_cuda else "KIND_CPU"

    def __get_tokens(self):
        token_type = ""
        if self.include_token_type:
            token_type = f"""    {{
        name: "token_type_ids"
        data_type: {self.input_type}
        dims: [-1, -1]
    }},
"""
        return f"""{{
        name: "input_ids"
        data_type: {self.input_type}
        dims: [-1, -1]
    }},
    {token_type}
    {{
        name: "attention_mask"
        data_type: {self.input_type}
        dims: [-1, -1]
    }}
"""

    def __instance_group(self):
        return f"""
instance_group [
    {{
      count: {self.nb_instance}
      kind: {self.kind}
    }}
]
""".strip()

    def get_model_conf(self) -> str:
        return f"""
name: "{self.model_folder_name}"
max_batch_size: {self.batch_size}
platform: "{self.inference_platform}"
default_model_filename: "model.bin"

input [
    {self.__get_tokens()}
]

output {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {self.nb_model_output}]
}}

{self.__instance_group()}
""".strip()

    def get_tokenize_conf(self):
        return f"""
name: "{self.tokenizer_folder_name}"
max_batch_size: {self.batch_size}
backend: "python"

input [
    {{
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }}
]

output [
    {self.__get_tokens()}
]

{self.__instance_group()}
""".strip()

    def get_inference_conf(self):
        input_token_type_ids = ""
        if self.include_token_type:
            input_token_type_ids = """
            {
                key: "token_type_ids"
                value: "token_type_ids"
            },
        """.strip()
        output_token_type_ids = ""
        if self.include_token_type:
            output_token_type_ids = """
            {
                key: "token_type_ids"
                value: "token_type_ids"
            },
        """.strip()
        return f"""
name: "{self.inference_folder_name}"
max_batch_size: {self.batch_size}
platform: "ensemble"

input [
    {{
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }}
]

output {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {self.nb_model_output}]
}}

ensemble_scheduling {{
    step [
        {{
            model_name: "{self.tokenizer_folder_name}"
            model_version: -1
            input_map {{
            key: "TEXT"
            value: "TEXT"
        }}
        output_map [
            {{
                key: "input_ids"
                value: "input_ids"
            }},
            {input_token_type_ids}
            {{
                key: "attention_mask"
                value: "attention_mask"
            }}
        ]
        }},
        {{
            model_name: "{self.model_folder_name}"
            model_version: -1
            input_map [
                {{
                    key: "input_ids"
                    value: "input_ids"
                }},
                {output_token_type_ids}
                {{
                    key: "attention_mask"
                    value: "attention_mask"
                }}
            ]
        output_map {{
                key: "output"
                value: "output"
            }}
        }}
    ]
}}
""".strip()

    def create_folders(self, tokenizer: PreTrainedTokenizer, model_path: str):
        """Create the models folder for Triton.

        This create model structure for Triton create the models endpoints
        and setup the ensemble of tokenizer+model using an custom python
        request handler.

        Parameters
        ----------
        tokenizer: PreTrainedTokenizer
            Transformers pre-trained tokenizer
        model_path: str
            Model full path
        """
        wd_path = Path(self.workind_directory)
        wd_path.mkdir(parents=True, exist_ok=True)
        for folder_name, conf_func in [
            (self.model_folder_name, self.get_model_conf),
            (self.tokenizer_folder_name, self.get_tokenize_conf),
            (self.inference_folder_name, self.get_inference_conf),
        ]:
            current_folder = wd_path.joinpath(folder_name)
            current_folder.mkdir(exist_ok=True)
            conf = conf_func()
            current_folder.joinpath("config.pbtxt").write_text(conf)
            version_folder = current_folder.joinpath("1")
            version_folder.mkdir(exist_ok=True)

        tokenizer_model_folder_path = wd_path.joinpath(
            self.tokenizer_folder_name
        ).joinpath("1")
        tokenizer.save_pretrained(str(tokenizer_model_folder_path.absolute()))
        tokenizer_model_path = (
            Path(__file__)
            .absolute()
            .parent.parent.joinpath("triton_models")
            .joinpath("python_tokenizer.py")
        )
        shutil.copy(
            str(tokenizer_model_path),
            str(Path(tokenizer_model_folder_path).joinpath("model.py")),
        )
        model_folder_path = wd_path.joinpath(self.model_folder_name).joinpath("1")
        shutil.copy(model_path, os.path.join(model_folder_path, "model.bin"))
