from typing import NamedTuple, Tuple, Optional, List


class ModelSchema(NamedTuple):
    input_names: List[str]
    input_shape: Tuple[int]
    output_names: List[str]
    output_shape: Tuple[int]
