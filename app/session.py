from pathlib import Path
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)


def create_model_for_provider(model_path: Path, num_threads: int = 1, provider: str = "CPUExecutionProvider"):
    options = SessionOptions()
    options.intra_op_num_threads = num_threads
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(model_path.as_posix(), options, providers=[provider])
    session.disable_fallback()
    return session
