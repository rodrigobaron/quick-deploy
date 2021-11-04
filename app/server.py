
import os
import logging
import json
from typing import List, Any

from fastapi import (
    FastAPI,
    Response,
    status
)
from session import create_model_for_provider


def setup_log():
    LOG_LEVEL_PARAM = os.getenv('LOG_LEVEL', 'INFO')

    logLevel = logging.INFO
    numeric_level = getattr(logging, LOG_LEVEL_PARAM.upper(), None)
    if isinstance(numeric_level, int):
        logLevel = numeric_level

    logging.basicConfig(
        level=logLevel,
        format='%(asctime)s %(thread)s %(pathname)s %(levelname)s %(message)s'
    )

app = FastAPI()
onnx_model = create_model_for_provider(Path('/onnx_server/model.onnx'))
#  TODO: Remove that 
from datasets import load_dataset
clinc = load_dataset("clinc_oos", "plus")
intents = clinc["test"].features["intent"]
pipe = OnnxPipeline(onnx_quantized_model, bert_tokenizer, intents=intents)

setup_log()

class InputRequest(BaseModel):
    instances: List[Any]

    class Config:
        schema_extra = {
            "example": {
                "instances": ["this is an example!"]
            }
        }

@app.get('/health_check')
async def health_check():
    return {"is_healthy": True}


@app.post('/predict')
async def predict(
        input_request: InputRequest,
        response: Response
    ):

    logging.info("Processing request..")
    return pipe(input_request.instances)
