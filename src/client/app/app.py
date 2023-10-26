import base64
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI
from pydantic import BaseModel
from tritonclient.utils import *

app = FastAPI()


class RequestModel(BaseModel):
    img: str


async def decode_img(img: str) -> np.ndarray:
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


async def inference(img: np.ndarray) -> np.ndarray:
    SERVER_URL = "triton-inference-server-svc.yolo:8001"
    MODEL_NAME = "YOLO"
    with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
        inputs = [
            grpcclient.InferInput(
                "INPUT_IMAGE", img.shape, np_to_triton_dtype(np.uint8)
            )
        ]
        inputs[0].set_data_from_numpy(img)
        outputs = [
            grpcclient.InferRequestedOutput("RESULTS"),
            grpcclient.InferRequestedOutput("VISUALIZE"),
        ]
        response = triton_client.infer(
            model_name=MODEL_NAME, inputs=inputs, outputs=outputs
        )
        response.get_response()
        RESULTS = response.as_numpy("RESULTS")
        VISUALIZE = response.as_numpy("VISUALIZE")
    return RESULTS, VISUALIZE


async def encode_img(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/")
async def main(request: RequestModel):
    img = await decode_img(request.img)
    START = time.time()
    RESULTS, VISUALIZE = await inference(img)
    END = time.time()
    VISUALIZE = await encode_img(VISUALIZE)
    return {
        "visualized_img": VISUALIZE,
        "results": RESULTS.tolist(),
        "process_time": END - START,
    }
