from fastapi import FastAPI
from pydantic import BaseModel

from main import main

app = FastAPI()


class RequestModel(BaseModel):
    img: str


@app.post("/")
def inference(request: RequestModel):
    img, results, pt = main(request.img)
    return {"visualized_img": img, "results": results.tolist(), "process_time": pt}
