from fastapi import FastAPI
from main import main
from pydantic import BaseModel

app = FastAPI()


class RequestModel(BaseModel):
    img: str


@app.post("/")
def inference(request: RequestModel):
    img, results, pt = main(request.img)
    return {"visualized_img": img, "results": results.tolist(), "process_time": pt}
