from fastapi import FastAPI
from main import main
from pydantic import BaseModel

app = FastAPI()


class RequestModel(BaseModel):
    img: str


@app.post("/")
async def inference(request: RequestModel):
    img, results, pt = await main(request.img)
    return {"visualized_img": img, "results": results.tolist(), "process_time": pt}
