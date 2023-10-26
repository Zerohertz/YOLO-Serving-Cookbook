import base64

import cv2
import numpy as np
import requests

import gradio as gr

URL = "http://fastapi-svc.yolo:80"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}


def encode_img(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def send_request(img):
    DATA = {"img": img}
    response = requests.post(URL, headers=HEADERS, json=DATA, verify=False)
    res = response.json()
    return res


def decode_img(img: str) -> np.ndarray:
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def main(img):
    img = encode_img(img)
    try:
        RESULTS = send_request(img)
    except Exception as e:
        return None, e, None
    img = decode_img(RESULTS["visualized_img"])
    return (
        img,
        "\n".join([str(tmp) for tmp in RESULTS["results"]]),
        RESULTS["process_time"],
    )


inputs = [
    gr.Image(label="입력 이미지", type="numpy"),
]
outputs = [
    gr.Image(label="출력 이미지", type="numpy"),
    gr.Textbox(label="API 응답", text_align="left", show_copy_button=True),
    gr.Textbox(label="Process Time", text_align="left", show_copy_button=True),
]

iface = gr.Interface(
    fn=main, inputs=inputs, outputs=outputs, title="YOLO", allow_flagging="never"
)

iface.launch(share=False, server_name="0.0.0.0")
