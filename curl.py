import base64

import cv2
import numpy as np
import requests

URL = "http://zerohertz.xyz:80/"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

if __name__ == "__main__":
    file_id = "test.jpg"
    img = cv2.imread(file_id)
    _, buffer = cv2.imencode(".jpg", img)
    DATA = {
        "img": base64.b64encode(buffer).decode("utf-8"),
    }
    response = requests.post(URL, headers=HEADERS, json=DATA, verify=False)
    res = response.json()
    img = res["visualized_img"]
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    cv2.imwrite("output.jpg", img)
    print(res["results"])
    print(res["process_time"])
