import base64
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests

SERIAL = True

URL = "http://zerohertz.xyz:80/"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}


def send_request(HEADERS, DATA):
    time.sleep(random.randrange(0, 20))
    response = requests.post(URL, headers=HEADERS, json=DATA, verify=False)
    res = response.json()
    print(
        f"""TOTAL: {res["process_time"]["total"]:.2f} | PREPROCESS: {res["process_time"]["preprocess"]:.2f} | INFERENCE: {res["process_time"]["inference"]:.2f} | POSTPROCESS: {res["process_time"]["postprocess"]:.2f} | VISUALIZE: {res["process_time"]["visualize"]:.2f}"""
    )
    print(res["results"][0])
    return response


if __name__ == "__main__":
    file_id = "test.jpg"
    img = cv2.imread(file_id)
    _, buffer = cv2.imencode(".jpg", img)
    DATA = {
        "img": base64.b64encode(buffer).decode("utf-8"),
    }
    max_workers = 100

    START = time.time()
    if SERIAL:
        for _ in range(max_workers):
            send_request(HEADERS, DATA)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses = list(
                executor.map(
                    send_request, [HEADERS] * max_workers, [DATA] * max_workers
                )
            )
    END = time.time()
    print("=" * 10, END - START, "=" * 10)
