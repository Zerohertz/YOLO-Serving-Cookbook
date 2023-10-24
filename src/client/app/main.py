import base64
import time
from typing import Tuple

import cv2
import numpy as np
import seaborn as sns
import tritonclient.grpc as grpcclient
import yaml
from tritonclient.utils import *


def load_image(IMAGE_PATH: str) -> np.ndarray:
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess(
    im: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
) -> (np.ndarray, Tuple, Tuple):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def inference(input_image: np.ndarray) -> np.ndarray:
    SERVER_URL = "server:8001"
    MODEL_NAME = "YOLO"
    cv2.imwrite("input_image.jpg", input_image)
    input_image = input_image.astype("float32")
    input_image = input_image.transpose((2, 0, 1))[np.newaxis, :] / 255.0
    input_image = np.ascontiguousarray(input_image)
    with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
        inputs = [
            grpcclient.InferInput(
                "images", input_image.shape, np_to_triton_dtype(np.float32)
            )
        ]
        inputs[0].set_data_from_numpy(input_image)
        outputs = [grpcclient.InferRequestedOutput("output0")]
        response = triton_client.infer(
            model_name=MODEL_NAME, inputs=inputs, outputs=outputs
        )
        response.get_response()
        output0 = response.as_numpy("output0")
    return output0


def postprocess(
    predictions: np.ndarray, r: Tuple, dwdh: Tuple, conf_thresh=0.25, iou_thresh=0.45
) -> np.ndarray:
    """
    Postprocess the YOLO predictions using numpy and cv2.

    Args:
    - predictions: the model output numpy array of shape (1, 25200, 85)
    - conf_thresh: confidence threshold
    - iou_thresh: IoU threshold for NMS

    Returns:
    - detections: list of detections with shape (x1, y1, x2, y2, obj_conf, class_conf, class_idx)
    """
    bboxes = predictions[..., :4]
    obj_conf = predictions[..., 4:5]
    class_scores = predictions[..., 5:]
    class_conf = np.max(class_scores, axis=-1, keepdims=True)
    class_pred = np.argmax(class_scores, axis=-1, keepdims=True)
    obj_mask = obj_conf > conf_thresh
    bboxes = bboxes[obj_mask[..., 0]]
    obj_conf = obj_conf[obj_mask[..., 0]]
    class_conf = class_conf[obj_mask[..., 0]]
    class_pred = class_pred[obj_mask[..., 0]]
    bboxes[..., :2] = bboxes[..., :2] - bboxes[..., 2:] / 2
    bboxes[..., 2:] = bboxes[..., :2] + bboxes[..., 2:]
    bboxes = np.clip(bboxes, 0, 640)
    bboxes -= dwdh * 2
    bboxes /= r * 2
    scores = (obj_conf * class_conf).squeeze()
    detections = np.concatenate([bboxes, obj_conf, class_conf, class_pred], axis=-1)
    indices = cv2.dnn.NMSBoxes(
        bboxes.tolist(), scores.tolist(), conf_thresh, iou_thresh
    )
    return detections[indices]


def visualize(img: np.ndarray, results: np.ndarray) -> np.ndarray:
    with open("/app/data/coco.yaml") as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)["names"]
    color = sns.color_palette("pastel", len(labels))
    color = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color]
    thickness = 2
    for result in results:
        pt1, pt2, obj_conf, class_conf, class_id = (
            result[0:2].astype(int),
            result[2:4].astype(int),
            result[4],
            result[5],
            int(result[6]),
        )
        cv2.rectangle(img, pt1, pt2, color[class_id], thickness)
        img = cv2.putText(
            img,
            f"[{labels[class_id]}] OBJ: {obj_conf:.2f} | CLASS: {class_conf:.2f}",
            pt1,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=color[class_id],
            thickness=2,
        )
    return img[:, :, ::-1]


def main(img):
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocess_start = time.time()
    input_image, r, dwdh = preprocess(img)
    preprocess_end = time.time()
    output = inference(input_image)
    inference_end = time.time()
    results = postprocess(output, r, dwdh)
    postprocess_end = time.time()
    img = visualize(img, results)
    visualize_end = time.time()

    process_time = {
        "preprocess": preprocess_end - preprocess_start,
        "inference": inference_end - preprocess_end,
        "postprocess": postprocess_end - inference_end,
        "visualize": visualize_end - postprocess_end,
        "total": visualize_end - preprocess_start,
    }

    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8"), results, process_time
