from typing import Tuple

import cv2
import numpy as np
import seaborn as sns
import triton_python_backend_utils as pb_utils
import yaml


class TritonPythonModel:
    def initialize(self, pbtxt_args):
        print("POST-PROCESSING")
        with open("/models/postprocess/1/coco.yaml", encoding="UTF-8") as f:
            self.labels = yaml.load(f, Loader=yaml.FullLoader)["names"]

    def execute(self, requests):
        responses = []
        for request in requests:
            INPUT_IMAGE = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            INPUT_IMAGE = INPUT_IMAGE.as_numpy()
            INFERENCE_FEATURES = pb_utils.get_input_tensor_by_name(
                request, "INFERENCE_FEATURES"
            )
            INFERENCE_FEATURES = INFERENCE_FEATURES.as_numpy()
            PRE_PROCESSING_RATIO = pb_utils.get_input_tensor_by_name(
                request, "PRE_PROCESSING_RATIO"
            )
            PRE_PROCESSING_RATIO = PRE_PROCESSING_RATIO.as_numpy().tolist()
            PRE_PROCESSING_DWDH = pb_utils.get_input_tensor_by_name(
                request, "PRE_PROCESSING_DWDH"
            )
            PRE_PROCESSING_DWDH = PRE_PROCESSING_DWDH.as_numpy().tolist()
            RESULTS = self.postprocess(
                INFERENCE_FEATURES, PRE_PROCESSING_RATIO, PRE_PROCESSING_DWDH
            )
            VIS = self.visualize(INPUT_IMAGE, RESULTS)
            output_0 = pb_utils.Tensor("RESULTS", np.array(RESULTS).astype(np.float32))
            output_1 = pb_utils.Tensor("VISUALIZE", np.array(VIS).astype(np.uint8))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_0, output_1]
            )
            responses.append(inference_response)
        return responses

    def postprocess(
        self,
        predictions: np.ndarray,
        r: Tuple,
        dwdh: Tuple,
        conf_thresh=0.25,
        iou_thresh=0.45,
    ) -> np.ndarray:
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

    def visualize(self, img: np.ndarray, results: np.ndarray) -> np.ndarray:
        color = sns.color_palette("pastel", len(self.labels))
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
                f"[{self.labels[class_id]}] OBJ: {obj_conf:.2f} | CLASS: {class_conf:.2f}",
                pt1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color[class_id],
                thickness=2,
            )
        return img[:, :, ::-1]

    def finalize(self):
        print("Cleaning up POST-PROCESSING Module...")
