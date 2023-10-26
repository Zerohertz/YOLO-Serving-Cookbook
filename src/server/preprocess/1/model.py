from typing import Tuple

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, pbtxt_args):
        print("PRE-PROCESSING")

    def execute(self, requests):
        responses = []
        for request in requests:
            INPUT_IMAGE = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            INPUT_IMAGE, PRE_PROCESSING_RATIO, PRE_PROCESSING_DWDH = self.preprocess(
                INPUT_IMAGE.as_numpy()
            )
            INPUT_IMAGE = INPUT_IMAGE.astype("float32")
            INPUT_IMAGE = INPUT_IMAGE.transpose((2, 0, 1))[np.newaxis, :] / 255.0
            INPUT_IMAGE = np.ascontiguousarray(INPUT_IMAGE)
            output_0 = pb_utils.Tensor(
                "PRE_PROCESSING_IMAGE", np.array(INPUT_IMAGE).astype(np.float32)
            )
            output_1 = pb_utils.Tensor(
                "PRE_PROCESSING_RATIO",
                np.array(PRE_PROCESSING_RATIO).astype(np.float32),
            )
            output_2 = pb_utils.Tensor(
                "PRE_PROCESSING_DWDH", np.array(PRE_PROCESSING_DWDH).astype(np.float32)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_0, output_1, output_2]
            )
            responses.append(inference_response)
        return responses

    def preprocess(
        self,
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

    def finalize(self):
        print("Cleaning up PRE-PROCESSING Module...")
