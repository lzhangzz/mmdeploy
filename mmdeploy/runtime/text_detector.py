# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy_python import TextDetector as _TextDetector
from typing import List
import numpy as np
class TextDetector:
    """Text detector.

    Args:
        model_path (str): Model path
        device_name (str): Device name (e.g. 'cpu', 'cuda', ...)
        device_id (int): Device id
    """

    def __init__(self,
                 model_path: str,
                 device_name: str = 'cpu',
                 device_id: int = 0):
        self._text_detector = _TextDetector(
            model_path=model_path,
            device_name=device_name,
            device_id=device_id)

    def __call__(self, image: np.ndarray):
        """Apply text detector on single image.

        Args:
            image (np.ndarray): 8-bit HWC format BGR image
        Returns:
            np.ndarray (n_dets, 9) containing the bboxes (x[..., :8]) and the scores (x[..., -1])
        """
        return self._text_detector(image)

    def batch(self, images: List[np.ndarray]):
        """Apply text detector on a batch of images.

        Args:
            images (List[np.ndarray]): list of 8-bit BGR image (HWC)
        Returns:
            List[np.ndarray]
        """
        return self._text_detector.batch(images)
