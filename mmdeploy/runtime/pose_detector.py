# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Optional

import numpy as np
from mmdeploy_python import PoseDetector as _PoseDetector


class PoseDetector:
    """Object detector.

    Args:
        model_path (str): Model path
        device_name (str): Device name (e.g. "cpu", "cuda", ...)
        device_id (int): Device id
    """

    def __init__(self,
                 model_path: str,
                 device_name: str = 'cpu',
                 device_id: int = 0):
        self._pose_detector = _PoseDetector(
            model_path=model_path,
            device_name=device_name,
            device_id=device_id)

    def __call__(self, image: np.ndarray, bboxes: Optional[np.ndarray] = None):
        """Apply pose detector on single image.

        Args:
            image (np.ndarray): 8-bit BGR image in HWC format
            bboxes (np.ndarray): bounding boxes (K, 4) in LTRB format,
        Returns:
            np.ndarray of shape (K, P, 3), where P is the number of key-points
        """
        if bboxes:
            return self._pose_detector(image, bboxes)
        else:
            return self._pose_detector(image)

    def batch(self,
              images: List[np.ndarray],
              bboxes: Optional[List[np.ndarray]] = None):
        """Apply pose detector on multiple images.

        Args:
            images (List[np.ndarray]): list of 8-bit BGR images in HWC format
            bboxes (List[np.ndarray]): list of bounding boxes for each image
        Returns:
            List[np.ndarray]
        """
        return self._pose_detector.batch(images, bboxes if bboxes else [])
