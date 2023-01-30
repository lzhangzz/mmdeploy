# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import numpy as np
from mmdeploy_python import Detector as _Detector


class Detector:
    """Object detector.

    Args:
        model_path (str): Model path
        device_name (str): Device name (e.g. 'cpu', 'cuda', ...)
        device_id (int): Device id
    """

    def __init__(self,
                 model_path: str,
                 device_name: str = 'cpu',
                 device_id: int = 0):
        self._detector = _Detector(
            model_path=model_path,
            device_name=device_name,
            device_id=device_id)

    def __call__(self, image: np.ndarray):
        """Apply detector on single image.

        Args:
            image (np.ndarray): 8-bit HWC format BGR image
        Returns:
            Tuple containing dets, labels and masks
        """
        return self._detector(image)

    def batch(self, images: List[np.ndarray]):
        """Apply detector on a batch of images.

        Args:
            images (List[np.ndarray]): list of 8-bit BGR image (HWC)
        Returns:
            List[Tuple] containing dets, labels, and masks
        """
        return self._detector.batch(images)
