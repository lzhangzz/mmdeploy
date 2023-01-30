# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Tuple

import numpy as np
from mmdeploy_python import Classifier as _Classifier


class Classifier:
    """Image classifier.

    Args:
        model_path (str): Model path
        device_name (str): Device name (e.g. "cpu", "cuda", etc...)
        device_id (int): Device id
    """

    def __init__(self,
                 model_path: str,
                 device_name: str = 'cpu',
                 device_id: int = 0):
        self._classifier = _Classifier(
            model_path=model_path,
            device_name=device_name,
            device_id=device_id)

    def __call__(self, image: np.ndarray) -> List[Tuple[int, float]]:
        """Apply classifier on single image.

        Args:
            image (numpy.array): 8-bit HWC format BGR image
        Returns:
            List[Tuple[int, float]]
        """
        return self._classifier(image)

    def batch(self, images: List[np.ndarray]) -> List[List[Tuple[int, float]]]:
        """Apply classifier on a batch of images.

        Args:
            images (List[np.ndarray]): list of 8-bit HWC format BGR images
        Returns:
            List[List[Tuple[int, float]]]
        """
        return self._classifier.batch(images)
