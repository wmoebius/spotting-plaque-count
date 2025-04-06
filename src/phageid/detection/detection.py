
from typing import List

import numpy as np
from numpy.typing import NDArray

from .detectors import GaussianDetector
from phageid.dtypes import ImageStack, PointStack


def detect_phage(images: ImageStack) -> PointStack:
    detector = GaussianDetector()
    return detector.detect(np.vstack(images))

def detect_
