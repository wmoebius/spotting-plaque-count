
import numpy as np
from numpy.typing import NDArray

from .detectors import GaussianDetector


def detect_phage(images: NDArray[np.number]) -> List[NDArray[np.number]]:
    detector = GaussianDetector()
    return detector.detect(images)
