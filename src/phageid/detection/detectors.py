from abc import ABC
import numpy as np
from numpy.typing import NDArray
from phageid.processing import Process
from phageid.processing.layers import (
    SubtractByFrame,
    Convolution,
    GaussianBlur,
    PeakFinder,
)
from phageid.processing.kernels import GaussianKernel
from phageid.utils import find_first_peak


class Detector(ABC):
    def detect() -> NDArray[np.int64]: ...


class GaussianDetector(Detector):
    process = Process(
        [
            SubtractByFrame(value=lambda x: find_first_peak(x, 256, False)),
            GaussianBlur(kernel_size=7, sigma=0.5),
            Convolution(kernel=GaussianKernel(size=13, sigma=2)),
            GaussianBlur(kernel_size=9, sigma=1),
            PeakFinder(min_distance=15, threshold_rel=0.0, threshold_abs=32),
        ]
    )

    def detect(self, data: NDArray[np.number], stack: bool = True) -> NDArray[np.int64]:
        points = self.process

        if not stack:
            return points
        else:
            return np.stack(points)
