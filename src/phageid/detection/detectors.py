from abc import ABC

from phageid.dtypes import ImageStack, PointStack
from phageid.processing import Process
from phageid.processing.kernels import GaussianKernel
from phageid.processing.layers import (
    AgglomeratePeaks,
    Convolution,
    GaussianBlur,
    PeakFinder,
    SubtractByFrame,
)
from phageid.utils import find_first_peak


class Detector(ABC):
    def __call__(self, stack: ImageStack) -> PointStack:
        ...


class GaussianDetector(Detector):
    process = Process(
        [
            SubtractByFrame(value=lambda x: find_first_peak(x, 256, False)),
            GaussianBlur(kernel_size=7, sigma=0.5),
            Convolution(kernel=GaussianKernel(size=13, sigma=2)),
            GaussianBlur(kernel_size=9, sigma=1),
            PeakFinder(min_distance=15, threshold_rel=0.0, threshold_abs=32),
            AgglomeratePeaks(min_distance=15),
        ]
    )

    def __call__(self, stack: ImageStack) -> PointStack:
        return self.process(stack)
