from abc import ABC
import numpy as np

from phageid.dtypes import ImageStack, PointStack
from phageid.processing import Process
from phageid.processing.kernels import RingKernel, GaussianKernel
from phageid.processing.layers import (
    AgglomeratePeaks,
    Convolution,
    PeakFinder,
    SubtractByFrame,
    Difference,
    Threshold,
)
from phageid.utils import find_first_peak


class Detector(ABC):
    def __call__(self, stack: ImageStack) -> PointStack:
        return self.process(stack)


class GaussianDetector(Detector):
    process = Process(
        [
            Threshold(
                thresh=lambda x: find_first_peak(x) * 1.1, above=True, allow_equal=False
            ),
            Convolution(kernel=GaussianKernel(size=21, sigma=3)),
            PeakFinder(threshold_abs=4.9, footprint=np.ones((3, 3))),
            AgglomeratePeaks(min_distance=15),
        ]
    )


class RingDetector(Detector):
    process = Process(
        [
            SubtractByFrame(value=lambda x: find_first_peak(x, 256, False)),
            Difference(separation=2),
            Convolution(kernel=RingKernel(size=25, thickness=4, blur=1)),
            PeakFinder(min_distance=10, threshold_rel=0.0, threshold_abs=3.3),
            AgglomeratePeaks(min_distance=15),
        ]
    )
