from abc import ABC
import numpy as np

from phageid.dtypes import ImageStack, PointStack
from phageid.processing import Process
from phageid.processing.kernels import GaussianKernel, RingKernel
from phageid.processing.layers import (
    AgglomeratePeaks,
    Convolution,
    PeakFinder,
    Threshold,
    Difference,
    SubtractByFrame,
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
    # define kernel
    # TODO: Tidy this up
    kernel = RingKernel(size=35, thickness=3, blur=1)
    kernel._kernel -= kernel._kernel.min()
    kernel._kernel /= kernel._kernel.sum()

    process = Process(
        [
            Difference(separation=2),
            SubtractByFrame(value=np.min),
            Threshold(
                thresh=lambda x: find_first_peak(x) + 5, above=True, allow_equal=False
            ),
            Convolution(kernel=kernel),
            PeakFinder(threshold_abs=0.65, footprint=np.ones((3, 3))),
            AgglomeratePeaks(min_distance=15),
        ]
    )
