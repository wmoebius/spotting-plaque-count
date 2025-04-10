import numpy as np

from phageid.dtypes import D_ImageStack, D_PointStack, ImageStack, PointStack
from phageid.processing.layers import AgglomeratePeaks

from .detectors import GaussianDetector, RingDetector

from tqdm import tqdm


def detect_phage(stack: ImageStack) -> PointStack:
    stack2 = stack.copy()

    gaus_detector = GaussianDetector()
    points_gaussian = gaus_detector(stack2)

    ring_detector = RingDetector()
    points_ring = ring_detector(stack)

    points_union = [np.vstack((pg, pr)) for pg, pr in zip(points_gaussian, points_ring)]

    points_union = AgglomeratePeaks(min_distance=15)(points_union)

    return points_union


def detect_dstack(d_stack: D_ImageStack) -> D_PointStack:
    d_points = {}

    for (i, j), stack in tqdm(d_stack.items()):
        points = detect_phage(stack)
        d_points[i, j] = points

    return d_points
