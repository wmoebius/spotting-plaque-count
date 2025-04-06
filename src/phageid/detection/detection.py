


from phageid.dtypes import D_ImageStack, D_PointStack, ImageStack, PointStack

from .detectors import GaussianDetector


def detect_phage(stack: ImageStack) -> PointStack:
    detector = GaussianDetector()
    return detector(stack)

def detect_dstack(d_stack: D_ImageStack) -> D_PointStack:
    d_points = {}

    for (i, j), stack in d_stack.items():
        points = detect_phage(stack)
        d_points[i, j] = points

    return d_points
