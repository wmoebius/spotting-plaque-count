from typing import Callable, Optional, Protocol, Tuple, TypeVar, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import peak_local_max

from phageid import logging
from phageid.dtypes import ImageStack, PointStack
from phageid.utils import image_to_cartesian

from .kernels import Kernel

T = TypeVar("T", bound="StackBase")

class Layer(Protocol[T]):
    def __call__(self, stack: T) -> T: ...


class ImageLayer(Layer):
    def __call__(self, stack: ImageStack) -> ImageStack: ...

class PointLayer(Layer):
    def __call__(self, stack: ImageStack | PointStack) -> ImageStack: ...


class Threshold(ImageLayer):
    def __init__(
        self, thresh: np.number, above: bool = True, allow_equal: bool = False
    ):
        self.thresh = thresh
        self.above = above
        self.allow_equal = allow_equal

    def __call__(self, stack: ImageStack) -> ImageStack:
        op = getattr(np, "greater") if self.above else getattr(np, "less")
        return op(stack, self.thresh).astype(np.int64)


class Normalise(ImageLayer):
    def __init__(self, range: Optional[Tuple[np.number, np.number]] = None):
        self.range = range

    def __call__(self, stack: ImageStack, **kwargs) -> ImageStack:
        stack_ = np.vstack(stack)
        stack_ = stack_ - stack_.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
        stack_ = stack_ / stack_.max(axis=(1, 2))[:, np.newaxis, np.newaxis]

        range = self.range if self.range is not None else (0.0, 1.0)

        scale = range[1] - range[0]

        stack_ = stack_ * scale + range[0]

        return stack_


class Erosion(ImageLayer):
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, stack: ImageStack) -> ImageStack:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        processed = [cv2.erode(
            s.astype(np.uint8), kernel, iterations=self.iterations
        ) for s in stack]
        return processed

class Dilation(Layer):
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, stack: ImageStack) -> ImageStack:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        processed = [cv2.dilate(
            s.astype(np.uint8), kernel, iterations=self.iterations
        ) for s in stack]
        return processed


class GaussianBlur(Layer):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, stack: ImageStack) -> ImageStack:
        for i, img in enumerate(stack):
            stack[i] = cv2.GaussianBlur(
                img.astype(np.float32), (self.kernel_size, self.kernel_size), self.sigma
            )
        return stack


class Difference(ImageLayer):
    def __init__(self, separation: int = 1):
        self.separation = separation

    def __call__(self, stack: ImageStack) -> ImageStack:
        stack_ = np.vstack(stack)
        num_stacks = stack_.shape[0]
        if self.separation < 1 or self.separation >= num_stacks:
            raise ValueError(f"Separation must be between 1 and {num_stacks - 1}.")

        differences = [
            stack_[i + self.separation] - stack_[i]
            for i in range(num_stacks - self.separation)
        ]
        return differences



class MaskReplace(ImageLayer):
    def __init__(
        self,
        mask: NDArray[np.int64],
        replace: Union[np.number, Callable],
        above: bool = True,
    ):
        self.mask = mask
        self.replace = replace

    def __call__(self, stack: ImageStack, **kwargs) -> ImageStack:
        stack_ = np.vstack(stack)

        if isinstance(self.replace, Callable):
            for i in range(stack_.shape[0]):
                stack_[i][self.mask] = self.replace(stack_[i])
        elif isinstance(self.replace, (float, int, np.nan)):
            for i in range(stack_.shape[0]):
                stack_[i][self.mask] = self.replace
        else:
            logging.error(
                "Bad argument to MaskReplace. replace should be number or callable, instead found:",
                type(self.replace),
            )
            raise ValueError(
                f"Bad argument to MaskReplace. replace should be number or callable, instead found: {type(self.replace)}"
            )

        return [im for im in stack_]


class ThreshReplace(ImageLayer):
    def __init__(
        self, thresh: float, replace: Union[np.number, str], above: bool = True
    ):
        self.thresh = thresh
        self.replace = replace
        self.above = above

    def __call__(self, stack: ImageStack) -> ImageStack:
        stack_ = np.stack(stack)
        if isinstance(self.replace, str):
            try:
                op = getattr(np, self.replace)
            except AttributeError:
                raise ValueError(
                    f"replace value parsed to ThreshReplace not valid. Received {self.replace}"
                )
            values = op(stack_, axis=(1, 2), keepdims=True)
        else:
            values = np.ones(stack_.shape[0]) * self.replace

        if self.above:
            mask = stack_ > self.thresh
        else:
            mask = stack_ < self.thresh

        # TODO: changed the below lines but did not test. Need to test and confirm.
        # TODO: This could easily be modified to use MaskReplace and reduce codebase
        # size and increase maintainability.
        # stack_[mask] = np.take_along_axis(
        stack_[:, mask] = np.take_along_axis(
            values, np.zeros_like(stack_, dtype=int), axis=0
        )[mask]

        return [im for im in stack_]


class SubtractByFrame(ImageLayer):
    def __init__(self, value: Callable | np.number):
        self.value = value

    def __call__(self, stack: ImageStack, **kwargs) -> ImageStack:
        stack_ = np.vstack(stack)

        if isinstance(self.value, Callable):
            for i in range(stack_.shape[0]):
                stack_[i] -= self.value(stack_[i])
        elif isinstance(self.value, (float, int, np.nan)):
            for i in range(stack_.shape[0]):
                stack_[i] -= self.value
        else:
            logging.error(
                f"Bad argument. value should be number or callable, instead found: {type(self.value)}",
            )
            raise ValueError(
                f"Bad argument. value should be number or callable, instead found: {type(self.value)}",
            )
        return [img for img in stack_]


class Subtract(Layer):
    def __init__(
        self, value: Optional[np.number] = None, operation: Optional[Callable] = None
    ):
        self.value = value
        self.operation = operation

    def __call__(self, stack: ImageStack, **kwargs) -> ImageStack:
        stack_ = np.vstack(stack)
        if self.value is not None and self.operation is None:
            stacks = stack_ - self.value
        elif self.value is None and self.operation is not None:
            stacks = stack_ - self.operation(stack_)
        else:
            msg = "Subtract layer should provide one of value or operation, not both and not neither."
            logging.error(msg)
            raise ValueError(msg)
        return [img for img in stack_]


class Convolution(ImageLayer):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, stack: ImageStack) -> ImageStack:
        return [self.kernel.convolve(img).astype(np.float64) for img in stack]


class PeakFinder(PointLayer):
    def __init__(self, min_distance: int, threshold_rel: float, threshold_abs: float):
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel
        self.threshold_abs = threshold_abs

    def __call__(self, stack: ImageStack) -> PointStack:
        peaks = [
            peak_local_max(
                im,
                min_distance=self.min_distance,
                threshold_rel=self.threshold_rel,
                threshold_abs=self.threshold_abs,
            )
            for im in stack
        ]
        return [image_to_cartesian(pk) for pk in peaks]
