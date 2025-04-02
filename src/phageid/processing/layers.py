from typing import Protocol
from numpy.typing import NDArray
import numpy as np
from typing import Union, Optional, Callable
import cv2
from phageid import logging
from typing import Tuple
from .kernels import Kernel
from skimage.feature import peak_local_max
from phageid.utils import image_to_cartesian


class Layer(Protocol):
    """
    Single detection layer base class.
    """

    def __call__(self, image: NDArray[np.number], **kwargs) -> NDArray[np.number]: ...


class Threshold(Layer):
    def __init__(
        self, thresh: np.number, above: bool = True, allow_equal: bool = False
    ):
        self.thresh = thresh
        self.above = above
        self.allow_equal = allow_equal

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        op_str = f"{'greater' if self.above else 'less'}{'_equal' if self.allow_equal else ''}"
        op = getattr(np, op_str)
        return op(image, self.thresh).astype(np.int64)


class Normalise(Layer):
    def __init__(self, range: Optional[Tuple[np.number, np.number]] = None):
        self.range = range

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        image = image - image.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
        image = image / image.max(axis=(1, 2))[:, np.newaxis, np.newaxis]

        range = self.range if self.range is not None else (0.0, 1.0)

        scale = range[1] - range[0]

        image = image * scale + range[0]

        return image


class Erosion(Layer):
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        processed = cv2.erode(
            image.astype(np.uint8), kernel, iterations=self.iterations
        )
        return processed.astype(np.float64)


class Dilation(Layer):
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        processed = cv2.dilate(
            image.astype(np.uint8), kernel, iterations=self.iterations
        )
        return processed.astype(np.float64)


class GaussianBlur(Layer):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        for i, img in enumerate(image):
            image[i] = cv2.GaussianBlur(
                img.astype(np.float32), (self.kernel_size, self.kernel_size), self.sigma
            )
        return image


class Difference(Layer):
    def __init__(self, separation: int = 1):
        self.separation = separation

    def __call__(self, images: NDArray[np.number]) -> NDArray[np.float64]:
        num_images = images.shape[0]
        if self.separation < 1 or self.separation >= num_images:
            raise ValueError(f"Separation must be between 1 and {num_images - 1}.")

        differences = [
            images[i + self.separation] - images[i]
            for i in range(num_images - self.separation)
        ]
        return np.stack(differences, axis=0).astype(np.float64)


class ThreshReplace(Layer):
    def __init__(
        self, thresh: float, replace: Union[np.number, str], above: bool = True
    ):
        self.thresh = thresh
        self.replace = replace
        self.above = above

    def __call__(self, images: NDArray[np.number]) -> NDArray[np.float64]:
        if isinstance(self.replace, str):
            try:
                op = getattr(np, self.replace)
            except AttributeError:
                raise ValueError(
                    f"replace value parsed to ThreshReplace not valid. Received {self.replace}"
                )
            values = op(images, axis=(1, 2), keepdims=True)
        else:
            values = np.ones(images.shape[0]) * self.replace

        if self.above:
            mask = images > self.thresh
        else:
            mask = images < self.thresh

        images[mask] = np.take_along_axis(
            values, np.zeros_like(images, dtype=int), axis=0
        )[mask]

        return images


class MaskReplace(Layer):
    def __init__(
        self,
        mask: NDArray[np.int64],
        replace: Union[np.number, Callable],
        above: bool = True,
    ):
        self.mask = mask
        self.replace = replace

    def __call__(self, images: NDArray[np.number]) -> NDArray[np.float64]:
        if isinstance(self.replace, Callable):
            for i in range(images.shape[0]):
                images[i][self.mask] = self.replace(images[i])
        elif isinstance(self.replace, (float, int, np.nan)):
            for i in range(images.shape[0]):
                images[i][self.mask] = self.replace
        else:
            logging.error(
                "Bad argument to MaskReplace. replace should be number or callable, instead found:",
                type(self.replace),
            )
            raise ValueError(
                f"Bad argument to MaskReplace. replace should be number or callable, instead found: {type(self.replace)}"
            )

        return images


class SubtractByFrame(Layer):
    def __init__(self, value: Callable | np.number):
        self.value = value

    def __call__(self, images: NDArray[np.number]) -> NDArray[np.float64]:
        if isinstance(self.value, Callable):
            for i in range(images.shape[0]):
                images[i] -= self.value(images[i])
        elif isinstance(self.value, (float, int, np.nan)):
            for i in range(images.shape[0]):
                images[i] -= self.value
        else:
            logging.error(
                f"Bad argument. value should be number or callable, instead found: {type(self.value)}",
            )
            raise ValueError(
                f"Bad argument. value should be number or callable, instead found: {type(self.value)}",
            )
        return images


class Subtract(Layer):
    def __init__(
        self, value: Optional[np.number] = None, operation: Optional[Callable] = None
    ):
        self.value = value
        self.operation = operation

    def __call__(self, images: NDArray[np.number]) -> NDArray[np.float64]:
        if self.value is not None and self.operation is None:
            images = images - self.value
        elif self.value is None and self.operation is not None:
            images = images - self.operation(images)
        else:
            msg = "Subtract layer should provide one of value or operation, not both and not neither."
            logging.error(msg)
            raise ValueError(msg)
        return images


class Convolution(Layer):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.number]:
        return np.stack(
            [self.kernel.convolve(img).astype(np.float64) for img in image], axis=0
        )


class PeakFinder(Layer):
    def __init__(self, min_distance: int, threshold_rel: float, threshold_abs: float):
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel
        self.threshold_abs = threshold_abs

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.number]:
        peaks = [
            peak_local_max(
                im,
                min_distance=self.min_distance,
                threshold_rel=self.threshold_rel,
                threshold_abs=self.threshold_abs,
            )
            for im in image
        ]
        return [image_to_cartesian(pk) for pk in peaks]
