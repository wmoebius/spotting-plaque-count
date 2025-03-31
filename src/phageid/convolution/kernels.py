from abc import ABC
import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d
from typing import Any, Optional, Callable
from scipy.ndimage import gaussian_filter


class Kernel(ABC):
    def __init__(
        self, size: int, kernel_op: Optional[Callable] = None, norm: bool = True
    ) -> None:
        self.size = size
        self.kernel_op = kernel_op
        self.norm = norm
        self._kernel = self._create()
        if norm:
            self._kernel = self._kernel - self._kernel.mean()

        super().__init__()

    def _create(self) -> np.ndarray:
        """
        create

        Args:
            size (int): size of kernel: (size, size)

        Raises:
            NotImplementedError: _description_

        Returns:
            npt.NDArray[float]: kernel as np array.
        """
        raise NotImplementedError

    def create(self) -> np.ndarray:
        if self.kernel_op is None:
            return self._create()
        else:
            return self.kernel_op(self._create())

    def convolve(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return convolve2d(
            image, self._kernel, mode="same", boundary="fill", fillvalue=1
        )


class GaussianKernel(Kernel):
    def __init__(
        self,
        size: int,
        sigma: Optional[float] = None,
        kernel_op: Optional[Callable] = None,
    ) -> None:
        self.sigma = sigma
        super().__init__(size, kernel_op=kernel_op)

    def _create(self) -> np.ndarray:
        """
        Creates a 2D Gaussian kernel that sums to 0 and when convolved with itself gives a result of 1.

        Args:
            size (int): The size of the kernel (must be odd).
            sigma (float): The standard deviation of the Gaussian.

        Returns:
            np.ndarray: A 2D Gaussian kernel with the desired properties.
        """
        sigma = self.sigma if self.sigma is not None else self.size / 5

        # Create a grid of (x, y) coordinates
        ax = np.linspace(-(self.size // 2), self.size // 2, self.size)
        ay = np.linspace(-(self.size // 2), self.size // 2, self.size)
        xx, yy = np.meshgrid(ax, ay)

        # Calculate the 2D Gaussian function
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Subtract the mean to make the sum 0
        kernel -= np.mean(kernel)

        # Normalize the kernel so that convolution with itself produces 1
        kernel /= np.sum(np.abs(kernel))  # Ensures normalization of kernel sum

        kernel = kernel * np.sqrt(1 / (kernel**2).sum())

        return kernel


class RingKernel(Kernel):
    def __init__(
        self,
        size: int,
        thickness: Optional[int] = None,
        blur: int = 1,
        kernel_op: Optional[Callable] = None,
    ) -> None:
        self.blur = blur
        self.thickness = thickness if thickness is not None else size // 5
        super().__init__(size, kernel_op=kernel_op)

    def _create(self) -> np.ndarray:
        r_outer = self.size // 2
        r_inner = r_outer - self.thickness

        # Create coordinate grids
        x = np.arange(self.size)
        y = np.arange(self.size)
        x_grid, y_grid = np.meshgrid(x, y)

        # Compute center of the array
        center = (self.size - 1) / 2

        # Compute Euclidean distance from the center
        distances = np.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2)
        kernel = (distances < r_outer) & (distances > r_inner)
        kernel = np.pad(kernel, pad_width=self.blur * 5)
        return gaussian_filter((kernel / kernel.sum()) * 2 - 1, self.blur)
