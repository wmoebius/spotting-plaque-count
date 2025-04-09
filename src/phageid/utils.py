from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from phageid.dtypes import D_ImageStack, D_PointStack, Image, Points


def image_to_cartesian(coordinates: np.ndarray) -> np.ndarray:
    """
    Converts image coordinates (row, column) to Cartesian coordinates (x, y).

    In image processing, coordinates are typically represented as (row, column),
    whereas in Cartesian form, they are represented as (x, y). This function swaps
    the two axes to follow Cartesian conventions.

    Args:
        coordinates (np.ndarray): A NumPy array of shape (N, 2), where each row
                                  represents an image coordinate (row, column).

    Returns:
        np.ndarray: A NumPy array of shape (N, 2) with converted Cartesian coordinates (x, y).
    """
    cartesian_coordinates = coordinates.copy()
    cartesian_coordinates[:, [0, 1]] = cartesian_coordinates[:, [1, 0]]
    return cartesian_coordinates


def normalise(array: NDArray) -> NDArray[np.float64]:
    rnge = array.max() - array.min()
    return (array - array.min()) / rnge


def find_first_peak(image, bins: Optional[int] = None, plot=False):
    """
    Finds the first peak in the histogram of intensity values in a NumPy array.

    Parameters:
    - image: np.ndarray, the input array of intensity values.
    - bins: int, number of bins for the histogram (default is 50).
    - plot: bool, whether to plot the histogram and highlight the peak (default is False).

    Returns:
    - first_peak_intensity: float or None, the intensity value of the first peak.
    """
    # Compute histogram
    image = image.astype(int)

    if bins is None:
        bins = int(image.max() - image.min())
    hist_values, bin_edges = np.histogram(image, bins=bins)

    # Find peaks
    peaks, _ = find_peaks(hist_values, prominence=5)

    if len(peaks) > 0:
        first_peak_bin = bin_edges[peaks[0]]  # First peak intensity

        if plot:
            # Plot histogram with detected peaks
            plt.plot(bin_edges[:-1], hist_values, label="Histogram")
            plt.scatter(
                bin_edges[peaks], hist_values[peaks], color="red", label="Peaks"
            )
            plt.axvline(
                first_peak_bin, color="green", linestyle="--", label="First Peak"
            )
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        return first_peak_bin
    else:
        return image.min()


def convert_image_stacks(d_stack: D_ImageStack) -> List[Dict[Tuple[int, int], Image]]:
    # TODO: remove the neccesity for this function
    if not d_stack:
        return []

    # Assume all lists in the dict are the same length
    stack_length = len(next(iter(d_stack.values())))

    # Transpose the data
    result: List[Dict[Tuple[int, int], Image]] = []
    for i in range(stack_length):
        frame: Dict[Tuple[int, int], Image] = {}
        for coord, image_stack in d_stack.items():
            frame[coord] = image_stack[i]
        result.append(frame)

    return result


def convert_point_stacks(d_stack: D_PointStack) -> List[Dict[Tuple[int, int], Points]]:
    # TODO: remove the neccesity for this function
    if not d_stack:
        return []

    # Assume all lists in the dict are the same length
    stack_length = len(next(iter(d_stack.values())))

    # Transpose the data
    result: List[Dict[Tuple[int, int], Points]] = []
    for i in range(stack_length):
        frame: Dict[Tuple[int, int], Points] = {}
        for coord, image_stack in d_stack.items():
            frame[coord] = image_stack[i]
        result.append(frame)

    return result
