import numpy as np
import numpy.typing as npt
from pathlib import Path
from scipy.signal import find_peaks

DIR_ROOT = Path(__file__).parent.parent.parent

FILE_CONFIG = DIR_ROOT / "config.toml"


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


def normalise(array: npt.NDArray) -> npt.NDArray[np.float64]:
    rnge = array.max() - array.min()
    return (array - array.min()) / rnge


def find_first_peak(image, bins=256, plot=False):
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
    hist_values, bin_edges = np.histogram(image, bins=np.arange(bins), range=(0, bins))

    # Find peaks
    peaks, _ = find_peaks(hist_values, prominence=10)

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
        return None
