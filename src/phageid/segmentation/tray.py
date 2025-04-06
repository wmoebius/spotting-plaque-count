from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from phageid import logging


def find_step_edges(step_array: np.ndarray) -> npt.NDArray[np.int32]:
    """
    Find the edges of a step function in a NumPy array.

    Parameters:
        step_array (ndarray): 1D NumPy array representing the step function.

    Returns:
        ndarray: Indices of the edges (steps) in the array.
    """
    diff = np.diff(step_array)
    edges = np.where(diff != 0)[0]
    return edges + 1  # Add 1 since diff is offset by one element


def split_into_quadrants(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a set of 2D coordinates into four quadrants based on their relation to the centroid.

    The centroid is calculated as the mean of all coordinates, and the quadrants are determined as:
    - Q1: Coordinates where both x and y are greater than the centroid.
    - Q2: Coordinates where both x and y are less than the centroid.
    - Q3: Coordinates where x > centroid_x and y < centroid_y.
    - Q4: Coordinates where x < centroid_x and y > centroid_y.# Copy selected text to clipboard (Ctrl+Shift+C)
    bindkey -M vicmd '^C' copy-to-clipboard

    # Paste text from clipboard (Ctrl+Shift+V)
    bindkey -M vicmd '^V' paste-from-clipboard

    # Define the copy-to-clipboard function
    copy-to-clipboard() {
      echo -n $BUFFER | xclip -selection clipboard
    }

    # Define the paste-from-clipboard function
    paste-from-clipboard() {
      BUFFER=$(xclip -selection clipboard -o)
      zle redisplay
    }

    :param coords:
        A 2D NumPy array of shape (n, 2), where each row represents a point (x, y).

    :returns:
        A tuple of four NumPy arrays, each containing the coordinates in one of the quadrants:
        (Q1, Q2, Q3, Q4).
    """
    centre = coords.mean(axis=0)
    print(centre)
    q1 = coords[((coords > centre) == (True, True)).all(axis=1)]
    q2 = coords[((coords > centre) == (False, False)).all(axis=1)]
    q3 = coords[((coords > centre) == (True, False)).all(axis=1)]
    q4 = coords[((coords > centre) == (False, True)).all(axis=1)]
    return q1, q2, q3, q4


def extract_subregion(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Extract a rectangular subregion from an image based on specified corner coordinates.

    This function identifies the bounding box defined by the given corner points and extracts
    the corresponding subregion from the input image.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D or 3D NumPy array representing the input image. For 3D arrays, the last dimension is
        typically the color channels (e.g., RGB).
    corners : numpy.ndarray
        A 2D NumPy array of shape (n, 2) where each row represents a (x, y) coordinate. These
        coordinates define the corners of the region to extract.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the extracted subregion of the image. The shape depends on the
        bounds and the input image dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.arange(100).reshape(10, 10)
    >>> corners = np.array([[2, 2], [5, 5]])
    >>> subregion = extract_subregion(image, corners)
    >>> print(subregion)
    [[22 23 24]
     [32 33 34]
     [42 43 44]]

    Notes
    -----
    - Ensure that the coordinates in `corners` are within the bounds of the input image.
    - For out-of-bound coordinates, unexpected behavior may occur unless explicitly handled.
    """
    # Get the minimum and maximum x and y coordinates
    min_x, min_y = np.min(corners, axis=0)
    max_x, max_y = np.max(corners, axis=0)

    if max_x > image.shape[1] or max_y > image.shape[0]:
        raise IndexError

    # Extract the subregion from the image
    subregion = image[min_y + 1 : max_y, min_x + 1 : max_x]

    return subregion


def extract_subregions(images: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return np.array(
        list(map(lambda image: extract_subregion(image, corners=corners), images))
    )


def extract_box(array, centre, size):
    """
    Extract a square box of size (m, m) from a 2D array centered on a given coordinate.

    Parameters:
    - array (np.ndarray): The input 2D array.
    - centre (tuple of int): The (row, col) coordinates of the center.
    - size (int): The size of the box (must be an odd or even integer).

    Returns:
    - np.ndarray: The extracted (m, m) box, padded if necessary.
    """
    half_size = size // 2
    col, row = centre

    # Define bounds of the box
    row_start = max(0, row - half_size)
    row_end = min(array.shape[0], row + half_size + 1)
    col_start = max(0, col - half_size)
    col_end = min(array.shape[1], col + half_size + 1)

    # Extract the box
    box = array[row_start:row_end, col_start:col_end]

    # Pad if the box is smaller than (m, m)
    if box.shape != (size, size):
        padded_box = np.zeros((size, size), dtype=array.dtype)
        r_start = max(0, half_size - row)
        r_end = r_start + box.shape[0]
        c_start = max(0, half_size - col)
        c_end = c_start + box.shape[1]
        padded_box[r_start:r_end, c_start:c_end] = box
        box = padded_box

    return box


def get_rect_corners(x_edges, y_edges):
    x_edges = np.sort(x_edges).reshape(-1, 2)
    y_edges = np.sort(y_edges).reshape(-1, 2)

    trays = []
    for xe in x_edges:
        for ye in y_edges:
            trays.append(np.array(list(product(xe, ye))))

    return trays


def validate_tray(tray):
    return tray.std() > 10


def segment_trays(images: List[npt.NDArray[np.number]], visualise: bool) -> List[npt.NDArray[np.number]]:

    # pad image
    pad_size = 10

    # get clearest view of phage sample spots
    image = np.vstack(images).min(axis=0).astype(int)
    image[image == 255] = 0
    image = np.pad(
        image,
        pad_size,
        mode="constant",
    )

    if visualise:
        plt.imshow(image)
        plt.show()

    # use pixel variance to fit step function to identify tray edges.
    xav = image.std(axis=0) > 10
    xav = gaussian_filter1d(xav, 10.0)
    x_coords = find_step_edges(xav)

    if visualise:
        plt.plot(image.mean(axis=0) / (image.mean(axis=0).max()))
        plt.plot(xav, c="C1")

        for xc in x_coords:
            plt.axvline(xc, c="r", linestyle=(0, (5, 10)), linewidth=1)

    yav = image.std(axis=1) > 10
    yav = gaussian_filter1d(yav, 10.0)
    y_coords = find_step_edges(yav)

    if visualise:
        plt.plot(image.mean(axis=1) / (image.mean(axis=1).max()), alpha=0.4)
        plt.plot(yav, c="C1")
        for yc in y_coords:
            plt.axvline(yc, c="r", linestyle=(0, (5, 10)), linewidth=1)

    # get corners from all intersections of step edges
    q_corners = get_rect_corners(x_edges=x_coords, y_edges=y_coords)

    if visualise:
        plt.imshow(np.float32(image), alpha=0.5, cmap="gray")
        for i, qc in enumerate(q_corners):
            plt.scatter(*qc.T, marker="x", s=50, c=f"C{i}")

    # extract tray subregions
    trays = [
        extract_subregions(images=np.asarray(images), corners=q_coord - pad_size)
        for q_coord in q_corners
    ]
    logging.info(f"detected {len(trays)} trays")

    for tray in trays:
        if visualise:
            plt.Figure()
            plt.imshow(tray[-1])
            plt.show()

    return trays
