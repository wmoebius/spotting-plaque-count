from typing import Dict, List, Tuple

import numpy as np

from phageid.dtypes import (
    D_ImageStack,
    D_PointStack,
    Image,
    Points,
)
from phageid.utils import convert_image_stacks, convert_point_stacks


def join_images(
    images: Dict[Tuple[int, int], np.ndarray],
    spacing: int = 10,
    background_color: int = 0,
) -> np.ndarray:
    """
    Joins a dictionary of images into a single image, arranging them based on their
    (row, col) positions and adding spacing between them.

    Args:
        images (dict[tuple[int, int], np.ndarray]): Dictionary where keys are (row, col) tuples
            representing the position of the image, and values are the image arrays.
        spacing (int): Number of pixels of spacing between images. Default is 10.
        background_color (int): Pixel value for the background. Default is 0 (black).

    Returns:
        np.ndarray: A single combined image as a NumPy array.
    """
    # Determine the layout size based on the maximum row and column indices
    rows = max(key[0] for key in images) + 1
    cols = max(key[1] for key in images) + 1

    # Find the maximum height and width for each row and column
    row_heights = [
        max(images.get((r, c), np.zeros((1, 1))).shape[0] for c in range(cols))
        for r in range(rows)
    ]
    col_widths = [
        max(images.get((r, c), np.zeros((1, 1))).shape[1] for r in range(rows))
        for c in range(cols)
    ]

    # Calculate the dimensions of the final combined image
    total_height = sum(row_heights) + spacing * (rows - 1)
    total_width = sum(col_widths) + spacing * (cols - 1)

    # Create the combined image with the specified background color
    combined_image = np.full(
        (total_height, total_width), background_color, dtype=np.float64
    )

    # Place each image into the combined image
    y_offset = 0
    for r in range(rows):
        x_offset = 0
        for c in range(cols):
            img = images.get((r, c))
            if img is not None:
                h, w = img.shape[:2]
                combined_image[y_offset : y_offset + h, x_offset : x_offset + w] = img
            x_offset += col_widths[c] + spacing
        y_offset += row_heights[r] + spacing

    return combined_image


def join_images_with_points(
    images: Dict[Tuple[int, int], Image],
    points: Dict[Tuple[int, int], Points],
    spacing: int = 10,
    background_color: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Joins a dictionary of images into a single image, arranging them based on their
    (row, col) positions and adding spacing between them.

    Args:
        images (dict[tuple[int, int], np.ndarray]): Dictionary where keys are (row, col) tuples
            representing the position of the image, and values are the image arrays.
        spacing (int): Number of pixels of spacing between images. Default is 10.
        background_color (int): Pixel value for the background. Default is 0 (black).

    Returns:
        np.ndarray: A single combined image as a NumPy array.
    """
    # Determine the layout size based on the maximum row and column indices
    rows = max(key[0] for key in images) + 1
    cols = max(key[1] for key in images) + 1

    points = points.copy()

    # Find the maximum height and width for each row and column
    row_heights = [
        max(images.get((r, c), np.zeros((1, 1))).shape[0] for c in range(cols))
        for r in range(rows)
    ]
    col_widths = [
        max(images.get((r, c), np.zeros((1, 1))).shape[1] for r in range(rows))
        for c in range(cols)
    ]

    # Calculate the dimensions of the final combined image
    total_height = sum(row_heights) + spacing * (rows - 1)
    total_width = sum(col_widths) + spacing * (cols - 1)

    # Create the combined image with the specified background color
    combined_image = np.full(
        (total_height, total_width), background_color, dtype=np.float64
    )
    combined_points = []

    # Place each image into the combined image
    y_offset = 0
    for r in range(rows):
        x_offset = 0
        for c in range(cols):
            img = images.get((r, c))
            pts = points.get((r, c))
            if img is not None:
                h, w = img.shape[:2]
                combined_image[y_offset : y_offset + h, x_offset : x_offset + w] = img
                adg = np.array((x_offset, y_offset)).reshape(1, -1)
                combined_points.append(pts + adg)

            x_offset += col_widths[c] + spacing
        y_offset += row_heights[r] + spacing

    return combined_image, np.vstack(combined_points)



def join_stacks_with_points(
    d_stacks: D_ImageStack,
    d_points: D_PointStack,
    spacing: int = 10,
    background_color: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:

    np.testing.assert_array_equal(
        np.asarray(d_stacks.keys()),
        np.asarray(d_points.keys()))

    l_dimages = convert_image_stacks(d_stacks)
    l_dpoints = convert_point_stacks(d_points)

    return [join_images_with_points(images, points, spacing, background_color)
        for images, points in zip(l_dimages, l_dpoints)]
