import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from typing import List


def video_image_sequence(images: List[np.ndarray], fps: int = 30) -> HTML:
    """
    Displays a sequence of images as a video interactively in a Jupyter notebook.

    Args:
        images (List[np.ndarray]): A list of images, where each image is
            a NumPy array (H x W x C) representing an RGB image.
        fps (int): Frames per second for the video playback. Default is 30.

    Returns:
        HTML: An HTML object containing the animation to display in a Jupyter notebook.
    """
    # get image colour range
    vmin = np.nanmin(np.asarray(images))
    vmax = np.nanmax(np.asarray(images))

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.axis("off")  # Turn off axes
    img_display = ax.imshow(images[0], vmin=vmin, vmax=vmax)  # Display the first image

    def update(frame_index: int):
        """
        Updates the displayed image for each frame.

        Args:
            frame_index (int): Index of the current frame to display.

        Returns:
            list: A list containing the updated image display object.
        """
        img_display.set_data(images[frame_index])
        return [img_display]

    # Create the animation
    interval = 1000 / fps  # Interval in milliseconds
    anim = FuncAnimation(fig, update, frames=len(images), interval=interval, blit=True)

    # Close the figure to prevent static rendering
    plt.close(fig)

    # Display the animation in the notebook
    return HTML(anim.to_jshtml())


def video_image_sequence_with_scatter(
    images: List[np.ndarray],
    points: List[np.ndarray],
    fps: int = 30,
    persist: bool = False,
) -> HTML:
    """
    Displays a sequence of images as a video interactively in a Jupyter notebook,
    overlaying scatter points on each frame.

    Args:
        images (List[np.ndarray]): A list of images, where each image is
            a NumPy array (H x W x C) representing an RGB image.
        points (List[np.ndarray]): A list of NumPy arrays (N x 2), where each array represents
            the scatter points' x and y coordinates for a corresponding frame.
        fps (int): Frames per second for the video playback. Default is 30.

    Returns:
        HTML: An HTML object containing the animation to display in a Jupyter notebook.
    """
    # Ensure the number of frames matches between images and points
    assert len(images) == len(points), (
        "The number of images and points must be the same."
    )

    # get image colour range
    vmin = np.nanmin(np.asarray(images))
    vmax = np.nanmax(np.asarray(images))

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.axis("off")  # Turn off axes
    img_display = ax.imshow(images[0], vmin=vmin, vmax=vmax)  # Display the first image
    img_scatter = ax.scatter([], [], c="red", s=10)  # Initialize scatter

    def update(frame_index: int):
        """
        Updates the displayed image and scatter points for each frame.

        Args:
            frame_index (int): Index of the current frame to display.

        Returns:
            list: A list containing the updated image and scatter objects.
        """
        img_display.set_data(images[frame_index])
        if not persist:
            img_scatter.set_offsets(points[frame_index])
        else:
            img_scatter.set_offsets(np.vstack(points[: frame_index + 1]))
        return [img_display, img_scatter]

    # Create the animation
    interval = 1000 / fps  # Interval in milliseconds
    anim = FuncAnimation(fig, update, frames=len(images), interval=interval, blit=True)

    # Close the figure to prevent static rendering
    plt.close(fig)

    # Display the animation in the notebook
    return HTML(anim.to_jshtml())


def join_images(
    images: dict[tuple[int, int], np.ndarray],
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
    images: dict[tuple[int, int], np.ndarray],
    points: dict[tuple[int, int], np.ndarray],
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
