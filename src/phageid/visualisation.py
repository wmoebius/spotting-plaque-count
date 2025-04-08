from typing import List

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def video_image_sequence(images: List[np.ndarray], fps: int = 30):
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
):
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
