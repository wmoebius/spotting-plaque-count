import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from cv2 import imread

from phageid import logging
from phageid.dtypes import ImageStack, PointStack
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm import tqdm


def has_placeholder(s: str) -> bool:
    return bool(re.search(r"(?<!{){[^{}]*}(?!})", s))


def parse_argument_dirs(input_dir: str, output_dir: Optional[str]) -> Tuple[Path, Path]:
    input_path = Path(input_dir)
    output_path = (
        input_path.parent / "phageid_out" if output_dir is None else Path(output_dir)
    )

    # validate input path
    if not input_path.is_dir():
        err = f"input directory: {input_dir} does not exist"
        logging.error(err)
        raise ValueError(err)

    # valudate output path
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
        logging.info(f"created output directory: {output_path}")

    return input_path, output_path


def parse_detection_args(
    input_file: str, output_dir: Optional[str]
) -> Tuple[Path, Path]:
    input_path = Path(input_file)

    output_path = (
        input_path.parent / "phageid_out" if output_dir is None else Path(output_dir)
    )

    # validate input path
    if not input_path.is_file():
        err = f"input file: {input_path} does not exist."
        logging.error(err)
        raise ValueError(err)

    # valudate output path
    if not output_path.is_dir():
        output_path.mkdir(parents=True)
        logging.info(f"created output directory: {output_path}")

    return input_path, output_path


def read_images(image_dir: Path) -> ImageStack:
    ## load images
    image_paths = sorted(list(image_dir.iterdir()))
    rgb_images = [imread(str(path)) for path in image_paths if path.is_file()]

    # basic preprocessing
    images = [image.mean(axis=2) for image in rgb_images]
    images = [np.where(image < 1, 1, image).astype(int) for image in images]
    del rgb_images
    return images


def read_stack(stack_path: Path) -> ImageStack:
    try:
        np_stack: np.ndarray = np.load(stack_path)
        logging.info(
            "read image data from {} with dimensions: {}".format(
                stack_path, np_stack.shape
            )
        )
        stack: ImageStack = [image for image in np_stack]
    except Exception as e:
        err = "Failed to read image from {} due to: {}".format(stack_path, e)
        logging.error(err)
        raise FileNotFoundError(err)

    return stack


def write_stack(stack: ImageStack, write_path: Path):
    images = np.stack(stack, axis=0)
    try:
        np.save(write_path, images)
        logging.info("wrote image file to {}".format(str(write_path)) + ".npy")
    except Exception as e:
        logging.error(
            "Failed to write image file to {} due to error: \n {}".format(write_path, e)
        )


def write_stacks(
    stacks: List[ImageStack], write_dir: Path, file_str: str = "image_{:03d}"
):
    for i, stack in enumerate(stacks):
        if not has_placeholder(file_str):
            # ensure there is a placeholder for the file name to prevent file overwriting
            file_str += "_{:03d}"

        # write to file path
        file_path = write_dir / file_str.format(i)
        write_stack(stack, file_path)


def write_plot(image: NDArray[np.number], points: NDArray[np.number], file_path: Path):
    # Create the plot
    plt.imshow(image, cmap="viridis")  # Use cmap='gray' for grayscale images
    plt.scatter(*points.T, c="r", s=1)  # red dots
    plt.axis("off")  # Optional: hide axes

    plt.savefig(file_path, bbox_inches="tight", pad_inches=0, dpi=500)
    plt.close()


def write_plots(images: ImageStack, points: PointStack, dir_path: Path):
    logging.info("writing outputs to {}".format(dir_path))
    for i in tqdm(range(1, len(images))):
        image = images[i]
        point = np.vstack(points[:i])

        file_path = dir_path / "phageid_image_t_{:03d}".format(i)

        write_plot(image, point, file_path)
