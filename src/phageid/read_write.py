import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from cv2 import imread

from phageid import logging
from phageid.dtypes import ImageStack


def has_placeholder(s: str) -> bool:
    return bool(re.search(r'(?<!{){[^{}]*}(?!})', s))

def parse_argument_dirs(input_dir: str, output_dir: Optional[str]) -> Tuple[Path, Path]:
    input_path = Path(input_dir)
    output_path = input_path.parent / "phageid_out" if output_dir is None else Path(output_dir)

    # validate input path
    if not input_path.is_dir():
        err = f"input directory: {input_dir} does not exist"
        logging.error(err)
        raise ValueError(err)

    # valudate output path
    if not output_path.is_dir():
        output_path.mkdir()
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


def write_stack(stack: ImageStack, write_path: Path)
    images = np.vstack(stack)
    try:
        np.save(write_path, images)
        logging.info("Wrote image file to {}".format(write_path))
    except Exception as e:
        logging.error("Failed to write image file to {} due to error: \n {}".format(write_path, e))

def write_stacks(stacks: List[ImageStack], write_dir: Path, file_str: str = "image_{:03d}"):
    for i, stack in enumerate(stacks):
        if not has_placeholder(file_str):
            # ensure there is a placeholder for the file name to prevent file overwriting
            file_str += "_{:03d}"

        # write to file path
        file_path = write_dir / file_str.format(i)
        write_stack(stack, file_path)
