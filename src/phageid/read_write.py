from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from cv2 import imread
from numpy.typing import NDArray

from phageid import logging


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


def read_images(image_dir: Path) -> List[NDArray[np.number]]:
    ## load images
    image_paths = sorted(list(image_dir.iterdir()))
    rgb_images = [imread(str(path)) for path in image_paths if path.is_file()]

    # basic preprocessing
    images = [image.mean(axis=2) for image in rgb_images]
    images = [np.where(image < 1, 1, image).astype(int) for image in images]
    del rgb_images
    return images


def write_images(images: NDArray[np.number], write_dir: Path, file_str: str = "image_{:03d}"):
        if write_dir.is_dir():
            for i, image in enumerate(images):

                file_path = write_dir / file_str.format(i)
                try:
                    np.save(file_path, image)
                    logging.info("Wrote image file to {}".format(file_path))
                except Exception as e:
                    logging.error("Failed to write image file to {} due to error: \n {}".format(file_path, e))
