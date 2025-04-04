from pathlib import Path
from typing import Optional, Tuple
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



def read_images(image_dir: Path):
