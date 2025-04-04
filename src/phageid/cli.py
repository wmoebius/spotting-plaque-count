from pathlib import Path
from typing import Optional, Tuple

import click

from phageid import logging
from phageid.detection import detect_phage
from phageid.segmentation import segment_trays as _segment_trays


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




@click.group()
def cli():
    """A simple CLI tool for segmentation and detection."""
    pass


@click.command()
@click.argument("input_dir", required=True, type=click.Path(exists=True))
@click.argument("output_dir", required=False, type=click.Path(), default=None)
@click.option(
    "--visualise",
    is_flag=True,
    default=False,
    help="Enable visualisation of segmentation process.",
)
def segment_trays(input_dir, output_dir, visualise):
    """Segment trays from raw images."""
    input_path, output_path = parse_argument_dirs(input_dir, output_dir)
    _segment_trays(input_dir, output_dir, visualise)


@click.command()
@click.argument("input_dir", required=True, type=click.Path(exists=True))
@click.argument("output_dir", required=False, type=click.Path(), default=None)
@click.option(
    "--visualise",
    is_flag=True,
    default=False,
    help="Enable visualisation of segmentation process.",
)
def detect(input_dir, output_dir):
    # format directories
    input_path, output_path = parse_argument_dirs(input_dir, output_dir)
    detect_phage(input_path, output_path)


# Add commands to CLI group
cli.add_command(segment_trays)
cli.add_command(detect)

if __name__ == "__main__":
    cli()
