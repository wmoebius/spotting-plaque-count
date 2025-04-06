
from typing import List

import click

from phageid.dtypes import D_ImageStack, ImageStack
from phageid.read_write import (
    parse_argument_dirs,
    parse_detection_args,
    read_images,
    read_stack,
    write_stacks,
)
from phageid.segmentation import segment_samples
from phageid.segmentation import segment_trays as _segment_trays


@click.group()
def cli():
    """A simple CLI tool for segmentation and detection."""
    pass


@click.command()
@click.argument("input_dir", required=True, type=click.Path(exists=True))
@click.argument("output_dir", required=False, type=click.Path(), default=None)
@click.argument("output_filename",
    required=False, type=click.Path(),
    default="tray_{}")
@click.option(
    "--visualise",
    is_flag=True,
    default=False,
    help="Enable visualisation of segmentation process.",
)
def segment_trays(input_dir, output_dir, output_filename, visualise):
    """Segment trays from raw images."""
    input_path, output_path = parse_argument_dirs(input_dir, output_dir)
    images: ImageStack = read_images(input_path)
    trays:  List[ImageStack]= _segment_trays(images, visualise)
    write_stacks(trays, output_path, output_filename)


@click.command()
@click.argument("input_file", required=True, type=click.Path(exists=True))
@click.argument("output_dir", required=False, type=click.Path(), default=None)
@click.option(
    "--visualise",
    is_flag=True,
    default=False,
    help="Enable visualisation of segmentation process.",
)
def detect(input_file, output_dir, visualise):
    # format directories
    input_path, output_path = parse_detection_args(input_file, output_dir)
    images: ImageStack = read_stack(input_path)
    d_samples: D_ImageStack = segment_samples(images, visualise=visualise)
    # d_points: D_PointStack = NotImplemented # detect points
    # combined: ImageStack = NotImplemented   # Mash them back together
    # write to disk (add option to this)


# Add commands to CLI group
cli.add_command(segment_trays)
cli.add_command(detect)

if __name__ == "__main__":
    cli()
