
import click

from phageid.detection import detect_phage
from phageid.read_write import parse_argument_dirs, read_images
from phageid.segmentation import segment_trays as _segment_trays


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
    images = read_images(input_path)
    # segment samples
    detections = detect_phage(images)
    # mash them back together
    # write to disk (add option to this)


# Add commands to CLI group
cli.add_command(segment_trays)
cli.add_command(detect)

if __name__ == "__main__":
    cli()
