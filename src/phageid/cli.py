import click
from phageid.segmentation import segment_trays as _segment_trays
from phageid.segmentation import segment_samples as _segment_samples
from pathlib import Path


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
    print(type(input_dir))

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else input_dir.parent / "segmented_trays"
    )
    input_dir = Path(input_dir)
    _segment_trays(input_dir, output_dir, visualise)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument(
    "output_dir", required=False, type=click.Path(path_type=Path), default=None
)
@click.option(
    "--visualise",
    is_flag=True,
    default=False,
    help="Enable visualisation of segmentation process.",
)
def segment_samples(input_file, output_dir, visualise):
    """Segment samples from tray images."""
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else input_file.parent / "segmented_samples"
    )
    input_file = Path(input_file)
    _segment_samples(input_file, output_dir, visualise)


@click.command()
@click.argument("sample_directory", type=click.Path(exists=True))
def detect():
    """Run detection on segmented samples."""
    click.echo("Running detection...")


# Add commands to CLI group
cli.add_command(segment_trays)
cli.add_command(segment_samples)
cli.add_command(detect)

if __name__ == "__main__":
    cli()
