import click
from phageid.segmentation import segment_trays


@click.group()
def cli():
    """A simple CLI tool for segmentation and detection."""
    pass


@click.command()
@click.argument("input_file", required=True, type=click.Path(exists=True))
@click.argument("output_dir", required=False, type=click.Path(), default=None)
def segment_plates(input_file, output_dir):
    """Segment plates in the input data."""

    output_dir = (
        output_dir if output_dir is not None else input_file.parent / "segmented_plates"
    )
    segment_trays(input_file, output_dir)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
def segment_samples():
    """Segment samples within the plates."""
    click.echo("Segmenting samples...")


@click.command()
@click.argument("sample_directory", type=click.Path(exists=True))
def detect():
    """Run detection on segmented samples."""
    click.echo("Running detection...")


# Add commands to CLI group
cli.add_command(segment_plates)
cli.add_command(segment_samples)
cli.add_command(detect)

if __name__ == "__main__":
    cli()
