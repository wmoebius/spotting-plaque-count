import click


@click.group()
def cli():
    """A simple CLI tool for segmentation and detection."""
    pass


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
def segment_plates(input):
    """Segment plates in the input data."""
    click.echo("Segmenting plates...")


@click.command()
@click.argument("plate_directory", type=click.Path(exists=True))
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
