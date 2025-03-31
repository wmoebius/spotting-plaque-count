import click


@click.group()
def cli():
    """A simple CLI tool for segmentation and detection."""
    pass


@click.command()
def segment_plates():
    """Segment plates in the input data."""
    click.echo("Segmenting plates...")


@click.command()
def segment_samples():
    """Segment samples within the plates."""
    click.echo("Segmenting samples...")


@click.command()
def detect():
    """Run detection on segmented samples."""
    click.echo("Running detection...")


# Add commands to CLI group
cli.add_command(segment_plates)
cli.add_command(segment_samples)
cli.add_command(detect)

if __name__ == "__main__":
    cli()
