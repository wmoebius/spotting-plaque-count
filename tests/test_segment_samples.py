from pathlib import Path
import pytest
from phageid.segmentation.samples import segment_samples
from shutil import rmtree


@pytest.mark.parametrize(
    "input_file",
    [
        (Path(__file__).parent / ".test_resources/example_trays/tray_location_0.npy"),
        (Path(__file__).parent / ".test_resources/example_trays/tray_location_1.npy"),
        (Path(__file__).parent / ".test_resources/example_trays/tray_location_2.npy"),
    ],
)
def test_sample_segmentation(input_file):
    assert input_file.is_file()
    output_dir = Path(__file__).parent / ".test_resources/example_trays/out"
    segment_samples(input_file=input_file, output_dir=output_dir, visualise=False)

    assert output_dir.is_dir()
    assert len(list(output_dir.iterdir())) > 0

    if output_dir.is_dir():
        rmtree(output_dir)
