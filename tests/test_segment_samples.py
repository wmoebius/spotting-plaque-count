from pathlib import Path

import pytest

from phageid.read_write import read_stack
from phageid.segmentation.samples import segment_samples


@pytest.mark.parametrize(
    "input_file",
    [
        (Path(__file__).parent / ".test_resources/example_trays/tray_location_0.npy"),
        (Path(__file__).parent / ".test_resources/example_trays/tray_location_1.npy"),
        (Path(__file__).parent / ".test_resources/example_trays/tray_location_2.npy"),
    ],
)
def test_sample_segmentation(input_file):
    # TODO: add config as an optional argument to segment_samples so that the
    # output can be tested more thoroughly without depending on config.toml
    # settings.
    assert input_file.is_file()
    images = read_stack(input_file)
    returned = segment_samples(images, visualise=False)

    assert returned is not None
    assert isinstance(returned, dict)
