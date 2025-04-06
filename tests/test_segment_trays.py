from pathlib import Path

import pytest

from typing import List
from phageid.read_write import read_images
from phageid.segmentation.trays import segment_trays
from phageid.dtypes import ImageStack


@pytest.mark.parametrize(
    ("input_dir", "n_trays"),
    [
        (Path(__file__).parent / ".test_resources/example_images/example_01", 3),
    ],
)
def test_tray_segmentation(input_dir, n_trays):
    assert input_dir.is_dir()

    images = read_images(input_dir)
    returned: List[ImageStack] = segment_trays(images=images, visualise=False)

    assert len(returned) == n_trays

    # check dimensions
    for stack in returned:
        assert len(stack) == 2
        for image in stack:
            h, w = image.shape
            assert h == pytest.approx(2900, abs=100)
            assert w == pytest.approx(1980, abs=25)
