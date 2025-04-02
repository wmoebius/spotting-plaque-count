import pytest
from pathlib import Path
from phageid.segmentation.tray import segment_trays
import numpy as np
import shutil


@pytest.mark.parametrize(
    ("input_dir", "n_trays"),
    [
        (Path(__file__).parent / ".test_resources/example_images/example_01", 3),
    ],
)
def test_tray_segmentation(input_dir, n_trays):
    output_dir = input_dir / "segmented_trays"
    assert input_dir.is_dir()

    segment_trays(input_dir=input_dir, output_dir=output_dir, visualise=False)

    assert output_dir.is_dir()

    images = [np.load(path) for path in output_dir.iterdir()]
    assert len(images) == n_trays

    # check dimensions
    for image in images:
        n, h, w = image.shape
        assert n == len([file for file in input_dir.iterdir() if file.is_file()])
        assert h == pytest.approx(2900, abs=100)
        assert w == pytest.approx(1980, abs=25)

    shutil.rmtree(output_dir)
