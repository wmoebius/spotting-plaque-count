from phageid.cli import detect
from pathlib import Path


def test_detect():
    input = Path("/home/finley/Temp/build/trays/tray_1.npy")
    assert input.is_file()

    detect.main([str(input), str(input.parent), "--visualise"])

    assert True
