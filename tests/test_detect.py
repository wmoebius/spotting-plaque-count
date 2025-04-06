from phageid.cli import detect
from phageid.utils import DIR_ROOT


def test_detect():
    input = DIR_ROOT / "test_out/tray_0.npy"

    assert input.is_file()

    detect.main([str(input), str(input.parent), "--visualise"])

    assert True
