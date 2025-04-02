import pytest
from phageid.convolution.kernels import GaussianKernel, RingKernel


# Test that the Gaussian and Ring kernels produce kernels of the correct size
@pytest.mark.parametrize("KernelType", [GaussianKernel, RingKernel])
@pytest.mark.parametrize("size", [3, 5, 7])  # Test different kernel sizes
def test_kernel_creation(KernelType, size):
    kernel = KernelType(size)
    assert kernel._kernel.mean() == pytest.approx(0.0, abs=1e-5)
