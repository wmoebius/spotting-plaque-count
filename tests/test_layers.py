import numpy as np

from phageid.dtypes import ImageStack
import numpy.testing as nptest


class TestErosionLayer:
    from phageid.processing.layers import Erosion

    def setup_class(cls):
        cls.erosion = cls.Erosion()

    def test_erosion_layer(self):
        input: ImageStack = [np.eye(10) for _ in range(5)]
        out = self.erosion(input)

        assert len(out) == 5

        for o in out:
            print(o.shape)
            assert o.shape == (10, 10)
            assert (o == 1.0).sum() < (input[0] == 1.0).sum()


class TestDilationLayer:
    from phageid.processing.layers import Dilation

    def setup_class(cls):
        cls.dilation = cls.Dilation()

    def test_dilation_layer(self):
        input: ImageStack = [np.eye(10) for _ in range(5)]
        out = self.dilation(input)

        assert len(out) == 5

        for o in out:
            print(o.shape)
            assert o.shape == (10, 10)
            assert (o == 1.0).sum() > (input[0] == 1.0).sum()


class TestSubtractByFrame:
    from phageid.processing.layers import SubtractByFrame

    def setup_class(cls):
        cls.operation = cls.SubtractByFrame
        cls.data = [np.ones((5, 5)) * (i + 1) for i in range(3)]

    def test_subtract_constant(self):
        sbf = self.operation(value=1)

        result = sbf(self.data)

        assert isinstance(result, list)

        for i, layer in enumerate(result):
            assert layer.shape == (5, 5)
            assert (layer == i).all()

    def test_subtract_callable(self):
        sbf = self.operation(value=np.mean)

        result = sbf(self.data)

        assert isinstance(result, list)

        for i, layer in enumerate(result):
            nptest.assert_array_almost_equal(layer, np.zeros_like(layer))
