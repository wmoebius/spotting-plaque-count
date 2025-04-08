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


class TestThreshReplace:
    from phageid.processing.layers import ThreshReplace

    def setup_class(cls):
        cls.operation = cls.ThreshReplace
        cls.data = [np.pad(np.ones((3, 3)), 1) for _ in range(5)]

    def test_threshold_constant(self):
        # replace all values below 0.5 with 1.0
        thresh = 0.5
        value = 1.0
        op = self.operation(thresh=thresh, value=value, above=False)

        result = op(self.data)

        nptest.assert_array_equal(np.stack(result), np.ones((5, 5, 5)))

    def test_threshold_callable(self):
        # replace all values below 0.5 with 1.0
        thresh = 0.5
        value = np.max
        op = self.operation(thresh=thresh, value=value, above=False)

        result = op(self.data)
        nptest.assert_array_equal(np.stack(result), np.ones((5, 5, 5)))

        value = np.max
        op = self.operation(thresh=thresh, value=value, above=True)

        result = op(self.data)
        nptest.assert_array_equal(np.stack(result), np.stack(self.data))
