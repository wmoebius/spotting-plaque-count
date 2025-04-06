import numpy as np

from phageid.dtypes import ImageStack


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
            assert (o == 1.).sum() < (input[0]==1.).sum()


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
            assert (o == 1.).sum() > (input[0]==1.).sum()
