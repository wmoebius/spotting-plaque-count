from abc import ABC
from typing import List

import numpy as np
from numpy.typing import NDArray

from .layers import Layer


class Process(ABC):
    """
    Process comprised of a combination of layers.
    """

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.number]:
        image = image.copy()
        for layer in self.layers:
            image = layer(image)
        return image
