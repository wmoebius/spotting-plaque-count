from abc import ABC
from .layers import Layer
from typing import List
from numpy.typing import NDArray
import numpy as np


class Process(ABC):
    """
    Process comprised of a combination of layers.
    """

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def _process(self, image: NDArray[np.number]) -> NDArray[np.number]:
        image = image.copy()
        for layer in self.layers:
            image = layer(image)
        return image
