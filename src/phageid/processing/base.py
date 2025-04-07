from abc import ABC
from typing import List

from phageid.dtypes import ImageStack, PointStack

from .layers import Layer, PointLayer


class Process(ABC):
    """
    Process comprised of a combination of layers.
    """

    def __init__(self, layers: List[Layer]):
        self.layers: List[Layer, ..., PointLayer] = layers

    def __call__(self, stack: ImageStack) -> PointStack:
        for layer in self.layers:
            stack = layer(stack)
        return stack
