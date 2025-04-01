from abc import ABC
import numpy as np
from numpy.typing import NDArray


class Detector(ABC):
    def __call__(self, image: NDArray[np.number]) -> NDArray[np.number]: ...
