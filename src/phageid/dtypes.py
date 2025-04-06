from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

type Image =NDArray[np.floating | np.integer]
# type ListImageStack = List[Image]
# type NpImageStack = Annotated[NDArray[np.number], 3]
# type ImageStack = ListImageStack | NpImageStack
type ImageStack = List[Image]
type Points= NDArray[np.int32]
type PointStack= List[Points]
type D_ImageStack = Dict[Tuple[int, int], ImageStack]
type D_PointStack = Dict[Tuple[int, int], PointStack]
