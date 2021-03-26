from typing import Dict, Generic, Tuple, TypeVar
from numpy import ndarray

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Obs(ndarray, Generic[Shape, DType]):
    """https://stackoverflow.com/a/64032593

    can't get it to work-- it doesn't like passing strings to Shape"""

    pass


Stepped = Tuple[
    Dict[str, ndarray],
    Dict[str, float],
    Dict[str, bool],
    Dict[str, str],
]
