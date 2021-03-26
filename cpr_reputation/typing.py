from typing import Dict, Generic, Tuple, TypeVar
from numpy import ndarray

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Obs(ndarray, Generic[Shape, DType]):
    pass

Stepped = Tuple[Dict[str, Obs["2,1", float]], Dict[str, float], Dict[str, bool], Dict[str, str]]
