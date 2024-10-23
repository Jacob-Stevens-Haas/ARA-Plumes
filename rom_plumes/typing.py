from typing import Any
from typing import List
from typing import Literal
from typing import NewType

import numpy as np
from numpy.typing import NBitBase

Frame = NewType("Frame", int)
Width = NewType("Width", int)
Height = NewType("Height", int)
Channel = NewType("Channel", int)

GrayImage = np.ndarray[tuple[Height, Width], np.dtype[np.uint8]]
ColorImage = np.ndarray[tuple[Height, Width, Channel], np.dtype[np.uint8]]
FloatImage = np.ndarray[tuple[Height, Width], np.dtype[np.floating[NBitBase]]]
GrayVideo = np.ndarray[tuple[Frame, Height, Width], np.dtype[np.uint8]]
ColorVideo = np.ndarray[tuple[Frame, Height, Width, Channel], np.dtype[np.uint8]]

NpFlt = np.dtype[np.floating[NBitBase]]

Y_pos = NewType("Y_pos", int)
X_pos = NewType("X_pos", int)
Contour_List = List[np.ndarray[tuple[int, Literal[2]], np.dtype[Any]]]
PlumePoints = np.ndarray[tuple[int, Literal[3]], NpFlt]

Bool1D = np.ndarray[tuple[int], np.dtype[np.bool_]]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Float3D = np.ndarray[tuple[int, int, int], NpFlt]
FloatND = np.ndarray[Any, NpFlt]

AX_FRAME = -3
AX_HEIGHT = -2
AX_WIDTH = -1
