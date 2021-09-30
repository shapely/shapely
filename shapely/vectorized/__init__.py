"""Provides multi-point element-wise operations such as ``contains``."""

# workaround DLL findings issues on Windows (https://github.com/Toblerity/Shapely/issues/1184)
from . import speedups

from ._vectorized import (contains, touches)
