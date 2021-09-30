"""Provides multi-point element-wise operations such as ``contains``."""

# workaround DLL findings issues on Windows (https://github.com/Toblerity/Shapely/issues/1184)
try:
    from . import speedups
except Exception:
    pass

from ._vectorized import (contains, touches)
