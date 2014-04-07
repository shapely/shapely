"""Provides multi-point element-wise operations such as ``contains``."""

# NOTE: The implementation of these operations is written in Cython
# at _vectorized.pyx
import _vectorized

try:
    from _vectorized import (contains, touches)
except ImportError:
    raise ImportError('The vectorized submodule is not been installed. '
                      'Cython is required at build time for vectorized '
                      'functionality.')
