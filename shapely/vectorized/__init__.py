"""Provides multi-point element-wise operations such as ``contains``."""

# NOTE: The implementation of these operations is written in Cython
# at _vectorized.pyx

try:
    from _vectorized import (contains, contains2d, )
except ImportError:
    raise ImportError('The vectorized submodule is not been installed. '
                      'Cython is required at build time for vectorized '
                      'functionality.')
