from pygeos import ufuncs, GEOM_DTYPE
import numpy as np

print(ufuncs.set_zero(np.empty((10,), dtype=GEOM_DTYPE)))
