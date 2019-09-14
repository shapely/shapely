from .ufuncs import GEOSException  # NOQA
from .ufuncs import Geometry  # NOQA
from .ufuncs import geos_version  # NOQA
from .geometry import *
from .creation import *
from .constructive import *
from .predicates import *
from .measurement import *
from .set_operations import *
from .linear import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
