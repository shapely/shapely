from .lib import GEOSException  # NOQA
from .lib import Geometry  # NOQA
from .lib import geos_version, geos_version_string  # NOQA
from .lib import geos_capi_version, geos_capi_version_string  # NOQA
from .decorators import UnsupportedGEOSOperation  # NOQA
from .geometry import *
from .creation import *
from .constructive import *
from .predicates import *
from .measurement import *
from .set_operations import *
from .linear import *
from .coordinates import *
from .strtree import *
from .io import *

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
