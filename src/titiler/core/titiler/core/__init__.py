"""titiler.core"""

import pyproj.datadir # noqa
from os import environ
environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

from . import dependencies, errors, factory, routing, utils, version  # noqa

from .factory import (  # noqa
    BaseTilerFactory,
    MultiBandTilerFactory,
    MultiBaseTilerFactory,
    TilerFactory,
)
from .version import __version__  # noqa
