"""titiler.application"""

import pyproj.datadir # noqa
from os import environ
environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

from .version import __version__  # noqa
