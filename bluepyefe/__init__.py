"""Init script"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from bluepyefe.extractor import *  # NOQA
from bluepyefe.tools import *  # NOQA
