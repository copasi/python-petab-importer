""" Utility package for importing PEtab files to COPASI

"""
from .convert_petab import *

try:
    from .PEtab import petab_gui
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.debug('PETabGui not available as PyQt5 is not installed')

from . import _version
__version__ = _version.get_versions()['version']
