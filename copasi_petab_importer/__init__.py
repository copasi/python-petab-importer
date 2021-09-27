""" Utility package for importing PEtab files to COPASI

"""
from .convert_petab import *
from .PEtab import petab_gui

from . import _version
__version__ = _version.get_versions()['version']
