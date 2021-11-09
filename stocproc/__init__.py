__MAJOR__ = 1
__MINOR__ = 0
__PATCH__ = 0


def version():
    """semantic version string with format 'MAJOR.MINOR' (https://semver.org/)"""
    return "{}.{}".format(__MAJOR__, __MINOR__)


def version_full():
    """semantic version string with format 'MAJOR.MINOR.PATCH' (https://semver.org/)"""
    return "{}.{}.{}".format(__MAJOR__, __MINOR__, __PATCH__)


import sys

if sys.version_info.major < 3:
    raise SystemError("no support for Python 2")

from .stocproc import logging_setup
from .stocproc import StocProc  # for typing
from .stocproc import StocProc_FFT
from .stocproc import StocProc_KLE
from .stocproc import StocProc_TanhSinh
