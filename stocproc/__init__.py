import sys

if sys.version_info.major < 3:
    raise SystemError("no support for Python 2")

from .stocproc import logging_setup
from .stocproc import StocProc_FFT
from .stocproc import StocProc_KLE
from .stocproc import StocProc_TanhSinh
