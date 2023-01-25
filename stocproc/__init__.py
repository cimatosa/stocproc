"""
The `stocproc` module provides direct access to the following samplers.

- [`KarhunenLoeve`][stocproc.samplers.KarhunenLoeve], slow but needs auto-correlation function only
- [`FastFourier`][stocproc.samplers.FastFourier], super fast, needs power spectral density too, should be
  considered to be the **default** sampler
- [`TanhSinh`][stocproc.samplers.TanhSinh], similar to `FastFourier`, but slower, can handle a singularity
  of the power spectral density at zero
- [`Cholesky`][stocproc.samplers.Cholesky], ???

They all derive from the abstract base class [`StocProc`][stocproc.samplers.StocProc].

In addition, the [`logging_setup`][stocproc.samplers.logging_setup] method is exposed which
allows to set the logging levels of the individual module ([`samplers`][stocproc.samplers],
[`method_ft`][stocproc.method_ft], [`method_kle`][stocproc.method_kle]).

"""

from .samplers import logging_setup
from .samplers import StocProc  # for typing
from .samplers import FastFourier
from .samplers import KarhunenLoeve
from .samplers import TanhSinh
from .samplers import Cholesky
