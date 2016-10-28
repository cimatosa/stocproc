#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stochastic Process Module
=========================

This module contains two different implementation for generating stochastic processes for a
given auto correlation function. Both methods are based on a time discrete process, however cubic
spline interpolation is assured to be valid within a given tolerance.

* simulate stochastic processes using Karhunen-Loève expansion :py:func:`stocproc.StocProc_KLE_tol`

  Setting up the class involves solving an eigenvalue problem which grows with
  the time interval the process is simulated on. Further generating a new process
  involves a multiplication with that matrix, therefore it scales quadratically with the
  time interval. Nonetheless it turns out that this method requires less random numbers
  than the Fast-Fourier method.

* simulate stochastic processes using Fast-Fourier method method :py:func:`stocproc.StocProc_FFT_tol`

  Setting up this class is quite efficient as it only calculates values of the
  associated spectral density. The number scales linear with the time interval of interest. However to achieve
  sufficient accuracy many of these values are required. As the generation of a new process is based on
  a Fast-Fouried-Transform over these values, this part is comparably lengthy.
"""

version = '0.2.0'

from .stocproc import StocProc_FFT_tol
from .stocproc import StocProc_KLE
from .stocproc import StocProc_KLE_tol