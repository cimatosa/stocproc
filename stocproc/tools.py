# -*- coding: utf8 -*-

from __future__ import print_function, division

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad

from .stocproc_c import auto_correlation as auto_correlation_c

import sys
import os
from warnings import warn
sys.path.append(os.path.dirname(__file__))
import numpy as np
from scipy.linalg import eigh as scipy_eigh
from collections import namedtuple

stocproc_key_type = namedtuple(typename    = 'stocproc_key_type', 
                               field_names = ['bcf', 't_max', 'ng', 'tol', 'cubatur_type', 'sig_min', 'ng_fac'] )


class ComplexInterpolatedUnivariateSpline(object):
    def __init__(self, x, y, k=2):
        self.re_spline = InterpolatedUnivariateSpline(x, np.real(y))
        self.im_spline = InterpolatedUnivariateSpline(x, np.imag(y))

    def __call__(self, t):
        return self.re_spline(t) + 1j * self.im_spline(t)

def auto_correlation_numpy(x, verbose=1):
    warn("use 'auto_correlation' instead", DeprecationWarning)
    
    # handle type error
    if x.ndim != 2:
        raise TypeError('expected 2D numpy array, but {} given'.format(type(x)))
    
    num_samples, num_time_points = x.shape
    
    x_prime = x.reshape(num_samples, 1, num_time_points)
    x       = x.reshape(num_samples, num_time_points, 1)
    
    if verbose > 0:
        print("calculate auto correlation function ...")
    res = np.mean(x * np.conj(x_prime), axis = 0), np.mean(x * x_prime, axis = 0)
    if verbose > 0:
        print("done!")
        
    return res

def auto_correlation(x, verbose=1):
    r"""Computes the auto correlation function for a set of wide-sense stationary stochastic processes
    
    Computes the auto correlation function for the given set :math:`{X_i(t)}` of stochastic processes:
    
    .. math:: \alpha(s, t) = \langle X(t)X^\ast(s) \rangle
    
    For wide-sense stationary processes :math:`\alpha` is independent of :math:`s`.
    
    :param x: 2D array of the shape (num_samples, num_time_points) containing the set of stochastic processes where each row represents one process
    
    :return: 2D array containing the correlation function as function of :math:`s, t` 
    """
    
    # handle type error
    if x.ndim != 2:
        raise TypeError('expected 2D numpy array, but {} given'.format(type(x)))
        
    if verbose > 0:
        print("calculate auto correlation function ...")
    res = auto_correlation_c(x)
    if verbose > 0:
        print("done!")
        
    return res   

def auto_correlation_zero(x, s_0_idx = 0):
    # handle type error
    if x.ndim != 2:
        raise TypeError('expected 2D numpy array, but {} given'.format(type(x)))
    
    num_samples = x.shape[0]
    x_s_0 = x[:,s_0_idx].reshape(num_samples,1)
    return np.mean(x * np.conj(x_s_0), axis = 0), np.mean(x * x_s_0, axis = 0)
