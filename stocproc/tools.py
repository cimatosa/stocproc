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


def complex_quad(func, a, b, **kw_args):
    func_re = lambda t: np.real(func(t))
    func_im = lambda t: np.imag(func(t))
    I_re = quad(func_re, a, b, **kw_args)[0]
    I_im = quad(func_im, a, b, **kw_args)[0]

    return I_re + 1j * I_im


def stochastic_process_fft(spectral_density, t_max, num_grid_points, num_samples, seed = None, verbose=1, omega_min=0):
    r"""Simulate Stochastic Process using FFT method
    
    This method works only for correlations functions of the form
    
    .. math:: \alpha(\tau) = \int_0^{\omega_\mathrm{max}} \mathrm{d}\omega \, \frac{J(\omega)}{\pi} e^{-\mathrm{i}\omega \tau}
    
    where :math:`J(\omega)` is a real non negative spectral density. 
    Then the intrgal can be approximated by the Riemann sum
    
    .. math:: \alpha(\tau) \approx \sum_{k=0}^{N-1} \Delta \omega \frac{J(\omega_k)}{\pi} e^{-\mathrm{i} k \Delta \omega \tau}
    
    For a process defined by
    
    .. math:: X(t) = \sum_{k=0}^{N-1} \sqrt{\frac{\Delta \omega J(\omega_k)}{\pi}} X_k \exp^{-\mathrm{i}\omega_k t}
    
    with compelx random variables :math:`X_k` such that :math:`\langle X_k \rangle = 0`, 
    :math:`\langle X_k X_{k'}\rangle = 0` and :math:`\langle X_k X^\ast_{k'}\rangle = \Delta \omega \delta_{k,k'}` it is easy to see
    that it fullfills the Riemann approximated correlation function.

    .. math:: 
        \begin{align}
            \langle X(t) X^\ast(s) \rangle = & \sum_{k,k'} \frac{\Delta \omega}{\pi} \sqrt{J(\omega_k)J(\omega_{k'})} \langle X_k X_{k'}\rangle \exp^{-\mathrm{i}\omega_k (t-s)} \\
                                           = & \sum_{k}    \frac{\Delta \omega}{\pi} J(\omega_k) \exp^{-\mathrm{i}\omega_k (t-s)} \\
                                           = & \alpha(t-s)
        \end{align}
    
    In order to use the sheme of the Discrete Fourier Transfrom (DFT) to calculate :math:`X(t)`
    :math:`t` has to be disrcetized as well. Some trivial rewriting leads
    
    .. math:: X(t_l) = \sum_{k=0}^{N-1} \sqrt{\frac{\Delta \omega J(\omega_k)}{\pi}} X_k e^{-\mathrm{i} 2 \pi \frac{k l}{N} \frac{\Delta \omega \Delta t}{ 2 \pi} N}
    
    For the DFT sheme to be applicable :math:`\Delta t` has to be chosen such that
    
    .. math:: 1 = \frac{\Delta \omega \Delta t}{2 \pi} N
    
    holds. Since :math:`J(\omega)` is real it follows that :math:`X(t_l) = X^\ast(t_{N-l})`.
    For that reason the stochastic process has only :math:`(N+1)/2` (odd :math:`N`) and
    :math:`(N/2 + 1)` (even :math:`N`) independent time grid points.
    
    Looking now from the other side, demanding that the process should run from 
    :math:`0` to :math:`t_\mathrm{max}` with :math:`n` equally distributed time grid points
    :math:`N = 2n-1` points for the DFT have to be considered. This also sets the time
    increment :math:`\Delta t = t_\mathrm{max} / (n-1)`.
    
    With that the frequency increment is determined by
    
    .. math:: \Delta \omega = \frac{2 \pi}{\Delta t N} 

    Implementing the above noted considerations it follows

    .. math:: X(l \Delta t) = DFT\left(\sqrt{\Delta \omega J(k \Delta \omega)} / \pi \times X_k\right) \qquad k = 0 \; ... \; N-1, \quad l = 0 \; ... \; n

    Note: since :math:`\omega_\mathrm{max} = N \Delta \omega = 2 \pi / \Delta t = 2 \pi (n-1) / t_\mathrm{max}`
    
    :param spectral_density: the spectral density :math:`J(\omega)` as callable function object
    :param t_max: :math:`[0,t_\mathrm{max}]` is the interval for which the process will be calculated
    :param num_grid_points: number :math:`n` of euqally distributed times :math:`t_k` on the intervall :math:`[0,t_\mathrm{max}]`
        for which the process will be evaluated
    :param num_samples: number of independent processes to be returned
    :param seed: seed passed to the random number generator used
    
    :return: returns the tuple (2D array of the set of stochastic processes, 
        1D array of time grid points). Each row of the stochastic process 
        array contains one sample of the stochastic process.
    """
    
    if verbose > 0:
        print("__ stochastic_process_fft __")
    
    n_dft = num_grid_points * 2 - 1
    delta_t = t_max / (num_grid_points-1)
    delta_omega = 2 * np.pi / (delta_t * n_dft)
    
    t = np.linspace(0, t_max, num_grid_points)
    omega_min_correction = np.exp(-1j * omega_min * t).reshape(1,-1)
      
    #omega axis
    omega = delta_omega*np.arange(n_dft)
    #reshape for multiplication with matrix xi
    sqrt_spectral_density = np.sqrt(spectral_density(omega + omega_min)).reshape((1, n_dft))
    if seed != None:
        np.random.seed(seed)
    if verbose > 0:
        print("  omega_max  : {:.2}".format(delta_omega * n_dft))
        print("  delta_omega: {:.2}".format(delta_omega))
        print("generate samples ...")
    #random complex normal samples
    xi = (np.random.normal(scale=1/np.sqrt(2), size = (2*num_samples*n_dft)).view(np.complex)).reshape(num_samples, n_dft)
    #each row contain a different integrand
    weighted_integrand = sqrt_spectral_density * np.sqrt(delta_omega / np.pi) * xi 
    #compute integral using fft routine
    z_ast = np.fft.fft(weighted_integrand, axis = 1)[:, 0:num_grid_points] * omega_min_correction
    #corresponding time axis
    
    if verbose > 0:
        print("done!")
    return z_ast, t
    
    
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
