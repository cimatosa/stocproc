#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2014 Richard Hartmann
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
"""
**Stochastic Process Module**
 
This module contains various methods to generate stochastic processes for a 
given correlation function. There are two different kinds of generators. The one kind
allows to generate the process for a given time grid, where as the other one generates a
time continuous process in such a way that it allows to "correctly" interpolate between the
solutions of the time discrete version.
 
    **time discrete methods:**
        :py:func:`stochastic_process_kle` 
        Simulate Stochastic Process using Karhunen-Loève expansion
        
            This method still needs explicit integrations weights for the 
            numeric integrations. For convenience you can use
             
                :py:func:`stochastic_process_mid_point_weight` simplest approach, for
                test reasons only, uses :py:func:`get_mid_point_weights` 
                to calculate the weights
                 
                :py:func:`stochastic_process_trapezoidal_weight` little more sophisticated,
                uses :py:func:`get_trapezoidal_weights_times` to calculate the weights
                 
                :py:func:`stochastic_process_simpson_weight`,
                **so far for general use**, uses :py:func:`get_simpson_weights_times` 
                to calculate the weights
     
         
        :py:func:`stochastic_process_fft`
        Simulate Stochastic Process using FFT method
         
    **time continuous methods:**
        :py:class:`StocProc` 
        Simulate Stochastic Process using Karhunen-Loève expansion and allows
        for correct interpolation. This class still needs explicit integrations 
        weights for the numeric integrations (use :py:func:`get_trapezoidal_weights_times`
        for general purposes).
         
        .. todo:: implement convenient classes with fixed weights
"""
from .stocproc_c import auto_correlation as auto_correlation_c

import sys
import os
from warnings import warn
sys.path.append(os.path.dirname(__file__))
import numpy as np
from scipy.linalg import eigh as scipy_eigh
 
    
def solve_hom_fredholm(r, w, eig_val_min, verbose=1):
    r"""Solves the discrete homogeneous Fredholm equation of the second kind
    
    .. math:: \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u(s) = \lambda u(t)
    
    Quadrature approximation of the integral gives a discrete representation of the
    basis functions and leads to a regular eigenvalue problem.
    
    .. math:: \sum_i w_i R(t_j-s_i) u(s_i) = \lambda u(t_j) \equiv \mathrm{diag(w_i)} \cdot R \cdot u = \lambda u 
    
    Note: If :math:`t_i = s_i \forall i` the matrix :math:`R(t_j-s_i)` is 
    a hermitian matrix. In order to preserve hermiticity for arbitrary :math:`w_i` 
    one defines the diagonal matrix :math:`D = \mathrm{diag(\sqrt{w_i})}` 
    with leads to the equivalent expression:
    
    .. math:: D \cdot R \cdot D \cdot D \cdot u = \lambda D \cdot u \equiv \tilde R \tilde u = \lambda \tilde u
    
    where :math:`\tilde R` is hermitian and :math:`u = D^{-1}\tilde u`    
    
    Setting :math:`t_i = s_i` and due to the definition of the correlation function the 
    matrix :math:`r_{ij} = R(t_i, s_j)` is hermitian.
    
    :param r: correlation matrix :math:`R(t_j-s_i)`
    :param w: integrations weights :math:`w_i` 
        (they have to correspond to the discrete time :math:`t_i`)
    :param eig_val_min: discards all eigenvalues and eigenvectos with
         :math:`\lambda_i < \mathtt{eig\_val\_min} / \mathrm{max}(\lambda)`
         
    :return: eigenvalues, eigenvectos (eigenvectos are stored in the normal numpy fashion, ordered in decreasing order)
    """
    
    # weighted matrix r due to quadrature weights
    if verbose > 0:
        print("build matrix ...")
#     d = np.diag(np.sqrt(w))
#     r = np.dot(d, np.dot(r, d))
    n = len(w)
    w_sqrt = np.sqrt(w)
    r = w_sqrt.reshape(n,1) * r * w_sqrt.reshape(1,n)
    
    if verbose > 0:
        print("solve eigenvalue equation ...")
    eig_val, eig_vec = scipy_eigh(r, overwrite_a=True)   # eig_vals in ascending

    # use only eigenvalues larger than sig_min**2
    
    min_idx = sum(eig_val < eig_val_min)
     
    eig_val = eig_val[min_idx:][::-1]
    eig_vec = eig_vec[:, min_idx:][:, ::-1]
    
    num_of_functions = len(eig_val)
    if verbose > 0:
        print("use {} / {} eigenfunctions (sig_min = {})".format(num_of_functions, len(w), np.sqrt(eig_val_min)))
    
    
    # inverse scale of the eigenvectors
#     d_inverse = np.diag(1/np.sqrt(w))
#     eig_vec = np.dot(d_inverse, eig_vec)
    eig_vec = np.reshape(1/w_sqrt, (n,1)) * eig_vec

    if verbose > 0:
        print("done!")
    
    
    return eig_val, eig_vec

def stochastic_process_kle(r_tau, t, w, num_samples, seed = None, sig_min = 1e-4, verbose=1):
    r"""Simulate Stochastic Process using Karhunen-Loève expansion
    
    Simulate :math:`N_\mathrm{S}` wide-sense stationary stochastic processes 
    with given correlation :math:`R(\tau) = \langle X(t) X^\ast(s) \rangle = R (t-s)`.
     
    Expanding :math:`X(t)` in a Karhunen–Loève manner 
    
    .. math:: X(t) = \sum_i X_i u_i(t)
    
    with
    
    .. math:: 
        \langle X_i X^\ast_j \rangle = \lambda_i \delta_{i,j} \qquad \langle u_i | u_j \rangle = 
        \int_0^{t_\mathrm{max}} u_i(t) u^\ast_i(t) dt = \delta_{i,j} 
    
    where :math:`\lambda_i` and :math:`u_i` are solutions of the following 
    homogeneous Fredholm equation of the second kind.
    
    .. math:: \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u(s) = \lambda u(t)
    
    Discrete solutions are provided by :py:func:`solve_hom_fredholm`.
    With these solutions and expressing the random variables :math:`X_i` through
    independent normal distributed random variables :math:`Y_i` with variance one
    the Stochastic Process at discrete times :math:`t_j` can be calculates as follows
    
    .. math:: X(t_j) = \sum_i Y_i \sqrt{\lambda_i} u_i(t_j)
    
    To calculate the Stochastic Process for abitrary times 
    :math:`t \in [0,t_\mathrm{max}]` use :py:class:`StocProc`.
    
    References:
        [1] Kobayashi, H., Mark, B.L., Turin, W.,
        2011. Probability, Random Processes, and Statistical Analysis,
        Cambridge University Press, Cambridge. (pp. 363)
    
        [2] Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 
        2007. Numerical Recipes 3rd Edition: The Art of Scientific Computing, 
        Auflage: 3. ed. Cambridge University Press, Cambridge, UK ; New York. (pp. 989)

    :param r_tau: function object of the one parameter correlation function :math:`R(\tau) = R (t-s) = \langle X(t) X^\ast(s) \rangle`
    :param t: list of grid points for the time axis
    :param w: appropriate weights to integrate along the time axis using the grid points given by :py:obj:`t`
    :param num_samples: number of stochastic process to sample
    :param seed: seed for the random number generator used
    :param sig_min: minimal standard deviation :math:`\sigma_i` the random variable :math:`X_i` 
        viewed as coefficient for the base function :math:`u_i(t)` must have to be considered as 
        significant for the Karhunen-Loève expansion (note: :math:`\sigma_i` results from the 
        square root of the eigenvalue :math:`\lambda_i`)
        
    :return: returns a 2D array of the shape (num_samples, len(t)). Each row of the returned array contains one sample of the
        stochastic process.
    """
    
    if verbose > 0:
        print("__ stochastic_process __")

    if seed != None:
        np.random.seed(seed)
    
    t_row = t
    t_col = t_row[:,np.newaxis]
    
    # correlation matrix
    # r_tau(t-s) -> integral/sum over s -> s must be row in EV equation
    r = r_tau(t_col-t_row)
    

    # solve discrete Fredholm equation
    # eig_val = lambda
    # eig_vec = u(t)
    eig_val, eig_vec = solve_hom_fredholm(r, w, sig_min**2)
    num_of_functions = len(eig_val)
    # generate samples
    sig = np.sqrt(eig_val).reshape(1, num_of_functions)               # variance of the random quantities of the  Karhunen-Loève expansion

    if verbose > 0:
        print("generate samples ...")
    x = np.random.normal(scale=1/np.sqrt(2), size=(2*num_samples*num_of_functions)).view(np.complex).reshape(num_of_functions, num_samples).T
                                                                      # random quantities all aligned for num_samples samples
    x_t_array = np.tensordot(x*sig, eig_vec, axes=([1],[1]))              # multiplication with the eigenfunctions == base of Karhunen-Loève expansion
    
    if verbose > 0:
        print("done!")
    
    return x_t_array

def _stochastic_process_alternative_samples(num_samples, num_of_functions, t, sig, eig_vec, seed):
    r"""used for debug and test reasons
    
    generate each sample independently in a for loop
    
    should be slower than using numpy's array operations to do it all at once  
    """
    np.random.seed(seed)
    x_t_array = np.empty(shape=(num_samples, len(t)), dtype = complex)
    for i in range(num_samples):
        x = np.random.normal(size=num_of_functions) * sig
        x_t_array[i,:] = np.dot(eig_vec, x.T)
    return x_t_array

def get_mid_point_weights(t_max, num_grid_points):
    r"""Returns the time gridpoints and wiehgts for numeric integration via **mid point rule**.
        
    The idea is to use :math:`N_\mathrm{GP}` time grid points located in the middle
    each of the :math:`N_\mathrm{GP}` equally distributed subintervals of :math:`[0, t_\mathrm{max}]`.
    
    .. math:: t_i = \left(i + \frac{1}{2}\right)\frac{t_\mathrm{max}}{N_\mathrm{GP}} \qquad i = 0,1, ... N_\mathrm{GP} - 1 

    The corresponding trivial weights for integration are
    
    .. math:: w_i = \Delta t = \frac{t_\mathrm{max}}{N_\mathrm{GP}} \qquad i = 0,1, ... N_\mathrm{GP} - 1
    
    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
        (note: this would corespond to an integration from :math:`0-\Delta t / 2`
        to :math:`t_\mathrm{max}+\Delta t /2`)  
    :param num_grid_points: number of 
    """
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.ones(num_grid_points)*delta_t
    return t, w

def stochastic_process_mid_point_weight(r_tau, t_max, num_grid_points, num_samples, seed = None, sig_min = 1e-4):
    r"""Simulate Stochastic Process using Karhunen-Loève expansion with **mid point rule** for integration
        
    The idea is to use :math:`N_\mathrm{GP}` time grid points located in the middle
    each of the :math:`N_\mathrm{GP}` equally distributed subintervals of :math:`[0, t_\mathrm{max}]`.
    
    .. math:: t_i = \left(i + \frac{1}{2}\right)\frac{t_\mathrm{max}}{N_\mathrm{GP}} \qquad i = 0,1, ... N_\mathrm{GP} - 1 

    The corresponding trivial weights for integration are
    
    .. math:: w_i = \Delta t = \frac{t_\mathrm{max}}{N_\mathrm{GP}} \qquad i = 0,1, ... N_\mathrm{GP} - 1
    
    Since the autocorrelation function depends solely on the time difference :math:`\tau` the static shift for :math:`t_i` does not
    alter matrix used to solve the Fredholm equation. So for the reason of convenience the time grid points are not centered at
    the middle of the intervals, but run from 0 to :math:`t_\mathrm{max}` equally distributed.    
    
    Calling :py:func:`stochastic_process` with these calculated :math:`t_i, w_i` gives the corresponding processes.        
    
    :param t_max: right end of the considered time interval :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: :math:`N_\mathrm{GP}` number of time grid points used for the discretization of the
        integral of the Fredholm integral (see :py:func:`stochastic_process`)
    
    :return: returns the tuple (set of stochastic processes, time grid points) 
        
    See :py:func:`stochastic_process` for other parameters
    """
    t,w = get_trapezoidal_weights_times(t_max, num_grid_points)    
    return stochastic_process_kle(r_tau, t, w, num_samples, seed, sig_min), t
    
def get_trapezoidal_weights_times(t_max, num_grid_points):
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.ones(num_grid_points)*delta_t
    w[0] /= 2
    w[-1] /= 2
    return t, w

def stochastic_process_trapezoidal_weight(r_tau, t_max, num_grid_points, num_samples, seed = None, sig_min = 1e-4):
    r"""Simulate Stochastic Process using Karhunen-Loève expansion with **trapezoidal rule** for integration
       
    .. math:: t_i = i \frac{t_\mathrm{max}}{N_\mathrm{GP}} = i \Delta t \qquad i = 0,1, ... N_\mathrm{GP} 

    The corresponding weights for integration are
    
    .. math:: w_0 = w_{N_\mathrm{GP}} = \frac{\Delta t}{2}, \qquad w_i = \Delta t = \qquad i = 1, ... N_\mathrm{GP} - 1
    
    Calling :py:func:`stochastic_process` with these calculated :math:`t_i, w_i` gives the corresponding processes.        
    
    :param t_max: right end of the considered time interval :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: :math:`N_\mathrm{GP}` number of time grid points used for the discretization of the
        integral of the Fredholm integral (see :py:func:`stochastic_process`)
    
    :return: returns the tuple (set of stochastic processes, time grid points) 
        
    See :py:func:`stochastic_process` for other parameters
    """ 
    t, w = get_trapezoidal_weights_times(t_max, num_grid_points)    
    return stochastic_process_kle(r_tau, t, w, num_samples, seed, sig_min), t

def get_simpson_weights_times(t_max, num_grid_points):
    if num_grid_points % 2 == 0:
        raise RuntimeError("simpson weight need odd number of grid points, but git ng={}".format(num_grid_points))
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::2] = 2/3*delta_t
    w[1::2] = 4/3*delta_t
    w[0]    = 1/3*delta_t
    w[-1]   = 1/3*delta_t
    return t, w

def stochastic_process_simpson_weight(r_tau, t_max, num_grid_points, num_samples, seed = None, sig_min = 1e-4):
    r"""Simulate Stochastic Process using Karhunen-Loève expansion with **simpson rule** for integration
    
    Calling :py:func:`stochastic_process` with these calculated :math:`t_i, w_i` gives the corresponding processes.        
    
    :param t_max: right end of the considered time interval :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: :math:`N_\mathrm{GP}` number of time grid points (need to be odd) used for the discretization of the
        integral of the Fredholm integral (see :py:func:`stochastic_process`)
    
    :return: returns the tuple (set of stochastic processes, time grid points) 
        
    See :py:func:`stochastic_process` for other parameters
    """ 
    t, w = get_simpson_weights_times(t_max, num_grid_points)    
    return stochastic_process_kle(r_tau, t, w, num_samples, seed, sig_min), t
   

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
    sqrt_spectral_density = np.sqrt(spectral_density(omega)).reshape((1, n_dft))
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
    return np.mean(x * x_s_0, axis = 0)
