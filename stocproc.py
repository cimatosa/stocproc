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
                so far for general use, uses :py:func:`get_trapezoidal_weights_times` 
                to calculate the weights
                
                .. todo:: implement Simpson etc.
    
        
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


import numpy as np
import time as tm

class StocProc(object):
    r"""Simulate Stochastic Process using Karhunen-Loève expansion 
    
    The :py:class:`StocProc` class provides methods to simulate stochastic processes 
    :math:`X(t)` in a given time interval :math:`[0,t_\mathrm{max}]` 
    with given autocorrelation function 
    :math:`R(\tau) = R(t-s) = \langle X(t)X^\ast(s)\rangle`. The method is similar to
    the one described and implemented in :py:func:`stochastic_process_kle`.
    
    The :py:class:`StocProc` class extends the functionality of the 
    :py:func:`stochastic_process_kle` routine by providing an interpolation
    method based on the numeric solution of the Fredholm equation.
    Since the time discrete solutions :math:`u_i(s_j)` of the Fredholm equation 
    are best interpolates using 
    
    .. math:: u_i(t) = \frac{1}{\lambda_i}\sum_j w_j R(t-s_j) u_i(s_j)
    
    with :math:`s_j` and :math:`w_j` being the time grid points and integration weights
    for the numeric integrations as well as :math:`\lambda_i` and :math:`u_i` being
    the eigenvalues the time discrete eigenvectors of the discrete Fredholm equation
    (see Ref. [1]).
    
    From that is follows that a good interpolation formula for the stochastic process
    is given by
    
    .. math:: X(t) = \sum_i \sqrt{\lambda_i} Y_i u_i(t) = \sum_{i,j} \frac{Y_i}{\sqrt{\lambda_i}}w_j R(t-s_j) u_i(s_j) 

    where the :math:`Y_i` are independent normal distributed random variables 
    with variance one.
    
    For extracting values of the Stochastic Process you may use:
        :py:func:`x`: returns the value of the Stochastic Process for a 
        single time :math:`t`
        
        :py:func:`x_t_array`: returns the value of the Stochastic Process for 
        all values of the `numpy.ndarray` a single time :math:`t_\mathrm{array}`.
        
        :py:func:`x_for_initial_time_grid`: returns the value of the Stochastic Process for 
        the times given to the constructor in order to discretize the Fredholm
        equation. This is equivalent to calling :py:func:`stochastic_process_kle` with the
        same weights :math:`w_i` and time grid points :math:`s_i`.
        
    To generate a new process call :py:func:`new_process`.
    
    To generate a new sample use :py:func:`new_process`.
    
    :param r_tau: function object of the one parameter correlation function 
        :math:`R(\tau) = R (t-s) = \langle X(t) X^\ast(s) \rangle`
    :param t: list of grid points for the time axis
    :param w: appropriate weights to integrate along the time axis using the 
        grid points given by :py:obj:`t`
    :param seed: seed for the random number generator used
    :param sig_min: minimal standard deviation :math:`\sigma_i` the random variable :math:`X_i = \sigma_i Y_i`, 
        viewed as coefficient for the base function :math:`u_i(t)`, must have to be considered as 
        significant for the Karhunen-Loève expansion (note: :math:`\sigma_i` results from the 
        square root of the eigenvalue :math:`\lambda_i`)
    
    For further reading see :py:func:`stochastic_process_kle`.
    
    References:
        [1] Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 
        2007. Numerical Recipes 3rd Edition: The Art of Scientific Computing, 
        Auflage: 3. ed. Cambridge University Press, Cambridge, UK ; New York. (pp. 989)
        
    """
    def __init__(self, r_tau, t, w, seed = None, sig_min = 1e-4):
        self._r_tau = r_tau
        self._num_gp = len(t)
        self._s = t
        self._w = w
        t_row = t.reshape(1, self._num_gp)
        t_col = t.reshape(self._num_gp, 1)
        # correlation matrix
        # r_tau(t-s) -> integral/sum over s -> s must be row in EV equation
        r = r_tau(t_col-t_row) 

        # solve discrete Fredholm equation
        # eig_val = lambda
        # eig_vec = u(t)
        self._eig_val, self._eig_vec = solve_hom_fredholm(r, w, sig_min**2)
        self._sqrt_eig_val = np.sqrt(self._eig_val)
        self._num_ev = len(self._eig_val)
        self._A = w.reshape(self._num_gp,1) * self._eig_vec / self._sqrt_eig_val.reshape(1, self._num_ev)

        self.new_process(seed) 

    def new_process(self, seed = None):
        r"""setup new process
        
        Generates a new set of independent normal random variables :math:`Y_i`
        which correspondent to the expansion coefficients of the
        Karhunen-Loève expansion for the stochastic process
        
        .. math:: X(t) = \sum_i \sqrt{\lambda_i} Y_i u_i(t)
        
        :param seed: a seed my be given which is passed to the random number generator
        """
        if seed != None:
            np.random.seed(seed)
        self._Y = np.random.normal(size = (self._num_ev,1))
        
    def x_for_initial_time_grid(self):
        r"""Get process on initial time grid
        
        Returns the value of the Stochastic Process for 
        the times given to the constructor in order to discretize the Fredholm
        equation. This is equivalent to calling :py:func:`stochastic_process_kle` with the
        same weights :math:`w_i` and time grid points :math:`s_i`.
        """
        tmp = self._Y * self._sqrt_eig_val.reshape(self._num_ev,1)
        return np.tensordot(tmp, self._eig_vec, axes=([0],[1])).flatten()
        
    def x(self, t):
        # self._Y                                      # (N_ev, 1   )
        tmp = self._Y*self._r_tau(t-self._s.reshape(1, self._num_gp))
                                                       # (N_ev, N_gp)
        # A                                            # (N_gp, N_ev)
        return np.tensordot(tmp, self._A, axes=([1,0],[0,1]))
        
    def x_t_array(self, t_array):
        t_array = t_array.reshape(1,1,len(t_array))    # (1   , 1   , N_t)
        tmp = (self._Y.reshape(self._num_ev,1,1) *  
               self._r_tau(t_array-self._s.reshape(1,self._num_gp,1)))
                                                       # (N_ev, N_gp, N_t)
        # A                                            # (N_gp, N_ev)
        # A_j,i = w_j / sqrt(lambda_i) u_i(s_j)
        return np.tensordot(tmp, self._A, axes=([1,0],[0,1]))

    
    def u_i(self, t_array, i):
        r"""get eigenfunction of index i
        
        Returns the i-th eigenfunction corresponding to the i-th eigenvalue
        of the discrete Fredholm equation using the interpolation scheme:
        
        .. math:: u_i(t) = \frac{1}{\lambda_i}\sum_j w_j R(t-s_j) u_i(s_j)
        
        :param t_array: 1D time array for which the eigenfunction :math:`u_i`
            will be evaluated.
        :param i: index of the eigenfunction
        :return: 1D array of length ``len(t_array)`` 
        """
        t_array = t_array.reshape(1,len(t_array))      # (1   , N_t)
        tmp = self._r_tau(t_array-self._s.reshape(self._num_gp,1))
                                                       # (N_gp, N_t)
        # A                                            # (N_gp, N_ev)
        # A_j,i = w_j / sqrt(lambda_i) u_i(s_j)                                                     
        return 1/self._sqrt_eig_val[i]*np.tensordot(tmp, self._A[:,i], axes=([0],[0]))

    def u_i_all(self, t_array):
        r"""get all eigenfunctions
        
        Returns all eigenfunctions of the discrete Fredholm equation using
        the interpolation scheme:
        
        .. math:: u_i(t) = \frac{1}{\lambda_i}\sum_j w_j R(t-s_j) u_i(s_j)
        
        :param t_array: 1D time array for which the eigenfunction :math:`u_i`
            will be evaluated.
        :return: 2D array of shape ``(len(t_array), number_of_eigenvalues=self._num_ev)``
             (note, the number of eigenvalues may be smaller than the number
             of grid points because of the selections mechanism implemented
             by the value of ``sig_min``)
        """
        t_array = t_array.reshape(1,len(t_array))      # (1   , N_t)
        tmp = self._r_tau(t_array-self._s.reshape(self._num_gp,1))
                                                       # (N_gp, N_t)
        # A                                            # (N_gp, N_ev)
        # A_j,i = w_j / sqrt(lambda_i) u_i(s_j)
        return np.tensordot(tmp, 1/self._sqrt_eig_val.reshape(1,self._num_ev) * self._A, axes=([0],[0]))

    def eigen_vector_i(self, i):
        r"""Returns the i-th eigenvector (solution of the discrete Fredhom equation)"""
        return self._eig_vec[:,i]
    
    def eigen_vector_i_all(self):
        r"""Returns all eigenvectors (solutions of the discrete Fredhom equation)
        
        Note: Note: The maximum number of eigenvalues / eigenfunctions is given by
        the number of time grid points passed to the constructor. But due to the
        threshold ``sig_min`` (see :py:class:`StocProc`) only those 
        eigenvalues and corresponding eigenfunctions which satisfy 
        :math:`\mathtt{sig_{toll}} \geq \sqrt{\lambda_i}` are kept.
        """
        return self._eig_vec

    def lambda_i(self, i):
        r"""Returns the i-th eigenvalue."""
        return self._eig_val[i]

    def lambda_i_all(self):
        r"""Returns all eigenvalues."""
        return self._eig_val
    
    def num_ev(self):
        r"""Returns the number of eigenvalues / eigenfunctions used
        
        Note: The maximum number of eigenvalues / eigenfunctions is given by
        the number of time grid points passed to the constructor. But due to the
        threshold ``sig_min`` (see :py:class:`StocProc`) only those 
        eigenvalues and corresponding eigenfunctions which satisfy 
        :math:`\mathtt{sig_{toll}} \geq \sqrt{\lambda_i}` are kept.
        """
        return self._num_ev
    
    def recons_corr(self, t_array):
        r"""computes the interpolated correlation functions
        
        For the Karhunen-Loève expansion of a stochastic process the
        correlation function can be expressed as follows:
        
        .. math:: R(t,s) = \langle X(t)X^\ast(s)\rangle = \sum_{n,m} \langle X_n X^\ast_m \rangle u_n(t) u^\ast_m(s) = \sum_n \lambda_n u_n(t) u^\ast_n(s) 
        
        With that one can do a consistency check whether the finite set of basis functions
        for the expansion (the solutions of the discrete Fredholm equation) is good
        enough to reproduce the given correlation function.   
        """
        u_i_all_t = self.u_i_all(t_array)                        #(N_gp, N_ev)
        u_i_all_ast_s = np.conj(u_i_all_t)                       #(N_gp, N_ev)
        lambda_i_all = self.lambda_i_all()                       #(N_ev)
        
        tmp = lambda_i_all.reshape(1, self._num_ev) * u_i_all_t  #(N_gp, N_ev)  
        
        return np.tensordot(tmp, u_i_all_ast_s, axes=([1],[1]))
    
def _mean_error(r_t_s, r_t_s_exact):
    r"""mean error of the correlation function as function of s
    
    .. math:: \mathtt{err} = \frac{1}{T}\int_0^T |r_\mathm{KLE}(t,r) - r_\mathrm{exact}(t,s)|^2 dt
    
    :return: the mean error ``err`` 
    """
    len_t, len_s = r_t_s.shape
    abs_sqare = np.abs(r_t_s - r_t_s_exact)**2
    
    abs_sqare[0,:] /= 2
    abs_sqare[-1,:] /= 2
    
    err = np.sum(abs_sqare, axis = 0) / len_t
    return err
    
def _max_error(r_t_s, r_t_s_exact):
    abs_sqare = np.abs(r_t_s - r_t_s_exact)**2
    
    err = np.max(abs_sqare, axis = 0)
    return err
    
    
def auto_grid_points(r_tau, t_max, ng_interpolation, tol = 1e-8, err_method = _max_error):
    err = 1
    ng = 1
    seed = None
    sig_min = 0
    t_large = np.linspace(0, t_max, ng_interpolation)
    print("start auto_grid_points, determine ng ...")

    #exponential increase to get below error threshold
    while err > tol:
        ng *= 2
        t, w = get_trapezoidal_weights_times(t_max, ng)
        stoc_proc = StocProc(r_tau, t, w, seed, sig_min)

        r_t_s = stoc_proc.recons_corr(t_large)
        r_t_s_exact = r_tau(t_large.reshape(ng_interpolation,1) - t_large.reshape(1, ng_interpolation))
        
        err = np.max(err_method(r_t_s, r_t_s_exact))

        print("    ng {} -> err {:.2e}".format(ng, err))
        
    ng_low = ng // 2
    ng_high = ng
    
    while (ng_high - ng_low) > 1:
        print("#"*40)
        print("    ng_l", ng_low)
        print("    ng_h", ng_high)
        ng = (ng_low + ng_high) // 2
        print("    ng", ng)
        t, w = get_trapezoidal_weights_times(t_max, ng)
        stoc_proc = StocProc(r_tau, t, w, seed, sig_min)

        r_t_s = stoc_proc.recons_corr(t_large)
        r_t_s_exact = r_tau(t_large.reshape(ng_interpolation,1) - t_large.reshape(1, ng_interpolation))
        
        err = np.max(err_method(r_t_s, r_t_s_exact))

        print("    ng {} -> err {:.2e}".format(ng, err))
        if err > tol:
            print("        err > tol!")
            print("        ng_l -> ", ng)
            ng_low = ng
        else:
            print("        err <= tol!")
            print("        ng_h -> ", ng)
            ng_high = ng
    

    return ng_high
        
def solve_hom_fredholm(r, w, eig_val_min):
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
         :math:`\lambda_i < \mathtt{eig\_val\_min}`
         
    :return: eigenvalues, eigenvectos (eigenvectos are stored in the normal numpy fashion, )
    """
    
    # weighted matrix r due to quadrature weights
    d = np.diag(np.sqrt(w))
    d_inverse = np.diag(1/np.sqrt(w))
    r = np.dot(d, np.dot(r, d))
    print("solve eigenvalue equation ...")
    eig_val, eig_vec = np.linalg.eigh(r)

    # use only eigenvalues larger than sig_min**2
    large_eig_val_idx = np.where(eig_val >= eig_val_min)[0]
    num_of_functions = len(large_eig_val_idx)
    print("use {} / {} eigenfunctions".format(num_of_functions, len(w)))
    eig_val = eig_val[large_eig_val_idx]
    eig_vec = eig_vec[:, large_eig_val_idx]
    
    # inverse scale of the eigenvectors
    eig_vec = np.dot(d_inverse, eig_vec)
    
    return eig_val, eig_vec

def stochastic_process_kle(r_tau, t, w, num_samples, seed = None, sig_min = 1e-4):
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

    print("__ stochastic_process __")
    print("pre calculations ...")
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

    
    print("generate samples ...")
    x = np.random.normal(size=(num_samples, num_of_functions)) * sig  # random quantities all aligned for num_samples samples
    x_t_array = np.tensordot(x, eig_vec, axes=([1],[1]))              # multiplication with the eigenfunctions == base of Karhunen-Loève expansion
    
    

    print("ALL DONE!\n")
    
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
    t,w = get_trapezoidal_weights_times(t_max, num_grid_points)    
    return stochastic_process_kle(r_tau, t, w, num_samples, seed, sig_min), t
   

def stochastic_process_fft(spectral_density, t_max, num_grid_points, num_samples, seed = None):
    r"""Simulate Stochastic Process using FFT method
    
    This method works only for correlations functions of the form
    
    .. math:: \alpha(\tau) = \int_0^{\omega_\mathrm{max}} \mathrm{d}\omega \, J(\omega) e^{-\mathrm{i}\omega \tau}
    
    where :math:`J(\omega)` is a real non negative spectral density. 
    Then the intrgal can be approximated by the Riemann sum
    
    .. math:: \alpha(\tau) \approx \sum_{k=0}^{N-1} \Delta \omega J(\omega_k) e^{-\mathrm{i} k \Delta \omega \tau}
    
    For a process defined by
    
    .. math:: X(t) = \sum_{k=0}^{N-1} \sqrt{J(\omega_k)} X_k \exp^{-\mathrm{i}\omega_k t}
    
    with random variables :math:`X_k` such that :math:`\langle X_k \rangle = 0` 
    and :math:`\langle X_k X_{k'}\rangle = \Delta \omega \delta_{k,k'}` it is easy to see
    that it fullfills the Riemann approximated correlation function.

    .. math:: 
        \begin{align}
            \langle X(t) X^\ast(s) \rangle = & \sum_{k,k'} \sqrt{J(\omega_k)J(\omega_{k'})} \langle X_k X_{k'}\rangle \exp{-\mathrm{i}\omega_k (t-s)} \\
                                           = & \sum_{k} \Delta \omega J(\omega_k) \exp{-\mathrm{i}\omega_k (t-s)} \\
                                           = & \alpha(t-s)
        \end{align}
    
    In order to use the sheme of the Discrete Fourier Transfrom (DFT) to calculate :math:`X(t)`
    :math:`t` has to be disrcetized as well. Some trivial rewriting leads
    
    .. math:: X(t_l) = \sum_{k=0}^{N-1} \sqrt{J(\omega_k)} X_k e^{-\mathrm{i} 2 \pi \frac{k l}{N} \frac{\Delta \omega \Delta t}{ 2 \pi} N}
    
    For the DFT sheme to be applicable :math:`\Delta t` has to be chosen such that
    
    .. math:: 1 = \frac{\Delta \omega \Delta t}{2 \pi} N
    
    holds. Since :math:`J(\omega)` is real it follows that :math:`X(t_l) = X^\ast(t_{N-l})`.
    For that reason the stochastic process has only :math:`(N+1)/2 \quad (N/2 + 1)` independent
    time grid points for odd (even) :math:`N`.
    
    Looking now from the other side, demanding that the process should run from 
    :math:`0` to :math:`t_\mathrm{max}` with :math:`n` equally distributed time grid points
    :math:`N = 2n-1` points for the DFT have to be considered. This also sets the time
    increment :math:`\Delta t = t_\mathrm{max} / (n-1)`.
    
    With that the frequency increment is determined by
    
    .. math:: \Delta \omega = \frac{2 \pi}{\Delta t N} 

    Implementing the above noted considerations it follows

    .. math:: X(l \Delta t) = DFT\left(\sqrt{J(k \Delta \omega)} X_k\right) \qquad k = 0 \; ... \; N-1, \quad l = 0 \; ... \; n

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
    print("__ stochastic_process_fft __")
    print("pre calculations ...")
    n_dft = num_grid_points * 2 - 1
    delta_t = t_max / (num_grid_points-1)
    delta_omega = 2 * np.pi / delta_t / n_dft
      
    #omega axis
    omega = delta_omega*np.arange(n_dft)
    #reshape for multiplication with matrix xi
    sqrt_spectral_density = np.sqrt(spectral_density(omega)).reshape((1, n_dft))
    if seed != None:
        np.random.seed(seed)
    print("  omega_max  : {:.2}".format(delta_omega * n_dft))
    print("  delta_omega: {:.2}".format(delta_omega))
    print("generate samples ...")
    #random normal samples
    xi = np.random.normal(size = (num_samples,n_dft))
    #each row contain a different integrand
    weighted_integrand = sqrt_spectral_density * xi * np.sqrt(delta_omega)
    #compute integral using fft routine
    z_ast = np.fft.rfft(weighted_integrand, axis = 1)
    #corresponding time axis
    t = np.linspace(0, t_max, num_grid_points)
    print("ALL DONE!\n")
    return z_ast, t
    
    
def auto_correlation(x, s_0_idx = 0):
    r"""Computes the auto correlation function for a set of wide-sense stationary stochastic processes
    
    Computes the auto correlation function for the given set :math:`{X_i(t)}` of stochastic processes:
    
    .. math:: \alpha(s, \tau) = \langle X(s+\tau)X^\ast(s) \rangle \qquad \tau = t-s
    
    For wide-sense stationary processes :math:`\alpha` is independent of :math:`s` so by default :math:`s` is set to :math:`s_0`.
    
    :param x: 2D array of the shape (num_samples, num_time_points) containing the set of stochastic processes where each row represents one process
    :param s_0_idx: time index of the reference time :math:`s`
    
    :return: 1D array containing the correlation function 
    """
    
    # handle type error
    if x.ndim != 2:
        raise TypeError('expected 2D numpy array, but {} given'.format(type(x)))
    
    num_samples = x.shape[0]
    x_s_0 = x[:,s_0_idx].reshape(num_samples,1)
    return np.mean(x * np.conj(x_s_0), axis = 0)

 
