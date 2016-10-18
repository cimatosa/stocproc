# -*- coding: utf8 -*-
"""
    advanced class to do all sort of things with KLE generated stochastic processes
"""

from __future__ import print_function, division

import functools
import numpy as np
import pickle
import sys

from .stocproc import solve_hom_fredholm
from .stocproc import get_mid_point_weights
from .stocproc import get_trapezoidal_weights_times
from .stocproc import get_simpson_weights_times
import gquad

from . import stocproc_c

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import product
from math import fsum

class ComplexInterpolatedUnivariateSpline(object):
    def __init__(self, x, y, k=2):
        self.re_spline = InterpolatedUnivariateSpline(x, np.real(y))
        self.im_spline = InterpolatedUnivariateSpline(x, np.imag(y))
        
    def __call__(self, t):
        return self.re_spline(t) + 1j*self.im_spline(t)
    
def complex_quad(func, a, b, **kw_args):
    func_re = lambda t: np.real(func(t))
    func_im = lambda t: np.imag(func(t))
    I_re = quad(func_re, a, b, **kw_args)[0]
    I_im = quad(func_im, a, b, **kw_args)[0]
    
    return I_re + 1j*I_im

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
    
    _dump_members = ['_r_tau', 
                     '_s',
                     '_w',
                     '_eig_val',
                     '_eig_vec']
    
    def __init__(self, 
                 r_tau      = None, 
                 t          = None, 
                 w          = None, 
                 seed       = None, 
                 sig_min    = 1e-4, 
                 fname      = None,
                 cache_size = 1024,
                 verbose    = 1,
                 align_eig_vec = False):
        
        self.verbose = verbose
        self._one_over_sqrt_2 = 1/np.sqrt(2)
        if fname is None:
            
            assert r_tau is not None
            self._r_tau = r_tau
            
            assert t is not None
            self._s = t
            self._num_gp = len(self._s)
            
            assert w is not None
            self._w = w

            r = StocProc._calc_corr_matrix(self._s, self._r_tau)            
            # solve discrete Fredholm equation
            # eig_val = lambda
            # eig_vec = u(t)
            self._eig_val, self._eig_vec = solve_hom_fredholm(r, w, sig_min**2, verbose=self.verbose)
            if align_eig_vec:
                for i in range(self._eig_vec.shape[1]):
                    s = np.sum(self._eig_vec[:,i])
                    phase  = np.exp(1j*np.arctan2(np.real(s), np.imag(s)))
                    self._eig_vec[:,i]/= phase

        else:
            self.__load(fname)

        self.__calc_missing()
            
        self.my_cache_decorator = functools.lru_cache(maxsize=cache_size, typed=False)
        self.x = self.my_cache_decorator(self._x)
        self.new_process(seed = seed)
        
    @staticmethod
    def _calc_corr_matrix(s, bcf):
        """calculates the matrix alpha_ij = bcf(t_i-s_j)
        
        calls bcf only for s-s_0 and reconstructs the rest
        """
        n_ = len(s)
        bcf_n_plus = bcf(s-s[0])
        #    [bcf(-3)    , bcf(-2)    , bcf(-1)    , bcf(0), bcf(1), bcf(2), bcf(3)]
        # == [bcf(3)^\ast, bcf(2)^\ast, bcf(1)^\ast, bcf(0), bcf(1), bcf(2), bcf(3)]        
        bcf_n = np.hstack((np.conj(bcf_n_plus[-1:0:-1]), bcf_n_plus))
        # we want
        # r = bcf(0) bcf(-1), bcf(-2)
        #     bcf(1) bcf( 0), bcf(-1)
        #     bcf(2) bcf( 1), bcf( 0)
        r = np.empty(shape=(n_,n_), dtype = np.complex128)
        for i in range(n_):
            idx = n_-1-i
            r[:,i] = bcf_n[idx:idx+n_]
        return r
            
        
    @classmethod
    def new_instance_by_name(cls, name, r_tau, t_max, ng, seed, sig_min, verbose=1, align_eig_vec=False):
        """create a new StocProc instance where the weights are given by name""" 
        known_names = ['trapezoidal', 'mid_point', 'simpson', 'gauss_legendre']
        
        if name == 'trapezoidal':
            ob = cls.new_instance_with_trapezoidal_weights(r_tau, t_max, ng, seed, sig_min, verbose, align_eig_vec)
        elif name == 'mid_point':
            ob = cls.new_instance_with_mid_point_weights(r_tau, t_max, ng, seed, sig_min, verbose, align_eig_vec)
        elif name == 'simpson':
            ob = cls.new_instance_with_simpson_weights(r_tau, t_max, ng, seed, sig_min, verbose, align_eig_vec)
        elif name == 'gauss_legendre':
            ob = cls.new_instance_with_gauss_legendre_weights(r_tau, t_max, ng, seed, sig_min, verbose, align_eig_vec)
        else:
            raise RuntimeError("unknown name '{}' to create StocProc instance\nknown names are {}".format(name, known_names))
        
        ob.name = name
        return ob

    @classmethod
    def new_instance_with_trapezoidal_weights(cls, r_tau, t_max, ng, seed=None, sig_min=0, verbose=1, align_eig_vec=False):
        """use trapezoidal weights (see :py:func:`get_trapezoidal_weights_times`)"""
        t, w = get_trapezoidal_weights_times(t_max, ng)
        return cls(r_tau, t, w, seed, sig_min, verbose=verbose, align_eig_vec=align_eig_vec)
    
    @classmethod
    def new_instance_with_simpson_weights(cls, r_tau, t_max, ng, seed=None, sig_min=0, verbose=1, align_eig_vec=False):
        """use simpson weights (see :py:func:`get_simpson_weights_times`)"""
        t, w = get_simpson_weights_times(t_max, ng)
        return cls(r_tau, t, w, seed, sig_min, verbose=verbose, align_eig_vec=align_eig_vec)

    @classmethod
    def new_instance_with_mid_point_weights(cls, r_tau, t_max, ng, seed=None, sig_min=0, verbose=1, align_eig_vec=False):
        """use mid-point weights (see :py:func:`get_mid_point_weights`)"""
        t, w = get_mid_point_weights(t_max, ng)
        return cls(r_tau, t, w, seed, sig_min, verbose=verbose, align_eig_vec=align_eig_vec)

    @classmethod    
    def new_instance_with_gauss_legendre_weights(cls, r_tau, t_max, ng, seed=None, sig_min=0, verbose=1, align_eig_vec=False):
        """use gauss legendre weights (see :py:func:`gauss_nodes_weights_legendre`)"""
        t, w = gquad.gauss_nodes_weights_legendre(n=ng, low=0, high=t_max)
        return cls(r_tau, t, w, seed, sig_min, verbose=verbose, align_eig_vec=align_eig_vec)
    
    def __load(self, fname):
        with open(fname, 'rb') as f:
            for m in self._dump_members:
                setattr(self, m, pickle.load(f))
                
    def __calc_missing(self):
        self._num_gp = len(self._s)
        self._sqrt_eig_val = np.sqrt(self._eig_val)
        self._num_ev = len(self._eig_val)
        self._A = self._w.reshape(self._num_gp,1) * self._eig_vec / self._sqrt_eig_val.reshape(1, self._num_ev) 

    
    def __dump(self, fname):
        with open(fname, 'wb') as f:
            for m in self._dump_members:
                pickle.dump(getattr(self, m), f, pickle.HIGHEST_PROTOCOL)
                
    def __getstate__(self):
        return [getattr(self, atr) for atr in self._dump_members]
    
    def __setstate__(self, state):
        for i, atr_value in enumerate(state):
            setattr(self, self._dump_members[i], atr_value)
        self.__calc_missing()

    def save_to_file(self, fname):
        self.__dump(fname)
        
    def get_name(self):
        if hasattr(self, 'name'):
            return self.name
        else:
            return 'unknown'

    def new_process(self, yi=None, seed=None):
        r"""setup new process
        
        Generates a new set of independent normal random variables :math:`Y_i`
        which correspondent to the expansion coefficients of the
        Karhunen-Loève expansion for the stochastic process
        
        .. math:: X(t) = \sum_i \sqrt{\lambda_i} Y_i u_i(t)
        
        :param seed: a seed my be given which is passed to the random number generator
        """
        if seed != None:
            np.random.seed(seed)

        self.clear_cache()
        if yi is None:
            if self.verbose > 1:
                print("generate samples ...")
            self._Y = np.random.normal(scale = self._one_over_sqrt_2, size=2*self._num_ev).view(np.complex).reshape(self._num_ev,1)
            if self.verbose > 1:
                print("done!")
        else:
            self._Y = yi.reshape(self._num_ev,1)

        self._a_tmp = np.tensordot(self._Y[:,0], self._A, axes=([0],[1]))
        
    def x_for_initial_time_grid(self):
        r"""Get process on initial time grid
        
        Returns the value of the Stochastic Process for 
        the times given to the constructor in order to discretize the Fredholm
        equation. This is equivalent to calling :py:func:`stochastic_process_kle` with the
        same weights :math:`w_i` and time grid points :math:`s_i`.
        """
        tmp = self._Y * self._sqrt_eig_val.reshape(self._num_ev,1) 
        if self.verbose > 1:
            print("calc process via matrix prod ...")
        res = np.tensordot(tmp, self._eig_vec, axes=([0],[1])).flatten()
        if self.verbose > 1:
            print("done!")
        
        return res
    
    def time_grid(self):
        return self._s
    
    def __call__(self, t):
        if isinstance(t, np.ndarray):
            return self.x_t_array(t)
        else:
            return self.x(t)

    def _x(self, t):
        """calculates the stochastic process at time t"""
        R = self._r_tau(t-self._s)
        res = np.tensordot(R, self._a_tmp, axes=([0],[0]))
        return res

    def get_cache_info(self):
        return self.x.cache_info()
    
    def clear_cache(self):
        self.x.cache_clear()

    def x_t_array(self, t_array):
        """calculates the stochastic process at several times [t_i]"""
        R = self._r_tau(t_array.reshape(1,-1,)-self._s.reshape(-1, 1))  # because t_array can be anything
                                                                        # it can not be optimized with _calc_corr_matrix
        res = np.tensordot(R, self._a_tmp, axes=([0],[0]))
        return res
    
    def x_t_mem_save(self, delta_t_fac, kahanSum=False):
        """calculate the stochastic process (SP) for a certain class fine grids
        
        when the SP is setup with n grid points, which means we know the eigenfunctions
        for the n discrete times t_i = i/(n-1)*t_max, i = 0,1,...n-1
        with delta_t = t_max/(n-1)
        it is efficient to calculate the interpolated process
        for finer grids with delta_t_fine = delta_t/delta_t_fac
        because we only need to know the bcf on the finer grid
        """
        a = delta_t_fac        
        N1 = len(self._s)
        N2 = a * (N1 - 1) + 1        
        T = self._s[-1]
        alpha_k = self._r_tau(np.linspace(-T, T, 2*N2 - 1))        
        return stocproc_c.z_t(delta_t_fac = delta_t_fac,
                              N1          = N1,
                              alpha_k     = alpha_k,
                              a_tmp       = self._a_tmp,
                              kahanSum    = kahanSum)
        
    def x_t_fsum(self, t):
        """slow fsum variant for testing / development reasons
        """
        alpha_k = self._r_tau(t - self._s)
        terms = np.asarray([self._Y[a] * alpha_k[i] * self._A[i, a] for (a,i) in product(range(self._num_ev), range(self._num_gp))])
        re = fsum(np.real(terms))
        im = fsum(np.imag(terms))
        
        return re + 1j*im        

    def u_i_mem_save(self, delta_t_fac, i):
        """efficient evaluation of the interpolated eigen function on special subgrids"""
        a = delta_t_fac        
        N1 = len(self._s)
        N2 = a * (N1 - 1) + 1        
        T = self._s[-1]
        alpha_k = self._r_tau(np.linspace(-T, T, 2*N2 - 1))

        return stocproc_c.eig_func_interp(delta_t_fac,
                                          self._s,
                                          alpha_k,
                                          self._w,
                                          self._eig_val[i],
                                          self._eig_vec[:, i])
        
    def u_i(self, t_array, i):
        r"""get eigenfunction of index i
        
        Returns the i-th eigenfunction corresponding to the i-th eigenvalue
        of the discrete Fredholm equation using the interpolation scheme:
        
        .. math:: u_i(t) = \frac{1}{\lambda_i}\sum_j w_j R(t-s_j) u_i(s_j)
        
        :param t_array: 1D time array for which the eigenfunction :math:`u_i`
            will be evaluated.
        :param i: index of the eigenfunction
        :return: 1D array of length ``len(t_array)`` 
        
        scales like len(t_array)*num_gp
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
             
        scales like len(t_array)*num_gp*num_ev
        """
        t_array = t_array.reshape(1,len(t_array))      # (1   , N_t)
        tmp = self._r_tau(t_array-self._s.reshape(self._num_gp,1))
                                                       # (N_gp, N_t)
        # A                                            # (N_gp, N_ev)
        # A_j,i = w_j / sqrt(lambda_i) u_i(s_j)
        return np.tensordot(tmp, 1/self._sqrt_eig_val.reshape(1,self._num_ev) * self._A, axes=([0],[0]))

    def u_i_all_mem_save(self, delta_t_fac):
        """efficient evaluation of the interpolated eigen function on special subgrids"""
        a = delta_t_fac        
        N1 = len(self._s)
        N2 = a * (N1 - 1) + 1        
        T = self._s[-1]
        alpha_k = self._r_tau(np.linspace(-T, T, 2*N2 - 1))
        
        return stocproc_c.eig_func_all_interp(delta_t_fac = delta_t_fac,
                                              time_axis   = self._s,
                                              alpha_k     = alpha_k, 
                                              weights     = self._w,
                                              eigen_val   = self._eig_val,
                                              eigen_vec   = self._eig_vec)
        
#         print("WARNING! this needs to be cythonized")
#         u_res = np.zeros(shape=(N2, self.num_ev()), dtype=np.complex)
#         for i in range(self.num_ev()):
#             for j in range(N2):
#                 for l in range(N1):
#                     k = j - a*l + N2-1
#                     u_res[j, i] += self._w[l] * alpha_k[k] * self._eig_vec[l, i]      
#  
#             u_res[:, i] /= self._eig_val[i]
#             
#         return u_res

    def t_mem_save(self, delta_t_fac):
        T = self._s[-1]
        N = len(self._s)
        return np.linspace(0, T, delta_t_fac*(N-1) + 1)

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
    
    def recons_corr_single_s(self, t_array, s):
        assert False, "something is wrong here"
        u_i_all_t = self.u_i_all(t_array)                        #(N_gp, N_ev)
        u_i_all_ast_s = np.conj(self.u_i_all(np.asarray([s])))   #(1, N_ev)
        lambda_i_all = self.lambda_i_all()                       #(N_ev)
        tmp = lambda_i_all.reshape(1, self._num_ev) * u_i_all_t  #(N_gp, N_ev)
        return np.tensordot(tmp, u_i_all_ast_s, axes=([1],[1]))[:,0]
    
    def recons_corr_memsave(self, delta_t_fac):
        u_i_all_t = self.u_i_all_mem_save(delta_t_fac)           #(N_gp, N_ev)
        u_i_all_ast_s = np.conj(u_i_all_t)                       #(N_gp, N_ev)
        lambda_i_all = self.lambda_i_all()                       #(N_ev)        
        tmp = lambda_i_all.reshape(1, self._num_ev) * u_i_all_t  #(N_gp, N_ev)  
        return np.tensordot(tmp, u_i_all_ast_s, axes=([1],[1]))    
    
    def get_num_ef(self, rel_threshold):
        G = self._sqrt_eig_val
        return get_num_ef(G, rel_threshold)    
    
    def get_largest_indices(self, rel_threshold):
        G = self._sqrt_eig_val
        return get_largest_indices(G, rel_threshold)
    
    def check_integral_eq(self, index, delta_t_fac = 4, num_t = 50):
        t = self.t_mem_save(delta_t_fac)
        u_t_discrete = self.u_i_mem_save(delta_t_fac, index)
        tmax = self._s[-1]
        G = self._sqrt_eig_val[index]
        bcf = self._r_tau
        data, norm = check_integral_eq(G, u_t_discrete, t, tmax, bcf, num_t)
        return u_t_discrete, data, norm

def get_num_ef(G, rel_threshold):
#     print("WARNING: debugging check for sorted G still active!")
#     g_old = np.Inf
#     for g in G:
#         assert g_old >= g
#         g_old = g
    
    # G must be in decreasing order
    return int(sum(G/max(G) >= rel_threshold))

def get_largest_indices(G, rel_threshold):
    
    print("WARNING: debugging check for sorted G still active!")
    g_old = np.Inf
    for g in G:
        assert g_old >= g
        g_old = g
    
    # G must be in decreasing order
    idx = sum(G/max(G) >= rel_threshold)
    idx_selection = np.arange(0, idx)
    return idx_selection      
    
def check_integral_eq(G, U, t_U, tmax, bcf, num_t=50, limit=5000, c=None):
    u_t = ComplexInterpolatedUnivariateSpline(t_U, U, k=3)
    data = np.empty(shape=(num_t, 2), dtype = np.complex128)
    tau = np.linspace(0, tmax, num_t)
    for i, tau_ in enumerate(tau): 
        data[i, 0] = complex_quad(lambda s: bcf(tau_-s) * u_t(s), 0, tmax, limit=limit)
        data[i, 1] = G**2*u_t(tau_)
        if c is not None:
            with c.get_lock():
                c.value += 1        
        
    norm = quad(lambda s: np.abs(u_t(s))**2, 0, tmax, limit=limit)[0]

    return data, norm 

def mean_error(r_t_s, r_t_s_exact):
    r"""mean error of the correlation function as function of s
    
    .. math:: \mathrm{err} = \frac{1}{T}\int_0^T |r_\mathrm{KLE}(t,r) - r_\mathrm{exact}(t,s)|^2 dt
    
    :return: the mean error ``err`` 
    """
    
    err = np.mean(np.abs(r_t_s - r_t_s_exact), axis = 0)
    return err
    
def max_error(r_t_s, r_t_s_exact):
    return np.max(np.abs(r_t_s - r_t_s_exact), axis = 0)

def max_rel_error(r_t_s, r_t_s_exact):
    return np.max(np.abs(r_t_s - r_t_s_exact) / np.abs(r_t_s_exact))

def recons_corr_and_get_bcf(T, ng, w, eig_val, eig_vec, bcf):
    """
        doing things here again for efficiency reasons
    """
    delta_t_fac = 2        
    N1 = ng
    N2 = delta_t_fac * (N1 - 1) + 1        
    alpha_k = bcf(np.linspace(-T, T, 2*N2 - 1))
    
    u_i_all_t =  stocproc_c.eig_func_all_interp(delta_t_fac = delta_t_fac,
                                                time_axis   = np.linspace(0, T, N1),
                                                alpha_k     = alpha_k, 
                                                weights     = w,
                                                eigen_val   = eig_val,
                                                eigen_vec   = eig_vec)    
    
    u_i_all_ast_s = np.conj(u_i_all_t)                  #(N_gp, N_ev)
    num_ev = len(eig_val)       
    tmp = eig_val.reshape(1, num_ev) * u_i_all_t  #(N_gp, N_ev)  
    recs_bcf = np.tensordot(tmp, u_i_all_ast_s, axes=([1],[1]))
    
    refc_bcf = np.empty(shape=(N2,N2), dtype = np.complex128)
    for i in range(N2):
        idx = N2-1-i
        refc_bcf[:,i] = alpha_k[idx:idx+N2]
    
    return recs_bcf, refc_bcf
      

def auto_grid_points(r_tau, t_max, tol = 1e-8, err_method = max_error, name = 'mid_point', sig_min = 1e-4, verbose=1): 
    err = 1
    c = 2
    seed = None
    err_method_name = err_method.__name__
    if verbose > 0:
        print("start auto_grid_points, determine ng ...")
    #exponential increase to get below error threshold
    while err > tol:
        c *= 2
        ng = 2*c + 1
        ng_fine = ng*2-1
        t_fine = np.linspace(0, t_max, ng_fine)
        if verbose == 1:
            print("ng:{} new proc ({}) ... ".format(ng, name), end='')
            sys.stdout.flush()
        if verbose > 1:
            print("#"*40)
            print("c", c, "ng", ng)
            print("new process with {} weights ...".format(name))
        stoc_proc = StocProc.new_instance_by_name(name, r_tau, t_max, ng, seed, sig_min, verbose-1)
        if verbose > 1:
            print("reconstruct correlation function ({} points)...".format(ng_fine))
        r_t_s, r_t_s_exact = recons_corr_and_get_bcf(T  = t_max, 
                                                     ng = ng, 
                                                     w       = stoc_proc._w, 
                                                     eig_val = stoc_proc._eig_val, 
                                                     eig_vec = stoc_proc._eig_vec,
                                                     bcf     = r_tau)
        if verbose > 1:
            print("calculate error using {} ...".format(err_method_name))
        err = np.max(err_method(r_t_s, r_t_s_exact))
        if verbose > 0:
            print("err {:.3e}".format(err))
        
    c_low = c // 2
    c_high = c
    
    while (c_high - c_low) > 1:
        if verbose > 1:
            print("#"*40)
            print("c_low", c_low)
            print("c_high", c_high)
        c = (c_low + c_high) // 2
        ng = 2*c + 1
        ng_fine = ng * 2 - 1
        if verbose > 1:
            print("c", c)
            print("ng", ng)
            print("ng_fine", ng_fine)

        t_fine = np.linspace(0, t_max, ng_fine)
        if verbose == 1:
            print("ng:{} new proc ({}) ... ".format(ng, name), end='')
            sys.stdout.flush()
            
        if verbose > 1:
            print("new process with {} weights ...".format(name))
        stoc_proc = StocProc.new_instance_by_name(name, r_tau, t_max, ng, seed, sig_min, verbose-1)
        if verbose > 1:
            print("reconstruct correlation function ({} points)...".format(ng_fine))
        r_t_s = stoc_proc.recons_corr(t_fine)
        if verbose > 1:
            print("calculate exact correlation function ...")
        r_t_s_exact = r_tau(t_fine.reshape(ng_fine,1) - t_fine.reshape(1, ng_fine))
        if verbose > 1:
            print("calculate error using {} ...".format(err_method_name))
        err = np.max(err_method(r_t_s, r_t_s_exact))
        if verbose > 0:
            print("err {:.3e}".format(err))
        if err > tol:
            if verbose > 1:
                print("    err > tol!")
                print("    c_low -> ", c)
            c_low = c
        else:
            if verbose > 1:
                print("    err <= tol!")
                print("    c_high -> ", c)
            c_high = c
    
    return ng      