from scipy.integrate import quad
from scipy.optimize import bisect

from functools import partial
from .stocproc_c import auto_correlation as auto_correlation_c
import sys
import os
from warnings import warn
sys.path.append(os.path.dirname(__file__))
import numpy as np
from collections import namedtuple

stocproc_key_type = namedtuple(typename    = 'stocproc_key_type', 
                               field_names = ['bcf', 't_max', 'ng', 'tol', 'cubatur_type', 'sig_min', 'ng_fac'] )


class ComplexInterpolatedUnivariateSpline(object):
    def __init__(self, x, y, k=3, noWarning=False):
        if not noWarning:
            raise DeprecationWarning("use fast cubic Spline (fcSpline) instead")
        from scipy.interpolate import InterpolatedUnivariateSpline
        self.re_spline = InterpolatedUnivariateSpline(x, np.real(y), k=k)
        self.im_spline = InterpolatedUnivariateSpline(x, np.imag(y), k=k)
    def __call__(self, t):
        return self.re_spline(t) + 1j * self.im_spline(t)


def complex_quad(func, a, b, **kw_args):
    func_re = lambda t: np.real(func(t))
    func_im = lambda t: np.imag(func(t))
    I_re = quad(func_re, a, b, **kw_args)[0]
    I_im = quad(func_im, a, b, **kw_args)[0]

    return I_re + 1j * I_im

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


class LorentzianOmega(object):
    def __init__(self, t_max):
        self.T = t_max
        self.delta = np.pi / self.T
        self.a = +np.pi / 2 / self.T
        self.special = False
        self.f_zero = lambda x: (x ** 2 - 1) * np.sin(x * self.T) - 2 * x * np.cos(x * self.T)
        self.cnt = 0

    def __next__(self):
        _a = self.a
        _b = self.a + self.delta
        if (_a < 1) and (_b > 1):
            if not self.special:
                _b = 1
                self.special = True
            else:
                _a = 1
                self.special = False

        r = bisect(self.f_zero, _a, _b)
        if not self.special:
            self.a = _b

        self.cnt += 1
        return r

    def asarray(self, num):
        r = np.empty(num)
        for i in range(num):
            r[i] = next(self)
        return r

class LorentzianEigenFunctions(object):
    def __init__(self, t_max, gamma, w, num):
        self._om_list_prime = LorentzianOmega(gamma*t_max).asarray(num)
        self._norm_list_prime = np.asarray([self._norm(om, gamma*t_max) for om in self._om_list_prime])
        self._c_prime = (self._om_list_prime**2 + 1)/2
        self.gamma = gamma
        self.w = w

    def _norm(self, om, t_max):
        tmp = 0.5*( (om**2 + 1)*t_max + (om**2 - 1)/om/2*np.sin(2*om*t_max) - np.cos(2*om*t_max) + 1 )
        return np.sqrt(tmp)

    def _u(self, t, om, gamma, norm_prime, w):
        t_prime = gamma*t
        return np.exp(-1j*w*t)*np.sqrt(gamma)*(np.sin(om*t_prime) + om*np.cos(om*t_prime))/norm_prime

    def get_eigfunc(self, index):
        return partial(self._u,
                       om = self._om_list_prime[index],
                       gamma = self.gamma,
                       norm_prime = self._norm_list_prime[index],
                       w = self.w)

    def get_eigval(self, index):
        return 1/(self._c_prime[index]*self.gamma)
