from __future__ import division, print_function

from .tools import ComplexInterpolatedUnivariateSpline

import logging
import numpy as np
from numpy.fft import rfft as np_rfft
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize import basinhopping
import sys
import warnings

warnings.simplefilter('error')
MAX_FLOAT = sys.float_info.max
log = logging.getLogger(__name__)


class FTReferenceError(Exception):
    pass


def find_integral_boundary(integrand, tol, ref_val, max_val, x0):
    """
        searches for the point x_0 where integrand(x_tol) = tol
        
        it is assumed that integrand(x) decays monotonic for all x > (ref_val+x0)
        if x0 is positive (x < (ref_val+x0) if x0 is negative)
        
        if x0 > 0: returns x_tol > ref_val (searches right of ref_val)
        if x0 < 0: returns x_tol < ref_val (searches left of ref_val)
        
        raise an error whenever 
            |x-ref_val|   > max_val or 
            1/|x-ref_val| > max_val
        this assured that the function does not search forever
    """
    _max_num_iteration = 100
    _i = 0
    assert x0 != 0
    I_at_ref = integrand(ref_val)
    if I_at_ref <= tol:
        raise ValueError("the integrand at ref_val needs to be greater that tol")

    log.info("ref_value for search: {} tol: {}".format(ref_val, tol))


    I = integrand(x0+ref_val)
    if I/I_at_ref > tol:
        log.debug("x={:.3e} I(x+ref_val) = {:.3e} > tol I(rev_val) -> veer x away from ref_value".format(x0, I))
        x = 2*x0
        I = integrand(x + ref_val)
        while I/I_at_ref > tol:
            log.debug("x={:.3e} I(x+ref_val) = {:.3e}".format(x, I))
            if abs(x) > max_val:
                raise RuntimeError("|x-ref_val| > max_val was reached")
            x *= 2
            I = integrand(x + ref_val)
            _i += 1
            if _i > _max_num_iteration:
                raise RuntimeError("iteration limit reached")

        log.debug("x={:.3e} I(x+ref_val) = {:.3e} < tol I(rev_val)".format(x, I))
        a = brentq(lambda x: integrand(x)-tol, x+ref_val, x0+ref_val)
        log.debug("found I(a={:.3e}) = {:.3e} = tol".format(a, integrand(a)))

    elif I/I_at_ref < tol:
        log.debug("x={:.3e} I(x+ref_val) = {:.3e} < tol I(rev_val) -> approach x towards ref_value".format(x0, I))
        x = x0/2
        I = integrand(x + ref_val)
        while I/I_at_ref < tol:
            log.debug("x={:.3e} I(x+ref_val) = {:.3e}".format(x, I))
            if (1/abs(x)) > max_val:
                raise RuntimeError("1/|x-ref_val| > max_val was reached")
            x /= 2
            I = integrand(x+ref_val)
            _i += 1
            if _i > _max_num_iteration:
                raise RuntimeError("iteration limit reached")

        log.debug("x={:.3e} I(x+ref_val) = {:.3e} > tol I(rev_val)".format(x, I))
        log.debug("search for root in interval [{:.3e},{:.3e}]".format(x0+ref_val, x+ref_val))
        a = brentq(lambda x_: integrand(x_)-tol, x+ref_val, x0+ref_val)
        log.debug("found I(a={:.3e}) = {:.3e} = tol I(rev_val)".format(a, integrand(a)))
    else:
        a = x0
        log.debug("I(ref_val) = tol -> no search necessary")

    log.info("return {:.5g}".format(a))
    return a

def find_integral_boundary_auto(integrand, tol, ref_val=0, max_val=1e6, 
                                ref_val_left=None, ref_val_right=None, 
                                max_val_left=None, max_val_right=None):
    
    ref_val_left  = ref_val if ref_val_left  is None else ref_val_left
    ref_val_right = ref_val if ref_val_right is None else ref_val_right
    max_val_left  = max_val if max_val_left  is None else max_val_left
    max_val_right = max_val if max_val_right is None else max_val_right

    log.info("trigger left search")
    a = find_integral_boundary(integrand, tol, ref_val=ref_val_left,  max_val=max_val_left,  x0=-1)
    log.info("trigger right search")
    b = find_integral_boundary(integrand, tol, ref_val=ref_val_right, max_val=max_val_right, x0=+1)
    return a,b

def fourier_integral_midpoint(integrand, a, b, N):
    """
        approximates int_a^b dx integrand(x) by the riemann sum with N terms
        and the most simplest uniform midpoint weights
    """
    #log.debug("integrate over [{:.3e},{:.3e}] using {} points".format(a,b,N))
    delta_x = (b-a)/N
    delta_k = 2*np.pi/(b-a)
    yl = integrand(np.linspace(a+delta_x/2, b+delta_x/2, N, endpoint=False))  
    fft_vals = np_rfft(yl)
    tau = np.arange(len(fft_vals))*delta_k
    #log.debug("yields d_x={:.3e}, d_k={:.3e} kmax={:.3e}".format(delta_x, delta_k, tau[-1]))
    return tau, delta_x*np.exp(-1j*tau*(a+delta_x/2))*fft_vals

def get_fourier_integral_simps_weighted_values(yl):
    N = len(yl)
    if N % 2 == 1:  # odd N  
        yl[1:  :2] *= 4   # the mid interval points
        yl[2:-2:2] *= 2   # points with left AND right interval
        return yl/3
        
    else:                 # all weight with an overall factor of 1/6
        yl[0]      *= 2   # the very first points
        yl[1:-1:2] *= 8   # the mid interval points (excluding the last)
        yl[2:-2:2] *= 4   # points with left AND right interval (excluding the last but one)
        yl[-2]     *= 5   # trapeziodal rule for the last two points 
        yl[-1]     *= 3
        return yl/6

def fourier_integral_simps(integrand, a, b, N):
    """
        approximates int_a^b dx integrand(x) by the riemann sum with N terms
        using simpson integration scheme        
    """
    delta_x = (b-a)/(N-1)
    delta_k = 2*np.pi/N/delta_x
    l = np.arange(0, N)    
    yl = integrand(a + l*delta_x)
    yl = get_fourier_integral_simps_weighted_values(yl)    
    
    fft_vals = np_rfft(yl)
    tau = np.arange(len(fft_vals))*delta_k
    return tau, delta_x*np.exp(-1j*tau*a)*fft_vals


def _relDiff(xRef, x):
    diff = np.abs(xRef - x)
    norm_xRef = np.abs(xRef)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.where(diff == 0, 0, diff / norm_xRef)
    idx0 = np.where(np.logical_and(norm_xRef == 0, diff != 0))
    res[idx0] = MAX_FLOAT
    return res


def _f_opt_b_only(x, a, integrand, N, t_max, ft_ref):
    b = x[0]
    if np.isnan(b):
        return 300
    tau, ft_tau = fourier_integral_midpoint(integrand, a, b, N)
    idx = np.where(tau <= t_max)
    ft_ref_tau = ft_ref(tau[idx])
    rd = np.max(_relDiff(ft_ref_tau, ft_tau[idx]))
    return np.log10(rd)


def _f_opt(x, integrand, N, t_max, ft_ref):
    a, b = x[0], x[1]
    if np.isnan(a) or np.isnan(b):
        return 300
    tau, ft_tau = fourier_integral_midpoint(integrand, a, b, N)
    idx = np.where(tau <= t_max)
    ft_ref_tau = ft_ref(tau[idx])
    rd = np.max(_relDiff(ft_ref_tau, ft_tau[idx]))
    return np.log10(rd)

class _AcceptTest(object):
    def __init__(self, N, tmax):
        self.tmax = tmax
        self._tmp = np.pi * N

    def __call__(self, x):
        a, b = x[0], x[1]
        tmax_new = self._tmp / (b - a)
        return tmax_new - self.tmax

class _AcceptTest_b_onbly(object):
    def __init__(self, N, a, tmax):
        self.tmax = tmax
        self.a = a
        self._tmp = np.pi * N

    def __call__(self, x):
        b = x[0]
        tmax_new = self._tmp / (b - self.a)
        return tmax_new - self.tmax

def opt_integral_boundaries(integrand, a, b, t_max, ft_ref, opt_b_only, N):
    log.debug("optimize integral boundary N:{} [{:.3e},{:.3e}]".format(N, a, b))
    if opt_b_only:
        act = _AcceptTest_b_onbly(N, a, t_max)
        r = basinhopping(_f_opt_b_only, [b], niter=150,
                         minimizer_kwargs={"method": "slsqp",
                                           "args": (a, integrand, N, t_max, ft_ref),
                                           "constraints": {"type": "ineq", "fun": act}},
                         stepsize=0.1 * (b - a))
        b = r.x[0]
    else:
        act = _AcceptTest(N, t_max)
        r = basinhopping(_f_opt, [a, b], niter=150,
                         minimizer_kwargs={"method"     : "slsqp",
                                           "args"       : (integrand, N, t_max, ft_ref),
                                           "constraints": {"type": "ineq", "fun": act}},
                         stepsize=0.1*(b-a))
        a, b = r.x[0], r.x[1]
    rd = 10 ** r.fun
    log.info("optimization yields max rd {:.3e} and new boundaries [{:.2e},{:.2e}]".format(rd, a, b))

    return rd, a, b

def get_N_a_b_for_accurate_fourier_integral(integrand, a, b, t_max, tol, ft_ref, opt_b_only, N_max = 2**20):
    """
        chooses N such that the approximated Fourier integral 
        meets the exact solution within a given tolerance of the
        relative deviation for a given interval of interest
    """
    log.info("error estimation up to tmax {:.3e} (tol={:.3e})".format(t_max, tol))
    if opt_b_only:
        I0 = quad(integrand, a, np.inf)[0]
    else:
        I0 = quad(integrand, -np.inf, np.inf)[0]
    ft_ref_0 = ft_ref(0)
    rd = np.abs(ft_ref_0 - I0) / np.abs(ft_ref_0)
    log.debug("ft_ref check yields rd {:.3e}".format(rd))
    if rd > 1e-6:
        raise FTReferenceError("it seems that 'ft_ref' is not the fourier transform of 'integrand'")
    
    i = 10
    while True:
        N = 2**i
        rd, a, b = opt_integral_boundaries(integrand=integrand, a=a, b=b, t_max=t_max, ft_ref=ft_ref,
                                           opt_b_only=opt_b_only, N=N)
        if rd < tol:
            log.info("reached rd ({:.3e}) < tol ({:.3e}), return N={}".format(rd, tol, N))
            return N, a, b
        if N > N_max:
            raise RuntimeError("maximum number of points for Fourier Transform reached")
        i += 1    

def get_dt_for_accurate_interpolation(t_max, tol, ft_ref):
    N = 32
    sub_sampl = 8
    
    while True:
        tau = np.linspace(0, t_max, N+1)
        ft_ref_n = ft_ref(tau)
        tau_sub = tau[::sub_sampl]
        
        ft_intp = ComplexInterpolatedUnivariateSpline(x = tau_sub, y = ft_ref_n[::sub_sampl], k=3)
        ft_intp_n = ft_intp(tau)
        
        d = np.max(np.abs(ft_intp_n-ft_ref_n))
        if d < tol:
            return t_max/(N/sub_sampl)
        N*=2


def calc_ab_N_dx_dt(integrand, intgr_tol, intpl_tol, t_max, a, b, ft_ref, opt_b_only, N_max = 2**20):
    N, a, b = get_N_a_b_for_accurate_fourier_integral(integrand, a, b,
                                                      t_max  = t_max,
                                                      tol    = intgr_tol,
                                                      ft_ref = ft_ref,
                                                      opt_b_only=opt_b_only,
                                                      N_max  = N_max)
    dt = get_dt_for_accurate_interpolation(t_max  = t_max,
                                           tol    = intpl_tol,
                                           ft_ref = ft_ref)
    
    dx = (b-a)/N
    N_min = 2*np.pi/dx/dt
    if N_min <= N:
        log.info("dt criterion fulfilled")
        return a, b, N, dx, dt
    else:
        log.info("down scale dx and dt to match new power of 2 N")
    N = 2**int(np.ceil(np.log2(N_min)))


    #scale = np.sqrt(N_min/N)
    #assert scale <= 1
    scale = 1

    dx_new = scale*dx
    b_minus_a = dx_new*N
    
    dt_new = 2*np.pi/dx_new/N
    if opt_b_only:
        b = a + b_minus_a
    else:
        delta = b_minus_a - (b-a)
        b += delta/2
        a -= delta/2

    rd, a, b = opt_integral_boundaries(integrand=integrand, a=a, b=b, t_max=t_max, ft_ref=ft_ref,
                                       opt_b_only=opt_b_only, N=N)

    log.debug("rd after final optimization:{:.3e}".format(rd))

    return a, b, N, dx_new, dt_new
        
        
    
    