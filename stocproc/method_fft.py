"""
    The method_fft module provides convenient function to
    setup a stochastic process generator using fft method


"""
from __future__ import division, print_function

#from .tools import ComplexInterpolatedUnivariateSpline
#from functools import lru_cache
import fcSpline
import logging
import numpy as np
from numpy.fft import rfft as np_rfft
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
import warnings
#warnings.simplefilter('error')
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
    x0 = float(x0)
    I_ref = integrand(ref_val)
    if I_ref < tol:
        # return find_integral_boundary(integrand = lambda x: 1/integrand(x),
        #                               tol = 1/tol,
        #                               ref_val=ref_val,
        #                               max_val=max_val,
        #                               x0=-x0)
        x_old = ref_val
        while True:
            _i += 1
            if _i > _max_num_iteration:
                raise RuntimeError("max number of iteration reached")
            x = ref_val - x0
            try:
                I_x = integrand(x)
            except Exception as e:
                raise RuntimeError("evaluation of integrand failed due to {}".format(e))

            if I_x > tol:
                break
            x0 *= 2
            x_old = x
        a = brentq(lambda x: integrand(x) - tol, x_old, x)
        return a
    elif I_ref > tol:
        x_old = ref_val
        while True:
            _i += 1
            if _i > _max_num_iteration:
                raise RuntimeError("max number of iteration reached")
            x = ref_val + x0
            try:
                I_x = integrand(x)
            except Exception as e:
                raise RuntimeError("evaluation of integrand failed due to {}".format(e))

            if I_x < tol:
                break
            x0 *= 2
            x_old = x
        a = brentq(lambda x: integrand(x) - tol, x_old, x)
        return a
    else:   # I_ref == tol
        return ref_val

def find_integral_boundary_auto(integrand, tol, ref_val=0, max_val=1e6, 
                                ref_val_left=None, ref_val_right=None, 
                                max_val_left=None, max_val_right=None):
    
    ref_val_left  = ref_val if ref_val_left  is None else ref_val_left
    ref_val_right = ref_val if ref_val_right is None else ref_val_right
    max_val_left  = max_val if max_val_left  is None else max_val_left
    max_val_right = max_val if max_val_right is None else max_val_right

    log.debug("trigger left search")
    a = find_integral_boundary(integrand, tol, ref_val=ref_val_left,  max_val=max_val_left,  x0=-1)
    log.debug("trigger right search")
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

def _absDiff(xRef, x):
    return np.max(np.abs(xRef - x))


def _f_opt_for_SLSQP_minimizer(x, integrand, a, b, N, t_max, ft_ref, diff_method, _f_opt_cache, b_only):
    key = float(x[0])
    if  key in _f_opt_cache:
        d, a_, b_ = _f_opt_cache[key]
        return np.log10(d)
    tol = 10**x

    try:
        if b_only:
            a_ = a
            b_ = find_integral_boundary(integrand, tol=tol, ref_val=b, max_val=1e6, x0=1)
        else:
            a_ = find_integral_boundary(integrand, tol=tol, ref_val=a, max_val=1e6, x0=-1)
            b_ = find_integral_boundary(integrand, tol=tol, ref_val=b, max_val=1e6, x0=1)
    except Exception as e:
        log.debug("Exception {} ({}) in _f_opt".format(type(e), e))
        # in case 'find_integral_boundary' failes
        d = 300
        _f_opt_cache[key] = d, None, None
        return d

    if a_ == b_:
        d = 300
        _f_opt_cache[key] = d, None, None
        return d
        

    tau, ft_tau = fourier_integral_midpoint(integrand, a_, b_, N)
    idx = np.where(tau <= t_max)
    ft_ref_tau = ft_ref(tau[idx])
    d = diff_method(ft_ref_tau, ft_tau[idx])
    _f_opt_cache[key] = d, a_, b_
    log.info("f_opt tol {} -> d {}".format(tol, d))
    return np.log10(d)

def _lower_contrs(x, integrand, a, b, N, t_max, ft_ref, diff_method, _f_opt_cache, b_only):
    _f_opt(x, integrand, a, b, N, t_max, ft_ref, diff_method, _f_opt_cache, b_only)
    d, a_, b_ = _f_opt_cache[float(x[0])]
    if (a_ is None) or (b_ is None):
        return -1
    v = N * np.pi / (b_ - a_) - t_max
    #log.debug("lower constr value {} for x {} (tol {})".format(v, 10**x, tol))
    return v


def _upper_contrs(x):
    #log.debug("upper constr value {}".format(-x))
    return -x


def _f_opt(x, integrand, a, b, N, t_max, ft_ref, diff_method, b_only):
    tol = x

    if b_only:
        a_ = a
        b_ = find_integral_boundary(integrand, tol=tol, ref_val=b, max_val=1e6, x0=1)
    else:
        a_ = find_integral_boundary(integrand, tol=tol, ref_val=a, max_val=1e6, x0=-1)
        b_ = find_integral_boundary(integrand, tol=tol, ref_val=b, max_val=1e6, x0=1)

    tau, ft_tau = fourier_integral_midpoint(integrand, a_, b_, N)
    idx = np.where(tau <= t_max)
    ft_ref_tau = ft_ref(tau[idx])
    d = diff_method(ft_ref_tau, ft_tau[idx])
    log.info("f_opt interval [{:.3e},{:.3e}] -> d {}".format(a_, b_, d))
    return d, a_, b_



def opt_integral_boundaries_use_SLSQP_minimizer(integrand, a, b, t_max, ft_ref, opt_b_only, N, diff_method):
    """
    this is very slow
    """
    log.info("optimize integral boundary N:{} [{:.3e},{:.3e}], please wait ...".format(N, a, b))

    _f_opt_cache = dict()
    args = (integrand, a, b, N, t_max, ft_ref, diff_method, _f_opt_cache, opt_b_only)
    x0 = np.log10(0.1*integrand(b))
    r = minimize(_f_opt_for_SLSQP_minimizer, x0 = x0, args = args,
                 method='SLSQP',
                 constraints=[{"type": "ineq", "fun": _lower_contrs, "args": args},
                              {"type": "ineq", "fun": _upper_contrs}])
    d, a_, b_ = _f_opt_cache[float(r.x)]
    if a_ is None or b_ is None:
        log.info("optimization with N {} failed".format(N))
        return d, a, b

    log.info("optimization with N {} yields max rd {:.3e} and new boundaries [{:.2e},{:.2e}]".format(N, d, a_, b_))
    return d, a_, b_

def opt_integral_boundaries(integrand, a, b, t_max, ft_ref, tol, opt_b_only, N, diff_method):
    log.info("optimize integral boundary N:{} [{:.3e},{:.3e}], please wait ...".format(N, a, b))


    args = (integrand, a, b, N, t_max, ft_ref, diff_method, opt_b_only)
    x0 = integrand(b)
    d1 = np.inf, None, None
    while True:
        d = _f_opt(x0, *args)
        log.info("opt int: J(w) min:{:.3e} and N:{} -> tol:{:.3e}".format(x0, N, d[0]))
        if d[0] < tol:
            log.info("return, cause tol of {} was reached".format(tol))
            return d
        x0 *= 0.1
        if d[0] > d1[0]:
            log.info("return cause further decrease of 'J(w) min' does not improove accuracy")
            return d
        if x0 < 1e-12:
            log.info("return cause 'J(w) min' < 1e-6")
            return d
        d1 = d



def get_N_a_b_for_accurate_fourier_integral(integrand, a, b, N_start, t_max, tol, ft_ref, opt_b_only, N_max = 2**20,
                                            diff_method=_absDiff):
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

    N = N_start
    while True:
        rd, a_new, b_new = opt_integral_boundaries(integrand=integrand, a=a, b=b, t_max=t_max, ft_ref=ft_ref, tol=tol,
                                                   opt_b_only=opt_b_only, N=N, diff_method=diff_method)
        #a = a_new
        #b = b_new

        if rd < tol:
            log.info("reached rd ({:.3e}) < tol ({:.3e}), return N={}".format(rd, tol, N))
            return N, a_new, b_new
        if N > N_max:
            raise RuntimeError("maximum number of points for Fourier Transform reached")
        N *= 2

def get_dt_for_accurate_interpolation(t_max, tol, ft_ref, diff_method=_absDiff):
    N = 32
    sub_sampl = 2
    
    while True:
        tau = np.linspace(0, t_max, N+1)
        ft_ref_n = ft_ref(tau)
        ft_intp = fcSpline.FCS(x_low = 0, x_high=t_max, y=ft_ref_n[::sub_sampl])
        ft_intp_n = ft_intp(tau)
        
        d = diff_method(ft_intp_n, ft_ref_n)
        log.info("acc interp N {} dt {:.3e} {:.3e} -> d {:.3e}".format(N, sub_sampl*tau[1], t_max/(N/sub_sampl), d))
        if d < tol:
            return t_max/(N/sub_sampl)
        N*=2


def calc_ab_N_dx_dt(integrand, intgr_tol, intpl_tol, t_max, a, b, ft_ref, opt_b_only, N_max = 2**20, diff_method=_absDiff):
    log.info("get_dt_for_accurate_interpolation, please wait ...")
    c = find_integral_boundary(lambda tau: np.abs(ft_ref(tau)) / np.abs(ft_ref(0)),
                                                   intgr_tol, 1, 1e6, 1)
    dt_tol = get_dt_for_accurate_interpolation(t_max=c,
                                               tol=intpl_tol,
                                               ft_ref=ft_ref,
                                               diff_method=diff_method)

    N_start = t_max / dt_tol
    N_start = 2 ** int(np.ceil(np.log2(N_start)))

    log.info("get_N_a_b_for_accurate_fourier_integral, please wait ...")
    N, a, b = get_N_a_b_for_accurate_fourier_integral(integrand, a, b,
                                                      N_start=N_start,
                                                      t_max  = t_max,
                                                      tol    = intgr_tol,
                                                      ft_ref = ft_ref,
                                                      opt_b_only=opt_b_only,
                                                      N_max  = N_max,
                                                      diff_method=diff_method)

    dx = (b-a)/N
    dt = 2*np.pi/dx/N
    if dt <= dt_tol:
        log.debug("dt criterion fulfilled")
        return a, b, N, dx, dt
    else:
        log.debug("down scale dx and dt to match new power of 2 N")

    N_min = 2*np.pi/dx/dt_tol
    N = 2**int(np.ceil(np.log2(N_min)))
    scale = np.sqrt(N_min/N)
    dx_new = scale*dx
    b_minus_a = dx_new*N
    dt_new = 2*np.pi/dx_new/N
    assert dt_new < dt_tol
    if opt_b_only:
        b = a + b_minus_a
    else:
        delta = b_minus_a - (b-a)
        b += delta/2
        a -= delta/2

    return a, b, N, dx_new, dt_new
