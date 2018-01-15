"""
    The method_fft module provides convenient function to
    setup a stochastic process generator using fft method


"""
from __future__ import division, print_function

#from .tools import ComplexInterpolatedUnivariateSpline
#from functools import lru_cache
import fcSpline
import logging
import mpmath
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
    log.debug("find_integral_boundary")
    log.debug("I_ref: {:.2e} at ref_val: {:.2e}".format(I_ref, ref_val))
    if I_ref < tol:
        log.debug("I_ref < tol: {:.2e}".format(tol))
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
                assert I_x is not np.nan
            except Exception as e:
                raise RuntimeError("evaluation of integrand failed due to {}".format(e))
            log.debug("x0: {:.2e} -> x: {:.2e} -> I_x: {:.2e}".format(x0, x, I_x))
            if I_x > tol:
                break
            x0 *= 1.3
            x_old = x
        a = brentq(lambda x: integrand(x) - tol, x_old, x)
        log.debug("found x_tol: {:.2e} I(x): {:.2}".format(float(a), float(integrand(a))))
        log.debug("done!")
        return a
    elif I_ref > tol:
        log.debug("I_ref > tol: {:.2e}".format(tol))
        x_old = ref_val
        while True:
            _i += 1
            if _i > _max_num_iteration:
                raise RuntimeError("max number of iteration reached")
            x = ref_val + x0
            try:
                I_x = integrand(x)
                assert I_x is not np.nan
            except Exception as e:
                raise RuntimeError("evaluation of integrand failed due to {}".format(e))
            log.debug("x0: {:.2e} -> x: {:.2e} -> I_x: {:.2e}".format(x0, x, I_x))
            if I_x < tol:
                break
            x0 *= 1.3
            x_old = x
        a = brentq(lambda x: integrand(x) - tol, x_old, x)
        log.debug("found x_tol: {:.2e} I(x): {:.2}".format(a, integrand(a)))
        log.debug("done!")
        return a
    else:   # I_ref == tol
        log.debug("I_ref = tol: {:.2e}".format(tol))
        log.debug("done!")
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

def wk(h, k):
    return float(0.5*mpmath.pi*h*mpmath.cosh(k*h)/(mpmath.cosh(0.5*mpmath.pi*mpmath.sinh(k*h))**2))

def yk(h, k):
    return float(1 / (mpmath.exp(mpmath.pi / 2 * mpmath.sinh(k * h)) * mpmath.cosh(mpmath.pi / 2 * mpmath.sinh(k * h))))

def fourier_integral_TanhSinh(integrand, w_max, tau, h, kmax):
    I, feed_back = fourier_integral_TanhSinh_with_feedback(integrand, w_max, tau, h, kmax)
    return I

def fourier_integral_TanhSinh_with_feedback(integrand, w_max, tau, h, kmax):
    """
    
    integrate from [0, w_max] the function
    integrand*exp(-1j*w*ti) for ti = dt*n, n in [0, N]
    
    w = w_max/2 (x + 1)     # maps the integral from [-1,1] to the integral [a, b]
    
    weights_k = (0.5 pi cosh(kh)/(cosh(0.5 pi sinh(kh))**2) = weights_minus_k
    x_k = 1-y_k = -x_minus_k  
    y_k = 1/( exp(pi/2 sinh(kh)) cosh(pi/2 np.sinh(kh)))
    
    I = sum_k weights_k * (b-a)/2 * (integrand(w(x_k)) + integrand(w(x_minus_k))) 
    
    :param integrand: 
    :param a: 
    :param b: 
    :param dt: 
    :param N: 
    :return: 
    """
    k_list = np.arange(kmax+1)
    weights_k = [wk(h, ki) for ki in k_list]
    y_k = [yk(h, ki) for ki in k_list]
    tmp1 = w_max/2
    I = []
    feed_back = "ok"
    for ti in tau:
        r = weights_k[0] * integrand(tmp1) * np.exp(-1j * tmp1 * ti)
        for i in range(1, kmax+1):
            if (y_k[i] * tmp1) == 0:
                log.debug("y_k is 0")
                feed_back = "max kmax reached"
                break

            r_tmp = weights_k[i] * (  integrand(y_k[i] * tmp1)     * np.exp(-1j*y_k[i]     * tmp1*ti)
                                    + integrand((2-y_k[i]) * tmp1) * np.exp(-1j*(2-y_k[i]) * tmp1*ti))
            if np.isnan(r_tmp):
                log.debug("integrand yields nan at {} or {}".format(y_k[i] * tmp1, (2-y_k[i]) * tmp1))
                feed_back = "integrand nan"
                break
            r += r_tmp
        I.append(tmp1*r)

    return np.asarray(I), feed_back



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
        a_ = 0
        b_ = find_integral_boundary(integrand, tol=tol, ref_val=1, max_val=1e6, x0=1)
    else:
        a_ = find_integral_boundary(integrand, tol=tol, ref_val=-1, max_val=1e6, x0=-1)
        b_ = find_integral_boundary(integrand, tol=tol, ref_val=1, max_val=1e6, x0=1)

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

def opt_integral_boundaries(integrand, t_max, ft_ref, tol, opt_b_only, diff_method):


    tol_0 = 2
    N_0 = 10
    i = 0
    while True:
        for j in range(0, i+1):
            J_w_min = 10**(-(tol_0 + i-j))
            N = 2**(N_0 + j)

            if opt_b_only:
                a_ = 0
                b_ = find_integral_boundary(integrand, tol=J_w_min, ref_val=1, max_val=1e6, x0=0.777)
            else:
                a_ = find_integral_boundary(integrand, tol=J_w_min, ref_val=-1, max_val=1e6, x0=-0.777)
                b_ = find_integral_boundary(integrand, tol=J_w_min, ref_val=1, max_val=1e6, x0=0.777)

            tau, ft_tau = fourier_integral_midpoint(integrand, a_, b_, N)
            idx = np.where(tau <= t_max)
            ft_ref_tau = ft_ref(tau[idx])
            d = diff_method(ft_ref_tau, ft_tau[idx])
            log.info("J_w_min:{:.2e} N {} yields: interval [{:.2e},{:.2e}] diff {:.2e}".format(J_w_min, N, a_, b_, d))
            if d < tol:
                log.info("return, cause tol of {} was reached".format(tol))
                return d, N, a_, b_
        i += 1




def get_N_a_b_for_accurate_fourier_integral(integrand, t_max, tol, ft_ref, opt_b_only, diff_method=_absDiff):

    """
    """
    if opt_b_only:
        I0 = quad(integrand, 0, np.inf)[0]
    else:
        I0 = quad(integrand, -np.inf, np.inf)[0]
    ft_ref_0 = ft_ref(0)
    rd = np.abs(ft_ref_0 - I0) / np.abs(ft_ref_0)
    log.debug("ft_ref check yields rd {:.3e}".format(rd))
    if rd > 1e-6:
        raise FTReferenceError("it seems that 'ft_ref' is not the fourier transform of 'integrand'")


    d, N, a_new, b_new = opt_integral_boundaries(integrand=integrand, t_max=t_max, ft_ref=ft_ref, tol=tol,
                                                 opt_b_only=opt_b_only, diff_method=diff_method)
    return N, a_new, b_new


def get_dt_for_accurate_interpolation(t_max, tol, ft_ref, diff_method=_absDiff):
    N = 32
    sub_sampl = 2
    
    while True:
        tau = np.linspace(0, t_max, N+1)
        ft_ref_n = ft_ref(tau)
        ft_intp = fcSpline.FCS(x_low = 0, x_high=t_max, y=ft_ref_n[::sub_sampl])
        ft_intp_n = ft_intp(tau)

        ft_ref_0 = abs(ft_ref(0))
        ft_ref_n /= ft_ref_0
        ft_intp_n /= ft_ref_0

        d = diff_method(ft_intp_n, ft_ref_n)
        log.info("acc interp N {} dt {:.2e} -> diff {:.2e}".format(N, sub_sampl*tau[1], d))
        if d < tol:
            return t_max/(N/sub_sampl)
        N*=2


def calc_ab_N_dx_dt(integrand, intgr_tol, intpl_tol, t_max, ft_ref, opt_b_only, diff_method=_absDiff):
    log.info("get_dt_for_accurate_interpolation, please wait ...")

    try:
        c = find_integral_boundary(lambda tau: np.abs(ft_ref(tau)) / np.abs(ft_ref(0)),
                                                       intgr_tol, 1, 1e6, 0.777)
    except RuntimeError:
        c = t_max

    c = min(c, t_max)
    dt_tol = get_dt_for_accurate_interpolation(t_max=c,
                                               tol=intpl_tol,
                                               ft_ref=ft_ref,
                                               diff_method=diff_method)

    log.info("requires dt < {:.3e}".format(dt_tol))

    log.info("get_N_a_b_for_accurate_fourier_integral, please wait ...")
    N, a, b = get_N_a_b_for_accurate_fourier_integral(integrand,
                                                      t_max       = t_max,
                                                      tol         = intgr_tol,
                                                      ft_ref      = ft_ref,
                                                      opt_b_only  = opt_b_only,
                                                      diff_method = diff_method)
    dx = (b-a)/N
    log.info("requires dx < {:.3e}".format(dx))


    dt = 2*np.pi/dx/N
    if dt > dt_tol:
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
    else:
        dt_new = dt
        dx_new = dx

    if dt_new*(N-1) < t_max:
        log.info("increase N to match dt_new*(N-1) < t_max")
        N_tmp = t_max/dt_new + 1
        N = 2 ** int(np.ceil(np.log2(N_tmp)))
        dx_new = 2*np.pi/N/dt_new

    return a, b, N, dx_new, dt_new
