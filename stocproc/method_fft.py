
from scipy.optimize import brentq, bisect
from numpy.fft import rfft as np_rfft
import numpy as np
import logging

log = logging.getLogger(__name__)

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
    assert x0 != 0
    if integrand(ref_val) <= tol:
        raise ValueError("the integrand at ref_val needs to be greater that tol")

    log.debug("ref_value for search: {} tol: {}".format(ref_val, tol))

    I = integrand(x0+ref_val)

    if I > tol:
        log.debug("x={:.3e} I(x+ref_val) = {:.3e} > tol -> veer x away from ref_value".format(x0, I))
        x = 2*x0
        I = integrand(x + ref_val)
        while I > tol:
            log.debug("x={:.3e} I(x+ref_val) = {:.3e}".format(x, I))
            if abs(x) > max_val:
                raise RuntimeError("|x-ref_val| > max_val was reached")
            x *= 2
            I = integrand(x + ref_val)

        log.debug("x={:.3e} I(x+ref_val) = {:.3e} < tol".format(x, I))
        a = brentq(lambda x: integrand(x)-tol, x+ref_val, x0+ref_val)
        log.debug("found I(a={:.3e}) = {:.3e} = tol".format(a, integrand(a)))

    elif I < tol:
        log.debug("x={:.3e} I(x+ref_val) = {:.3e} < tol -> approach x towards ref_value".format(x0, I))
        x = x0/2
        I = integrand(x + ref_val)
        while I < tol:
            log.debug("x={:.3e} I(x+ref_val) = {:.3e}".format(x, I))
            if (1/abs(x)) > max_val:
                raise RuntimeError("1/|x-ref_val| > max_val was reached")
            x /= 2
            I = integrand(x+ref_val)

        log.debug("x={:.3e} I(x+ref_val) = {:.3e} > tol".format(x, I))
        log.debug("search for root in interval [{:.3e},{:.3e}]".format(x0+ref_val, x+ref_val))
        a = brentq(lambda x_: integrand(x_)-tol, x+ref_val, x0+ref_val)
        log.debug("found I(a={:.3e}) = {:.3e} = tol".format(a, integrand(a)))
    else:
        a = x0
        log.debug("I(ref_val) = tol -> no search necessary")

    log.debug("return a={:.5g}".format(a))
    return a

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

def fourier_integral(integrand, a, b, N):
    """
        approximates int_a^b dx integrand(x) by the riemann sum with N terms
        and the most simplest uniform midpoint weights
    """
    log.debug("integrate over [{:.3e},{:.3e}] using {} points".format(a,b,N))
    delta_x = (b-a)/N
    delta_k = 2*np.pi/(b-a)
    yl = integrand(np.linspace(a+delta_x/2, b+delta_x/2, N, endpoint=False))  
    fft_vals = np_rfft(yl)
    tau = np.arange(len(fft_vals))*delta_k
    log.debug("yields d_x={:.3e}, d_k={:.3e} kmax={:.3e}".format(delta_x, delta_k, tau[-1]))
    return tau, delta_x*np.exp(-1j*tau*(a+delta_x/2))*fft_vals
    

def get_N_for_accurate_fourier_integral(integrand, a, b, t_max, ft_ref, tol=1e-3, N_max = 2**20):
    """
        chooses N such that the approximated Fourier integral 
        meets the exact solution within a given tolerance of the
        relative deviation for a given interval of interest
    """

    log.debug("FFT accuracy for k in [0, {:.3e}] (tol={:.3e})".format(t_max, tol))

    i = 10
    while True:
        N = 2**i 
        tau, ft_tau = fourier_integral(integrand, a, b, N)
        idx = np.where(tau <= t_max)
        ft_ref_tau = ft_ref(tau[idx])
        rd = np.max(np.abs(ft_tau[idx] - ft_ref_tau) / np.abs(ft_ref_tau))
        log.debug("N:{} found rel dif of {:.3e}".format(N, rd))
        if rd < tol:
            log.debug("reached rd ({:.3e}) < tol ({:.3e}), return N={}".format(rd, tol, N))
            return N
        if N > N_max:
            raise RuntimeError("maximum number of points for Fourier Transform reached")
        
        i += 1

