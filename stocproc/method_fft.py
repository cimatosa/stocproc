
from scipy.optimize import brentq
from numpy.fft import rfft as np_rfft
import numpy as np
from math import fsum

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
       
    # find the left boundary called a
    if integrand(x0+ref_val) > tol:
        x = 2*x0
        while integrand(x+ref_val) > tol:
            if abs(x-ref_val) > max_val:
                raise RuntimeError("|x-ref_val| > max_val was reached")
            x *= 2            
        a = brentq(lambda x: integrand(x)-tol, x+ref_val, x0+ref_val)
    elif integrand(x0+ref_val) < tol:
        x = x0/2
        while integrand(x+ref_val) < tol:
            if (1/abs(x-ref_val)) > max_val:
                raise RuntimeError("1/|x-ref_val| > max_val was reached")
            x /= 2
        a = brentq(lambda x: integrand(x)-tol, x+ref_val, x0+ref_val)
    else:
        a = x0
    return a

def find_integral_boundary_auto(integrand, tol, ref_val=0, max_val=1e6, 
                                ref_val_left=None, ref_val_right=None, 
                                max_val_left=None, max_val_right=None):
    
    ref_val_left  = ref_val if ref_val_left  is None else ref_val_left
    ref_val_right = ref_val if ref_val_right is None else ref_val_right
    max_val_left  = max_val if max_val_left  is None else max_val_left
    max_val_right = max_val if max_val_right is None else max_val_right
    
    a = find_integral_boundary(integrand, tol, ref_val=ref_val_left,  max_val=max_val_left,  x0=-1)
    b = find_integral_boundary(integrand, tol, ref_val=ref_val_right, max_val=max_val_right, x0=+1)
    return a,b

def fourier_integral(integrand, a, b, N):
    """
        approximates int_a^b dx integrand(x) by the riemann sum with N terms
        
    """
    delta_x = (b-a)/N
    delta_k = 2*np.pi/(b-a)    
    yl = integrand(np.linspace(a+delta_x/2, b+delta_x/2, N, endpoint=False))  
    fft_vals = np_rfft(yl)
    tau = np.arange(len(fft_vals))*delta_k
    return tau, delta_x*np.exp(-1j*tau*(a+delta_x/2))*fft_vals
    

def get_N_for_accurate_fourier_integral(integrand, a, b, t_0, t_max, tol, ft_ref, N_max = 2**15):
    """
        chooses N such that the approximated Fourier integral 
        meets the exact solution within a given tolerance of the
        relative deviation for a given interval of interest
    """
    i = 10
    while True:
        N = 2**i 
        tau, ft_tau = fourier_integral(integrand, a, b, N)
        idx = np.where(tau <= t_max)
        ft_ref_tau = ft_ref(tau[idx])
        rd = np.abs(ft_tau[idx] - ft_ref_tau) / np.abs(ft_ref_tau)
        if rd < tol:
            return N
        
        i += 2    


    
        
        
    
    