import sys
import os

import numpy as np
import math
from scipy.special import gamma as gamma_func
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
from math import fsum

import pathlib
p = pathlib.PosixPath(os.path.abspath(__file__))
sys.path.insert(0, str(p.parent.parent))

import stocproc as sp

import logging
logging.basicConfig(level=logging.DEBUG)



def test_find_integral_boundary():
    def f(x):
        return np.exp(-(x)**2)
    
    tol = 1e-10
    b = sp.method_fft.find_integral_boundary(integrand=f, tol=tol, ref_val=0, x0=+1, max_val=1e6)
    a = sp.method_fft.find_integral_boundary(integrand=f, tol=tol, ref_val=0, x0=-1, max_val=1e6)
    assert a != b
    assert abs(f(a)-tol) < 1e-14
    assert abs(f(b)-tol) < 1e-14
    
    b = sp.method_fft.find_integral_boundary(integrand=f, tol=tol, ref_val=0, x0=b+5, max_val=1e6)
    a = sp.method_fft.find_integral_boundary(integrand=f, tol=tol, ref_val=0, x0=a-5, max_val=1e6)
    assert a != b
    assert abs(f(a)-tol) < 1e-14
    assert abs(f(b)-tol) < 1e-14    
    
    def f2(x):
        return np.exp(-(x)**2)*x**2
    
    tol = 1e-10
    b = sp.method_fft.find_integral_boundary(integrand=f2, tol=tol, ref_val=1, x0=+1, max_val=1e6)
    a = sp.method_fft.find_integral_boundary(integrand=f2, tol=tol, ref_val=-1, x0=-1, max_val=1e6)
    assert a != b
    assert abs(f2(a)-tol) < 1e-14
    assert abs(f2(b)-tol) < 1e-14
    
    b = sp.method_fft.find_integral_boundary(integrand=f2, tol=tol, ref_val=1, x0=b+5, max_val=1e6)
    a = sp.method_fft.find_integral_boundary(integrand=f2, tol=tol, ref_val=-1, x0=a-5, max_val=1e6)
    assert a != b
    assert abs(f2(a)-tol) < 1e-14, "diff {}".format(abs(f2(a)-tol))
    assert abs(f2(b)-tol) < 1e-14, "diff {}".format(abs(f2(b)-tol)) 
    
    def f3(x):
        return np.exp(-(x-5)**2)*x**2
    
    tol = 1e-10
    b = sp.method_fft.find_integral_boundary(integrand=f3, tol=tol, ref_val=5, x0=+1, max_val=1e6)
    a = sp.method_fft.find_integral_boundary(integrand=f3, tol=tol, ref_val=5, x0=-1, max_val=1e6)
    assert a != b
    assert abs(f3(a)-tol) < 1e-14
    assert abs(f3(b)-tol) < 1e-14
    
    b = sp.method_fft.find_integral_boundary(integrand=f3, tol=tol, ref_val=5, x0=b+5, max_val=1e6)
    a = sp.method_fft.find_integral_boundary(integrand=f3, tol=tol, ref_val=5, x0=a-5, max_val=1e6)
    assert a != b
    assert abs(f3(a)-tol) < 1e-14, "diff {}".format(abs(f3(a)-tol))
    assert abs(f3(b)-tol) < 1e-14, "diff {}".format(abs(f3(b)-tol))
    
def fourier_integral_trapz(integrand, a, b, N):
    """
        approximates int_a^b dx integrand(x) by the riemann sum with N terms

        this function is here and not in method_fft because it has almost no
        advantage over the modpoint method. so only for testing purposes.
    """
    yl = integrand(np.linspace(a, b, N+1, endpoint=True))
    yl[0] = yl[0]/2
    yl[-1] = yl[-1]/2
    
    delta_x = (b-a)/N
    delta_k = 2*np.pi*N/(b-a)/(N+1)
    
    fft_vals = np.fft.rfft(yl)
    tau = np.arange(len(fft_vals))*delta_k

    return tau, delta_x*np.exp(-1j*tau*a)*fft_vals

def fourier_integral_simple_test(integrand, a, b, N):
    delta_x = (b-a)/N
    delta_k = 2*np.pi/(b-a)

    x = np.linspace(a, b, N, endpoint = False) + delta_x/2
    k = np.arange(N//2+1)*delta_k
    
    k_np = 2*np.pi*np.fft.rfftfreq(N, delta_x)
    
    kdif = np.max(np.abs(k_np[1:] - k[1:])/k[1:])
    if kdif > 1e-15:
        print("WARNING |rfftfreq - k| = {}".format(kdif))
    
    yl = integrand(x)
    res = np.empty(shape=(N//2+1,), dtype=np.complex128)
    for i in range(N//2+1):
        tmp = yl*np.exp(-1j*x*k[i])
        res[i] = delta_x*(math.fsum(tmp.real) + 1j*math.fsum(tmp.imag))
        
    return k, res

def fourier_integral_trapz_simple_test(integrand, a, b, N):
    delta_x = (b-a)/N
    delta_k = 2*np.pi*N/(b-a)/(N+1)
    
    x = np.linspace(a, b, N+1, endpoint = True)
    k = np.arange((N+1)//2+1)*delta_k
       
    yl = integrand(x)
    yl[0] = yl[0]/2
    yl[-1] = yl[-1]/2
    
    res = np.empty(shape=((N+1)//2+1,), dtype=np.complex128)
    for i in range((N+1)//2+1):
        tmp = yl*np.exp(-1j*x*k[i])
        res[i] = delta_x*(math.fsum(tmp.real) + 1j*math.fsum(tmp.imag))
        
    return k, res
        
def test_fourier_integral_finite_boundary():
    intg = lambda x: x**2
    a = -1.23
    b = 4.87
      
    ft_ref = lambda k: (np.exp(-1j*a*k)*(2j - a*k*(2 + 1j*a*k)) + np.exp(-1j*b*k)*(-2j + b*k*(2 + 1j*b*k)))/k**3
    N = 2**18
    N_test = 100
    tau, ft_n = sp.method_fft.fourier_integral_midpoint(intg, a, b, N)
    ft_ref_n = ft_ref(tau)
    tau = tau[1:N_test]
    ft_n = ft_n[1:N_test]
    ft_ref_n = ft_ref_n[1:N_test]
    rd = np.max(np.abs(ft_n - ft_ref_n)/np.abs(ft_ref_n))
#     print(rd) 
    assert rd < 4e-6
      
    N = 2**18
    N_test = 100
    tau, ft_n = fourier_integral_trapz(intg, a, b, N)
    ft_ref_n = ft_ref(tau)
    tau = tau[1:N_test]
    ft_n = ft_n[1:N_test]
    ft_ref_n = ft_ref_n[1:N_test]
     
    rd = np.max(np.abs(ft_n - ft_ref_n)/np.abs(ft_ref_n))
#     print(rd)
    assert rd < 4e-6
    
    
    N = 512
    tau, ft_n = sp.method_fft.fourier_integral_midpoint(intg, a, b, N)
    ft_ref_n = ft_ref(tau)
   
    k, ft_simple = fourier_integral_simple_test(intg, a, b, N)
    assert np.max(np.abs(ft_simple-ft_n)) < 1e-11
    
#     plt.plot(np.abs(ft_simple-ft_n))
#     plt.yscale('log')
#     plt.show()
    
    
    N = 512
    tau, ft_n = fourier_integral_trapz(intg, a, b, N)
    ft_ref_n = ft_ref(tau)
   
    k, ft_simple = fourier_integral_trapz_simple_test(intg, a, b, N)
    assert np.max(np.abs(ft_simple-ft_n)) < 1e-11
    
#     plt.plot(np.abs(ft_simple-ft_n))
#     plt.yscale('log')
#     plt.show()    
#     sys.exit()



    N = 1024
    tau, ft_n = sp.method_fft.fourier_integral_midpoint(intg, a, b, N)
    ft_ref_n = ft_ref(tau)
    rd = np.abs(ft_ref_n-ft_n) / np.abs(ft_ref_n)
    idx = np.where(np.logical_and(tau < 75, np.isfinite(rd)))
    tau = tau[idx]
    rd = rd[idx]
    plt.plot(tau, rd, label='trapz N:{}'.format(N))
    mrd_trapz = np.max(rd)
     
    N = 513
    tau, ft_n = sp.method_fft.fourier_integral_simps(intg, a, b, N)
    ft_ref_n = ft_ref(tau)
    rd = np.abs(ft_ref_n-ft_n) / np.abs(ft_ref_n)
    idx = np.where(np.logical_and(tau < 75, np.isfinite(rd)))    
    tau = tau[idx]
    rd = rd[idx]
    plt.plot(tau, rd, label='simps N:{}'.format(N))
    mrd_simps = np.max(rd)    
    
    assert mrd_simps < mrd_trapz, "mrd_simps ({:.3e}) >= mrd_trapz ({:.3e})".format(mrd_simps, mrd_trapz)
        
#     plt.grid()
#     plt.legend(loc='lower right')
#     plt.yscale('log')
#     plt.show()
    

def osd(w, s, wc):
    if not isinstance(w, np.ndarray):
        if w < 0:
            return 0
        else:
            return w**s*np.exp(-w/wc)
    else:
        res = np.zeros(shape=w.shape)
        
        w_flat = w.flatten()
        idx_pos = np.where(w_flat > 0)
        fv_res = res.flat
        fv_res[idx_pos] = w_flat[idx_pos]**s*np.exp(-w_flat[idx_pos]/wc)
        
        return res    

    
def test_fourier_integral_infinite_boundary():
    s = 0.5
    wc = 4
    intg = lambda x: osd(x, s, wc)
    bcf_ref = lambda t:  gamma_func(s + 1) * wc**(s+1) * (1 + 1j*wc * t)**(-(s+1))
    
#     w = np.linspace(0, 10, 1500)
#     plt.plot(w, intg(w))
#     plt.show()
#     sys.exit()
    
    a,b = sp.method_fft.find_integral_boundary_auto(integrand=intg, tol=1e-12, ref_val=1)
    print(a,b)

    for N in [2**16, 2**18, 2**20]:
    
        tau, bcf_n = sp.method_fft.fourier_integral_midpoint(intg, a, b, N=N)
        bcf_ref_n = bcf_ref(tau)
        
        tau_max = 5
        idx = np.where(tau <= tau_max)
        tau = tau[idx]
        bcf_n = bcf_n[idx]
        bcf_ref_n = bcf_ref_n[idx]
        
        rd_trapz = np.abs(bcf_ref_n-bcf_n)/np.abs(bcf_ref_n)
        p, = plt.plot(tau, rd_trapz, label="trapz N {}".format(N))
        
        
        tau, bcf_n = sp.method_fft.fourier_integral_simps(intg, a, b=b, N=N-1)
        bcf_ref_n = bcf_ref(tau)
         
        idx = np.where(tau <= tau_max)
        
        tau = tau[idx]
        bcf_n = bcf_n[idx]
        bcf_ref_n = bcf_ref_n[idx]
         
        rd = np.abs(bcf_ref_n-bcf_n)/np.abs(bcf_ref_n)
        plt.plot(tau, rd, label="simps N {}".format(N), color=p.get_color(), ls='--')
        
        t_ = 3
        print(a,b)
        
        x_simps, dx = np.linspace(a,b,N-1, endpoint=True, retstep=True)
        I = sp_int.simps(intg(x_simps)*np.exp(-1j*x_simps*t_), dx=dx)
        err = np.abs(I-bcf_ref(t_))/np.abs(bcf_ref(t_))
        plt.plot(t_, err, marker='o', color='g')
  
    assert np.max(rd_trapz) < 5*1e-4, "max rd_trapz = {:.3e}".format(np.max(rd_trapz))
    
#     plt.legend(loc='lower right')
#     plt.grid()    
#     plt.yscale('log')
#     plt.show()
    
def test_get_N_for_accurate_fourier_integral():
    s = 0.5
    wc = 4
    intg = lambda x: osd(x, s, wc)
    bcf_ref = lambda t:  gamma_func(s + 1) * wc**(s+1) * (1 + 1j*wc * t)**(-(s+1))
    
    a,b = sp.method_fft.find_integral_boundary_auto(integrand=intg, tol=1e-12, ref_val=1)
    N = sp.method_fft.get_N_for_accurate_fourier_integral(intg, a, b, t_max=40, tol=1e-3, ft_ref=bcf_ref, N_max = 2**20, method='simps')
    print(N)

def test_get_dt_for_accurate_interpolation():
    s = 0.5
    wc = 4
    intg = lambda x: osd(x, s, wc)
    bcf_ref = lambda t:  gamma_func(s + 1) * wc**(s+1) * (1 + 1j*wc * t)**(-(s+1))
    dt = sp.method_fft.get_dt_for_accurate_interpolation(t_max=40, tol=1e-4, ft_ref=bcf_ref)
    print(dt)
    
def test_sclicing():
    yl = np.ones(10, dtype=int)
    yl = sp.method_fft.get_fourier_integral_simps_weighted_values(yl)
    assert yl[0] == 2/6
    assert yl[1] == 8/6
    assert yl[2] == 4/6
    assert yl[3] == 8/6
    assert yl[4] == 4/6
    assert yl[5] == 8/6
    assert yl[6] == 4/6
    assert yl[7] == 8/6
    assert yl[8] == 5/6
    assert yl[9] == 3/6
    
def test_calc_abN():
    s = 0.5
    wc = 4
    intg = lambda x: osd(x, s, wc)
    bcf_ref = lambda t:  gamma_func(s + 1) * wc**(s+1) * (1 + 1j*wc * t)**(-(s+1))
    
    tol = 1e-3
    tmax=40

    a,b = sp.method_fft.find_integral_boundary_auto(integrand=intg, tol=1e-12, ref_val=1)    
    ab, N, dx, dt = sp.method_fft.calc_ab_N_dx_dt(integrand = intg, 
                                                  intgr_tol = tol, 
                                                  intpl_tol = tol, 
                                                  tmax = tmax, 
                                                  a    = 0, 
                                                  b    = b, 
                                                  ft_ref = bcf_ref)
    assert (np.abs(dx*dt*N - np.pi*2)) < 1e-15
    
    
    
if __name__ == "__main__":
    test_find_integral_boundary()
    test_fourier_integral_finite_boundary()
    test_fourier_integral_infinite_boundary()
    test_get_N_for_accurate_fourier_integral()
    test_get_dt_for_accurate_interpolation()
    test_sclicing()
    test_calc_abN()