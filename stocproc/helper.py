# -*- coding: utf8 -*-
from __future__ import print_function, division

import numpy as np
from scipy.optimize import brentq
from . import class_stocproc 

def get_param_single_lorentz(tmax, dw_max, eta, gamma, wc, x=1e-4, verbose=0):
    d = gamma * np.sqrt(1/x - 1)
    w_min = wc - d
    w_max = wc + d
    
    if verbose > 0:
        print('w_min :{:.3}'.format(w_min))
        print('w_max :{:.3}'.format(w_max))
    
    C = (w_max - w_min)*tmax / 2 / np.pi
    
    N = int(np.ceil((2 + C)/2 + np.sqrt( (2+C)**2 / 4 - 1)))
    dw = (w_max - w_min)/N
    if verbose > 0:
        print('N: {}'.format(N))
        print('-> dw: {:.3}'.format(dw))
    
    if dw <= dw_max:
        if verbose > 0:
            print('dw <= dw_max: {:.3}'.format(dw_max))
        return N, w_min, tmax
    else:
        if verbose > 0:
            print('dw > dw_max: {:.3}'.format(dw_max))
            print('adjust tmax and N to fulfill dw <= dw_max')
        N = int(np.ceil((w_max - w_min) / dw_max)) - 1
        dt = 2*np.pi / (dw_max*N)
        tmax_ = dt*N
        if verbose > 0:
            print('N: {}'.format(N))
            print('-> tmax: {:.3}'.format(tmax_))
        assert tmax_ > tmax, "tmax_={} > tmax={} FAILED".format(tmax_, tmax)
        return N, w_min, tmax

def get_param_ohmic(t_max, spec_dens, x=1e-12, verbose=0):
    fmin_spec_dens = lambda w: abs(spec_dens(w)) - spec_dens.maximum_val()*x
    w_pos = spec_dens.maximum_at()
    w_neg = 2*w_pos
    while fmin_spec_dens(w_neg) > 0:
        w_neg *= 2

    omega_max = brentq(fmin_spec_dens, w_pos, w_neg)
    if verbose > 0:
        print("omega_max from threshold condition: J(w_max) = x = {:.3g} <-> w_max = {:.3g}".format(x, omega_max))

    dw = np.pi / t_max
    
    if verbose > 0:
        print("dw:{:.3}".format(dw))

    ng_omega = np.ceil(omega_max / dw)  # next larger integer
    ng_omega = np.ceil(ng_omega / 2) * 2 - 1                   # next lager odd integer
    ng_t     = (ng_omega + 1) / 2                              # so this becomes an integer
    delta_t  = 2 * np.pi / (dw * ng_omega)
    sp_t_max    = ng_t * delta_t
    if verbose > 0:
        print("result ng_t: {}".format(ng_t))
        print("result sp_t_max: {:.3g}".format(sp_t_max))
    
    return ng_t, sp_t_max

def show_ohmic_sp(ng_t, sp_t_max, spec_dens, seed, ax, t_max):
    try:
        n = len(ng_t)
    except:
        n = None
        
    try:
        m = len(sp_t_max)
    except:
        m = None
        
    if (n is None) and m is not None:
        ng_t = [ng_t] * m
    elif (n is not None) and m is None:
        sp_t_max = [sp_t_max] * n
    elif (n is not None) and (m is not None):
        if n != m:
            raise ValueError("len(ng_t) == len(sp_t_max) FAILED")
    else:
        ng_t = [ng_t]
        sp_t_max = [sp_t_max]
        
    
    for i in range(len(ng_t)):
        spfft = class_stocproc.StocProc_FFT(spectral_density = spec_dens, 
                                            t_max            = sp_t_max[i], 
                                            num_grid_points  = ng_t[i],
                                            seed             = seed)
        
        spfft.new_process()
    
        
    
        t = np.linspace(0, sp_t_max[i], ng_t[i])
        t_interp = np.linspace(0, sp_t_max[i], 10*ng_t[i])
        
        t = t[np.where(t < t_max)]
        t_interp = t_interp[np.where(t_interp < t_max)]
        
        eta = spfft(t)
        eta_interp = spfft(t_interp)
        
        p, = ax.plot(t_interp, np.real(eta_interp))
        ax.plot(t_interp, np.imag(eta_interp), color=p.get_color(), ls='--')
        ax.plot(t, np.real(eta), color=p.get_color(), ls='', marker = '.')
        ax.plot(t, np.imag(eta), color=p.get_color(), ls='', marker = '.')
    
    
    