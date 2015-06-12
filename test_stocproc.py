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
"""Test Suite for Stochastic Process Module stocproc.py
"""

import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d
import time as tm
import pickle as pc
import matplotlib.pyplot as plt
import functools

import sys
import os

path = os.path.dirname(__file__)
sys.path.append(path)

import stocproc as sp

def corr(tau, s, gamma_s_plus_1):
    """ohmic bath correlation function"""
    return (1 + 1j*(tau))**(-(s+1)) * gamma_s_plus_1

def spectral_density(omega, s):
    return omega**s * np.exp(-omega)

def test_stochastic_process_KLE_correlation_function():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 30
    # number of subintervals
    # leads to N+1 grid points
    num_grid_points = 300
    # number of samples for the stochastic process
    num_samples = 100000
    
    seed = 0
    sig_min = 1e-4
    s_0_idx = 0
    
    x_t_array_KLE, t = sp.stochastic_process_trapezoidal_weight(r_tau, t_max, num_grid_points, num_samples, seed, sig_min)
    autoCorr_KLE = sp.auto_correlation(x_t_array_KLE, s_0_idx)
    tau = t - t[s_0_idx]
    c = r_tau(tau)
    
    plt.plot(tau, np.real(autoCorr_KLE), color='b')
    plt.plot(tau, np.imag(autoCorr_KLE), color='r')
    plt.plot(tau, np.real(c), color='b', ls='--')
    plt.plot(tau, np.imag(c), color='r', ls='--')
    
    plt.grid()
    plt.show()    

    max_diff = np.max(np.abs(c - autoCorr_KLE)) 
    assert max_diff < 1e-2, "KLE max diff: {}".format(max_diff)

def test_stochastic_process_FFT_correlation_function():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)    
    spectral_density_omega = lambda omega : spectral_density(omega, s_param)
    # time interval [0,T]
    t_max = 20
    # number of subintervals
    # leads to N+1 grid points
    num_grid_points = 300
    # number of samples for the stochastic process
    num_samples = 10000
    
    seed = 0
    sig_min = 1e-3
    s_0_idx = num_grid_points/3
    
    x_t_array_FFT, t = sp.stochastic_process_fft(spectral_density_omega, t_max, num_grid_points, num_samples, seed)
    autoCorr_FFT = sp.auto_correlation(x_t_array_FFT, s_0_idx)
    tau = t - t[s_0_idx]
    c = r_tau(tau)
    
    plt.plot(tau, np.real(autoCorr_FFT), color='b')
    plt.plot(tau, np.imag(autoCorr_FFT), color='r')
    plt.plot(tau, np.real(c), color='b', ls='--')
    plt.plot(tau, np.imag(c), color='r', ls='--')
    
    plt.grid()
    plt.show()
    
  
    #max_diff = np.max(np.abs(c - autoCorr_FFT))
    #assert max_diff < 1e-2, "FFT max diff: {}".format(max_diff)
    
def test_stochastic_process_KLE_interpolation():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    ng = 100
    
    num_samples = 1
    seed = 0
    sig_min = 1e-4
    s_0_idx = 0

    x_t_array_1, t = sp.stochastic_process_trapezoidal_weight(r_tau, t_max, ng, num_samples, seed, sig_min)
    w = sp.get_trapezoidal_weights_times(t_max, ng)[1]
    stoc_proc = sp.StocProc(r_tau, t, w, seed, sig_min)
    x_t_array_2 = stoc_proc.x_t_array(t)
    x_t_array_3 = stoc_proc.x_for_initial_time_grid()

    assert np.allclose(x_t_array_1, x_t_array_2), "stochastic_process_kle vs. StocProc Class, TEST FAILED"
    assert np.all(x_t_array_1 == x_t_array_3), "StocProc Class: interpolation at grid points vs. on grid point values from eigenfunctions, TEST FAILED"
    
    finer_t = np.linspace(0, t[-1], len(t)+1)
    
    x_t_array_fine = stoc_proc.x_t_array(finer_t)
    
    for i, t_i in enumerate(finer_t):
        assert stoc_proc.x(t_i) == x_t_array_fine[i], "StocProc Class: single time value interpolation vs. time array interpolation"

def test_stocProc_eigenfunction_extraction():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    ng = 10
    
    seed = 0
    sig_min = 1e-4

    t, w = sp.get_trapezoidal_weights_times(t_max, ng)
    stoc_proc = sp.StocProc(r_tau, t, w, seed, sig_min)
    
    t_large = np.linspace(t[0], t[-1], int(8.7*ng))
    ui_all = stoc_proc.u_i_all(t_large)
    
    for i in range(ng):
        ui = stoc_proc.u_i(t_large, i)
        assert np.max(np.abs(ui - ui_all[:,i])) < 1e-15 
        
def test_orthonomality():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    ng = 30
    
    seed = 0
    sig_min = 1e-4

    t, w = sp.get_trapezoidal_weights_times(t_max, ng)
    stoc_proc = sp.StocProc(r_tau, t, w, seed, sig_min)

    # check integral norm of eigenfunctions (non interpolated eigenfunctions)    
    ev = stoc_proc.eigen_vector_i_all()
    max_diff = np.max(np.abs(1 - np.sum(w.reshape(ng,1) * ev * np.conj(ev), axis = 0)))
    assert max_diff < 1e-14, "max_diff {}".format(max_diff) 
   
    
    # check integral norm of interpolated eigenfunctions and orthonomality
    t_large = np.linspace(t[0], t[-1], int(8.7*ng))
    ui_all = stoc_proc.u_i_all(t_large)

    # scale first an last point to end up with trapezoidal integration weights    
    ui_all[ 0,:] /= np.sqrt(2)
    ui_all[-1,:] /= np.sqrt(2)
    
    # does the summation for all pairs of functions (i,j)
    # multiply by Delta t gives the integral values
    # so for an orthonomal set this should lead to the unity matrix 
    f = np.tensordot(ui_all, np.conj(ui_all), axes = ([0],[0])) * (t_large[1]-t_large[0])
    diff = np.abs(np.diag(np.ones(ng)) - f)
    
    diff_assert = 0.1
    idx1, idx2 = np.where(diff > diff_assert)
    
    if len(idx1) > 0:
        print("orthonomality test FAILED at:")
        for i in range(len(idx1)):
            print("    ({}, {}) diff to unity matrix: {}".format(idx1[i],idx2[i], diff[idx1[i],idx2[i]]))
        raise Exception("test_orthonomality FAILED!")
    
def test_auto_grid_points():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    ng_interpolation = 1000
    tol = 1e-16
    
    ng = sp.auto_grid_points(r_tau, t_max, ng_interpolation, tol)
    print(ng)
    
def test_chache():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    
    t_max = 10
    ng = 50
    seed = 0
    sig_min = 1e-8
     
    stocproc = sp.StocProc.new_instance_with_trapezoidal_weights(r_tau, t_max, ng, seed, sig_min)
    
    t = {}
    t[1] = 3
    t[2] = 4
    t[3] = 5
    
    total = 0
    misses = len(t.keys())
    for t_i in t.keys():
        for i in range(t[t_i]):
            total += 1
            stocproc.x(t_i)
        
    ci = stocproc.get_cache_info()
    assert ci.hits == total - misses
    assert ci.misses == misses
    
def test_dump_load():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    r_tau = functools.partial(corr, s=s_param, gamma_s_plus_1=gamma_s_plus_1) 
    
    t_max = 10
    ng = 50
    seed = 0
    sig_min = 1e-8
     
    stocproc = sp.StocProc.new_instance_with_trapezoidal_weights(r_tau, t_max, ng, seed, sig_min)
    
    t = np.linspace(0,4,30)
    
    x_t = stocproc.x_t_array(t)
    
    fname = 'test_stocproc.dump'
    
    stocproc.save_to_file(fname)
    
    stocproc_2 = sp.StocProc(seed = seed, fname = fname)
    x_t_2 = stocproc_2.x_t_array(t)
    
    assert np.all(x_t == x_t_2)
    
    
    
    
def show_auto_grid_points_result():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 3
    ng_interpolation = 1000
    tol = 1e-8
    seed = None
    sig_min = 0
    
    t_large = np.linspace(0, t_max, ng_interpolation)
    
    name = 'mid_point'
    name = 'trapezoidal'
    name = 'gauss_legendre'
    
    ng = sp.auto_grid_points(r_tau, t_max, ng_interpolation, tol, name=name)

    t, w = sp.get_trapezoidal_weights_times(t_max, ng)
    stoc_proc = sp.StocProc(r_tau, t, w, seed, sig_min)
    r_t_s = stoc_proc.recons_corr(t_large)
    
    r_t_s_exact = r_tau(t_large.reshape(ng_interpolation,1) - t_large.reshape(1, ng_interpolation))
    
    diff = sp._mean_error(r_t_s, r_t_s_exact)
    diff_max = sp._max_error(r_t_s, r_t_s_exact)

    plt.plot(t_large, diff)
    plt.plot(t_large, diff_max)
    plt.yscale('log')
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
#     test_stochastic_process_KLE_correlation_function()
    test_stochastic_process_FFT_correlation_function()
#     test_stochastic_process_KLE_interpolation()
#     test_stocProc_eigenfunction_extraction()
#     test_orthonomality()
#     test_auto_grid_points()
#     show_auto_grid_points_result()
#     test_chache()
#     test_dump_load()
    pass