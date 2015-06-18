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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import functools

import sys
import os

import pathlib
p = pathlib.PosixPath(os.path.abspath(__file__))
sys.path.insert(0, str(p.parent.parent / 'stocproc'))

import stocproc as sp

def corr(tau, s, gamma_s_plus_1):
    """ohmic bath correlation function"""
    return (1 + 1j*(tau))**(-(s+1)) * gamma_s_plus_1

def spectral_density(omega, s):
    return omega**s * np.exp(-omega)

def test_stochastic_process_KLE_correlation_function(plot=False):
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    num_grid_points = 200
    # number of samples for the stochastic process
    num_samples = 10000
    
    seed = 0
    sig_min = 0
    
    x_t_array_KLE, t = sp.stochastic_process_trapezoidal_weight(r_tau, t_max, num_grid_points, num_samples, seed, sig_min)
    autoCorr_KLE_conj, autoCorr_KLE_not_conj = sp.auto_correlation(x_t_array_KLE)
    
    t_grid = np.linspace(0, t_max, num_grid_points)
    ac_true = r_tau(t_grid.reshape(num_grid_points, 1) - t_grid.reshape(1, num_grid_points))
    
    max_diff_conj = np.max(np.abs(ac_true - autoCorr_KLE_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))
    
    max_diff_not_conj = np.max(np.abs(autoCorr_KLE_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))
    
    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))
        
        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))
        
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,12))
        ax[0,0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")        
        ax[0,0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[0,1].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0,1].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
                
        ax[1,0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,0].imshow(np.real(autoCorr_KLE_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[1,1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,1].imshow(np.imag(autoCorr_KLE_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[2,0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,0].imshow(np.real(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[2,1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,1].imshow(np.imag(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        plt.show()    
    
    assert max_diff_not_conj < 3e-2
    assert max_diff_conj < 3e-2

        
            

def test_stochastic_process_FFT_correlation_function(plot = False):
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)    
    spectral_density_omega = lambda omega : spectral_density(omega, s_param)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    num_grid_points = 256
    # number of samples for the stochastic process
    num_samples = 10000
    
    seed = 0
    
    x_t_array_FFT, t = sp.stochastic_process_fft(spectral_density_omega, t_max, num_grid_points, num_samples, seed)
    autoCorr_KLE_conj, autoCorr_KLE_not_conj = sp.auto_correlation(x_t_array_FFT)
    
    t_grid = np.linspace(0, t_max, num_grid_points)
    ac_true = r_tau(t_grid.reshape(num_grid_points, 1) - t_grid.reshape(1, num_grid_points))
    
    max_diff_conj = np.max(np.abs(ac_true - autoCorr_KLE_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))
    
    max_diff_not_conj = np.max(np.abs(autoCorr_KLE_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))
    
    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))
        
        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))
        
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,12))
        ax[0,0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")        
        ax[0,0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[0,1].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0,1].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
                
        ax[1,0].set_title(r"FFT $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,0].imshow(np.real(autoCorr_KLE_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[1,1].set_title(r"FFT $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,1].imshow(np.imag(autoCorr_KLE_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[2,0].set_title(r"FFT $\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,0].imshow(np.real(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[2,1].set_title(r"FFT $\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,1].imshow(np.imag(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        plt.show()    
    
    assert max_diff_not_conj < 3e-2
    assert max_diff_conj < 3e-2
    
def test_func_vs_class_KLE_FFT():
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    
    J = lambda w : spectral_density(w, s_param)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    ng = 200
    
    num_samples = 1
    seed = 0
    sig_min = 0

    x_t_array_func, t = sp.stochastic_process_trapezoidal_weight(r_tau, t_max, ng, num_samples, seed, sig_min)
    stoc_proc = sp.StocProc.new_instance_by_name(name    = 'trapezoidal', 
                                                 r_tau   = r_tau, 
                                                 t_max   = t_max, 
                                                 ng      = ng, 
                                                 seed    = seed, 
                                                 sig_min = sig_min)
    x_t_array_class = stoc_proc.x_for_initial_time_grid()

    print(np.max(np.abs(x_t_array_func - x_t_array_class)))
    assert np.all(x_t_array_func == x_t_array_class), "stochastic_process_kle vs. StocProc Class not identical"

    x_t_array_func, t = sp.stochastic_process_fft(spectral_density  = J,
                                                  t_max             = t_max, 
                                                  num_grid_points   = ng, 
                                                  num_samples       = num_samples, 
                                                  seed              = seed)
    
    stoc_proc = sp.StocProc_FFT(spectral_density = J,
                                t_max            = t_max,
                                num_grid_points  = ng,
                                seed             = seed)
    
    x_t_array_class = stoc_proc.get_z()    
    
    print(np.max(np.abs(x_t_array_func - x_t_array_class)))
    assert np.all(x_t_array_func == x_t_array_class), "stochastic_process_fft vs. StocProc Class not identical"

    
def test_stochastic_process_KLE_interpolation(plot=False):
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    ng = 60
    ng_fine = ng*3
    
    seed = 0
    sig_min = 1e-5
    
    stoc_proc = sp.StocProc.new_instance_by_name(name    = 'trapezoidal', 
                                                 r_tau   = r_tau, 
                                                 t_max   = t_max, 
                                                 ng      = ng, 
                                                 seed    = seed, 
                                                 sig_min = sig_min)

    
    finer_t = np.linspace(0, t_max, ng_fine)
    
    ns = 6000
    
    x_t_samples = np.empty(shape=(ns, ng_fine), dtype=np.complex)

    print("generate samples ...")
    for n in range(ns):
        stoc_proc.new_process()
        x_t_samples[n] = stoc_proc(finer_t)
    print("done!")
    ac_kle_int_conj, ac_kle_int_not_conj = sp.auto_correlation(x_t_samples)
    
    t_grid = np.linspace(0, t_max, ng_fine)
    ac_true = r_tau(t_grid.reshape(ng_fine, 1) - t_grid.reshape(1, ng_fine))
    
    max_diff_conj = np.max(np.abs(ac_true - ac_kle_int_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))
    
    max_diff_not_conj = np.max(np.abs(ac_kle_int_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))
    
    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))
        
        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,12))
        ax[0,0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")        
        ax[0,0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[0,1].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0,1].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
                
        ax[1,0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,0].imshow(np.real(ac_kle_int_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[1,1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,1].imshow(np.imag(ac_kle_int_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[2,0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,0].imshow(np.real(ac_kle_int_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[2,1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,1].imshow(np.imag(ac_kle_int_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[0,2].set_title(r"FFT log rel diff")
        im02 = ax[0,2].imshow(np.log10(np.abs(ac_kle_int_conj - ac_true) / np.abs(ac_true)), interpolation='none')
        divider02 = make_axes_locatable(ax[0,2])
        cax02 = divider02.append_axes("right", size="10%", pad=0.05)
        cbar02 = plt.colorbar(im02, cax=cax02)
        
        ax[1,2].set_title(r"FFT rel diff")
        im12 = ax[1,2].imshow(np.abs(ac_kle_int_conj - ac_true) / np.abs(ac_true), interpolation='none')
        divider12 = make_axes_locatable(ax[1,2])
        cax12 = divider12.append_axes("right", size="10%", pad=0.05)
        cbar12 = plt.colorbar(im12, cax=cax12)        
        
        ax[2,2].set_title(r"FFT abs diff")
        im22 = ax[2,2].imshow(np.abs(ac_kle_int_conj - ac_true), interpolation='none')
        divider22 = make_axes_locatable(ax[2,2])
        cax22 = divider22.append_axes("right", size="10%", pad=0.05)
        cbar22 = plt.colorbar(im22, cax=cax22)          
        
        plt.show()    
    
    assert max_diff_not_conj < 4e-2
    assert max_diff_conj < 4e-2    
    
def test_stocproc_KLE_splineinterpolation(plot=False):
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    # time interval [0,T]
    t_max = 15
    # number of subintervals
    # leads to N+1 grid points
    ng_fredholm   = 60
    ng_kle_interp = ng_fredholm*3 
    ng_fine       = ng_fredholm*30  
    
    seed = 0
    sig_min = 1e-5
    stoc_proc = sp.StocProc_KLE(r_tau, t_max, ng_kle_interp, ng_fredholm, seed, sig_min)
  
    finer_t = np.linspace(0, t_max, ng_fine)
    
    ns = 6000
    
    ac_conj     = np.zeros(shape=(ng_fine, ng_fine), dtype=np.complex)
    ac_not_conj = np.zeros(shape=(ng_fine, ng_fine), dtype=np.complex)
    print("generate samples ...")    
    for n in range(ns):
        stoc_proc.new_process()
        x_t = stoc_proc(finer_t)
        ac_conj     += x_t.reshape(ng_fine, 1) * np.conj(x_t.reshape(1, ng_fine))
        ac_not_conj += x_t.reshape(ng_fine, 1) * x_t.reshape(1, ng_fine)
    print("done!")
    ac_conj /= ns
    ac_not_conj /= ns
    
    t_grid = np.linspace(0, t_max, ng_fine)
    ac_true = r_tau(t_grid.reshape(ng_fine, 1) - t_grid.reshape(1, ng_fine))
    
    max_diff_conj = np.max(np.abs(ac_true - ac_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))
    
    max_diff_not_conj = np.max(np.abs(ac_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))
    
    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))
        
        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,12))
        ax[0,0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")        
        ax[0,0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[0,1].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0,1].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
                
        ax[1,0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,0].imshow(np.real(ac_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[1,1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,1].imshow(np.imag(ac_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[2,0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,0].imshow(np.real(ac_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[2,1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,1].imshow(np.imag(ac_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[0,2].set_title(r"FFT log rel diff")
        im02 = ax[0,2].imshow(np.log10(np.abs(ac_conj - ac_true) / np.abs(ac_true)), interpolation='none')
        divider02 = make_axes_locatable(ax[0,2])
        cax02 = divider02.append_axes("right", size="10%", pad=0.05)
        cbar02 = plt.colorbar(im02, cax=cax02)
        
        ax[1,2].set_title(r"FFT rel diff")
        im12 = ax[1,2].imshow(np.abs(ac_conj - ac_true) / np.abs(ac_true), interpolation='none')
        divider12 = make_axes_locatable(ax[1,2])
        cax12 = divider12.append_axes("right", size="10%", pad=0.05)
        cbar12 = plt.colorbar(im12, cax=cax12)        
        
        ax[2,2].set_title(r"FFT abs diff")
        im22 = ax[2,2].imshow(np.abs(ac_conj - ac_true), interpolation='none')
        divider22 = make_axes_locatable(ax[2,2])
        cax22 = divider22.append_axes("right", size="10%", pad=0.05)
        cbar22 = plt.colorbar(im22, cax=cax22)          
        
        plt.show()    
    
    assert max_diff_not_conj < 4e-2
    assert max_diff_conj < 4e-2        
    
def test_stochastic_process_FFT_interpolation(plot=False):
    s_param = 1
    gamma_s_plus_1 = gamma(s_param+1)
    # two parameter correlation function -> correlation matrix
    r_tau = lambda tau : corr(tau, s_param, gamma_s_plus_1)
    J = lambda w : spectral_density(w, s_param)
    # time interval [0,T]
    t_max = 30
    # number of subintervals
    # leads to N+1 grid points
    ng = 100
    ng_fine = ng*3
    
    seed = 0
    
    stoc_proc = sp.StocProc_FFT(spectral_density    = J,
                                t_max               = t_max, 
                                num_grid_points     = ng,
                                seed                = seed)
    
    finer_t = np.linspace(0, t_max, ng_fine)
    
    ns = 10000
    
    ac_conj     = np.zeros(shape=(ng_fine, ng_fine), dtype=np.complex)
    ac_not_conj = np.zeros(shape=(ng_fine, ng_fine), dtype=np.complex)
    print("generate samples ...")    
    for n in range(ns):
        stoc_proc.new_process()
        x_t = stoc_proc(finer_t)
        ac_conj     += x_t.reshape(ng_fine, 1) * np.conj(x_t.reshape(1, ng_fine))
        ac_not_conj += x_t.reshape(ng_fine, 1) * x_t.reshape(1, ng_fine)
    print("done!")
    ac_conj /= ns
    ac_not_conj /= ns
    
    t_grid = np.linspace(0, t_max, ng_fine)
    ac_true = r_tau(t_grid.reshape(ng_fine, 1) - t_grid.reshape(1, ng_fine))
    
    max_diff_conj = np.max(np.abs(ac_true - ac_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))
    
    max_diff_not_conj = np.max(np.abs(ac_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))
    
    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))
        
        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,12))
        fig.suptitle("ns:{}, ng:{}, ng_fine:{}".format(ns, ng, ng_fine))
        ax[0,0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")        
        ax[0,0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[0,1].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0,1].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
                
        ax[1,0].set_title(r"FFT $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,0].imshow(np.real(ac_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[1,1].set_title(r"FFT $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1,1].imshow(np.imag(ac_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[2,0].set_title(r"FFT $\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,0].imshow(np.real(ac_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real)
        ax[2,1].set_title(r"FFT $\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2,1].imshow(np.imag(ac_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag)
        
        ax[0,2].set_title(r"FFT log rel diff")
        im02 = ax[0,2].imshow(np.log10(np.abs(ac_conj - ac_true) / np.abs(ac_true)), interpolation='none')
        divider02 = make_axes_locatable(ax[0,2])
        cax02 = divider02.append_axes("right", size="10%", pad=0.05)
        cbar02 = plt.colorbar(im02, cax=cax02)
        
        ax[1,2].set_title(r"FFT rel diff")
        im12 = ax[1,2].imshow(np.abs(ac_conj - ac_true) / np.abs(ac_true), interpolation='none')
        divider12 = make_axes_locatable(ax[1,2])
        cax12 = divider12.append_axes("right", size="10%", pad=0.05)
        cbar12 = plt.colorbar(im12, cax=cax12)        
        
        ax[2,2].set_title(r"FFT abs diff")
        im22 = ax[2,2].imshow(np.abs(ac_conj - ac_true), interpolation='none')
        divider22 = make_axes_locatable(ax[2,2])
        cax22 = divider22.append_axes("right", size="10%", pad=0.05)
        cbar22 = plt.colorbar(im22, cax=cax22)        
        
#         fig2, ax2 = plt.subplots(nrows=1, ncols=1)
#       
#         
#         i = 30
#         tau = t_grid - t_grid[i]
#         
#         ax2.plot(t_grid, np.abs(r_tau(tau)))
#         
#         ax2.plot(t_grid, np.abs(np.mean(x_t_samples*np.conj(x_t_samples[:,i].reshape(ns,1)), axis=0)))
        
        plt.show()    
    
    assert max_diff_not_conj < 4e-2
    assert max_diff_conj < 4e-2    

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
    tol = 1e-8
    
    ng = sp.auto_grid_points(r_tau, t_max, ng_interpolation, tol, sig_min=0)
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
            stocproc(t_i)
        
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
    t_max = 15
    ng_interpolation = 1000
    tol = 1e-8
    seed = None
    sig_min = 0
    
    t_large = np.linspace(0, t_max, ng_interpolation)
    
    name = 'mid_point'
#     name = 'trapezoidal'
#     name = 'gauss_legendre'
    
    ng = sp.auto_grid_points(r_tau, t_max, ng_interpolation, tol, name=name, sig_min=sig_min)

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
#     test_stochastic_process_KLE_correlation_function(plot=False)
#     test_stochastic_process_FFT_correlation_function(plot=False)
#     test_func_vs_class_KLE_FFT()
#     test_stochastic_process_KLE_interpolation(plot=False)
#     test_stocproc_KLE_splineinterpolation(plot=False)
#     test_stochastic_process_FFT_interpolation(plot=False)
#     test_stocProc_eigenfunction_extraction()
    test_orthonomality()
    test_auto_grid_points()
    show_auto_grid_points_result()
    test_chache()
    test_dump_load()
    pass