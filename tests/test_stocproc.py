#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Suite for Stochastic Process Module stocproc.py
"""

import numpy as np
from scipy.special import gamma
import pickle

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    print("matplotlib not found -> any plotting will crash")

import functools
import time

import sys
import os

import pathlib
p = pathlib.PosixPath(os.path.abspath(__file__))
sys.path.insert(0, str(p.parent.parent))

import stocproc as sp
import warnings
warnings.simplefilter('default')

_S_ = 0.6
_GAMMA_S_PLUS_1 = gamma(_S_ + 1)


def corr(tau):
    """ohmic bath correlation function"""
    return (1 + 1j*(tau))**(-(_S_+1)) * _GAMMA_S_PLUS_1 / np.pi

def spectral_density(omega):
    return omega**_S_ * np.exp(-omega)

def stocproc_metatest(stp, num_samples, tol, corr, plot):
    print("generate samples")
    t = np.linspace(0, stp.t_max, 487)
    x_t_array_KLE = np.empty(shape=(num_samples, 487), dtype=np.complex128)
    for i in range(num_samples):
        stp.new_process()
        x_t_array_KLE[i] = stp(t)

    autoCorr_KLE_conj, autoCorr_KLE_not_conj = sp.tools.auto_correlation(x_t_array_KLE)

    ac_true = corr(t.reshape(487, 1) - t.reshape(1, 487))

    max_diff_conj = np.max(np.abs(ac_true - autoCorr_KLE_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))

    max_diff_not_conj = np.max(np.abs(autoCorr_KLE_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))

    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))

        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))

        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 14))
        ax[0, 0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0, 0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real, cmap="seismic")
        ax[0, 1].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0, 1].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag, cmap="seismic")

        ax[1, 0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1, 0].imshow(np.real(autoCorr_KLE_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real,
                        cmap="seismic")
        ax[1, 1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1, 1].imshow(np.imag(autoCorr_KLE_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag,
                        cmap="seismic")

        ax[2, 0].set_title(r"KLE $\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2, 0].imshow(np.real(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real,
                        cmap="seismic")
        ax[2, 1].set_title(r"KLE $\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[2, 1].imshow(np.imag(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag,
                        cmap="seismic")

        ax[3, 0].set_title(r"abs diff $\langle x(t) x^\ast(s) \rangle$")
        cax = ax[3, 0].imshow(np.log10(np.abs(autoCorr_KLE_conj - ac_true)), interpolation='none', cmap="inferno")
        fig.colorbar(cax, ax=ax[3, 0])
        ax[3, 1].set_title(r"abs diff $\langle x(t) x(s) \rangle$")
        cax = ax[3, 1].imshow(np.log10(np.abs(autoCorr_KLE_not_conj)), interpolation='none', cmap="inferno")
        fig.colorbar(cax, ax=ax[3, 1])

        plt.tight_layout()
        plt.show()

    assert max_diff_not_conj < tol
    assert max_diff_conj < tol


def test_stochastic_process_KLE_correlation_function(plot=False):
    """
        generate samples using StocPrioc_KLE class

        compare <X_t X_s> and <X_t X^ast_s> from samples with true ac functions

        no interpolation at all
    """
    t_max = 15
    num_grid_points = 101
    num_samples = 1000
    tol = 3e-2
    stp = sp.StocProc_KLE(r_tau       = corr,
                          t_max       = t_max,
                          ng_fredholm = num_grid_points,
                          ng_fac      = 4,
                          seed        = 0)
    stocproc_metatest(stp, num_samples, tol, corr, plot)


def test_stochastic_process_KLE_tol_correlation_function(plot=False):
    """
        generate samples using FFT method

        compare <X_t X_s> and <X_t X^ast_s> from samples with true ac functions

        no interpolation at all
    """

    t_max = 15
    num_samples = 1000
    tol = 3e-2
    stp = sp.StocProc_KLE_tol(tol         = 1e-2,
                              r_tau       = corr,
                              t_max       = t_max,
                              ng_fac      = 4,
                              seed        = 0)
    stocproc_metatest(stp, num_samples, tol, corr, plot)


            
def test_stochastic_process_FFT_correlation_function(plot = False):
    """
        generate samples using FFT method
        
        compare <X_t X_s> and <X_t X^ast_s> from samples with true ac functions
        
        no interpolation at all
    """

    t_max = 15
    num_samples = 1000
    tol = 3e-2
    stp = sp.StocProc_FFT_tol(spectral_density = spectral_density,
                              t_max            = t_max,
                              bcf_ref          = corr,
                              intgr_tol        = 1e-2,
                              intpl_tol        = 1e-2,
                              seed             = 0)
    stocproc_metatest(stp, num_samples, tol, corr, plot)

def test_stocproc_dump_load():
    t_max = 15
    num_grid_points = 101

    t0 = time.time()
    stp = sp.StocProc_KLE(r_tau       = corr,
                          t_max       = t_max,
                          ng_fredholm = num_grid_points,
                          ng_fac      = 4,
                          seed        = 0)
    t1 = time.time()
    dt1 = t1 - t0
    stp.new_process()
    x = stp()

    bin_data = pickle.dumps(stp)
    t0 = time.time()
    stp2 = pickle.loads(bin_data)
    t1 = time.time()
    dt2 = t1 - t0
    assert dt2 / dt1 < 0.1 # loading should be way faster

    stp2.new_process()
    x2 = stp2()
    assert np.all(x == x2)

    t0 = time.time()
    stp = sp.StocProc_KLE_tol(r_tau=corr,
                              t_max=t_max,
                              tol=1e-2,
                              ng_fac=4,
                              seed=0)
    t1 = time.time()
    dt1 = t1 - t0
    stp.new_process()
    x = stp()

    bin_data = pickle.dumps(stp)
    t0=time.time()
    stp2 = pickle.loads(bin_data)
    t1 = time.time()
    dt2 = t1 - t0
    assert dt2 / dt1 < 0.1 # loading should be way faster

    stp2.new_process()
    x2 = stp2()

    assert np.all(x == x2)

    t0 = time.time()
    stp = sp.StocProc_FFT_tol(spectral_density, t_max, corr, seed=0)
    t1 = time.time()
    dt1 = t1-t0

    stp.new_process()
    x = stp()

    bin_data = pickle.dumps(stp)
    t0 = time.time()
    stp2 = pickle.loads(bin_data)
    t1 = time.time()
    dt2 = t1 - t0

    assert dt2/dt1 < 0.1 # loading should be way faster

    stp2.new_process()
    x2 = stp2()

    assert np.all(x == x2)

def test_lorentz_SD(plot=False):
    _WC_ = 1
    def lsp(w):
        return 1/(1 + (w - _WC_)**2)# / np.pi

    def lac(t):
        return np.exp(- np.abs(t) - 1j*_WC_*t)


    t_max = 15
    num_samples = 5000
    tol = 3e-2
    stp = sp.StocProc_KLE(r_tau=lac,
                          t_max=t_max,
                          ng_fredholm=1025,
                          ng_fac=2,
                          seed=0)
    t = stp.t
    stp.new_process()
    plt.plot(t, np.abs(stp()))


    stp = sp.StocProc_FFT_tol(lsp, t_max, lac, negative_frequencies=True, seed=0, intgr_tol=1e-2, intpl_tol=1e-9)
    t = stp.t
    stp.new_process()
    plt.plot(t, np.abs(stp()), ls='', marker='.')

    stp = sp.StocProc_FFT_tol(lsp, t_max, lac, negative_frequencies=True, seed=0, intgr_tol=1e-2, intpl_tol=1e-11)
    t = stp.t
    stp.new_process()
    plt.plot(t, np.abs(stp()), ls='', marker='.')


    plt.show()
    return

    print("generate samples")
    t = np.linspace(0, stp.t_max, 487)
    t_fine = np.linspace(0, stp.t_max, 4870)

    t_ref_list = [0,1,7]



    # for num_samples in [100, 1000]:
    #     x_t0 = np.zeros(shape=(3), dtype=np.complex128)
    #     for i in range(num_samples):
    #         stp.new_process()
    #         x_t = stp(t)
    #         for j, ti in enumerate(t_ref_list):
    #             x_t0[j] += np.conj(stp(ti)) * stp(ti)
    #     x_t0 /= num_samples
    #     print(np.abs(1-x_t0))

    num_samples = 1000
    x_t_array = np.zeros(shape=(487, 3), dtype=np.complex128)
    for i in range(num_samples):
        stp.new_process()
        x_t = stp(t)
        for j, ti in enumerate(t_ref_list):
            x_t_array[:, j] += x_t * np.conj(stp(ti))

    x_t_array /= num_samples
    for j, ti in enumerate(t_ref_list):
        p, = plt.plot(t, np.abs(x_t_array[:, j]))
        plt.plot(t_fine, np.abs(lac(t_fine-ti)), color=p.get_color())

    plt.show()




    #num_samples = 1000
    #stp = sp.StocProc_FFT_tol(lsp, t_max, lac, negative_frequencies=True, seed=0, intgr_tol=1e-2, intpl_tol=1e-2)
    #stocproc_metatest(stp, num_samples, tol, lambda tau:lac(tau)/np.pi, plot)



def test_subohmic_SD(plot=False):

    def lsp(omega):
        return omega ** _S_ * np.exp(-omega)

    def lac(tau):
        return (1 + 1j*(tau))**(-(_S_+1)) * _GAMMA_S_PLUS_1 / np.pi


    t_max = 15
    stp = sp.StocProc_KLE(r_tau=lac,
                          t_max=t_max,
                          ng_fredholm=1025,
                          ng_fac=2,
                          seed=0)
    t = stp.t
    stp.new_process()
    plt.plot(t, np.abs(stp()))


    stp = sp.StocProc_FFT_tol(lsp, t_max, lac, negative_frequencies=False, seed=0, intgr_tol=1e-3, intpl_tol=1e-3)
    t = stp.t
    stp.new_process()
    plt.plot(t, np.abs(stp()))
    plt.show()
    return



    print("generate samples")
    t = np.linspace(0, stp.t_max, 487)
    t_fine = np.linspace(0, stp.t_max, 4870)

    t_ref_list = [0,1,7]

    num_samples = 1000
    x_t_array = np.zeros(shape=(487, 3), dtype=np.complex128)
    for i in range(num_samples):
        stp.new_process()
        x_t = stp(t)
        for j, ti in enumerate(t_ref_list):
            x_t_array[:, j] += x_t * np.conj(stp(ti))

    x_t_array /= num_samples
    for j, ti in enumerate(t_ref_list):
        p, = plt.plot(t, np.abs(x_t_array[:, j]))
        plt.plot(t_fine, np.abs(lac(t_fine-ti)), color=p.get_color())

    plt.show()





if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    #logging.basicConfig(level=logging.INFO)
    # test_stochastic_process_KLE_correlation_function(plot=True)
    # test_stochastic_process_FFT_correlation_function(plot=True)
    # test_stochastic_process_KLE_tol_correlation_function(plot=True)
    # test_stocproc_dump_load()


    test_lorentz_SD(plot=True)
    # test_subohmic_SD()
    pass
