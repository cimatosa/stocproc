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
    return (1 + 1j * (tau)) ** (-(_S_ + 1)) * _GAMMA_S_PLUS_1 / np.pi
def spectral_density(omega):
    return omega ** _S_ * np.exp(-omega)

_WC_ = 2
def lac(t):
    """lorenzian bath correlation function"""
    return np.exp(- np.abs(t) - 1j * _WC_ * t)

def lsd(w):
    return 1 / (1 + (w - _WC_) ** 2)


def stocproc_metatest(stp, num_samples, tol, corr, plot):
    print("generate samples")
    N = 287
    t = np.linspace(0, stp.t_max, N)
    x_t_array_KLE = np.empty(shape=(num_samples, N), dtype=np.complex128)
    for i in range(num_samples):
        stp.new_process()
        x_t_array_KLE[i] = stp(t)

    autoCorr_KLE_conj, autoCorr_KLE_not_conj = sp.tools.auto_correlation(x_t_array_KLE)

    ac_true = corr(t.reshape(N, 1) - t.reshape(1, N))

    max_diff_conj = np.max(np.abs(ac_true - autoCorr_KLE_conj))
    print("max diff <x(t) x^ast(s)>: {:.2e}".format(max_diff_conj))

    max_diff_not_conj = np.max(np.abs(autoCorr_KLE_not_conj))
    print("max diff <x(t) x(s)>: {:.2e}".format(max_diff_not_conj))

    if plot:
        v_min_real = np.floor(np.min(np.real(ac_true)))
        v_max_real = np.ceil(np.max(np.real(ac_true)))

        v_min_imag = np.floor(np.min(np.imag(ac_true)))
        v_max_imag = np.ceil(np.max(np.imag(ac_true)))

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 10))
        ax[0, 0].set_title(r"exact $\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0, 0].imshow(np.real(ac_true), interpolation='none', vmin=v_min_real, vmax=v_max_real, cmap="seismic")
        ax[1, 0].set_title(r"exact $\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1, 0].imshow(np.imag(ac_true), interpolation='none', vmin=v_min_imag, vmax=v_max_imag, cmap="seismic")

        ax[0, 1].set_title(r"$\mathrm{re}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[0, 1].imshow(np.real(autoCorr_KLE_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real,
                        cmap="seismic")
        ax[1, 1].set_title(r"$\mathrm{im}\left(\langle x(t) x^\ast(s) \rangle\right)$")
        ax[1, 1].imshow(np.imag(autoCorr_KLE_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag,
                        cmap="seismic")

        ax[0, 2].set_title(r"$\mathrm{re}\left(\langle x(t) x(s) \rangle\right)$")
        ax[0, 2].imshow(np.real(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_real, vmax=v_max_real,
                        cmap="seismic")
        ax[1, 2].set_title(r"$\mathrm{im}\left(\langle x(t) x(s) \rangle\right)$")
        ax[1, 2].imshow(np.imag(autoCorr_KLE_not_conj), interpolation='none', vmin=v_min_imag, vmax=v_max_imag,
                        cmap="seismic")

        ax[0, 3].set_title(r"abs diff $\langle x(t) x^\ast(s) \rangle$")
        cax = ax[0, 3].imshow(np.log10(np.abs(autoCorr_KLE_conj - ac_true)), interpolation='none', cmap="inferno")
        fig.colorbar(cax, ax=ax[0, 3])
        ax[1, 3].set_title(r"abs diff $\langle x(t) x(s) \rangle$")
        cax = ax[1, 3].imshow(np.log10(np.abs(autoCorr_KLE_not_conj)), interpolation='none', cmap="inferno")
        fig.colorbar(cax, ax=ax[1, 3])

        plt.tight_layout()
        plt.show()

    assert max_diff_not_conj < tol
    assert max_diff_conj < tol


def test_stochastic_process_KLE_correlation_function(plot=False):
    """
        generate samples using FFT method

        compare <X_t X_s> and <X_t X^ast_s> from samples with true ac functions

        no interpolation at all
    """

    t_max = 15
    num_samples = 2000
    tol = 3e-2
    stp = sp.StocProc_KLE(tol=1e-2, r_tau=corr, t_max=t_max, ng_fac=4, seed=0)
    stocproc_metatest(stp, num_samples, tol, corr, plot)


def test_stochastic_process_FFT_correlation_function(plot=False):
    """
        generate samples using FFT method
        
        compare <X_t X_s> and <X_t X^ast_s> from samples with true ac functions
        
        no interpolation at all
    """

    t_max = 15
    num_samples = 2000
    tol = 3e-2
    stp = sp.StocProc_FFT(spectral_density=spectral_density, t_max=t_max, bcf_ref=corr, intgr_tol=1e-2, intpl_tol=1e-2,
                          seed=0)
    stocproc_metatest(stp, num_samples, tol, corr, plot)


def test_stocproc_dump_load():
    t_max = 15

    ##  STOCPROC KLE  ##
    ####################
    t0 = time.time()
    stp = sp.StocProc_KLE(tol=1e-2, r_tau=corr, t_max=t_max, ng_fac=4, seed=0)
    t1 = time.time()
    dt1 = t1 - t0
    stp.new_process()
    x = stp()

    bin_data = pickle.dumps(stp)
    t0 = time.time()
    stp2 = pickle.loads(bin_data)
    t1 = time.time()
    dt2 = t1 - t0
    assert dt2 / dt1 < 0.1  # loading should be way faster

    stp2.new_process()
    x2 = stp2()
    assert np.all(x == x2)

    ##  STOCPROC FFT  ##
    ####################
    t0 = time.time()
    stp = sp.StocProc_FFT(spectral_density, t_max, corr, seed=0)
    t1 = time.time()
    dt1 = t1 - t0

    stp.new_process()
    x = stp()

    bin_data = pickle.dumps(stp)
    t0 = time.time()
    stp2 = pickle.loads(bin_data)
    t1 = time.time()
    dt2 = t1 - t0

    assert dt2 / dt1 < 0.1  # loading should be way faster

    stp2.new_process()
    x2 = stp2()

    assert np.all(x == x2)


def test_many(plot=False):
    import logging
    logging.basicConfig(level=logging.INFO)
    t_max = 15
    num_samples = 5000
    tol = 5e-2

    sd = spectral_density
    ac = corr

    stp = sp.StocProc_FFT(sd, t_max, ac, negative_frequencies=False, seed=0, intgr_tol=5e-3, intpl_tol=5e-3)
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='full', meth='simp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='random', meth='simp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='full', meth='fp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='random', meth='fp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)


    t_max = 15
    num_samples = 12000
    tol = 5e-2

    sd = lsd
    ac = lac

    stp = sp.StocProc_FFT(sd, t_max, ac, negative_frequencies=True, seed=0, intgr_tol=5e-3, intpl_tol=5e-3)
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='full', meth='simp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='random', meth='simp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='full', meth='fp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)

    stp = sp.StocProc_KLE(tol=5e-3, r_tau=ac, t_max=t_max, ng_fac=1, seed=0, diff_method='random', meth='fp')
    stocproc_metatest(stp, num_samples, tol, ac, plot)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # test_stochastic_process_KLE_correlation_function(plot=False)
    # test_stochastic_process_FFT_correlation_function(plot=False)
    # test_stocproc_dump_load()

    test_many(plot=False)
    pass
