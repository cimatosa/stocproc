r"""
A collection of auto (bath) correlation functions (ACF)

In principle the ACF $\alpha(\tau)$ can be any kind of positive integral kernel.
When a using Fourier Transformation based approach to samples a stochastic process, that positive kernel
is related to a positive functions, the so-called spectral density $J(\omega)$, via Fourier transform.
Here the normalization convention

$$
    \alpha(\tau) = \frac{1}{\pi} \int_{-\infty}^{\infty} d \omega J(\omega) e^{-i \omega \tau}
$$

is used.

This module contains the following classes of spectral densities:
    - Lorentzians
    - (sub/super) Ohmic functions
"""

from scipy.special import gamma
import numpy as np
from functools import cache

cached_gamma_s_over_pi = cache(lambda s: gamma(s) / np.pi)


def ohmic_acf(tau, eta, s, wc):
    r"""
    Ohmic auto correlation function

    Related to the spectral density

    $$
        J(\omega) = \eta \omega^s e^{-\omega / \omega_c}
    $$

    the ACF reads

    $$
        \alpha(\tau) = \frac{\eta}{\pi} \Gamma(s+1) \left( \frac{\omega_c}{1 + i \omega_c \tau} \right)^{s+1} \; .
    $$

    """
    return eta * cached_gamma_s_over_pi(s) * (wc / (1+1j*wc*tau))**(s+1)


def ohmic_sd(w, eta, s, wc):
    r"""
    Ohmic spectral density

    defines as

    $$
        J(\omega) = \eta \omega^s e^{-\omega / \omega_c} \; .
    $$
    """
    return eta * w**s * np.exp(-w/wc)


def lorentzian_acf(tau, eta, gamma, w0):
    r"""
    Lorentzian auto correlation function

    Related to the spectral density

    $$
        J(\omega) = \eta \frac{\gamma}{\gamma^2 + (\omega - \omega_0)^2}
    $$

    the ACF reads

    $$
        \alpha(\tau) = \eta e^{-\gamma |\tau| - \i \omega_0 t}
    $$

    """
    return eta * np.exp(-gamma * np.abs(tau) - 1j * w0 * tau)


def lorentzian_sd(w, eta, gamma, w0):
    r"""
        Lorentzian spectral density

        defines as

        $$
            J(\omega) = \eta \frac{\gamma}{\gamma^2 + (\omega - \omega_0)^2}
        $$
    """
    return eta * gamma / (gamma**2 + (w - w0) ** 2)
