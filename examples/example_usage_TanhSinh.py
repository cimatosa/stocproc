import numpy as np
from functools import partial
import stocproc as sp
import mpmath
from scipy.special import gamma


def alpha(t, s, wc, beta):
    """thermal sub-Ohmic spectral correlation function"""
    if np.isscalar(t):
        zeta = np.complex128(
            mpmath.zeta(s + 1, (1 + beta * wc + 1j * wc * t) / (beta * wc))
        )
        return gamma(s + 1) * zeta / beta ** (s + 1)

    r = np.empty(shape=t.shape, dtype=np.complex128)
    for i, ti in enumerate(t):
        zeta = np.complex128(
            mpmath.zeta(s + 1, (1 + beta * wc + 1j * wc * ti) / (beta * wc))
        )
        r[i] = gamma(s + 1) * zeta / beta ** (s + 1)
    return r


def spec_dens(w, s, wc, beta):
    """thermal sub-Ohmic spectral density"""
    return (w**s * np.exp(-w / wc)) / (np.exp(beta * w) - 1)


wc = 5
s = 0.6
beta = 1
al = partial(alpha, s=s, wc=wc, beta=beta)
sd = partial(spec_dens, s=s, wc=wc, beta=beta)


t_max = 30
my_sp = sp.StocProc_FFT(
    alpha=al,
    t_max=t_max,
    spectral_density=sd,
    positive_frequencies_only=True,  # default is False
)
print(my_sp.get_num_y())

my_sp = sp.StocProc_TanhSinh(
    alpha=al,
    t_max=t_max,
    spectral_density=sd,
    positive_frequencies_only=True,  # default is False
)
