import numpy as np
from functools import partial
import stocproc as sp


def alpha(t, wc):
    """Ohmic correlation function, wc: cutoff frequency"""
    return (wc / (1 + 1j * wc * t)) ** 2


def spec_dens(w, wc):
    """
    Ohmic spectral density with exponential cutoff at the cutoff frequency wc

    Note that for w < 0, the spectral density is zero which is accounted for
    by setting `positive_frequencies_only` to `True`.
    """
    return w * np.exp(-w / wc)


wc = 5
al = partial(alpha, wc=wc)
sd = partial(spec_dens, wc=wc)

t_max = 30
my_sp = sp.StocProc_FFT(
    alpha=al,
    t_max=t_max,
    spectral_density=sd,
    positive_frequencies_only=True,  # default is False
)
my_sp.new_process()
print(my_sp(2.3))
print(my_sp.get_num_y())
