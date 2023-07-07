import numpy as np
import stocproc as sp
import matplotlib.pyplot as plt
import pathlib
import shutil


def lsd(w):
    # Lorentzian spectral density
    return 1 / (1 + (w - _WC_) ** 2)


def exp_ac(t):
    # exponential auto correlation function
    # note there is a factor of one over pi in the definition of the auto correlation function
    # exp_ac(t) = 1/pi int_{-infty}^infty d w  lsd(w) exp(-i w t)
    return np.exp(-np.abs(t) - 1j * _WC_ * t)


_WC_ = 5
t_max = 10
# setup process generator (using FFT method)
stp = sp.StocProc_FFT(
    spectral_density=lsd,
    t_max=t_max,
    alpha=exp_ac,
    intgr_tol=1e-2,  # integration error control parameter
    intpl_tol=1e-2,  # interpolation error control parameter
    negative_frequencies=True,  # due to the Lorentzian spec. dens.
    seed=1,  # fixed a particular process
)
# generate a new process
stp.new_process()
t = np.linspace(0, t_max, 250)
# return the values of the process for the time t
zt = stp(t)

plt.figure(figsize=(10, 4))
plt.plot(t, zt.real, label="real")
plt.plot(t, zt.imag, label="imag")
plt.legend(ncol=2, loc="upper right")
plt.title("stochastic process with exponential autocorrelation function")
plt.xlabel("time")
plt.ylabel("process $z(t)$")
plt.axhline(0, color="0.5", lw=1)
plt.tight_layout()
plt.savefig("example_simple_out.png")
