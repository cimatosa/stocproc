import sys
import pathlib

# add stocproc package location to path
sys.path.insert(0, pathlib.Path(__file__).parent.parent)
import matplotlib.pyplot as plt
import numpy as np
import stocproc as sp

_WC_ = 5


def lsd(w):
    """Lorenzian spectral density"""
    return 1 / (1 + (w - _WC_) ** 2)


def lac(t):
    """the corresponding Lorenzian correlation function

    note there is a factor of one over pi in the deficition
    of the correlation function:
    lac(t) = 1/pi int_{-infty}^infty d w  lsd(w) exp(-i w t)
    """
    return np.exp(-np.abs(t) - 1j * _WC_ * t)


t_max = 10
print("setup process generator")
stp = sp.StocProc_FFT_tol(
    lsd, t_max, lac, negative_frequencies=True, seed=0, intgr_tol=1e-2, intpl_tol=1e-2
)

fig = plt.figure()
print("generate single process")
stp.new_process()
zt = stp()  # get discrete process
t = stp.t  # and the natural time axis for the discrete process
plt.plot(stp.t, np.real(stp()), color="k", label="$\mathrm{real}(z(t))$")
plt.plot(stp.t, np.imag(stp()), color="k", ls="--", label="$\mathrm{imag}(z(t))$")
plt.xlim([0, 10])
plt.legend(ncol=2, loc="upper right")
plt.title("stochastic process with exponential autocorrelaition")
plt.xlabel("time")
plt.ylabel("process $z(t)$")
plt.grid()
plt.savefig("proc.png")

ns = 5000
print("generate {} samples".format(ns))
# choose time axis 4 time finer than the natural discrete axis
tfine = np.linspace(0, t_max, (stp.num_grid_points - 1) * 4 + 1)
corr = np.zeros(shape=len(tfine), dtype=np.complex128)
# tells that we want to calculate <z(t) z^\ast(tref)>
tref = 2
# calculates the auto correlation
for i in range(ns):
    stp.new_process()
    zt = stp(tfine)
    corr += zt * np.conj(stp(tref))
corr /= ns

fig = plt.figure()
aclab = r"\langle z(t)z^\ast(t_\mathrm{ref})\rangle"
plt.plot(
    tfine,
    np.real(lac(tfine - tref)),
    color="r",
    label=r"$\mathrm{{real}}\left({}\right)$".format(aclab),
)
plt.plot(
    tfine,
    np.imag(lac(tfine - tref)),
    color="r",
    ls="--",
    label=r"$\mathrm{{imag}}\left({}\right)$".format(aclab),
)
plt.plot(tfine, np.real(corr), color="k", label="exact auto correlation")
plt.plot(tfine, np.imag(corr), color="k", ls="--")
plt.legend(loc="upper right")
plt.title("auto correlation using {} samples".format(ns))
plt.xlabel("time")
plt.ylabel(r"auto correlation ($t_\mathrm{{ref}}={}$)".format(tref))
plt.grid()
plt.savefig("ac.png")

plt.show()
