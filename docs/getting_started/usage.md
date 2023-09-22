## Usage

First of all the auto-correlation function needs to be defined.
```python
from functools import partial
import stocproc as sp

def alpha(t, wc):
    """Ohmic correlation function, wc: cutoff frequency"""
    return (wc/(1+1j*wc*t))**2
```
It has to be passed to the constructor of any stochastic process generator as 
a `Callable` with the time arguments only.
```python
al = partial(alpha, wc=5)
```
Secondly, the time interval $[0, t_\mathrm{max}]$ over which the stochastic process is generated
needs to be specified.
```python
t_max = 3
```

### StocProc_KLE

That suffices to instantiate a stochastic process generator using the 
[**Karhunen–Loève expansion**](#1-karhunenloeve-expansion) algorithm.
```python
my_sp = sp.StocProc_KLE(alpha=al, t_max=t_max)
```
By calling [`new_process()`][stocproc.samplers.StocProc.new_process] a new stochastic process is generated.
```python
my_sp.new_process()
```
Its value for any $t \in [0, t_\mathrm{max}]$ it obtained by 
simply [calling the `my_sp` instance][stocproc.samplers.StocProc.__call__].
```python
t = 2.3
z_t = my_sp(t)
```
As default, the tolerance is set to `1e-2` which amounts in this example to an expansion using 35 terms,
which means that there are 35 independent complex values and Gaussian distributed random variable $Y_k$
involved to sample the process.
the number random variables involved is accessible by [`get_num_y`][stocproc.samplers.StocProc.get_num_y].
```python
my_sp.get_num_y()
# 35
```
Increasing the accuracy by setting the tolerance to `1e-3` yields  45 terms.
```python
my_sp = sp.StocProc_KLE(alpha=al, t_max=t_max, tol=1e-3)
my_sp.get_num_y()
# 45
```

([full example code](../pythonsnippets/example_usage_KLE.md))

!!! warning
    The initialization of the `StocProc_KLE` class is rather costly.
    In particular for large time intervals the method becomes unpractical.
    To partly cope with that problem, a `StocProc_KLE` instance can be pickled 
    (see also the [Caching](#caching) section).

!!! Info
    The Karhunen–Loève expansion requires the least amount of
    expansion terms and with that the least amount of random numbers.
    From the perspective of simply generating stochastic processes, this is
    not very relevant, but it may be in others contexts.

### StocProc_FFT

To use the fast Fourier transform algorithm implemented in `StocProc_FFT`, the spectral
density $J(\omega)$ needs to be given in addition to the auto-correlation function $\alpha(\tau)$.
Recall, they are related by Fourier transform with the following convention

$$
    \alpha(\tau) = \int_{-\infty}^{\infty} \mathrm{d} \omega S(\omega) e^{-i\omega\tau}\; .
$$

For the above example, the spectral density take the following form.
```python
def spec_dens(w, wc):
    """
    Ohmic spectral density with exponential cutoff at the cutoff frequency wc

    Note that for w < 0, the spectral density is zero which is accounted for
    by setting `positive_frequencies_only` to `True`.
    """
    return w * np.exp(-w/wc)
```
Instantiating the `StocProc_FFT` class is significantly faster compared to the `StocProc_KLE` class.
```python
t_max = 30
my_sp = sp.StocProc_FFT(
    alpha=al, 
    t_max=t_max, 
    spectral_density=sd, 
    positive_frequencies_only=True  # default is False
)
my_sp.new_process()
z_t = my_sp(2.3)
my_sp.get_num_y()
# 1024
```
Note that in our example the power spectral density is zero for negative frequencies which is
accounted for by setting `positive_frequencies_only` to `True`.
As the fast Fourier transform algorthm is used, the number of random variables used to sample
the stochastic process is always a power of 2.

([full example code](../pythonsnippets/example_usage_FFT.md))


!!! info
    The `StocProc_FFT` class can efficiently handle large time intervals 
    and can be considered as the default method to use.

### StocProc_TanhSinh

If the power spectral density diverges somewhere, the simple quadrature scheme inherent to the
fast Fourier transform method converges very bad.

For a singularity at zero, the function can still be integrated efficiently using 
[Tanh-Sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature).
For the Fourier integral this has the drawback, that the nodes are not distributed equally anymore which
prohibits the use for the *fast* Fourier method.
Still, this approach is well applicable.

Borrowed from open quantum system dynamics, let's consider a sub-Ohmic spectral density with thermal occupation, i.e.

$$
    S(\omega) = \frac{\omega^s e^{-\omega /\omega_c}}{e^{\beta\omega} - 1} \; .
$$ 

with $0 < s < 1$ and inverse temperature $0 < \beta < \infty$.
The related auto-correlation function can be expressed in terms of the Gamma function $\Gamma$ and the 
Hurwitz-zeta function $\zeta$.

$$ 
    \alpha(\tau) = \frac{1}{\beta^{s+1}}\Gamma(s+1) 
    \zeta\left(s+1, \frac{1 + \beta\omega_c + i\omega_c\tau}{\beta\omega_c}\right) 
$$

### Scaling the process

### Control of accuracy

### Caching

### Examples

### Logging