r"""
Stochastic Process Module
=========================


This module allows to sample Gaussian stochastic processes which are wide-sense stationary,
continuous in time and complex valued. In particular for a given auto correlation function
:math:`\alpha(\tau)` the stochastic process :math:`z(t)` obeys.

.. math:: \langle z(t) \rangle = 0 \qquad \langle z(t)z(s) \rangle = 0 \qquad \langle z(t)z^\ast(s) \rangle = \alpha(t-s)

Here :math:`\langle \cdot \rangle` denotes the ensemble average.


So far two different approaches to sample such processes have been implemented.

:doc:`Karhunen-Loève expansion (KLE) </StocProc_KLE>`
-----------------------------------------------------

The Karhunen-Loève expansion makes use of the spectral representation of the auto correlation function viewed as a
semi-positive integral kernel which allows for the decomposition.

.. math:: \alpha(t-s) = \sum_n \lambda_n u_n(t) u_n^\ast(s)

The Eigenvalues :math:`\lambda_n` and the Eigenfunction :math:`u_n(t)` of :math:`\alpha(t-s)` are defined in
terms of the homogeneous Fredholm equation.

.. math:: \int_0^T \mathrm{d} s \; \alpha(t-s) u_n(s) = \lambda_n u_n(t)

For complex valued Gaussian distributed and independent random variables :math:`Z_n` with
:math:`\langle Z_n \rangle = 0`,  :math:`\langle Z_n Z_m \rangle = 0` and
:math:`\langle Z_n Z^\ast_m\rangle = \delta_{nm}` a stochastic process defined as

.. math:: z(t) = \sum_n Z_n \sqrt{\lambda_n} u_n(t)

obeys the required statistic. Note, the upper integral limit :math:`T` sets the time interval for which the
stochastic process :math:`z(t) \; t \in [0,T]` is defined.

The KLE approach is implemented by the class :py:class:`stocproc.StocProc_KLE`.
It is numerically feasible if :math:`T` is not too large in comparision to a typical decay time of the
auto correlation function.
Implementation details can be found in the class documentation of :py:class:`stocproc.StocProc_KLE`.

::

    import stocproc as sp
    import numpy as np
    stp = sp.StocProc_KLE(alpha = lambda t: np.exp(- np.abs(t) - 1j*5*t),
                          t_max = 2)
    stp.new_process()
    zt = stp()


:doc:`Fourier transform (FT) </StocProc_FFT>`
---------------------------------------------

This approach utilizes the relation of the auto correlation function and its Fourier transform.

.. math:: \alpha(\tau) = \frac{1}{\pi} \int_{-\infty}^{\infty} \mathrm{d}\omega \; J(\omega) e^{-\mathrm{i}\omega\tau} \qquad J(\omega) \geq 0

Discretizing the integral yields an approximate expression for the auto correlation function.

.. math:: \alpha(\tau) \approx \sum_n w_n \frac{J(\omega_n)}{\pi} e^{-\mathrm{i}\omega_n\tau}

For complex valued Gaussian distributed and independent random variables :math:`Z_n` with
:math:`\langle Z_n \rangle = 0`,  :math:`\langle Z_n Z_m \rangle = 0` and
:math:`\langle Z_n Z^\ast_m\rangle = \delta_{nm}` a stochastic process defined as

.. math:: z(t) = \sum_n Z_n \sqrt{\frac{w_n J(\omega_n)}{\pi}} e^{-\mathrm{i}\omega_n t}

obeys the required statistics up to an accuracy of the integral discretization.

Equally distributed nodes :math:`\omega_n` allow for an evaluation of the stochastic process
using the Fast Fourier Transform algorithm (see :py:class:`stocproc.StocProc_FFT` for an implementation).

For spectral densities :math:`J(\omega)` with a singularity at :math:`\omega=0` the TanhSinh integration
scheme is more suitable. Such an implementation and its details can be found at :py:class:`stocproc.StocProc_TanhSinh`.



To implement additional methods



This module contains two different implementation for generating stochastic processes for a
given auto correlation function (:doc:`Karhunen-Loève expansion </StocProc_KLE>`
and :doc:`Fast-Fourier method </StocProc_FFT>`).
Both methods are based on a time discrete process, however cubic
spline interpolation is assured to be valid within a given tolerance.

Documentation Overview
----------------------

.. toctree::
   :maxdepth: 3

   stocproc
   example

a Simple Example
----------------

The example will setup a process generator for an exponential auto correlation function
and sample a single realization. ::

    def lsd(w):
        # Lorenzian spectral density
        return 1/(1 + (w - _WC_)**2)


    def exp_ac(t):
        # exponential auto correlation function
        # note there is a factor of one over pi in the
        # definition of the auto correlation function:
        # exp_ac(t) = 1/pi int_{-infty}^infty d w  lsd(w) exp(-i w t)
        return np.exp(- np.abs(t) - 1j*_WC_*t)

    _WC_ = 5
    t_max = 10
    print("setup process generator")
    stp = sp.StocProc_FFT(spectral_density = lsd,
                          t_max = t_max,
                          bcf_ref = exp_ac,
                          intgr_tol=1e-2,
                          intpl_tol=1e-2):

    print("generate single process")
    stp.new_process()
    zt = stp()    # get discrete process

The full example can be found :doc:`here </example>`.

.. image:: ../../examples/proc.*

Averaging over 5000 samples yields the auto correlation function which is in good agreement
with the exact auto correlation.

.. image:: ../../examples/ac.*
"""

__version__ = "0.2.1"

import sys
if sys.version_info.major < 3:
    raise SystemError("no support for Python 2")


from .stocproc import StocProc_FFT
from .stocproc import StocProc_KLE
from .stocproc import StocProc_TanhSinh
