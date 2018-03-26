Methods to Sample Stochastic Processes
======================================


This module allows to sample Gaussian stochastic processes which are wide-sense stationary,
continuous in time and complex valued. In particular for a given auto correlation function
:math:`\alpha(\tau)` the stochastic process :math:`z(t)` obeys.

.. math:: \langle z(t) \rangle = 0 \qquad \langle z(t)z(s) \rangle = 0 \qquad \langle z(t)z^\ast(s) \rangle = \alpha(t-s)

Here :math:`\langle \cdot \rangle` denotes the ensemble average.


So far two different approaches to sample such processes have been implemented.

Karhunen-Loève Expansion (KLE)
------------------------------

The Karhunen-Loève expansion makes use of the spectral representation of the auto correlation function viewed as a
semi-positive integral kernel which allows for the decomposition.

.. math:: \alpha(t-s) = \sum_n \lambda_n u_n(t) u_n^\ast(s)

The Eigenvalues :math:`\lambda_n` and the Eigenfunction :math:`u_n(t)` of :math:`\alpha(t-s)` are defined in
terms of the homogeneous Fredholm equation.

.. math:: \int_0^T \mathrm{d} s \; \alpha(t-s) u_n(s) = \lambda_n u_n(t)

For complex valued Gaussian distributed and independent random variables :math:`Y_n` with
:math:`\langle Y_n \rangle = 0`,  :math:`\langle Y_n Y_m \rangle = 0` and
:math:`\langle Y_n Y^\ast_m\rangle = \delta_{nm}` a stochastic process defined as

.. math:: z(t) = \sum_n Y_n \sqrt{\lambda_n} u_n(t)

obeys the required statistic. Note, the upper integral limit :math:`T` sets the time interval for which the
stochastic process :math:`z(t) \; t \in [0,T]` is defined.

In principal the sum is infinite. Nonetheless, a finite subset of summands can be found to yield a very good
approximation of the preset auto correlations functions.
Secondly when solving the Fredholm equation numerically, the integral is approximated in terms of a sum with
integration weights :math:`w_i`,
which in turn yields a matrix Eigenvalue problem with discrete "Eigenfunctions"
([NumericalRecipes]_ Chap. 19.1).
Comparing the preset auto correlation function with the approximate auto correlation function
using a finite set of :math:`N` discrete Eigenfunctions

.. math:: \sum_{n=1}^N \lambda_n u_n(t) u_n^\ast(s)

where :math:`u_n(t)` is the interpolated discrete Eigenfunction ([NumericalRecipes]_ eq. 19.1.3)

.. math:: u_n(t) = \sum_i \frac{w_i}{\lambda_n} \alpha(t-s_i) u_{n,i}

allows for an error estimation.


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


Fourier Transform Methods
-------------------------

This approach utilizes the relation of the auto correlation function and its Fourier transform.

.. math:: \alpha(\tau) = \frac{1}{\pi} \int_{-\infty}^{\infty} \mathrm{d}\omega \; J(\omega) e^{-\mathrm{i}\omega\tau} \qquad J(\omega) \geq 0

Discretizing the integral yields an approximate expression for the auto correlation function.

.. math:: \alpha(\tau) \approx \sum_n w_n \frac{J(\omega_n)}{\pi} e^{-\mathrm{i}\omega_n\tau}

For complex valued Gaussian distributed and independent random variables :math:`Y_n` with
:math:`\langle Y_n \rangle = 0`,  :math:`\langle Y_n Y_m \rangle = 0` and
:math:`\langle Y_n Y^\ast_m\rangle = \delta_{nm}` a stochastic process defined as

.. math:: z(t) = \sum_n Y_n \sqrt{\frac{w_n J(\omega_n)}{\pi}} e^{-\mathrm{i}\omega_n t}

obeys the required statistics up to an accuracy of the integral discretization.

To ensure efficient evaluation of the stochastic process the continuous time property is realized only approximately
by interpolating a pre calculated discrete time process.
However, the error caused by the cubic spline interpolation can be explicitly controlled
(usually by the `intpl_tol` parameter). Error values of one percent and below are easily achievable.


Fast Fourier Transform (FFT)
````````````````````````````

Equally distributed nodes :math:`\omega_n` allow for an evaluation of the stochastic process
using the Fast Fourier Transform algorithm (see :py:class:`stocproc.StocProc_FFT` for an implementation).

TanhSinh Intgeration (TanhSinh)
```````````````````````````````

For spectral densities :math:`J(\omega)` with a singularity at :math:`\omega=0` the TanhSinh integration
scheme is more suitable. Such an implementation and its details can be found at :py:class:`stocproc.StocProc_TanhSinh`.


.. [NumericalRecipes] Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 2007. Numerical Recipes 3rd Edition: The Art of Scientific Computing, Auflage: 3. ed. Cambridge University Press, Cambridge, UK ; New York.
