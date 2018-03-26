StocProc
========

The StocProc module is a Python3 module allowing to sample
Gaussian stochastic processes :math:`z(t)` which are wide-sense stationary,
continuous in time and complex valued. In particular for a given auto correlation function
:math:`\alpha(\tau)` the stochastic process obeys:

.. math:: \langle z(t) \rangle = 0 \qquad \langle z(t)z(s) \rangle = 0 \qquad \langle z(t)z^\ast(s) \rangle = \alpha(t-s)

Here :math:`\langle \cdot \rangle` denotes the ensemble average.

The so far implemented methods (KLE, FFT, TanhSinh) to generate such processes provide an error control mechanism
which ensures that the auto correlation function of the numerically generated stochastic processes does correspond
to the preset auto correlation function :math:`\alpha(\tau)`.


Example
-------

The example will setup a process generator for an exponential auto correlation function
and sample a single realization.

::

    def lsd(w):
        # Lorenzian spectral density
        return 1/(1 + (w - _WC_)**2)


    def exp_ac(t):
        # exponential auto correlation function
        # note there is a factor of one over pi in the definition of the auto correlation function
        # exp_ac(t) = 1/pi int_{-infty}^infty d w  lsd(w) exp(-i w t)
        return np.exp(- np.abs(t) - 1j*_WC_*t)

    _WC_ = 5
    t_max = 10
    print("setup process generator")
    stp = sp.StocProc_FFT(spectral_density = lsd,
                          t_max = t_max,
                          alpha = exp_ac,
                          intgr_tol=1e-2,       # integration error control parameter
                          intpl_tol=1e-2):      # interpolation error control parameter

    print("generate single process")
    stp.new_process()
    zt = stp()    # get discrete process

.. image:: ../../examples/proc.*
    :width: 500px
    :align: center
    :height: 300px

The full example code can be found :doc:`here </example>`.


How to Install
--------------

Install the latest version via pip ::

    pip install stocproc

or fetch the bleeding edge version from the `git repository <https://github.com/cimatosa/stocproc>`_ ::

    git clone https://github.com/cimatosa/stocproc.git

and install the package invoking ``setup.py``.

Note: The stocproc module depends on `fcSpline <https://github.com/cimatosa/fcSpline>`_ , a fast cubic spline interpolator for equally distributed nodes.


Documentation
-------------

.. toctree::
   :maxdepth: 3

   methods
   api
   example
