"""
The `samplers` module implements the samplers [`KarhunenLoeve`][stocproc.samplers.KarhunenLoeve],
[`FastFourier`][stocproc.samplers.FastFourier], [`TanhSinh`][stocproc.samplers.TanhSinh] and
[`Cholesky`][stocproc.samplers.Cholesky].
They all inherit from the abstract base class [`StocProc`][stocproc.samplers.StocProc]
which takes care of sampling a new process, evaluating the process at any time by interpolation and possibly
caching. Any new sampler should be a subclass of [`StocProc`][stocproc.samplers.StocProc].
"""

# python imports
import abc
from collections.abc import Callable
from functools import partial
import logging
from typing import Optional, Union

# third party imports
import fcSpline
import numpy as np
import numpy.random
from numpy.typing import NDArray
import scipy.linalg
import scipy.optimize

# module imports
from . import method_kle
from . import method_ft
from . import util


ONE_OVER_SQRT_2 = 1 / np.sqrt(2)

log = logging.getLogger(__name__)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))


def logging_setup(
    sh_level: int = logging.INFO,
    smpl_log_level: int = logging.INFO,
    kle_log_level: int = logging.INFO,
    ft_log_level: int = logging.INFO,
) -> None:
    """
    Controls the logging levels

    On loading the `stocproc` module this function is called with its default argument
    which amount to informative logging.

    Parameters:
        sh_level: the logging level for the (output) StreamHandler
        smpl_log_level: the logging level of the [`samplers`][stocproc.samplers] module log
        kle_log_level: the logging level of the KLE helper functions log (see [`method_kle`][stocproc.method_kle])
        ft_log_level: the logging level of the FT helper functions log (see [`method_ft`][stocproc.method_ft])
    """
    sh.setLevel(sh_level)

    log.addHandler(sh)
    log.setLevel(smpl_log_level)

    method_kle.log.addHandler(sh)
    method_kle.log.setLevel(kle_log_level)

    method_ft.log.addHandler(sh)
    method_ft.log.setLevel(ft_log_level)


logging_setup()


class StocProc(abc.ABC):
    r"""
    Interface definition for stochastic process implementations

    A new implementation for a stochastic process sampler (generator)
    should subclass `StocProc` and overwrite
    [`calc_z`][stocproc.samplers.StocProc.calc_z], [`get_num_y`][stocproc.samplers.StocProc.calc_z]
    and optionally [`calc_z_dot`][stocproc.samplers.StocProc.calc_z_dot].

    For the $N$ equally spaced times $t_n = n \frac{t_{max}}{N-1}$ with $n = 0 \dots N-1$,
    the function [`calc_z`][stocproc.samplers.StocProc.calc_z] should map $M$ independent complex valued and Gaussian
    distributed random variables $Y_m$ with $\langle Y_m \rangle = 0 = \langle Y_m Y_{m'} \rangle$ and
    $\langle Y_m Y^\ast_{m'} = \delta_{m m'}\rangle$) to the discrete time stochastic process $z_n = z(t_n)$.
    The method [`get_num_y`][stocproc.samplers.StocProc.calc_z] needs to return the number $M$ of random variables
    $Y_m$ required as input for [`calc_z`][stocproc.samplers.StocProc.calc_z].

    Just like [`calc_z`][stocproc.samplers.StocProc.calc_z], the method
    [`calc_z_dot`][stocproc.samplers.StocProc.calc_z_dot] needs to be implemented.
    It needs to map $M$ independent complex valued and Gaussian distributed random variables $Y_m$ with
    $\langle Y_m \rangle = 0 = \langle Y_m Y_{m'} \rangle$ and
    $\langle Y_m Y^\ast_{m'} = \delta_{m m'}\rangle$) to the derivative os the stochastic process
    at discrete times, i.e., $\dot z_n = \dot z(t_n)$.

    Having implemented these methods, the [`StocProc`][stocproc.samplers.StocProc] class provides
    convenient functions such as:

    - [`__call__`][stocproc.samplers.StocProc.__call__] evaluate the stochastic process for any time within the
      interval $[0, t_\mathrm{max}]$ using [cubic spline interpolation](https://github.com/cimatosa/fcSpline)
    - [`get_time`][stocproc.samplers.StocProc.get_time] returns the times $t_n$
    - [`get_z`][stocproc.samplers.StocProc.get_z] returns the discrete stochastic process $z_n$
    - [`new_process`][stocproc.samplers.StocProc.new_process] draws new samples $Y_m$ and updates $z_n$ as well as the
      cubic spline interpolator
    - [`set_scale`][stocproc.samplers.StocProc.set_scale] set the scalar pre factor $\eta$ of the auto-correlation
      function which scales the stochastic process such that $\langle z(t) z^\ast(s)\rangle = \eta \alpha(t-s)$.

    Each instance of a stochastic process has its own random number generator instance.
    Its default amounts to the Generator object returned by numpy.random.default_rng.
    It is seeded with 0.
    Calling [`init_rng`][stocproc.StocProc.init_rng] reinits the Generator class, possibly with a different seed.

    Parameters:
        t_max: specifies the upper bound of the time interval
        num_grid_points: number of equidistant grid points $N$

    !!! Note
        [`new_process`][stocproc.samplers.StocProc.new_process] is **not** called on init.
        If you want to obtain a particular realization of the stochastic process, a new sample needs to be drawn by
        calling [`new_process`][stocproc.samplers.StocProc.new_process].
        Otherwise, a `RuntimeError` is raised.
    """
    _use_cache = False

    def __init__(self, t_max=None, num_grid_points=None):
        # time axes
        self.t_max = t_max
        self.num_grid_points = num_grid_points
        self.t = np.linspace(0, t_max, num_grid_points)

        self._z = None
        self._z_dot = None
        self._interpolator = None
        self._interpolator_dot = None
        self._proc_cnt = 0

        # these can be changed by the 'config_rng' method
        self._rng_class = np.random.default_rng
        self._seed = 0
        self._rand_skip = 1000
        self._rng = None
        self.init_rng(
            random_numer_gen=self._rng_class,
            seed=self._seed,
            rand_skip=self._rand_skip
        )

        log.debug(f"init StocProc with t_max {t_max} and {num_grid_points} grid points")

    def __call__(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Evaluates the stochastic process

        If $t$ is not `None`, cubic spline interpolation is used to evaluate $z(t)$
        based on the discrete realization of the process $z_n = z(t_n)$.
        The time argument $t$ may be a single value or array like.

        If $t$ is `None`, the discrete process $z_n$ is returned.

        Parameters:
            t: the time (or times as array) for which to evaluate the stochastic process

        Returns:
            the value of the stochastic process $z(t)$
        """
        if self._z is None:
            raise RuntimeError(
                "StocProc has NO random data yet, call 'new_process' to generate a new random process!"
            )

        if t is None:
            return self._z
        else:
            return self._interpolator(t)

    def dot(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Returns the derivative of the stochastic process.

        Works the same as [`__call__`][stocproc.StocProc.__call__] in all other regards.
        """
        if self._z_dot is None:
            raise RuntimeError(
                "StocProc has NO random data yet, call 'new_process' to generate a new random process!"
            )

        if t is None:
            return self._z_dot
        else:
            return self._interpolator_dot(t)

    @abc.abstractmethod
    def calc_z(self, y: np.ndarray) -> np.ndarray:
        r"""*abstract method*

        An implementation needs to map $M$ independent complex valued and Gaussian
        distributed random variables $Y_m$ (with $\langle Y_m \rangle = 0 = \langle Y_m Y_{m'} \rangle$ and
        $\langle Y_m Y^\ast_{m'} = \delta_{m m'}\rangle$) to the discrete time stochastic
        process $z_n = z(t_n)$.

        Parameters:
            y: M independent complex valued and Gaussian distributed random numbers with zero mean and variance one.

        Returns:
            the discrete time stochastic process $z_n$ as a numpy array of complex numbers
        """
        pass

    @abc.abstractmethod
    def get_num_y(self):
        r"""*abstract method*

        An implementation needs to return the number $M$ of random variables $Y_m$ required as
        input for the method [`calc_z`][stocproc.StocProc.calc_z]
        """
        pass

    def calc_z_dot(self, y: np.ndarray) -> Union[np.ndarray, None]:
        r"""*abstract method (optional)*

        An implementation needs to map $M$ independent complex
        valued and Gaussian distributed random variables $Y_m$
        (with $\langle Y_m \rangle = 0 = \langle Y_m Y_{m'}\rangle$ and
        $\langle Y_m Y^\ast_{m'} = \delta_{mm'}\rangle$) to the discrete time
        derivative of stochastic process $z_n = z(t_n)$.

        Parameters:
            y: M independent complex valued and Gaussian distributed random numbers with zero mean and variance one.

        Returns:
            the derivative of the discrete time stochastic process $z_n$ as a numpy array of complex numbers
        """
        return None

    def get_time(self) -> np.ndarray:
        r"""
        Returns:
             the times $t_n$ for which the discrete time stochastic process is defined $z_n = z(t_n)$.
        """
        return self.t

    def get_z(self) -> np.ndarray:
        r"""
        Returns:
             the discrete time stochastic process $z_n = z(t_n)$.
        """
        return self._z

    def init_rng(
        self,
        random_numer_gen=None,
        seed: float = None,
        rand_skip: int = None
    ):
        """
        Set instance wide parameters which control the random number generation.
        If a parameter is `None` (default), it is ignored.

        Parameters:
            random_numer_gen: a random number generator class (numpy.random.Generator), default is to use
                numpy.random.default_rng
            seed: the seed passed to the instantiation of the generator class (is actually passed through SeedSequence
                to map any integer to a suitable random number generator seed.
            rand_skip: discard the first 'rand_skip' numbers
        """
        if random_numer_gen is not None:
            self._rng_class = random_numer_gen
        if seed is not None:
            self._seed = seed
        if rand_skip is not None:
            self._rand_skip = rand_skip
        self._rng = self._rng_class(self._seed)

    def new_process(
        self,
        y: Optional[np.ndarray] = None,
        scale: float = 1,
        seed: int = None,
        rand_skip: int = None
    ) -> None:
        r"""Generate a new realization of the stochastic process.

        If `y` is not `None`, use `y` as input to generate the new realization by calling
        [`calc_z`][stocproc.StocProc.calc_z].

        If `y` is `None`, draw new random numbers to generate the new realization.

        If `seed` is not `None`, reinit the random number generator with seed before drawing new random numbers.
        Discard the first `rand_skip` samples.

        Scale the process with square root of `scale`.

        Parameters:
            y: M independent complex valued and Gaussian distributed random numbers with zero mean and variance one.
            scale: Scale the process with square root of `scale`.
            seed: reinit the random number generator with seed
            rand_skip: discard the first `rand_skip` samples.
        """
        # clean up old data
        del self._interpolator
        self._interpolator = None
        del self._z
        self._z = None
        del self._z_dot
        self._z_dot = None

        self._proc_cnt += 1

        # reinit random number generator
        if seed is not None:
            self.init_rng(
                seed=seed,
                rand_skip=rand_skip
            )

        if y is None:
            # draw new random numbers
            y = self._rng.normal(scale=ONE_OVER_SQRT_2, size=2 * self.get_num_y())
            if y.dtype != np.float64:
                raise RuntimeError(
                    f"Expect that numpy.random.normal returns with dtype float64, but it is {y.dtype}"
                )
            y = y.view(np.complex128)
        else:
            len_y = len(y)
            num_y = self.get_num_y()
            if len_y != num_y:
                raise RuntimeError(f"the length of 'y' ({len_y}) needs to be {num_y}")

        # generate the new process (scaled version)
        self._z = np.sqrt(scale) * self.calc_z(y)
        self._interpolator = fcSpline.FCS(x_low=0, x_high=self.t_max, y=self._z)

        # .. and optionally its derivative
        self._z_dot = self.calc_z_dot(y)
        if self._z_dot is not None:
            self._z_dot *= np.sqrt(scale)
            self._interpolator_dot = fcSpline.FCS(
                x_low=0, x_high=self.t_max, y=self._z_dot
            )


class KarhunenLoeve(StocProc):
    r"""
    A class to simulate stochastic processes using Karhunen-Loève expansion (KLE) method.

    The idea is that any stochastic process can be expressed in terms of the KLE

    $$ Z(t) = \sum_i \sqrt{\lambda_i} Y_i u_i(t) . $$

    Here $Y_i$ are independent complex valued Gaussian random variables with variance one, i.e.,
    $\langle Y_i Y_j \rangle = \delta_{ij}$. $\lambda_i$, $u_i(t)$ are the
    eigenvalues / eigenfunctions of the following homogeneous Fredholm equation

    $$ \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u_i(s) = \lambda_i u_i(t) . $$

    The positive integral kernel $R(\tau)$ amount to the auto correlation of the
    stochastic processes, i.e., $\langle Z(t)Z^\ast(s) \rangle = R(t-s)$.

    For a numeric treatment, the integral equation needs to be discretized
    (see [`method_kle.solve_hom_fredholm`][stocproc.method_kle.solve_hom_fredholm] for details)
    which leads to a regular matrix eigenvalue problem.
    The accuracy of the generated process, by means of the accuracy of its auto correlation function, depends on
    to factors. First, the accuracy of the eigenvalues and eigenfunction which depends on the integral discritization.
    Second, the truncation of the infinite sum of the KLE.
    Both errors can in principle be made arbitrarily small.

    Note that the property of representing the integral kernel in terms of the eigenfunction, i.e,

    $$ R(t-s) = \sum_i \lambda_i u_i(t) u_i^\ast(s) $$

    is used to find a suitable discritization and the number of required eigenfunctions such that
    the sum represents the actual kernel up to a given tolerance.
    See [`method_kle.auto_ng`][stocproc.method_kle.auto_ng] for details.

    Parameters:
        :param r_tau: the idesired auto correlation function of a single parameter tau
        :param t_max: specifies the time interval [0, t_max] for which the processes in generated
        :param tol: maximal deviation of the auto correlation function of the sampled processes from
            the given auto correlation r_tau.
        :param ngfac: specifies the fine grid to use for the spline interpolation, the intermediate points are
            calculated using integral interpolation
        :param meth: the method for calculation integration weights and times, a callable or one of the following strings
            'midpoint' ('midp'), 'trapezoidal' ('trapz'), 'simpson' ('simp'), 'fourpoint' ('fp'),
            'gauss_legendre' ('gl'), 'tanh_sinh' ('ts')
        :param diff_method: either 'full' or 'random', determines the points where the above success criterion is evaluated,
            'full': full grid in between the fine grid, such that the spline interpolation error is expected to be maximal
            'random': pick a fixed number of random times t and s within the interval [0, t_max]
        :param dm_random_samples: the number of random times used for diff_method 'random'
        :param seed: if not None seed the random number generator on init of this class with seed
        :param align_eig_vec: assures that :math:`re(u_i(0)) \leq 0` and :math:`im(u_i(0)) = 0` for all i

        .. note ::
           To circumvent the time consuming initializing the StocProc class can be saved and loaded using
           the standard python pickle module. The :py:func:`get_key` method may be used identify the
           Process class by its parameters (r_tau, t_max, tol).

        .. seealso ::
           Details on how to solve the homogeneous Fredholm equation: :py:func:`stocproc.method_kle.solve_hom_fredholm`

           Details on the error estimation and further clarification of the parameters ng_fac, meth,
           diff_method, dm_random_samples can be found at :py:func:`stocproc.method_kle.auto_ng`.

    """

    def __init__(
        self,
        alpha: util.CplxFnc,
        t_max: float,
        tol: float = 1e-2,
        ng_fac: int = 4,
        meth: Union[str, Callable[[float, int], tuple[NDArray, NDArray, bool]]] = 'midpoint',
        diff_method: str = "full",
        dm_random_samples: float = 10**4,
    ):
        sqrt_lambda_ui_fine, t = method_kle.auto_ng(
            acf=alpha,
            t_max=t_max,
            ng_fac=ng_fac,
            meth=meth,
            tol=tol,
            diff_method=diff_method,
            dm_random_samples=dm_random_samples,
        )
        num_ev, ng = sqrt_lambda_ui_fine.shape

        super(KarhunenLoeve, self).__init__(t_max=t_max, num_grid_points=ng)
        self.num_ev = num_ev
        self.sqrt_lambda_ui_fine = sqrt_lambda_ui_fine

    def __getstate__(self):
        return (
            self.sqrt_lambda_ui_fine,
            self.t_max,
        )

    def __setstate__(self, state):
        sqrt_lambda_ui_fine, t_max = state
        num_ev, ng = sqrt_lambda_ui_fine.shape
        super(KarhunenLoeve, self).__init__(t_max=t_max, num_grid_points=ng)
        self.num_ev = num_ev
        self.sqrt_lambda_ui_fine = sqrt_lambda_ui_fine

    def calc_z(self, y):
        r"""evaluate :math:`z_k = \sum_i \lambda_i Y_i u_{ik}`"""
        return np.tensordot(y, self.sqrt_lambda_ui_fine, axes=([0], [0])).flatten()

    def get_num_y(self):
        """The number of independent random variables Y is given by the number of used eigenfunction
        to approximate the auto correlation kernel.
        """
        return self.num_ev


class FastFourier(StocProc):
    r"""Generate Stochastic Processes using the Fast Fourier Transform (FFT) method

    This method uses the relation of the auto correlation function ``alpha`` to the non negative real valued
    spectral density (``spectral_density``) :math:`J(\omega)`.
    The integral can be approximated by a discrete integration scheme

    .. math::
        \alpha(\tau) = \int_{-\infty}^{\infty} \mathrm{d}\omega \, \frac{J(\omega)}{\pi} e^{-\mathrm{i}\omega \tau}
        \approx \sum_{k=0}^{N-1} w_k \frac{J(\omega_k)}{\pi} e^{-\mathrm{i} \omega_k \tau}

    where the weights :math:`\omega_k` depend on the particular integration scheme. For a process defined as

    .. math:: z(t) = \sum_{k=0}^{N-1} \sqrt{\frac{w_k J(\omega_k)}{\pi}} Y_k \exp^{-\mathrm{i}\omega_k t}

    with independent complex random variables :math:`Y_k` such that :math:`\langle Y_k \rangle = 0`,
    :math:`\langle Y_k Y_{k'}\rangle = 0` and :math:`\langle Y_k Y^\ast_{k'}\rangle = \delta_{k,k'}`
    it is easy to see that its auto correlation function will be exactly the approximated auto correlation function.

    .. math::
        \begin{align}
            \langle z(t) z^\ast(s) \rangle = & \sum_{k,k'} \frac{1}{\pi} \sqrt{w_k w_{k'} J(\omega_k)J(\omega_{k'})} \langle Y_k Y_{k'}\rangle \exp(-\mathrm{i}(\omega_k t - \omega_k' s)) \\
                                           = & \sum_{k}    \frac{w_k}{\pi} J(\omega_k) e^{-\mathrm{i}\omega_k (t-s)} \\
                                           \approx & \alpha(t-s)
        \end{align}

    To calculate :math:`z(t)` the Discrete Fourier Transform (DFT) can be applied as follows:

    .. math:: z(t_l) = e^{-\mathrm{i}\omega_\mathrm{min} t_l} \sum_{k=0}^{N-1} \sqrt{\frac{w_k J(\omega_k)}{\pi}} Y_k  e^{-\mathrm{i} 2 \pi \frac{k l}{N} \frac{\Delta \omega \Delta t}{ 2 \pi} N}

    However, this requires that :math:`\omega_k` takes the form :math:`\omega_k = \omega_\mathrm{min} + k \Delta \omega`
    with :math:`\Delta \omega = (\omega_\mathrm{max} - \omega_\mathrm{min}) / (N-1)` which limits
    the integration schemes to those with equidistant nodes.

    For the DFT scheme to be applicable :math:`\Delta t` has to be chosen such that
    :math:`2\pi = N \Delta \omega \Delta t` holds.
    Since :math:`J(\omega)` is real it follows that :math:`z(t_l) = z^\ast(t_{N-l})`.
    For that reason the stochastic process has only :math:`(N+1)/2` (odd :math:`N`) and
    :math:`(N/2 + 1)` (even :math:`N`) independent time grid points.

    To generate a process with given auto correlation function on the interval :math:`[0, t_{max}]`
    requires that the auto correlation function approximation is valid for all :math:`t` in :math:`[0, t_{max}]`.

    This is ensured by automatically determining the number of sumands N and the integral
    boundaries :math:`\omega_\mathrm{min}` and :math:`\omega_\mathrm{max}` such that
    discrete Fourier transform of the spectral density matches the preset auto correlation function
    within the tolerance `intgr_tol` for all discrete :math:`t_l \in [0, t_{max}]`.

    As the time continuous process is generated via cubic spline interpolation, the deviation
    due to the interpolation is controlled by the parameter ``intpl_tol``. The maximum time step :math:`\Delta t`
    is chosen such that the interpolated valued at each half step :math:`t_i + \Delta t /2` differs at
    most ``intpl_tol`` from the exact value of the auto correlation function.

    If not fulfilled already N and the integration boundaries are increased such that the :math:`\Delta t`
    criterion from the interpolation is met.

    See :py:func:`stocproc.method_ft.calc_ab_N_dx_dt` for implementation details on how the
    tolerance criterion is met. Since the pre calculation may become time consuming the :py:class:`StocProc_FFT`
    class can be pickled and unpickled. To identify a particular instance a unique key is formed by the tuple
    ``(alpha, t_max, intgr_tol, intpl_tol)``.
    It is advisable to use :py:func:`get_key` with keyword arguments to generate such a tuple.


    :param spectral_density: the spectral density :math:`J(\omega)` as callable function object
    :param t_max: :math:`[0,t_\mathrm{max}]` is the interval for which the process will be calculated
    :param alpha: a callable which evaluates the Fourier integral exactly
    :param intgr_tol: tolerance for the integral approximation
    :param intpl_tol: tolerance for the interpolation
    :param seed: if not None, use this seed to seed the random number generator
    :param negative_frequencies: if False, keep :math:`\omega_\mathrm{min} = 0` otherwise
       find a negative :math:`\omega_\mathrm{min}` appropriately just like :math:`\omega_\mathrm{max}`

    """
    def __init__(
        self,
        spectral_density,
        t_max,
        alpha,
        intgr_tol=1e-2,
        intpl_tol=1e-2,
        seed=None,
        positive_frequencies_only=False,
        scale=1,
        calc_deriv: bool = False,
    ):
        self.key = self.get_key(
            alpha=alpha, t_max=t_max, intgr_tol=intgr_tol, intpl_tol=intpl_tol
        )

        ft_ref = alpha

        if positive_frequencies_only:
            log.info("non neg freq only")
            a, b, N, dx, dt = method_ft.calc_ab_n_dx_dt(
                integrand=spectral_density,
                intgr_tol=intgr_tol,
                intpl_tol=intpl_tol,
                t_max=t_max,
                ft_ref=ft_ref,
                opt_b_only=True,
            )
        else:
            log.info("use neg freq")
            a, b, N, dx, dt = method_ft.calc_ab_n_dx_dt(
                integrand=spectral_density,
                intgr_tol=intgr_tol,
                intpl_tol=intpl_tol,
                t_max=t_max,
                ft_ref=ft_ref,
                opt_b_only=False,
            )

        d = abs(2 * np.pi - N * dx * dt)
        if d >= 1e-12:
            log.fatal("method_ft.calc_ab_N_dx_dt returned inconsistent data!")
            raise RuntimeError("d = {:.3e} < 1e-12 FAILED!".format(d))

        log.info("Fourier Integral Boundaries: [{:.3e}, {:.3e}]".format(a, b))
        log.info("Number of Nodes            : {}".format(N))
        log.info("yields dx                  : {:.3e}".format(dx))
        log.info("yields dt                  : {:.3e}".format(dt))
        log.info("yields t_max               : {:.3e}".format((N - 1) * dt))

        num_grid_points = int(np.ceil(t_max / dt)) + 1

        if num_grid_points > N:
            log.fatal("num_grid_points and number of points used for FFT inconsistent!")
            raise RuntimeError(
                "num_grid_points = {} <= N_DFT = {}  FAILED!".format(num_grid_points, N)
            )

        t_max = (num_grid_points - 1) * dt
        super().setup(
            t_max=t_max,
            num_grid_points=num_grid_points,
            seed=seed,
            scale=scale,
            calc_deriv=calc_deriv,
        )
        self.omega_min = a + dx / 2
        self.omega_k = dx * np.arange(N) + self.omega_min
        self.yl = spectral_density(self.omega_k) * dx / np.pi
        self.yl = np.sqrt(self.yl)

        self.omega_min_correction = np.exp(
            (-1j * self.omega_min * self.t)
        )  # self.t is from the parent class

    @staticmethod
    def get_key(t_max, alpha, intgr_tol=1e-2, intpl_tol=1e-2):
        """
        Returns the tuple ``(alpha, t_max, intgr_tol, intpl_tol)`` which uniquely identifies a particular
        :py:class:`StocProc_FFT` instance
        """
        return "fft", alpha, t_max, intgr_tol, intpl_tol

    def __getstate__(self):
        return (
            self.yl,
            self.num_grid_points,
            self.omega_min,
            self.omega_min_correction,
            self.omega_k,
            self.t_max,
            self._seed,
            self.scale,
            self.key,
            self.calc_deriv,
        )

    def __setstate__(self, state):
        (
            self.yl,
            num_grid_points,
            self.omega_min,
            self.omega_min_correction,
            self.omega_k,
            t_max,
            seed,
            scale,
            self.key,
            calc_deriv,
        ) = state
        super().setup(
            t_max=t_max,
            num_grid_points=num_grid_points,
            seed=seed,
            scale=scale,
            calc_deriv=calc_deriv,
        )

    def calc_z(self, y):
        r"""Calculate the discrete time stochastic process using FFT algorithm

        .. math::
            z_n = z(t_n) = e^{-\mathrm{i}\omega_\mathrm{min} t_n} \mathrm{FFT}\left( \sqrt{\frac{w_k J(\omega_k)}{\pi}} Y_k \right)

        and return values :math:`z_n` with :math:`t_n <= t_\mathrm{max}`.
        """

        z_fft = np.fft.fft(self.yl * y)
        z = z_fft[0 : self.num_grid_points] * self.omega_min_correction

        return z

    def calc_z_dot(self, y: np.ndarray) -> np.ndarray:
        r"""Calculate the discrete time stochastic process derivative using FFT algorithm
        and return values :math:`\dot{z}_n` with :math:`t_n <= t_\mathrm{max}`.
        """

        z_dot_fft = np.fft.fft(-1j * self.omega_k * self.yl * y)
        z_dot = z_dot_fft[0 : self.num_grid_points] * self.omega_min_correction
        return z_dot

    def get_num_y(self):
        r"""The number of independent random variables :math:`Y_m` is given by the number of discrete nodes
        used by the Fast Fourier Transform algorithm.
        """
        return len(self.yl)


class TanhSinh(StocProc):
    r"""Simulate Stochastic Process using TanhSinh integration for the Fourier Integral"""
    _use_cache = True

    def __post_init__(
        self,
        spectral_density,
        t_max,
        alpha,
        intgr_tol=1e-2,
        intpl_tol=1e-2,
        seed=None,
        positive_frequencies_only=False,
        scale=1,
        calc_deriv=False,
    ):
        self.key = "ts", alpha, t_max, intgr_tol, intpl_tol

        ft_ref = alpha

        if positive_frequencies_only:
            log.info("non neg freq only")
            try:
                log.info("get_dt_for_accurate_interpolation, please wait ...")
                c = method_ft.find_integral_boundary(
                    integrand=lambda tau: np.abs(ft_ref(tau)) / np.abs(ft_ref(0)),
                    direction='right',
                    tol= intgr_tol
                )
            except RuntimeError:
                c = t_max

            c = min(c, t_max)
            dt_tol = method_ft.get_dt_for_accurate_interpolation(
                t_max=c, tol=intpl_tol, ft_ref=ft_ref
            )
            log.info("requires dt < {:.3e}".format(dt_tol))
        else:
            raise NotImplementedError

        N = int(np.ceil(t_max / dt_tol)) + 1
        log.info("yields N = {} (time domain)".format(N))

        log.info("find accurate discretisation in frequency domain")
        wmax = method_ft.find_integral_boundary(
            spectral_density, tol=intgr_tol / 10, ref_val=1, max_val=1e6, x0=0.777
        )
        log.info("wmax:{}".format(wmax))

        sd_over_pi = partial(SD_over_pi, J=spectral_density)

        t_max_ts = method_ft.get_t_max_for_singularity_ts(
            sd_over_pi, 0, wmax, intgr_tol
        )

        tau = np.linspace(0, t_max, 35)

        n = 16
        d = intgr_tol + 1
        while d > intgr_tol:
            n *= 2
            I = method_ft.fourier_integral_tanhsinh(
                f=sd_over_pi, x_max=wmax, n=n, tau_l=tau, t_max_ts=t_max_ts
            )
            bcf_ref_t = alpha(tau)

            d = np.abs(bcf_ref_t - I) / abs(bcf_ref_t[0])
            d = np.max(d)
            log.debug("n:{} d:{} tol:{}".format(n, d, intgr_tol))

        tau = np.linspace(0, (N - 1) * dt_tol, N)
        log.debug(
            "perform numeric check of entire time axis [{},{}] N:{}".format(
                0, (N - 1) * dt_tol, N
            )
        )
        num_FT = method_ft.fourier_integral_tanhsinh(
            f=sd_over_pi, x_max=wmax, n=n, tau_l=tau, t_max_ts=t_max_ts
        )

        bcf_ref_t = alpha(tau)
        d = np.max(np.abs(num_FT - bcf_ref_t) / np.abs(bcf_ref_t[0]))
        if d > intgr_tol:
            log.error("numeric check over entire time axis failed")
            import matplotlib.pyplot as plt

            plt.plot(tau, num_FT.real, label="ts intr bcf real")
            plt.plot(tau, num_FT.imag, label="ts intr bcf imag")

            plt.plot(tau, bcf_ref_t.real, label="bcf ref real")
            plt.plot(tau, bcf_ref_t.imag, label="bcf ref imag")

            plt.figure()
            d_tau = np.abs(num_FT - bcf_ref_t) / np.abs(bcf_ref_t[0])
            plt.plot(tau, d_tau)
            plt.yscale("log")

            plt.show()

        assert d <= intgr_tol, "d:{}, intgr_tol:{}".format(d, intgr_tol)
        log.debug("done!")

        yk, wk = method_ft.tanhsinh_get_x_and_w(n, wmax, t_max_ts)
        self.omega_k = yk
        self.fl = np.sqrt(wk * spectral_density(self.omega_k) / np.pi)
        super().setup(
            t_max=t_max,
            num_grid_points=N,
            seed=seed,
            scale=scale,
            calc_deriv=calc_deriv,
        )

    @staticmethod
    def get_key(t_max, alpha, intgr_tol=1e-2, intpl_tol=1e-2):
        return "ts", alpha, t_max, intgr_tol, intpl_tol

    def __getstate__(self):
        return (
            self.fl,
            self.omega_k,
            self.num_grid_points,
            self.t_max,
            self._seed,
            self.scale,
            self.key,
            self.calc_deriv,
        )

    def __setstate__(self, state):
        (
            self.fl,
            self.omega_k,
            num_grid_points,
            t_max,
            seed,
            scale,
            self.key,
            calc_deriv,
        ) = state

        super().setup(
            t_max=t_max,
            num_grid_points=num_grid_points,
            seed=seed,
            scale=scale,
            calc_deriv=calc_deriv,
        )

    def calc_z(self, y):
        r"""calculate

        .. math::
            Z(t_l) = sum_k \sqrt{\frac{w_k J(\omega_k)}{\pi}} Y_k e^{-\i \omega_k t_l}
        """
        # z = np.empty(shape=self.num_grid_points, dtype=np.complex128)
        # for i, ti in enumerate(self.t):
        #     z[i] = np.sum(self.fl*y*np.exp(-1j*self.omega_k*ti))

        tmp1 = self.fl * y
        tmp2 = -1j * self.omega_k
        z = np.fromiter(
            (np.sum(tmp1 * np.exp(tmp2 * t)) for t in self.t), dtype=tmp2.dtype
        )

        return z

    def calc_z_dot(self, y: np.ndarray) -> np.ndarray:
        r"""calculate the derivative

        .. math::
            Z(t_l) = sum_k \sqrt{\frac{w_k J(\omega_k)}{\pi}} (-\i \omega_k) Y_k e^{-\i \omega_k t_l}
        """
        # z = np.empty(shape=self.num_grid_points, dtype=np.complex128)
        # for i, ti in enumerate(self.t):
        #     z[i] = np.sum(self.fl*y*np.exp(-1j*self.omega_k*ti))

        tmp1 = self.fl * y
        tmp2 = -1j * self.omega_k

        pre = tmp1 * tmp2
        z_dot = np.fromiter(
            (np.sum(pre * np.exp(tmp2 * t)) for t in self.t), dtype=tmp2.dtype
        )

        return z_dot

    def calc_z_map(self, y):
        r"""calculate

        .. math::
            Z(t_l) = sum_k \sqrt{\frac{w_k J(\omega_k)}{\pi}} Y_k e^{-\i \omega_k t_l}
        """
        # z = np.empty(shape=self.num_grid_points, dtype=np.complex128)
        tmp1 = self.fl * y
        tmp2 = -1j * self.omega_k
        z = map(lambda ti: tmp1 * np.exp(tmp2 * ti), self.t)
        return np.asarray(z)

    def get_num_y(self):
        return len(self.fl)


BCF = Callable[[NDArray[np.floating]], NDArray[np.complex128]]


class Cholesky(StocProc):
    r"""Generate Stochastic Processes using the cholesky decomposition."""

    def __post_init__(
        self,
        t_max: float,
        alpha: BCF,
        intgr_tol=1e-2,
        intpl_tol=1e-2,
        chol_tol=1e-2,
        correlation_cutoff=1e-3,
        seed=None,
        scale=1,
        calc_deriv: bool = False,
        max_iterations: int = 10,
    ):
        del intgr_tol  # not used for now
        self.key = (
            "chol",
            alpha,
            t_max,
            intpl_tol,
            chol_tol,
            correlation_cutoff,
            calc_deriv,
            max_iterations,
        )

        steps = int(t_max / intpl_tol) + 1

        self.t: NDArray[np.float128] = np.linspace(0, t_max, steps, dtype=np.float128)
        """The times at which the stochastic process will be sampled."""

        super().setup(
            t_max=t_max,
            num_grid_points=len(self.t),
            seed=seed,
            scale=scale,
            calc_deriv=calc_deriv,
        )

        cutoff_sol = scipy.optimize.root(
            lambda t: np.abs(alpha(t)) - correlation_cutoff, x0=[0.001]
        )

        if not cutoff_sol.success:
            raise RuntimeError(
                f"Could not find a suitable cutoff time. Scipy says '{cutoff_sol.message}'."
            )

        self.t_chol = np.linspace(
            0,
            cutoff_sol.x[0] * 2,
            int(((cutoff_sol.x[0]) / intpl_tol) + 1) * 2,
            dtype=np.float128,
        )

        mat, tol = self.stable_cholesky(alpha, self.t_chol, max_iterations)
        log.info(f"Achieved a deviation of {tol} in the cholesky decomposition.")
        if mat is None or tol > chol_tol:
            raise RuntimeError(
                f"The tolerance of {chol_tol} could not be reached. We got as far as {tol}."
            )

        self.chol_matrix: NDArray[np.complex128] = mat
        if calc_deriv:
            self.chol_deriv = np.gradient(mat, self.t, axis=0)

        self.t_chol = self.t_chol

        self.chunk_size = len(self.t_chol) // 2

        self.patch_matrix: NDArray[np.complex128] = scipy.linalg.inv(
            self.chol_matrix[: self.chunk_size, : self.chunk_size]
        )

        self.num_chunks = int(len(self.t) / self.chunk_size) + 1

    @staticmethod
    def stable_cholesky(
        α: BCF, t, max_iterations: int = 100, starteps: Optional[float] = None
    ):
        t = np.asarray(t)
        tt, ss = np.meshgrid(t, t, sparse=False)
        Σ = α(np.array(tt - ss))
        eye = np.eye(len(t))

        eps: float = 0.0

        L = None
        reversed = False
        for _ in range(max_iterations):
            log.debug(f"Trying ε={eps}.")
            try:
                L = scipy.linalg.cholesky(Σ + eps * eye, lower=True, check_finite=False)

                if eps == 0 or reversed:
                    break

                eps /= 2

            except scipy.linalg.LinAlgError as _:
                if eps == 0:
                    eps = starteps or np.finfo(np.float64).eps * 4
                else:
                    eps = eps * 2
                    reversed = True

        return L, (np.max(np.abs((L @ L.T.conj() - Σ) / Σ)) if L is not None else -1)

    def calc_z(self, y: NDArray[np.complex128]):
        assert y.shape == (self.get_num_y(),)

        res = np.empty(self.chunk_size * self.num_chunks, dtype=np.complex128)
        y = np.pad(y, (0, len(res) - len(y)), "constant")

        offset = len(self.t_chol)

        res[0:offset] = self.chol_matrix @ y[:offset]
        y_curr = np.empty(self.chunk_size * 2, dtype=np.complex128)
        last_values = res[offset // 2 : offset]

        for i in range(self.num_chunks - 2):
            next_offset = offset + self.chunk_size

            y_curr[0 : self.chunk_size] = self.patch_matrix @ last_values
            y_curr[self.chunk_size : self.chunk_size * 2] = y[offset:next_offset]

            res[offset:next_offset] = (self.chol_matrix @ y_curr)[
                self.chunk_size : self.chunk_size * 2
            ]

            last_values = res[offset:next_offset]
            offset = next_offset

        return res[0 : len(self.t)]

    def calc_z_dot(self, y: np.ndarray) -> np.ndarray:
        r"""Calculate the discrete time stochastic process derivative using FFT algorithm
        and return values :math:`\dot{z}_n` with :math:`t_n <= t_\mathrm{max}`.
        """

        z_dot_fft = np.fft.fft(-1j * self.omega_k * self.yl * y)
        z_dot = z_dot_fft[0 : self.num_grid_points] * self.omega_min_correction
        return z_dot

    def get_num_y(self):
        r"""The number of independent random variables :math:`Y_m` is given by the number of discrete nodes
        used by the Fast Fourier Transform algorithm.
        """

        return len(self.t)


def alpha_times_pi(tau, alpha):
    return alpha(tau) * np.pi


def SD_over_pi(w, J):
    return J(w) / np.pi
