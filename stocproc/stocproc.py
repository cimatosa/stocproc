import abc
from functools import partial
import numpy as np
import time

from . import method_kle
from . import method_ft
import fcSpline

import logging

log = logging.getLogger(__name__)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))


def loggin_setup(
    sh_level=logging.INFO,
    sp_log_level=logging.INFO,
    kle_log_level=logging.INFO,
    ft_log_level=logging.INFO,
):
    """
    controls the logging levels

    On loading the ``stocproc`` module this function is called with its default argument
    which ammount to informative logging.

    :param sh_level: the logging level for the (output) StreamHandler
    :param sp_log_level: the logging level of the stocproc module log
    :param kle_log_level: the logging level of the KLE helper functions log (see :doc:`method_kle`)
    :param ft_log_level: the logging level of the FT helper functions log (see :doc:`method_ft`)
    """
    sh.setLevel(sh_level)

    log.addHandler(sh)
    log.setLevel(sp_log_level)

    method_kle.log.addHandler(sh)
    method_kle.log.setLevel(kle_log_level)

    method_ft.log.addHandler(sh)
    method_ft.log.setLevel(ft_log_level)


loggin_setup()


class StocProc(abc.ABC):
    r"""
    Interface definition for stochastic process implementations

    A new implementation for a stochastic process generator should subclass :py:class:`StocProc` and
    overwrite :py:func:`calc_z` and :py:func:`get_num_y`.

    Depending on the equally spaced times :math:`t_n = n \frac{t_{max}}{N-1}` with :math:`n = 0 \dots N-1`
    (:math:`N` = number of grid points),
    the function :py:func:`calc_z` should map :math:`M` independent complex valued and Gaussian
    distributed random variables :math:`Y_m` (with :math:`\langle Y_m \rangle = 0 = \langle Y_m Y_{m'} \rangle` and
    :math:`\langle Y_m Y^\ast_{m'} = \delta_{m m'}\rangle`) to the discrete time stochastic process :math:`z_n = z(t_n)`.

    :py:func:`get_num_y` needs to return the number :math:`M` of random variables :math:`Y_m` required as
    input for :py:func:`calc_z`.

    Having implemented :py:func:`calc_z` and :py:func:`get_num_y` the :py:class:`StocProc` provides
    convenient functions such as:

        - :py:func:`__call__`: evaluate the stochastic process for any time within the interval
          :math:`[0, t_{max}]` using cubic spline interpolation
        - :py:func:`get_time`: returns the times :math:`t_n`
        - :py:func:`get_z`: returns the discrete stochastic process :math:`z_n`
        - :py:func:`new_process`: draw new samples :math:`Y_m` and update :math:`z_n` as well as the
          cubic spline interpolator
        - :py:func:`set_scale`: set a scalar pre factor for the auto correlation function
          which scales the stochastic process such that :math:`\langle z(t) z^\ast(s)\rangle = \text{scale} \; \alpha(t-s)`.

    :param t_max: specifies the upper bound of the time interval
    :param num_grid_points: number of grid points :math:`N`
    :param seed: if not ``None`` seed the random number generator with ``seed``
    :param t_axis: an explicit definition of times t_k (may be non equidistant)
    :param scale: passes ``scale`` to :py:func:`set_scale`

    Note: :py:func:`new_process` is **not** called on init. If you want to evaluate a particular
    realization of the stocastic process, a new sample needs to be drawn by calling :py:func:`new_process`.
    Otherwise a ``RuntimeError`` is raised.

    """

    def __init__(self, t_max=None, num_grid_points=None, seed=None, scale=1):
        self.t_max = t_max
        self.num_grid_points = num_grid_points
        self.t = np.linspace(0, t_max, num_grid_points)

        self._z = None
        self._interpolator = None
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        self._one_over_sqrt_2 = 1 / np.sqrt(2)
        self._proc_cnt = 0
        self.scale = scale
        assert not np.isnan(scale)
        self.sqrt_scale = np.sqrt(self.scale)
        log.debug(
            "init StocProc with t_max {} and {} grid points".format(
                t_max, num_grid_points
            )
        )

    def __call__(self, t=None):
        r"""Evaluates the stochastic process.

        If ``t`` is not ``None`` cubic spline interpolation is used to evaluate :math:`z(t)` based on the
        discrete realization of the process :math:`z_n = z(t_n)`. ``t`` may be a single value or array like.

        If ``t`` is ``None`` the discrete process :math:`z_n` is returned.
        """
        if self._z is None:
            raise RuntimeError(
                "StocProc has NO random data, call 'new_process' to generate a new random process"
            )

        if t is None:
            return self._z
        else:
            return self._interpolator(t)

    @abc.abstractmethod
    def calc_z(self, y):
        r"""*abstract method*

        An implementation needs to map :math:`M` independent complex valued and Gaussian
        distributed random variables :math:`Y_m` (with :math:`\langle Y_m \rangle = 0 = \langle Y_m Y_{m'} \rangle` and
        :math:`\langle Y_m Y^\ast_{m'} = \delta_{m m'}\rangle`) to the discrete time stochastic process :math:`z_n = z(t_n)`.

        :return: the discrete time stochastic process :math:`z_n`, array of complex numbers
        """
        pass

    def _calc_scaled_z(self, y):
        r"""scaled the discrete process z with sqrt(scale), such that <z_i z^ast_j> = scale bcf(i,j)"""
        return self.sqrt_scale * self.calc_z(y)

    @abc.abstractmethod
    def get_num_y(self):
        r"""*abstract method*

        An implementation needs to return the number :math:`M` of random variables :math:`Y_m` required as
        input for :py:func:`calc_z`.
        """
        pass

    def get_time(self):
        r"""Returns the times :math:`t_n` for which the discrete time stochastic process may be evaluated."""
        return self.t

    def get_z(self):
        r"""Returns the discrete time stochastic process :math:`z_n = z(t_n)`."""
        return self._z

    def new_process(self, y=None, seed=None):
        r"""Generate a new realization of the stochastic process.

        If ``seed`` is not ``None`` seed the random number generator with ``seed`` before drawing new random numbers.

        If ``y`` is ``None`` draw new random numbers to generate the new realization. Otherwise use ``y`` as input to
        generate the new realization.
        """
        t0 = time.time()

        # clean up old data
        del self._interpolator
        del self._z

        self._proc_cnt += 1
        if seed != None:
            log.info("use fixed seed ({}) for new process".format(seed))
            np.random.seed(seed)
        if y is None:
            # random complex normal samples
            y = np.random.normal(
                scale=self._one_over_sqrt_2, size=2 * self.get_num_y()
            ).view(np.complex)
        else:
            if len(y) != self.get_num_y():
                raise RuntimeError(
                    "the length of 'y' ({}) needs to be {}".format(
                        len(y), self.get_num_y()
                    )
                )

        self._z = self._calc_scaled_z(y)
        log.debug(
            "proc_cnt:{} new process generated [{:.2e}s]".format(
                self._proc_cnt, time.time() - t0
            )
        )
        t0 = time.time()
        self._interpolator = fcSpline.FCS(x_low=0, x_high=self.t_max, y=self._z)
        log.debug("created interpolator [{:.2e}s]".format(time.time() - t0))

    def set_scale(self, scale):
        r"""
        Set a scalar pre factor for the auto correlation function
        which scales the stochastic process such that :math:`\langle z(t) z^\ast(s)\rangle = \text{scale} \; \alpha(t-s)`.
        """
        self.scale = scale
        self.sqrt_scale = np.sqrt(scale)


class StocProc_KLE(StocProc):
    r"""
    A class to simulate stochastic processes using Karhunen-Loève expansion (KLE) method.
    The idea is that any stochastic process can be expressed in terms of the KLE

    .. math:: Z(t) = \sum_i \sqrt{\lambda_i} Y_i u_i(t)

    where :math:`Y_i` and independent complex valued Gaussian random variables with variance one
    (:math:`\langle Y_i Y_j \rangle = \delta_{ij}`) and :math:`\lambda_i`, :math:`u_i(t)` are
    eigenvalues / eigenfunctions of the following homogeneous Fredholm equation

    .. math:: \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u_i(s) = \lambda_i u_i(t)

    for a given positive integral kernel :math:`R(\tau)`. It turns out that the auto correlation of the
    stocastic processes :math:`\langle Z(t)Z^\ast(s) \rangle = R(t-s)` is given by that kernel.

    For the numeric implementation the integral equation will be discretized
    (see :py:func:`stocproc.method_kle.solve_hom_fredholm` for details) which leads to a regular matrix
    eigenvalue problem.
    The accuracy of the generated  process in terms of its auto correlation function depends on
    the quality of the eigenvalues and eigenfunction and thus of the number of discritization points.
    Further for a given threshold there is only a finite number of eigenvalues above that threshold,
    provided that the number of discritization points is large enough.

    Now the property of representing the integral kernel in terms of the eigenfunction

    .. math :: R(t-s) = \sum_i \lambda_i u_i(t) u_i^\ast(s)

    is used to find the number of discritization points and the number of used eigenfunctions such that
    the sum represents the kernel up to a given tolerance (see :py:func:`stocproc.method_kle.auto_ng`
    for details).
    """

    def __init__(
        self,
        alpha,
        t_max,
        tol=1e-2,
        ng_fac=4,
        meth="fourpoint",
        diff_method="full",
        dm_random_samples=10 ** 4,
        seed=None,
        align_eig_vec=False,
        scale=1,
    ):
        r"""
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
        key = alpha, t_max, tol

        sqrt_lambda_ui_fine, t = method_kle.auto_ng(
            corr=alpha,
            t_max=t_max,
            ngfac=ng_fac,
            meth=meth,
            tol=tol,
            diff_method=diff_method,
            dm_random_samples=dm_random_samples,
        )

        # inplace alignment such that re(ui(0)) >= 0 and im(ui(0)) = 0
        if align_eig_vec:
            method_kle.align_eig_vec(sqrt_lambda_ui_fine)

        state = sqrt_lambda_ui_fine, t_max, len(t), seed, scale, key
        self.__setstate__(state)

    @staticmethod
    def get_key(r_tau, t_max, tol=1e-2):
        return r_tau, t_max, tol

    # def get_key(self):
    #     """Returns the tuple (r_tau, t_max, tol) which should suffice to identify the process in order to load/dump
    #     the StocProc class.
    #     """
    #     return self.key

    def __bfkey__(self):
        return self.key

    def __getstate__(self):
        return (
            self.sqrt_lambda_ui_fine,
            self.t_max,
            self.num_grid_points,
            self._seed,
            self.scale,
            self.key,
        )

    def __setstate__(self, state):
        sqrt_lambda_ui_fine, t_max, num_grid_points, seed, scale, self.key = state
        num_ev, ng = sqrt_lambda_ui_fine.shape
        super().__init__(
            t_max=t_max, num_grid_points=num_grid_points, seed=seed, scale=scale
        )
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


class StocProc_FFT(StocProc):
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
        negative_frequencies=False,
        scale=1,
    ):
        self.key = self.get_key(
            alpha=alpha, t_max=t_max, intgr_tol=intgr_tol, intpl_tol=intpl_tol
        )

        ft_ref = partial(alpha_times_pi, alpha=alpha)

        if not negative_frequencies:
            log.info("non neg freq only")
            a, b, N, dx, dt = method_ft.calc_ab_N_dx_dt(
                integrand=spectral_density,
                intgr_tol=intgr_tol,
                intpl_tol=intpl_tol,
                t_max=t_max,
                ft_ref=ft_ref,
                opt_b_only=True,
            )
        else:
            log.info("use neg freq")
            a, b, N, dx, dt = method_ft.calc_ab_N_dx_dt(
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
        super().__init__(
            t_max=t_max, num_grid_points=num_grid_points, seed=seed, scale=scale
        )

        self.yl = spectral_density(dx * np.arange(N) + a + dx / 2) * dx / np.pi
        self.yl = np.sqrt(self.yl)
        self.omega_min_correction = np.exp(
            -1j * (a + dx / 2) * self.t
        )  # self.t is from the parent class

    @staticmethod
    def get_key(t_max, alpha, intgr_tol=1e-2, intpl_tol=1e-2):
        """
        Returns the tuple ``(alpha, t_max, intgr_tol, intpl_tol)`` which uniquely identifies a particular
        :py:class:`StocProc_FFT` instance
        """
        return alpha, t_max, intgr_tol, intpl_tol

    def __getstate__(self):
        return (
            self.yl,
            self.num_grid_points,
            self.omega_min_correction,
            self.t_max,
            self._seed,
            self.scale,
            self.key,
        )

    def __setstate__(self, state):
        (
            self.yl,
            num_grid_points,
            self.omega_min_correction,
            t_max,
            seed,
            scale,
            self.key,
        ) = state
        super().__init__(
            t_max=t_max, num_grid_points=num_grid_points, seed=seed, scale=scale
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

    def get_num_y(self):
        r"""The number of independent random variables :math:`Y_m` is given by the number of discrete nodes
        used by the Fast Fourier Transform algorithm.
        """
        return len(self.yl)


class StocProc_TanhSinh(StocProc):
    r"""Simulate Stochastic Process using TanhSinh integration for the Fourier Integral"""

    def __init__(
        self,
        spectral_density,
        t_max,
        alpha,
        intgr_tol=1e-2,
        intpl_tol=1e-2,
        seed=None,
        negative_frequencies=False,
        scale=1,
    ):
        self.key = alpha, t_max, intgr_tol, intpl_tol

        if not negative_frequencies:
            log.info("non neg freq only")
            log.info("get_dt_for_accurate_interpolation, please wait ...")
            try:
                ft_ref = partial(alpha_times_pi, alpha=alpha)
                c = method_ft.find_integral_boundary(
                    lambda tau: np.abs(ft_ref(tau)) / np.abs(ft_ref(0)),
                    intgr_tol,
                    1,
                    1e6,
                    0.777,
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
            I = method_ft.fourier_integral_TanhSinh(
                f=sd_over_pi, x_max=wmax, n=n, tau_l=tau, t_max_ts=t_max_ts
            )
            bcf_ref_t = alpha(tau)

            d = np.abs(bcf_ref_t - I) / abs(bcf_ref_t[0])
            d = np.max(d)
            print("n:{} d:{} tol:{}".format(n, d, intgr_tol))

        tau = np.linspace(0, (N - 1) * dt_tol, N)
        log.info(
            "perform numeric check of entire time axis [{},{}] N:{}".format(
                0, (N - 1) * dt_tol, N
            )
        )
        num_FT = method_ft.fourier_integral_TanhSinh(
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
        print("done!")

        yk, wk = method_ft.get_x_w_and_dt(n, wmax, t_max_ts)
        self.omega_k = yk
        self.fl = np.sqrt(wk * spectral_density(self.omega_k) / np.pi)

        super().__init__(t_max=t_max, num_grid_points=N, seed=seed, scale=scale)

    @staticmethod
    def get_key(t_max, alpha, intgr_tol=1e-2, intpl_tol=1e-2):
        return alpha, t_max, intgr_tol, intpl_tol

    def __getstate__(self):
        return (
            self.fl,
            self.omega_k,
            self.num_grid_points,
            self.t_max,
            self._seed,
            self.scale,
            self.key,
        )

    def __setstate__(self, state):
        self.fl, self.omega_k, num_grid_points, t_max, seed, scale, self.key = state
        super().__init__(
            t_max=t_max, num_grid_points=num_grid_points, seed=seed, scale=scale
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
            map(lambda ti: np.sum(tmp1 * np.exp(tmp2 * ti)), self.t),
            dtype=np.complex128,
        )
        return z

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


def alpha_times_pi(tau, alpha):
    return alpha(tau) * np.pi


def SD_over_pi(w, J):
    return J(w) / np.pi
