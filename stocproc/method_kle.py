"""
Tests related to the submodule stocproc.method_kle
"""

# python imports
import logging
import time
from typing import Callable, Union

# third party imports
from numba import njit
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh as scipy_eigh
import fastcubicspline


# module imports
from stocproc import stocproc_c
from . import gquad
from . import util

log = logging.getLogger(__name__)


def solve_hom_fredholm(
    r: NDArray, w: NDArray, small_weights_problem=False
) -> tuple[NDArray, NDArray]:
    r"""Solves the discrete homogeneous Fredholm equation of the second kind

    $$ \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u(s) = \lambda u(t) $$

    Quadrature approximation of the integral gives a discrete representation
    which leads to the regular eigenvalue problem,

    $$ \sum_i w_i R(t_j-s_i) u(s_i) = \lambda u(t_j)
       \equiv \mathrm{diag(w_i)} \cdot R \cdot u = \lambda u . $$

    Note that, if $t_i = s_i \forall i$ the matrix $R_{ij} = R(t_j-s_i)$ is hermitian.
    In order to preserve hermiticity for arbitrary $w_i$, one defines the diagonal matrix
    $D = \mathrm{diag(\sqrt{w_i})}$  with leads to the equivalent expression:

    $$ D \cdot R \cdot D \cdot D \cdot u = \lambda D \cdot u \equiv \tilde R \tilde u = \lambda \tilde u $$

    where $\tilde R$ is hermitian and $u = D^{-1}\tilde u$.

    Note that if the weights become very small (i.g., for the tanh-sinh scheme)
    $u = D^{-1}\tilde u$ recovers the auto correlation matrix poorly.
    In that case it is advantageous to use $u(t_i) = \frac{1}{\lambda} R(t_i-t_j) \sqrt{w_j} \tilde u(t_j)$.
    If `small_weights_problem` is set to `True`, this expression will be used.

    Parameters:
        r: hermitian correlation matrix $R_{ij} = R(t_j-s_i)$
        w: integrations weights $w_i$ (they have to correspond to the discrete time $t_i$)
        small_weights_problem: if `True`, use the alternative expression (numerically slower)
            $u(t_i) = \frac{1}{\lambda} R(t_i-t_j) \sqrt{w_j} \tilde u(t_j)$ to
            calculate the eigen functions.

    Returns:
        a tuple of eigenvalues and eigenvectos, decreasingly ordered with respec to the magnitude of the eigenvalues

    !!! Info

        There are various convenient functions to calculate the integration weights and times
        to approximate the integral over the interval [0, t_max] using `ng` points.

        - [`stocproc.method_kle.get_nodes_weights_mid_point`][]
        - [`stocproc.method_kle.get_nodes_weights_trapezoidal`][]
        - [`stocproc.method_kle.get_nodes_weights_simpson`][]
        - [`stocproc.method_kle.get_nodes_weights_four_point`][]
        - [`stocproc.method_kle.get_nodes_weights_gauss_legendre`][]
        - [`stocproc.method_kle.get_nodes_weights_tanh_sinh`][]

    !!! Note

        It has been noticed that the performance of the various weights depends on the auto correlation
        function. As default one should use the 'simpson weights'. 'four point', 'gauss legendre' and 'tanh sinh'
        might perform better for auto correlation function that decay slowly. Their advantage becomes evident
        for a large numbers of grid points only. So if one cares about relative differences below 1e-4
        the more sophisticated weights are suitable.
    """
    t0 = time.time()

    n = len(w)
    w_sqrt = np.sqrt(w)
    # weighted matrix r due to quadrature weights
    if small_weights_problem:
        acf_matrix = r.copy()
    r = w_sqrt.reshape(n, 1) * r * w_sqrt.reshape(1, n)
    eig_val, eig_vec = scipy_eigh(r, overwrite_a=True)  # eig_vals in ascending

    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]
    if small_weights_problem:
        eig_vec = np.einsum(
            "ij, j, k, jk -> ik", acf_matrix, w_sqrt, 1 / eig_val, eig_vec
        )
    else:
        eig_vec = np.reshape(1 / w_sqrt, (n, 1)) * eig_vec

    dt = time.time() - t0
    log.debug(f"discrete fredholm equation of size {n} solved in {dt:.2e}s")
    return eig_val, eig_vec


def align_eig_vec(eig_vec: NDArray):
    """
    Rotate eigenvectors in the complex plane so $u(0) = u_0$ is real and positive.

    Modify the given parameter in place.

    Parameters:
        eig_vec: 2D numpy array with the `i`th eigen vectors at `eig_vec[:, i]`
    """
    for i in range(eig_vec.shape[1]):
        phase = np.exp(1j * np.arctan2(np.real(eig_vec[0, i]), np.imag(eig_vec[0, i])))
        eig_vec[:, i] /= phase


def _calc_corr_matrix(s: NDArray, acf: util.CplxFnc, is_equi=None) -> NDArray:
    r"""
    calculates the correlation matrix $\alpha_ij = \alpha(t_i-s_j)$

    For equidistant grid points $t_i$, exploit Toeplitz structure
    $\alpha_ij = \alpha(\Delta t (i-j)) = \alpha_{i-j}$ to efficiently construct the correlation matrix.

    Parameters:
        s: the time axes
        acf: the auto correlation function
        is_equi: tell whether `s` is an equidistant grid. If `None` analyze `s` to decide.
    Returns:
        the correlation matrix

    !!! ToDo

        speedup the non-equidistant case
    """
    if is_equi is None:
        is_equi = is_axis_equidistant(s)

    if is_equi:
        size_s = len(s)
        acf_s = acf(s)
        # this is [acf(-n dt), ..., acf(-dt), acf(0), acf(dt), ..., acf(n dt) ]
        acf_ms_s = np.hstack((acf_s[1:].conj()[::-1], acf_s))
        # we want
        # r = acf(   0) acf(-dt), acf(-2 dt)
        #     acf(  dt) acf(  0), acf(  -dt)
        #     acf(2 dt) acf( dt), acf(    0)
        r = np.empty(shape=(size_s, size_s), dtype=np.complex128)
        for i in range(size_s):
            idx = size_s - 1 - i
            # i-th column
            r[:, i] = acf_ms_s[idx : idx + size_s]
    else:
        return acf(s.reshape(-1, 1) - s.reshape(1, -1))
    return r


def get_nodes_weights_mid_point(
    t_max: float, num_grid_points: int
) -> tuple[NDArray, NDArray, bool]:
    r"""
    nodes and weights of the **mid-point rule** (equal integration weights)

    The $N$ equally spaced grid points with indices $i = 0, 1, ... N - 1$ are located at
    $t_i = i \Delta t$ with $\Delta t = \frac{t_\mathrm{max}}{N-1}$.
    The corresponding weights are $w_i = \Delta t$.

    Note that referring to the mid-point integration scheme, this actually
    corresponds to the integration from $-\Delta t / 2$ to $t_\mathrm{max} + \Delta t/2$.

    Parameters:
        t_max: end of the interval for the time grid $[0,t_\mathrm{max}]$
        num_grid_points: number of grid points $N$
    Returns:
        location of the grid points, corresponding weights and `True` since the grid points are spaced equidistantly
    """
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep=True)
    w = np.ones(num_grid_points) * delta_t
    return t, w, True


def get_nodes_weights_trapezoidal(
    t_max: float, num_grid_points: int
) -> tuple[NDArray, NDArray, bool]:
    r"""
    nodes and weights of the **trapezoidal rule**

    The $N$ equally spaced grid points with indices $i = 0, 1, ... N - 1$ are located at
    $t_i = i \Delta t$ with $\Delta t = \frac{t_\mathrm{max}}{N-1}$.
    The corresponding weights are $w_0 = w_{N-1} = \Delta t /2$ and $w_i = \Delta t$ for $0 < i < N - 1$.

    Parameters:
        t_max: end of the interval for the time grid $[0,t_\mathrm{max}]$
        num_grid_points: number of grid points $N$
    Returns:
        location of the grid points, corresponding weights and `True` since the grid points are spaced equidistantly
    """
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep=True)
    w = np.ones(num_grid_points) * delta_t
    w[0] /= 2
    w[-1] /= 2
    return t, w, True


def get_nodes_weights_simpson(
    t_max: float, num_grid_points: int
) -> tuple[NDArray, NDArray, bool]:
    r"""
    nodes and weights **simpson rule**

    The $N$ equally spaced grid points with indices $i = 0, 1, ... N - 1$ are located at
    $t_i = i \Delta t$ with $\Delta t = \frac{t_\mathrm{max}}{N-1}$.
    The corresponding weights are

    $$
            w_0      = w_{N-1} =  \Delta t / 3, \qquad
            w_{\mathrm{even}}  = 2\Delta t / 3, \qquad
            w_{\mathrm{odd}}   = 4\Delta t / 3.
    $$

    $N$ must be odd.

    Parameters:
        t_max: end of the interval for the time grid $[0,t_\mathrm{max}]$
        num_grid_points: number of grid points $N$
    Returns:
        location of the grid points, corresponding weights and `True` since the grid points are spaced equidistantly
    """
    if num_grid_points % 2 != 1:
        raise RuntimeError(
            "simpson weights needs grid points ng such that ng = 2*k+1, but git ng={}".format(
                num_grid_points
            )
        )
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep=True)
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::2] = 2 / 3 * delta_t
    w[1::2] = 4 / 3 * delta_t
    w[0] = 1 / 3 * delta_t
    w[-1] = 1 / 3 * delta_t
    return t, w, True


def get_nodes_weights_four_point(
    t_max: float, num_grid_points: int
) -> tuple[NDArray, NDArray, bool]:
    r"""
    nodes and weights **four-point Newton-Cotes rule**

    The $N$ equally spaced grid points with indices $i = 0, 1, ... N - 1$ are located at
    $t_i = i \Delta t$ with $\Delta t = \frac{t_\mathrm{max}}{N-1}$.
    The corresponding weights are

    $$
            w_0    = w_{N-1} =  28 \Delta t / 90, \quad
            w_{+0}           =  56 \Delta t / 90, \quad
            w_{+1} = w_{+3}  = 128 \Delta t / 90, \quad
            w_{+2}           =  48 \Delta t / 90.
    $$

    $N$ must be a multiple of 4 plus 1.

    Parameters:
        t_max: end of the interval for the time grid $[0,t_\mathrm{max}]$
        num_grid_points: number of grid points $N$
    Returns:
        location of the grid points, corresponding weights and `True` since the grid points are spaced equidistantly
    """
    if num_grid_points % 4 != 1:
        raise RuntimeError(
            "four point weights needs grid points ng such that ng = 4*k+1, but git ng={}".format(
                num_grid_points
            )
        )
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep=True)
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::4] = 56 / 90 * delta_t
    w[1::4] = 128 / 90 * delta_t
    w[2::4] = 48 / 90 * delta_t
    w[3::4] = 128 / 90 * delta_t
    w[0] = 28 / 90 * delta_t
    w[-1] = 28 / 90 * delta_t
    return t, w, True


def get_nodes_weights_gauss_legendre(
    t_max: float, num_grid_points: int
) -> tuple[NDArray, NDArray, bool]:
    r"""
    Calculate the nodes and weights for **Gauss integration**
    by expanding the function in terms of Legendre Polynomials, known as
    [Gauss–Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

    See [`stocproc.gquad`][] for details in the implementation.

    Parameters:
        t_max: end of the interval for the time grid $[0,t_\mathrm{max}]$
        num_grid_points: number of grid points $N$
    Returns:
        location of the grid points, corresponding weights and
            `False` since the grid points are **not spaced equidistantly**
    """
    t, w = gquad.gauss_nodes_weights_legendre(n=num_grid_points, a=0, b=t_max)
    return t, w, False


def get_nodes_weights_tanh_sinh(
    t_max: float, num_grid_points: int
) -> tuple[NDArray, NDArray, bool]:
    r"""
    Calculate the nodes and weights for **Tanh-Sinh integration**.

    As elucidated in *Tanh-Sinh High-Precision Quadrature - David H. Bailey*,
    the idea is to transform the integral over a finite interval, i.e, $t \in [-1, 1]$ via
    the variable transformation $t = \tanh(\pi/2 \sinh(x))$ to an integral over the entire
    real axis $x \in [-\infty,\infty]$ but where the transformed integrand decays rapidly
    such that the simple midpoint rule performs very well on the transformed axes.

    For a fixed small parameter h the location of the grid points read $t_i = \tanh(\pi/2 \sinh(i h)$
    with corresponding weights $w_i = \frac{\pi/2 \cosh(ih)}{\cosh^2(\pi/2 \sinh(i h))}$
    where i can be any integer. For a given number of grid points N, h is chosen such that
    $w_{(N-1)/2} < 10^{-14}$ which implies odd N. With that particular $h$, $t_i$ and
    $w_i$ are calculated for $-(N-1)/2 \leq i \leq (N-1)/2$. Afterwards, the $t_i$ are linearly
    scaled such that $t_{-(N-1)/2} = 0$ and $t_{(N-1)/2} = t_\mathrm{max}$.

    Parameters:
        t_max: end of the interval for the time grid $[0,t_\mathrm{max}]$
        num_grid_points: number of grid points $N$
    Returns:
        location of the grid points, corresponding weights and
            `False` since the grid points are **not spaced equidistantly**
    """

    def get_h_of_n(n):
        r"""
        returns the stepsize h for sinh tanh quad for a given number of points N
        such that the smallest weights are about 1e-14
        """
        a = 16.12087683080651
        b = -2.393599730652087
        c = 6.536936185577097
        d = -1.012504470475915
        if n < 4:
            raise ValueError("only tested for N >= 4")
        return a * n**b + c * n**d

    h = get_h_of_n(num_grid_points)
    if num_grid_points % 2 != 1:
        raise RuntimeError(
            "sinh tanh weights needs grid points ng such that ng = 2*k+1, but git ng={}".format(
                num_grid_points
            )
        )
    kmax = (num_grid_points - 1) / 2
    k = np.arange(0, kmax + 1)
    w = h * np.pi / 2 * np.cosh(k * h) / np.cosh(np.pi / 2 * np.sinh(k * h)) ** 2
    w = np.hstack((w[-1:0:-1], w)) * t_max / 2

    tmp = np.pi / 2 * np.sinh(h * k)
    y_plus = 1 / (np.exp(tmp) * np.cosh(tmp))
    t = np.hstack((y_plus[-1:0:-1], (2 - y_plus))) * t_max / 2
    return t, w, False


def interpolate_eigenfunction(
    acf: util.CplxFnc,
    t: NDArray,
    w: NDArray,
    eig_vec: NDArray,
    eig_val: float,
    t_fine: NDArray,
):
    r"""
    interpolate the discrete eigenfunction of the Fredholm equation according to

    $$
        u(t) = \frac{1}{\lambda } \int_0^T \mathrm{d}s \alpha(t-s) u(s) = \sum_i w_i \alpha(t-s_i) u(s_i) .
    $$

    Parameters:
        acf: the auto correlation function
        t: the time axes $s_i$ of the discrete eigen function (integration nodes)
        w: the integration weights
        eig_vec: the discrete eigen function (vector) $u(s_i)$
        eig_val: the eigenvalue $\lambda$
        t_fine: the fine time axes for which the eigen function is evaluated using Fredholm interpolation
    Returns:
        the values of eigen function obtained by interpolation for the times given in `t_fine`
    """
    return np.asarray([np.sum(acf(ti - t) * w * eig_vec) for ti in t_fine]) / eig_val


def subdivide_axis(t: NDArray, ng_fac: int) -> NDArray:
    r"""
    Subdivide the $t$ axes $[t_0, t_1, \dots t_{n-1}]$ to obtain a finder grid.

    Each interval $[t_i, t_{i+1}]$ is divided into `ng_fac` sub-intervals of equal width.
    Therefore, the returned grid has `(n-1) ng_fac + 1` points.
    Note that, $t$ does **not** need to be equally spaced.

    Parameters:
        t: the axis (grid) of which to create a finer sub-grid
        ng_fac: the number of sub-intervals per interval $[t_i, t_{i+1}]$
    Returns:
         the finer grid
    """
    n = len(t)
    if not isinstance(t, np.ndarray):
        t = np.asarray(t)
    t_fine = np.empty(shape=((n - 1) * ng_fac + 1))
    t_fine[::ng_fac] = t
    for l in range(1, ng_fac):
        t_fine[l::ng_fac] = (l * t[1:] + (ng_fac - l) * t[:-1]) / ng_fac
    return t_fine


def auto_ng(
    acf: util.CplxFnc,
    t_max: float,
    ng_fac: int = 2,
    quad_scheme: Union[
        str, Callable[[float, int], tuple[NDArray, NDArray, bool]]
    ] = get_nodes_weights_mid_point,
    tol: float = 1e-3,
    diff_method: str = "full",
    dm_random_samples: float = 10**4,
    ret_eig_vals: bool = False,
    relative_difference: bool = False,
):
    r"""
    Exponentially increase the number of grid points of the discrete
    Fredholm equation until the desired accuracy is met.
    The accuracy is determined from the deviation of the approximated
    auto correlation of the Karhunen-Loève expansion from the given
    reference auto correlation.

    $$
        \Delta(n) = \max_{t,s \in [0,t_\mathrm{max}]}
        \left( \Big | R(t-s) - \sum_{i=1}^{n} \lambda_i u_i(t) u_i^\ast(s) \Big | \right )
    $$

    Parameters:
        acf:
            the reference auto correlation function $R$
        t_max:
            specifies the interval [0, t_max] of the stochastic process
        ng_fac:
            specifies the fine grid to use to check against spline interpolation.
            A value of `n` means that the fine grid splits an interval of the rough grid into `n` sub-intervals.
            The reference value of the eigenfunctions at the intermediate points (new points of thefine grid)
            are calculated using integral interpolation (see book Numerical Recipes - Fredholm Equation [1]).
        quad_scheme:
            the method used to calculate the integration weights and times,
            a callable or one of the following strings:
            `midpoint` (`midp`), `trapezoidal` (`trapz`), `simpson` (`simp`), `fourpoint` (`fp`),
            `gauss_legendre` (`gl`), `tanh_sinh` (`ts`)
        tol:
            defines the success criterion max(abs(corr_exact - corr_reconstr)) < tol
        diff_method:
            either `full` or `random`, determines the points where the above success criterion is evaluated,
            `full`: full grid in between the fine grid, such that
            the spline interpolation error is expected to be maximal
            `random`: pick a fixed number of random times t and s within the interval [0, t_max]
        dm_random_samples:
            the number of random times used for diff_method `random`
        ret_eig_vals:
            if `True`, return also the eigen values
        relative_difference:
            if `True`, use relative difference instead of absolute difference for the success criterion

    Returns:
        an array containing the necessary eigenfunctions of the Karhunen-Loève expansion for sampling the
            stochastic processes (shape=(num_eigen_functions, num_grid_points) and the related time axes,
            which is not necessarily equidistant.

    The procedure works as follows:
        1) **discritize the integral**
            Solve the discrete Fredholm equation on a grid with `ng` points.
            This gives `ng` eigenvalues/vectors where each `ng`-dimensional vector approximates the continuous
            eigenfunction ($t, u_i(t) \leftrightarrow t_k, u_{ik}$ where the $t_k$ depend on the integration weights
            method. Thus, they are not necessarily equally spaced).

        2) **interpolate to finer grid**
            Since solving the eigenvalue problem is numerically expensive, we approximate the eigenfunction
            on a finer grid with $ng_\mathrm{fine} = ng_\mathrm{fac}(ng-1)+1$ grid points, using

           $$
               u_i(t) = \frac{1}{\lambda_i} \int_0^{t_\mathrm{max}} \mathrm{d}s \; \alpha(t-s) u_i(s)
                        \approx \frac{1}{\lambda_i} \sum_k w_k \alpha(t-s_k) u_{ik} .
           $$

           According to the book Numerical Recipes [1], this interpolation should perform better that simple
           spline interpolation. However, it turns that this is not the general case. E.g., for an exponential
           auto correlation functions the spline interpolation performs better. For that reason it might be
           useful to set `ngfac` to 1 which will skip the interpolation.

        3) **spline interpolation**
            Based on the eigenfunction on the fine grid, set up a cubic spline interpolation.
            Use the spline interpolation to estimate the deviation $\Delta(n)$. When using
            `diff_method = 'full'`, the maximization is performed over all $t'_i, s'_j$ where
            $t'_i = (t_i + t_{i+1})/2$ and $s'_i = (s_i + s_{i+1})/2$ with $i,j$ beeing
            the indices of the fine grid. It is expected that the interpolation error is maximal
            when being in between the reference points.


        4) **increase the number of eigenfunctions**

        Calculate the deviation $\Delta(n)$ for increasing value of $n$
        (error when using the first $n$ eigenfunctions).
        If there is no $n$ which fulfills the success criterion, double the number of grid points,
        $2 ng-1 \rightarrow ng$, and start over again.

    !!! Note

        The scaling of the error of the various integration methods does not correspond to the scaling of
        the number of eigenfunctions to use in order to reconstruct the auto correlation function within
        a given tolerance. Surprisingly it turns out that in general the most trivial **mid-point method** performs
        quite well. If other methods suite better needs to be check in every case.

    [1] Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P.,
    2007. Numerical Recipes 3rd Edition: The Art of Scientific Computing,
    Auflage: 3. ed. Cambridge University Press, Cambridge, UK ; New York. (pp. 990)

    !!! ToDo

        There is certainly room for cython as well as improvements on readability of the
        current implementation.
    """

    time_start = time.time()

    if diff_method == "full":
        alpha_ref = t_rand = s_rand = None
    elif diff_method == "random":
        np.random.seed(0)
        t_rand = np.random.rand(dm_random_samples) * t_max
        s_rand = np.random.rand(dm_random_samples) * t_max
        alpha_ref = acf(t_rand - s_rand)
    else:
        raise ValueError(
            "unknown diff_method '{}', use 'full' or 'random'".format(diff_method)
        )
    log.info(f"start auto_ng (tol: {tol}, diff_method: {diff_method}, rel. diff.: {relative_difference}")

    time_fredholm = 0
    time_calc_ac = 0
    time_integr_intp = 0
    time_spline = 0
    time_calc_diff = 0

    if isinstance(quad_scheme, str):
        quad_scheme = quad_scheme_name_to_function(quad_scheme)

    k = 4
    # double number of grid points for the discrete eigen value approximation
    # of the Fredholm equation
    while True:
        k += 1
        ng = 2**k + 1
        t, w, is_equi = quad_scheme(t_max, ng)

        # construct the auto correlation matrix
        t0 = time.time()
        r = _calc_corr_matrix(t, acf, is_equi)
        time_calc_ac += time.time() - t0

        # solve the discrete fredholm equation
        t0 = time.time()
        _eig_val, _eig_vec = solve_hom_fredholm(r, w)
        time_fredholm += time.time() - t0

        t_fine = subdivide_axis(t, ng_fac)   # setup fine
        ts_fine = subdivide_axis(t_fine, 2)  # and super fine time grid

        # this is needed for efficient integral interpolation
        if is_equi:
            t0 = time.time()
            acf_t_fine = acf(t_fine)
            # this is [acf(-n dt), ..., acf(-dt), acf(0), acf(dt), ..., acf(n dt) ]
            # from -tmax untill tmax on the fine grid
            alpha_k = np.hstack((acf_t_fine[1:].conj()[::-1], acf_t_fine))
            time_calc_ac += time.time() - t0
        else:
            alpha_k = None

        if diff_method == "full":
            alpha_ref = _calc_corr_matrix(ts_fine, acf, is_equi=is_equi)

        diff = -alpha_ref
        if relative_difference:
            abs_alpha_res = np.abs(alpha_ref)
        else:
            abs_alpha_res = 1

        # subsequently add interpolated (by the KLE-kernel) eigen functions
        # to reconstruct the ACF
        sqrt_lambda_ui_fine_all = []
        md = None
        for i in range(ng):
            eig_vec = _eig_vec[:, i]
            if _eig_val[i] < 0:
                # print(ng, i)
                break
            sqrt_eval = np.sqrt(_eig_val[i])
            if ng_fac != 1:
                t0 = time.time()
                # when using sqrt_lambda instead of lambda we get sqrt_lamda time u
                # which is the quantity needed for the stochastic process generation
                if not is_equi:
                    sqrt_lambda_ui_fine = interpolate_eigenfunction(
                        acf=acf,
                        t=t,
                        w=w,
                        eig_vec=eig_vec,
                        eig_val=sqrt_eval,
                        t_fine=t_fine,
                    )
                else:
                    sqrt_lambda_ui_fine = stocproc_c.eig_func_interp(
                        delta_t_fac=ng_fac,
                        time_axis=t,
                        alpha_k=np.asarray(alpha_k, dtype=np.complex128),
                        weights=w,
                        eigen_val=sqrt_eval,
                        eigen_vec=eig_vec,
                    )

                time_integr_intp += time.time() - t0
            else:
                sqrt_lambda_ui_fine = eig_vec * sqrt_eval

            sqrt_lambda_ui_fine_all.append(sqrt_lambda_ui_fine)

            # setup cubic spline interpolator
            t0 = time.time()
            if not is_equi:
                sqrt_lambda_ui_spl = util.ComplexInterpolatedUnivariateSpline(
                    t_fine, sqrt_lambda_ui_fine, noWarning=True
                )
            else:
                sqrt_lambda_ui_spl = fastcubicspline.FCS(
                    x_low=0, x_high=t_max, y=sqrt_lambda_ui_fine
                )
            time_spline += time.time() - t0

            # calculate the max deviation
            t0 = time.time()
            if diff_method == "random":
                ui_t = sqrt_lambda_ui_spl(t_rand)
                ui_s = sqrt_lambda_ui_spl(s_rand)
                diff += ui_t * np.conj(ui_s)
            elif diff_method == "full":
                ui_super_fine = sqrt_lambda_ui_spl(ts_fine)
                diff += np.outer(ui_super_fine, np.conj(ui_super_fine))
            md = np.max(np.abs(diff) / abs_alpha_res)
            time_calc_diff += time.time() - t0

            log.debug("num evec {} -> max diff {:.3e}".format(i + 1, md))

            if md < tol:
                time_total = (
                    time_calc_diff
                    + time_spline
                    + time_integr_intp
                    + time_calc_ac
                    + time_fredholm
                )
                time_overall = time.time() - time_start
                time_rest = time_overall - time_total

                log.info(
                    "calc_ac {:.3%}, fredholm {:.3%}, integr_intp {:.3%}, spline {:.3%}, calc_diff {:.3%}, rest {:.3%}".format(
                        time_calc_ac / time_overall,
                        time_fredholm / time_overall,
                        time_integr_intp / time_overall,
                        time_spline / time_overall,
                        time_calc_diff / time_overall,
                        time_rest / time_overall,
                    )
                )
                log.info(
                    "auto ng SUCCESSFUL with {} grid points -> max diff {:.3e} < tol {:.3e}, used eig. vec. {}".format(
                        ng, md, tol, ng, i + 1
                    )
                )
                if ret_eig_vals:
                    return (
                        np.asarray(sqrt_lambda_ui_fine_all),
                        t_fine,
                        _eig_val[: len(sqrt_lambda_ui_fine_all)],
                    )
                else:
                    return np.asarray(sqrt_lambda_ui_fine_all), t_fine

        # We have added all eigen function, but the error is still too large.
        # This means that the discretization is not fine enough.
        log.info(f"{ng} grid points -> final deviation of {md:.3e} > tol {tol:.3e}")

def is_axis_equidistant(ax: NDArray):
    r"""
    Return `True` if the axes `ax` is equidistant, i.e., $t_n = n \Delta t$.
    """
    ax = np.asarray(ax)
    d = ax[1:] - ax[:-1]
    return np.max(np.abs(d - d[0])) < 1e-15


def quad_scheme_name_to_function(
    quad_scheme_name: str,
) -> Callable[[float, int], tuple[NDArray, NDArray, bool]]:
    """
    Convenient access by name to the generator functions for the integration nodes and weights.

    The parameters `quad_scheme_name` (str) may be:

    * `midpoint` or `midp`: [stocproc.method_kle.get_nodes_weights_mid_point][]
    * `trapezoidal` or `trapz`: [stocproc.method_kle.get_nodes_weights_trapezoidal][]
    * `simpson` or `simp`: [stocproc.method_kle.get_nodes_weights_simpson][]
    * `fourpoint` or `fp`: [stocproc.method_kle.get_nodes_weights_four_point][]
    * `gauss_legendre` or `gl`: [stocproc.method_kle.get_nodes_weights_gauss_legendre][]
    * `tanh_sinh` or `ts`: [stocproc.method_kle.get_nodes_weights_tanh_sinh][]

    Parameters:
        quad_scheme_name: a string naming the integration scheme
    Returns:
        the function that generated the nodes and weights for that integration scheme
    """
    if (quad_scheme_name == "midpoint") or (quad_scheme_name == "midp"):
        return get_nodes_weights_mid_point
    elif (quad_scheme_name == "trapezoidal") or (quad_scheme_name == "trapz"):
        return get_nodes_weights_trapezoidal
    elif (quad_scheme_name == "simpson") or (quad_scheme_name == "simp"):
        return get_nodes_weights_simpson
    elif (quad_scheme_name == "fourpoint") or (quad_scheme_name == "fp"):
        return get_nodes_weights_four_point
    elif (quad_scheme_name == "gauss_legendre") or (quad_scheme_name == "gl"):
        return get_nodes_weights_gauss_legendre
    elif (quad_scheme_name == "tanh_sinh") or (quad_scheme_name == "ts"):
        return get_nodes_weights_tanh_sinh
    else:
        raise ValueError(
            "unknown method to get integration weights '{}'".format(quad_scheme_name)
        )
