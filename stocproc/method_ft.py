"""
    The method_fft module provides convenient function to
    setup a stochastic process generator using fft method
"""
# python import
import logging
from multiprocessing import Pool
import sys
from typing import Callable, Union
import warnings

# third party
import fastcubicspline
import numpy as np
from numpy.fft import rfft as np_rfft
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq

# stocproc module imports
from . import util

# warnings.simplefilter('error')
MAX_FLOAT = sys.float_info.max
log = logging.getLogger(__name__)


class FTReferenceError(Exception):
    pass


def find_integral_boundary(
    integrand: util.CplxFnc,
    direction: str,
    tol: float = 1e-10,
    ref_val: float = 0,
    max_num_iteration: int = 100,
) -> float:
    """
    Searches for the point x_tol where for the integrand f, |f(x_tol)| = tol holds.
    This should be useful to approximate an indefinite integral by an integral with finite bounds.

    If direction is 'right', search for x_tol >= ref_val, i.e., right of ref_val.
    If direction is 'left', search for x_tol <= ref_val, i.e., left of ref_val.

    It is assumed that |f(x)| decays monotonically for all |x| > |x_tol|.

    Note that if f(ref_val) == 0, we increase (decrease) ref_val in steps of 0.1 if
    direction is 'right' ('left') until f(ref_val) != 0.
    This is useful for Ohmic SD, and even mor for shifted Ohmic SD, where J(0) = 0.

    If, however, 0 < |f(ref_val)| < tol, the integrand behaves strangely and a RuntimeError is raised.

    Parameters:
        integrand: the real valued function f(x) for which to find the bound
        direction: if 'left', search left of ref_value, if 'right' search right
        tol: a small value which defines the bound x_tol by |f(x_tol)| = tol
        ref_val: the initial value for starting the boundary search
        max_num_iteration: allows to pose a failure condition for ill posed problems
    Returns:
        the x boundary, i.e., the x value for which |f(x)| = tol holds
    """
    i = 0
    if direction.lower() == "right":
        x_step = 1
    elif direction.lower() == "left":
        x_step = -1
    else:
        raise ValueError("direction must be either 'right' or 'left'")

    def abs_integrand(x):
        return abs(integrand(x))

    I_ref = abs_integrand(ref_val)

    log.debug(
        f"find_integral_boundary for f(x), start at x={ref_val:.2e} and search {direction}"
    )
    log.debug(f"f({ref_val:.2e} =  {I_ref:.2e}")

    if I_ref == 0:
        log.debug("f(ref_val) == 0 -> change ref_val in small steps")
        for i in range(max_num_iteration):
            ref_val += 0.1 * x_step
            I_ref = abs_integrand(ref_val)
            log.debug(f"  f({ref_val:.2e} =  {I_ref:.2e}")
            if I_ref != 0:
                log.debug("  f(ref_val) > 0 -> stop")
                break
        if I_ref == 0:
            raise RuntimeError(
                f"f(ref_val={ref_val:.2}) is still zero after {max_num_iteration} iterations"
            )

    if I_ref < tol:
        log.debug(f"f(ref_val) < tol: {tol:.2e}")
        raise RuntimeError(
            f"f(ref_val)={I_ref:.2e} must not be smaller than tol: {tol:.2e}"
        )

    if I_ref == tol:
        log.debug(f"f(ref_val) = tol: {tol:.2e}")
        log.debug("done!")
        return ref_val

    log.debug(f"f(ref_val) > tol: {tol:.2e} -> start search")
    x = ref_val
    while True:
        i += 1
        if i > max_num_iteration:
            raise RuntimeError(f"max number of iteration {max_num_iteration} reached")
        x_old = x
        x = ref_val + x_step
        I_x = abs_integrand(x)
        log.debug(f"x_step: {x_step:.2e} -> x: {x:.2e} -> f(x): {I_x:.2e}")
        if I_x < tol:
            break
        x_step *= 1.3
    x_tol = brentq(lambda _x: abs_integrand(_x) - tol, x_old, x)
    I_x_tol = abs_integrand(x_tol)
    log.debug(f"found x_tol: {x_tol:.2e} f(x): {I_x_tol:.2}")
    log.debug("done!")
    return x_tol


def find_integral_boundary_auto(
    integrand: util.CplxFnc,
    tol: float = 1e-10,
    ref_val: float = 0,
    ref_val_left: Union[None, float] = None,
    ref_val_right: Union[None, float] = None,
    max_num_iteration: int = 100,
) -> tuple[float, float]:
    """
    find the left and right boundary of the function 'integrand',
    i.e. f(x_tol_left) = tol and f(x_tol_right) = tol.

    See 'find_integral_boundary' for details on the behavior and possible exceptions.

    Parameters:
        integrand: the real valued function f(x) for which to find the bound
        tol: a small value which defines the bound x_tol by f(x_tol) = tol
        ref_val: the initial value for starting the boundary search
        ref_val_left: if not None, overwrites ref_val for the 'left' search
        ref_val_right: if not None, overwrites ref_val for the 'lright' search
        max_num_iteration: allows to pose a failure condition for ill posed problems

    Returns:
        the left and the right boundary
    """

    ref_val_left = ref_val if ref_val_left is None else ref_val_left
    ref_val_right = ref_val if ref_val_right is None else ref_val_right

    log.debug("trigger left search")
    a = find_integral_boundary(
        integrand,
        direction="left",
        tol=tol,
        ref_val=ref_val_left,
        max_num_iteration=max_num_iteration,
    )
    log.debug("trigger right search")
    b = find_integral_boundary(
        integrand,
        direction="right",
        tol=tol,
        ref_val=ref_val_right,
        max_num_iteration=max_num_iteration,
    )
    return a, b


def fourier_integral_midpoint_fft(
    integrand: util.CplxFnc,
    a: float,
    b: float,
    n: int,
) -> tuple[NDArray, NDArray]:
    """
    Approximates F(t_i) int_a^b dx integrand(x) exp(i x t_i) by the riemann sum
    with N terms (the simplest uniform midpoint weights).

    Use FFT algorithm to perform the summation. This also yields the times t_i.

    Parameters:
        integrand: the function of which to approximate the Fourier integral
        a: left integration boundary
        b: right integration boundary
        n: number of equidistant grid points of the x-axes

    Returns:
        a tuple with
            1) the times t_i (due to the FFT formalism) as numpy array
            2) the values F(t_i) of the Fourier integral
    """
    delta_x = (b - a) / n
    delta_k = 2 * np.pi / (b - a)
    yl = integrand(np.linspace(a + delta_x / 2, b + delta_x / 2, n, endpoint=False))
    fft_vals = np_rfft(yl)
    tau = np.arange(len(fft_vals)) * delta_k
    return tau, delta_x * np.exp(-1j * tau * (a + delta_x / 2)) * fft_vals


def simpson_weights(n: int) -> NDArray:
    """
    Compute the weights w_i for the simpson integration scheme for equidistant nodes
    (piece-wise quadratic approximation of the integrand).
    This method also works for an EVEN number of grid points.

    Note that the step size dx is not yet included, so the formula for the actual
    integral reads

        F = int_a^b dx f(x) approx  dx sum_i=1^n w_i f(x_i)

    with n equidistant x_i and x_1 = a, x_n = b, dx = x_i+1 - x_i

    Parameters:
        n: number of grid points (nodes)
    Returns:
        weights for the simpson integration scheme
    """
    w = np.empty(shape=n, dtype=np.float64)
    n_odd = n if n % 2 == 1 else n - 1

    # first, use n_odd points "only"
    w[1:n_odd:2] = 4 / 3  # points at interval middle
    w[2 : n_odd - 1 : 2] = 2 / 3  # points connecting two interval (factor 2)
    w[0] = 1 / 3  # starting point
    w[n_odd - 1] = 1 / 3  # end point, in case of odd number of points

    # if we have an even number of points,
    # we can still put a quadratic function through the last three points
    # and evaluate the integral from n-1 to n, which yields the
    # following correction
    if n != n_odd:
        w[-3] -= 1 / 12
        w[-2] += 8 / 12
        w[-1] = 5 / 12

    return w


def fourier_integral_simps_fft(
    integrand: util.CplxFnc,
    a: float,
    b: float,
    n: int,
) -> tuple[NDArray, NDArray]:
    """
    approximates F(t_i) int_a^b dx integrand(x) exp(i x t_i) by the riemann sum with n terms
    using simpson integration scheme

    Parameters:
        integrand: the function of which to evaluate the Fourier integral
        a: lower integral bound
        b: upper integral bound
        n: number of gird points (nodes)
    Returns:
        a tuple with
            1) the times t_i (due to the FFT formalism) as numpy array
            2) the values F(t_i) of the Fourier integral
    """
    delta_x = (b - a) / (n - 1)
    delta_k = 2 * np.pi / n / delta_x
    l = np.arange(0, n)
    yl = integrand(a + l * delta_x)
    wl = simpson_weights(n)

    fft_vals = np_rfft(wl * yl)
    tau = np.arange(len(fft_vals)) * delta_k
    return tau, delta_x * np.exp(-1j * tau * a) * fft_vals


def tanhsinh_g_ast_of_t(t: float) -> float:
    """
    Return the variable transformation g_ast(t) for the tanh-sinh integration scheme.

    The variable transformation for the tanh-sinh integration scheme reads

        x(t) = b/2  (1-g(t))
        with
        g(t) = tanh(pi/2 sinh(t))

    If x goes from [0, b] then t goes from [inf, -inf].
    The singularity at x=0 is stretched to infinity.

    More suitable for numerics x(t) is expressed as

        x(t) = b/2  g_ast(t)
        with
        g_ast(t) = 1 / ( exp(pi/2 sinh(t)) cosh(pi/2 sinh(t)) )

    Parameters:
        t: the new integration variable
    Returns:
        g_ast(t)
    """
    s_t = np.pi / 2 * np.sinh(t)
    return 1 / (np.exp(s_t) * np.cosh(s_t))


def tanhsinh_w_t(t: float) -> float:
    """
    Return the weight function for the tanh_sinh integration scheme.

    The variable transformation for the tanh-sinh integration scheme reads

        x(t) = b/2  (1-g(t))
        with
        g(t) = tanh(pi/2 sinh(t))

    which implies that the effective integrand becomes

        f(x(t)) d x(t) / dt = w(t) f(x(t))
        with
        w(t) =  pi cosh(t)/(2 cosh^2(pi/2 sinh(t)))

    Parameters:
        t: the new integration variable
    Returns:
        the "weight" function w(t)
    """
    s_t = np.sinh(t) * np.pi / 2
    return np.pi * np.cosh(t) / 2 / np.cosh(s_t) ** 2


def tanhsinh_t_max_for_singularity(
    f: util.CplxFnc,
    a: float,
    b: float,
    tol: float,
    max_num_iterations: int = 50,
) -> float:
    """
    Numeric integration of a function with a singularity at the boundary 'a' is efficiently
    done using tanh-sinh integration scheme.
    For that, the integral over [a,b] is mapped to an integral over [-inf, inf] with a
    new effective function to integrate

    The 'get_t_max_for_singularity_ts' procedure determines the boundary for that new effective function.

    Note that using floating point numbers, this mapping does not provide much benefit if the singularity is
    not at zero. So we raise a warning whenever a != 0.

    In particular t_tol is returned which fulfils |w_(t_tol) I(x(t_tol))| < tol.
    Note that t starts at 3. If the above criterion is not met, t is increased by 0.1 iteratively,
    otherwise that t is returned as t_tol.
    An RuntimeError is raised if max_num_iterations are reached.

    Parameters:
        f: the function of interest, with a possible singularity at 'a'
        a: left integration boundary
        b: right integration boundary
        tol: tolerance, from which on we dare to neglect contributions to the integral.
        max_num_iterations: maximum number of iteration allowed. If reached, a RuntimeError is raised.
    """
    if a != 0:
        log.warning(
            "a should be 0 for the tanh-sinh integration scheme to be beneficial!"
        )

    sc = (b - a) / 2
    t = 3
    log.debug("get_t_max_for_singularity_ts")
    for i in range(max_num_iterations):
        g_t = tanhsinh_g_ast_of_t(t)
        w_t = tanhsinh_w_t(t)
        f_x = f(a + sc * g_t)
        tmp = abs(sc * f_x * w_t)
        log.debug(f"at t={t} we have |w(t) f(x(t))| = {tmp:.2e}")
        if tmp < tol:
            log.debug(
                f"criterion |w(t) f(x(t))| < tol = {tol} was met -> return t_tol={t}"
            )
            return t
        t += 0.1


def tanhsinh_x_and_w(n: int, x_max: float, t_max: float) -> tuple[NDArray, NDArray]:
    """
    Return nodes x and corresponding weights w for tanh-sinh integration such that

        int_0^x_max dx f(x) = sum_(i=1)^n w_i f(x_i)

    The integral over x from [0, x_max] is mapped to an integral over t from [-3, t_max]
    using n equidistant point on the t axes.
    Note that values smaller t=-3 correspond to x values so close to x_max that they cannot be
    resolved using floating points numbers (difference smaller than 1e-16).

    Since t -> infinity corresponds to x -> 0, we can resolve values up  to 1e-300, so the choice
    depends on the integrand (see get_t_max_for_singularity_ts).

    Parameters:
        n: number of equidistant grid points on the t-axes
        x_max: the upper integral bound on the x-axes (corresponds to t -> -infinity, t_min can be set to 3)
        t_max: the lower integral bound is x=0, which actually corresponds to t -> infinity, but is approximated
        by a finite positive value of t_max
    Returns:
        a tuple
            1) the nodes on the x-axes and
            2) the corresponding weights
    """
    t_l, d_t = np.linspace(-3, t_max, n, retstep=True)
    s_t = np.sinh(t_l) * np.pi / 2
    x = x_max / 2 / (np.exp(s_t) * np.cosh(s_t))
    w = x_max / 2 * d_t * np.pi * np.cosh(t_l) / 2 / np.cosh(s_t) ** 2
    return x, w


def fourier_integral_tanhsinh_x0_0(
    f: util.CplxFnc, x_max: float, n: int, tau_l: NDArray, t_max_ts: float
) -> NDArray:
    """
    Evaluate the Fourier integral of f(x) for a list of values tau_l,

        F(tau) = int_0^x_max dx f(x) e^{-j x tau}

    Use tanh-sinh integration scheme to discretize the integral.
    Thus, the method is suited function f with a singularity at the lower integral boundary.

        F_l = sum_i w_k f(x_i) exp(-j x_i tau_l)

    where x_k (w_k) are the nodes (weights) of the tanh-sinh integration scheme.

    For the tanh-sinh scheme, the lower bound (x=0) is mapped to infinity.
    The needed cutoff of the t-axes is given by t_max_ts (see 'tanhsinh_get_x_and_w' for details).
    As suitable cutoff is returned by 'get_t_max_for_singularity_ts'.

    Parameters:
        f: the function of which to calculate the Fourier integral
        tau_l: the times for which the Fourier integral is evaluated
        n: number of equidistant grid points on the t-axes
        x_max: the upper integral bound on the x-axes (corresponds to t -> -infinity, t_min can be set to 3)
        t_max_ts: the lower integral bound is x=0, which actually corresponds to t -> infinity, but is approximated
        by a finite positive value of t_max
    Returns:
        a numpy array containing the values of the Fourier integral obtained by the tanh-sinh scheme
    """
    x, w = tanhsinh_x_and_w(n, x_max, t_max_ts)
    f_x = f(x)
    size_x = n
    size_tau = len(tau_l)
    return np.einsum(
        "i, i, ij",
        w,
        f_x,
        np.exp(-1j * x.reshape(size_x, -1) * tau_l.reshape(-1, size_tau)),
    )


def fourier_integral_tanhsinh(
    integrand: util.CplxFnc, a: float, b: float, n: int, tau: NDArray
) -> NDArray:
    """
    Approximates F(t_i) int_a^b dx integrand(x) exp(i x t_i) using the tanh-sinh integration
    scheme with n terms.
    Note that the tanh-sinh scheme can handle singularities at the lower bound 'a' efficiently,
    if a = 0.
    If 'a' is not zero, a warning will be shown.

    The FFT algorithm is NOT used. Therefore, the time t_i need to be specified by the parameter tau.

    Parameters:
        integrand: the function of which to approximate the Fourier integral
        a: left integration boundary
        b: right integration boundary
        n: number of grid points
        tau: the times t_i for which to evaluate the Fourier integral.

    Returns:
        the values F(t_i) of the Fourier integral
    """
    t_max = tanhsinh_t_max_for_singularity(integrand, a, b, tol=1e-16)
    F_t = fourier_integral_tanhsinh_x0_0(
        f=lambda x: integrand(x + a), x_max=b - a, n=n, tau_l=tau, t_max_ts=t_max
    )
    return F_t * np.exp(-1j * tau * a)


def _rel_diff(x_ref: NDArray, x: NDArray) -> NDArray:
    """
    Compute the relative difference between x and the reference value x_ref, i.e. abs(x-x_ref) / abs(x_ref).
    Avoid nans by
        - returning zero whenever x-x_ref = 0 and
        - returning MAX_FLOAT when x != x_ref and x_ref = 0.

    The function is provided for convenience to be passed as argument 'diff_method' to
    other functions of this module.
    """
    diff = np.abs(x_ref - x)
    norm_x_ref = np.abs(x_ref)
    with warnings.catch_warnings():
        # ignore warning which may occur if norm_x_ref is zero
        warnings.simplefilter("ignore")
        res = np.where(diff == 0, 0, diff / norm_x_ref)
    idx0 = np.where(np.logical_and(norm_x_ref == 0, diff != 0))
    res[idx0] = MAX_FLOAT
    return res


def _abs_diff(x_ref: NDArray, x: NDArray) -> NDArray:
    """
    Compute the absolute difference abs(x - x_Ref).

    The function is provided for convenience to be passed as argument 'diff_method' to
    other functions of this module.
    """
    return np.abs(x_ref - x)


def get_suitable_a_b_n_for_fourier_integral(
    integrand: util.CplxFnc,
    k_max: float,
    ft_ref: util.CplxFnc,
    tol: float,
    opt_b_only: bool,
    diff_method: Callable[[NDArray, NDArray], NDArray],
    ref_val_left: float = 0,
    ref_val_right: float = 0,
    max_num_iteration: int = 100,
) -> tuple[float, float, int]:
    """
    Determine the integral boundaries 'a' and 'b', as well as the number of grid points 'n'
    of the numeric Fourier integral

        int_a^b dx f(x) e^(-ixk) = sum_l=1^n w_l f_l

    such that the sum approximates the actual Fourier transform with bounds -inf and +inf
    up to a given tolerance 'tol' over a finite range of k values [0, k_max].

    The half-sided Fourier transform with bound [0, inf] is used when setting 'opt_b_only' to True.

    A measure of difference between the numeric Fourier integral and the reference values returned
    by 'ft_ref' needs to be returned by the function passed as 'diff_method'.
    Predefined measures are the relative ('_ref_diff') as well as th absolute difference ('_abs_diff').

    Parameters:
        integrand: the function of which to calculate the Fourier integral
        k_max: specifies the range of k values [0, k_max] to be used for calculating the difference
        ft_ref: the exact Fourier transform as callable function (reference values for calculating the difference)
        tol: largest allowed difference between the numeric integral and the reference values
        opt_b_only: if True, use the half-sided Fourier transform (implies a = 0)
        diff_method: a callable which returns a measure of difference between the numeric integral and
        the reference value of the Fourier transform.
        ref_val_left: the initial value for starting the left boundary search (for  details see 'find_integral_boundary')
        ref_val_right: the initial value for starting the right boundary search (for  details see 'find_integral_boundary')
        max_num_iteration: allows to pose a failure condition for ill posed problems (for  details see 'find_integral_boundary')

    Returns:
         as tuple
            1) lower bound 'a' (is zero if 'opt_b_only' is True)
            2) upper bound 'b'
            3) number of grid points needed for the integral discretization
    """

    if opt_b_only:
        I0 = quad(integrand, 0, np.inf)[0]
    else:
        I0 = quad(integrand, -np.inf, np.inf)[0]

    ft_ref_0 = ft_ref(0)
    rd = np.abs(ft_ref_0 - I0) / np.abs(ft_ref_0)
    log.debug(f"check Fourier integral at w=0 yields a relative difference of {rd:.3e}")
    log.debug(f"I_0 = {I0}")
    log.debug(f"ft_ref_0 = {ft_ref_0}")
    if rd > 1e-6:
        raise FTReferenceError(
            "It seems that 'ft_ref' is not the fourier transform of 'integrand'!"
        )

    log10_tol_0 = -2
    log2_n_0 = 5
    # 'i' roughly scales the interval [a, b] by exponentially decreasing tol ~ 10^-i -> f(a) = tol = f(b)
    # 'j' increases the number of grid points exponentially
    i = 0

    # we actually traverse 'diagonally', i.e., i + j = const
    # start with
    #   tol = 10^-2 and n = 2^5 (first level)
    # next (second level)
    #   tol=10^-3 and n = 2^5 (enlarge [a,b] interval)
    #   tol=10^-2 and n = 2^6 (increase n)
    # and so on
    while True:
        d_old = None
        log.debug(f"at diagonal level i+j={i}")

        for j in range(0, i + 1):
            fx_min = 10 ** (log10_tol_0 - i + j)
            n = 2 ** (log2_n_0 + j)
            log.debug(f"check i={i} (tol={tol:.2e}) and j={j} (n={n})")

            if opt_b_only:
                a = 0
            else:
                a = find_integral_boundary(
                    integrand=integrand,
                    direction="left",
                    tol=fx_min,
                    ref_val=ref_val_left,
                    max_num_iteration=max_num_iteration,
                )
            b = find_integral_boundary(
                integrand=integrand,
                direction="right",
                tol=fx_min,
                ref_val=ref_val_right,
                max_num_iteration=max_num_iteration,
            )

            k, ft_k = fourier_integral_midpoint_fft(integrand, a, b, n)
            idx = np.where(k <= k_max)
            ft_ref_k = ft_ref(k[idx])
            d = np.max(diff_method(ft_ref_k, ft_k[idx]))
            log.debug(
                f"fx_min:{fx_min:.2e} yields: interval [{a:.2e},{b:.2e}] diff {d:.2e}"
            )

            if d_old is not None and d > d_old:
                log.debug(
                    "increasing N while shrinking the interval does lower the error -> try next level"
                )
                break
            else:
                d_old = d

            if d < tol:
                log.debug(f"{d:2e} = d < tol {tol:.2e} reached")
                return a, b, n
        i += 1


def get_dt_for_accurate_interpolation(
    t_max: float,
    tol: float,
    ft_ref: util.CplxFnc,
    diff_method: Callable[[NDArray, NDArray], NDArray] = _abs_diff,
):
    r"""
    Determine the spacing of a uniform discretization of the given interval $[0, t_\mathrm{max}]$ which is
    necessary to ensure a given tolerance threshold for qubic spline interpolation of a function over that interval.

    For a given grid $t_i$ we estimate the interpolation error $\epsilon_i$ by the error
    at the center of interval $[t_i, t_{i+1}]$, i.e., $x_i = (t_i + t_{i+1})/2$.
    If the overall error is below `tol`

    $$
        \max_i \epsilon_i < \mathrm{tol}, \qquad \epsilon_i = \mathrm{diff}(f(x_i), f_\mathrm{interp}(x_i))
    $$

    the interval length $\mathrm{dt} = t_{i+1} - t_i$ is returned.

    Parameters:
        t_max: specifies the interval of consideration [0, t_max]
        tol: tolerance
        ft_ref: the function (callable: float -> complex)
        diff_method: a measure to quantify the difference between the reference value and
            the interpolated value (predefined functions are `_abs_diff` and `_rel_diff`)
    Returns:
        A step size 'dt' for which the interpolation between the resulting discretization fulfills
            the condition max_t diff_method(interp_ft(t), ft_ref(t)) < tol.
    """
    n = 16

    tau = np.linspace(0, t_max, n + 1)
    ft_ref_n_old = ft_ref(tau)
    ft_ref_0 = abs(ft_ref(0))

    while True:
        n *= 2

        tau = np.linspace(0, t_max, n + 1)
        ft_ref_n = np.empty(shape=(n + 1,), dtype=np.complex128)

        ft_ref_n[::2] = ft_ref_n_old

        try:
            with Pool() as pool:
                ft_ref_n_new = np.asarray(pool.map(ft_ref, tau[1::2]))
        except Exception as e:
            log.warning(f"could not call 'ft_ref' in parallel (mp.pool.map) ({e})")
            ft_ref_n_new = ft_ref(tau[1::2])

        ft_ref_n[1::2] = np.array(ft_ref_n_new)

        ft_intp = fastcubicspline.FCS(x_low=0, x_high=t_max, y=ft_ref_n_old)
        ft_intp_n_new = ft_intp(tau[1::2])

        ft_ref_n_new /= ft_ref_0
        ft_intp_n_new /= ft_ref_0

        d = np.max(diff_method(ft_intp_n_new, ft_ref_n_new))
        dt = 2 * tau[1]
        log.debug(
            f"interpolation with step size dt {dt:.2e} estimates a difference of {d:.2e}"
        )
        if d < tol:
            return dt

        ft_ref_n_old = ft_ref_n


def calc_ab_n_dx_dt(
    integrand: util.CplxFnc,
    intgr_tol: float,
    intpl_tol: float,
    t_max: float,
    ft_ref: util.CplxFnc,
    opt_b_only: bool,
    diff_method: Callable[[NDArray, NDArray], NDArray] = _abs_diff,
) -> tuple[float, float, int, float, float]:
    r"""
    Calculate the parameters for the FFT method such that the error tolerance is met.

    Free parameters are:

        - :math:`\omega_{max}` (here also called b):
          the improper Fourier integral has to be approximated by a definite integral with upper
          bound :math:`\omega_{max}`
        - :math:`\omega_{min}` (here also called a):
          if `opt_b_only` is `False` also the lower bound has to be chosen such that the
          improper Fourier integral is well approximated. if `opt_b_only` is `True` the lower bound is
          fixed at zero.
        - :math:`N`: the number of nodes used to numerically evaluate the definite integral from
          :math:`\omega_{min}` to :math:`\omega_{max}`. Increasing :math:`N` lowers the error
          of the numeric integration scheme.

    Since :math:`\Delta \omega` is determined by the above parameters
    :math:`\Delta \omega = (\omega_{max}-\omega_{min})/(N-1)`, also the time grid with spacing
    :math:`\Delta t =  2\pi/(N \Delta\omega)` is fixed by the relation imposed by FFT algorithm.

    In particular two criterion have to be met.

    1) The error of the numeric Fourier integral is smaller than 'intgr_tol'.
    2) The error due to interpolation is smaller than 'intpl_tol'.

    Note that the error is calculated using 'diff_method'

    Parameters:
        integrand: the function of which to evaluate the Fourier integral
        intgr_tol: the tolerance for the integration error
        intpl_tol: the tolerance for the interpolation error
        t_max: specifies the interval [0, t_max] over which the Fourier integral needs to be evaluated
        ft_ref: the Fourier transform as callable.
        opt_b_only: if 'True', use the half-sided Fourier transform (fix 'a' to 0)
        diff_method: a measure to quantify the difference between the reference and the numeric value
        (predefined functions are '_abd_diff' and '_rel_diff')
    Returns:
        a tuple with
            1) the lower integral bound 'a'
            2) the upper integral bound 'b'
            3) the number of grid point (nodes) 'n'
            4) the step size in the integrand domain 'delta omega'
            5) the step size in the Fourier domain 'delta t'
    """
    log.debug("call get_dt_for_accurate_interpolation (may take some time) ...")

    # when calculating the interpolation error, we only consider times [0, t_max_for_dt]
    # where 't_max_for_dt' is the minimum of
    #   1) t_tol, the time, where the Fourier transform (normed by its value at tau=0)
    #      has decayed to the 'intgr_tol' threshold
    #   2) t_max, the maximum time of interest
    try:
        t_tol = find_integral_boundary(
            lambda tau: np.abs(ft_ref(tau)) / np.abs(ft_ref(0)),
            direction="right",
            tol=intgr_tol,
        )
    except RuntimeError:
        t_tol = t_max
    t_max_for_dt = min(t_tol, t_max)
    dt_tol = get_dt_for_accurate_interpolation(
        t_max=t_max_for_dt, tol=intpl_tol, ft_ref=ft_ref, diff_method=diff_method
    )
    log.debug(
        f"get_dt_for_accurate_interpolation returns the condition dt < {dt_tol:.3e}"
    )

    log.debug("call get_suitable_a_b_n_for_fourier_integral (may take some time) ...")
    a, b, n = get_suitable_a_b_n_for_fourier_integral(
        integrand=integrand,
        k_max=t_max,
        tol=intgr_tol,
        ft_ref=ft_ref,
        opt_b_only=opt_b_only,
        diff_method=diff_method,
    )
    dx = (b - a) / n
    log.debug(
        f"get_suitable_a_b_n_for_fourier_integral returns condition dx < {dx:.3e}"
    )

    dt = 2 * np.pi / dx / n
    log.debug(f"dx={dt:.3e} -> dt={dx:.3e}")
    # check if this dt obeys the above condition due to interpolation (get_dt_for_accurate_interpolation)
    if dt > dt_tol:
        log.debug(f"dt does not meet the interpolation condition, dt < {dt_tol:.3e}")
        log.debug(
            " -> increase n (to powers of 2) such that dx and dt decrease simultaneously"
        )
        n_min = 2 * np.pi / dx / dt_tol
        n = 2 ** int(np.ceil(np.log2(n_min)))
        scale = np.sqrt(n_min / n)
        dx_new = scale * dx
        b_minus_a = dx_new * n
        dt_new = 2 * np.pi / dx_new / n
        assert dt_new < dt_tol

        if opt_b_only:
            b = a + b_minus_a
        else:
            delta = b_minus_a - (b - a)
            b += delta / 2
            a -= delta / 2
    else:
        dt_new = dt
        dx_new = dx

    # we may need to further increase n to reach for the entire interval [0, t_max]
    if dt_new * (n - 1) < t_max:
        log.debug("increase n further to match dt (n-1) < t_max")
        n_tmp = t_max / dt_new + 1
        n = 2 ** int(np.ceil(np.log2(n_tmp)))
        dx_new = 2 * np.pi / n / dt_new

    return a, b, n, dx_new, dt_new
