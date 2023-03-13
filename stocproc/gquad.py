r"""
A module to generate the nodes and weights for Guassian quadrature

A quadrature scheme approximates an integral by the finite sum

$$
    \int_a^b \mathrm{d}x \, f(x) \approx \sum_{i=1}^n w_i f(x_i)
$$

where $x_i$ are called nodes and $w_i$ weights.


## Gaussian quadrature based on Laguerre-polynomials

The formalism is suited for integrals of the kind

$$
    \int_0^\infty \mathrm{d}x\, f(x) e^{-x} \, .
$$

The nodes and weights are returned by [`stocproc.gquad.gauss_nodes_weights_laguerre`][].
(See, e.g., [Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature) for details.)


## Gaussian quadrature based on Legendre-polynomials

The formalism is suited for integrals of the kind

$$
    \int_{-1}^1 \mathrm{d}x\, f(x) \, .
$$

The nodes and weights are returned by [`stocproc.gquad.gauss_nodes_weights_legendre`][].
(See, e.g., [Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature) for details.)


!!! Note

    The code was inspired by pyOrthpol [`pypi.python.org/pypi/orthpol`](https://pypi.python.org/pypi/orthpol)
    as well as the original fortran resource from:

        Gautschi, W. (1994). Algorithm 726:
        "ORTHPOL – a package of routines for generating orthogonal
        polynomials and Gauss-type quadrature rules."
        ACM Transactions on Mathematical Software (TOMS),
        20(1), 21–62.  doi:10.1145/174603.174605
"""

import numpy as np
import numpy.polynomial as pln
from numpy.typing import NDArray
from scipy.linalg import eig_banded
from scipy.special import gamma


def _recur_laguerre(n: int, alpha: float = 0.0) -> tuple[NDArray, NDArray]:
    r"""
    Calculate the recursion coefficients `a` and `b` leading to the
    Laguerre polynomials motivated by the Gauss quadrature
    formula for integrals with exponential weights $\sim exp(-x)$.

    see Theodore Seio Chihara, "An Introduction to Orthogonal Polynomials", 1978, p.217

    Parameters:
        n: number of coefficients
        alpha: parameter for the generalized Laguerre polynomials
    Returns:
        the recursion coefficients `a` and `b`
    """
    n_range = np.arange(n)
    a = 2 * n_range + alpha + 1
    b = n_range * (n_range + alpha)
    b[0] = gamma(alpha + 1.0)
    return a, b


def _gauss_nodes_weights(a: NDArray, b: NDArray) -> tuple[NDArray, NDArray]:
    r"""
    Calculate the nodes and weights for given recursion coefficients `a` and `b` defining the
    orthogonal polynomials with respect to a given weight function.
    Assume that this weight function is normalized to 1.

    see Walter Gautschi, Algorithm 726:
    ORTHPOL;  a Package of Routines for Generating Orthogonal Polynomials and Gauss-type Quadrature Rules, 1994

    Parameters:
        a: recursion coefficient `a`
        a: recursion coefficient `b`  (`a` and `b` must have the same size)
    Returns:
        the nodes and weights of the related Gaussian quadrature
    """
    assert len(a) == len(b)

    a_band = np.vstack((np.sqrt(b), a))
    w, v = eig_banded(a_band)

    # eigenvalues
    nodes = w
    # first component of each eigenvector
    weights = b[0] * v[0, :] ** 2
    # the pre-factor b[0] from the original paper
    # accounts for the weights of un-normalized weight functions
    return nodes, weights


def gauss_nodes_weights_laguerre(n: int, alpha: float = 0.0) -> tuple[NDArray, NDArray]:
    r"""
    Return the nodes $x_i$ and weights $w_i$ of the Gauss-Laguerre quadrature suitable to
    approximate

    $$
        \int_0^\infty \mathrm{d} x \, f(x) x^\alpha \exp(-x) \approx \sum_{i=1}^n w_i f(x_i)
    $$

    Parameters:
        n: number of nodes / weights to calculate
        alpha: Gaussian quadrature using generalized Laguerre polynomials with exponent $\alpha$
    Returns:
        a tuple containing the nodes and weights as numpy NDArrays.
    """
    a, b = _recur_laguerre(n, alpha)
    return _gauss_nodes_weights(a, b)


def _recur_legendre(n: int) -> tuple[NDArray, NDArray]:
    """
    Calculate the recursion coefficients `a` and `b` for the Legendre polynomials

    see Theodore Seio Chihara, "An Introduction to Orthogonal Polynomials", 1978, p.217
    """
    n_range = np.arange(n, dtype=np.float64)
    a = np.zeros(n)
    b = n_range**2 / ((2 * n_range - 1) * (2 * n_range + 1))
    b[0] = 2
    return a, b


def gauss_nodes_weights_legendre(
    n: int, a: float = -1.0, b: float = 1.0
) -> tuple[NDArray, NDArray]:
    r"""
    Return the nodes $x_i$ and weights $w_i$ of the Gauss-Legendre quadrature suitable to
    approximate

    $$
        \int_a^b \mathrm{d}x \, f(x)  \approx \sum_{i=1}^n w_i f(x_i)
    $$

    Parameters:
        n: number of nodes / weights to calculate
        a: lower integral bound
        b: upper integral bound
    Returns:
        a tuple containing the nodes and weights as numpy NDArrays.
    """
    rec_a, rec_b = _recur_legendre(n)
    x, w = _gauss_nodes_weights(rec_a, rec_b)
    fac = (b - a) / 2
    return (x + 1) * fac + a, fac * w


def _get_poly(a: NDArray, b: NDArray) -> list["pln.Polynomial"]:
    r"""
    Calculate the list of polynomials generated by the recursion coefficients `a` and `b` defined as

    $$
        p_0(x) = 0 \, , \quad p_1(x) = 1 \, , \quad
        p_n(x) = (x - a_{n-2}) p_{n-1}(x) - b_{n-2} p_{n-2}(x) \quad n > 1 \, .
    $$

    Parameters:
        a: list of coefficients `a`
        b: list of coefficients `b` (`a` and `b` must have the same size)
    Returns:
        a list `P` of instances of Polynomial representing the polynomials $p_n$ where $\texttt{P[n]} = p_{n+2}$
    """
    n = len(a)
    assert len(b) == n

    p = []

    p.append(0)
    p.append(pln.Polynomial(coef=(1,)))

    x = pln.Polynomial(coef=(0, 1))

    for i in range(n):
        p_i = (x - a[i]) * p[-1] - b[i] * p[-2]
        p.append(p_i)

    return p[1:]
