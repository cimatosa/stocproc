import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.special import gamma
from scipy.optimize import minimize
import time
from . import stocproc_c
from . import gquad
from . import tools

import logging
log = logging.getLogger(__name__)

def solve_hom_fredholm(r, w, eig_val_min=None):
    r"""Solves the discrete homogeneous Fredholm equation of the second kind
    
    .. math:: \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u(s) = \lambda u(t)
    
    Quadrature approximation of the integral gives a discrete representation of the
    basis functions and leads to a regular eigenvalue problem.
    
    .. math:: \sum_i w_i R(t_j-s_i) u(s_i) = \lambda u(t_j) \equiv \mathrm{diag(w_i)} \cdot R \cdot u = \lambda u 
    
    Note: If :math:`t_i = s_i \forall i` the matrix :math:`R(t_j-s_i)` is 
    a hermitian matrix. In order to preserve hermiticity for arbitrary :math:`w_i` 
    one defines the diagonal matrix :math:`D = \mathrm{diag(\sqrt{w_i})}` 
    with leads to the equivalent expression:
    
    .. math:: D \cdot R \cdot D \cdot D \cdot u = \lambda D \cdot u \equiv \tilde R \tilde u = \lambda \tilde u
    
    where :math:`\tilde R` is hermitian and :math:`u = D^{-1}\tilde u`    
    
    Setting :math:`t_i = s_i` and due to the definition of the correlation function the 
    matrix :math:`r_{ij} = R(t_i, s_j)` is hermitian.
    
    :param r: correlation matrix :math:`R(t_j-s_i)`
    :param w: integrations weights :math:`w_i` 
        (they have to correspond to the discrete time :math:`t_i`)
    :param eig_val_min: discards all eigenvalues and eigenvectos with
         :math:`\lambda_i < \mathtt{eig\_val\_min} / \mathrm{max}(\lambda)`
         
    :return: eigenvalues, eigenvectos (eigenvectos are stored in the normal numpy fashion, ordered in decreasing order)
    """
    t0 = time.time()
    # weighted matrix r due to quadrature weights
    n = len(w)
    w_sqrt = np.sqrt(w)
    r = w_sqrt.reshape(n,1) * r * w_sqrt.reshape(1,n)
    eig_val, eig_vec = scipy_eigh(r, overwrite_a=True)   # eig_vals in ascending

    if eig_val_min is None:
        min_idx = 0
    else:
        min_idx = sum(eig_val < eig_val_min)

    eig_val = eig_val[min_idx:][::-1]
    eig_vec = eig_vec[:, min_idx:][:, ::-1]
    log.debug("use {} / {} eigenfunctions".format(len(eig_val), len(w)))
    eig_vec = np.reshape(1/w_sqrt, (n,1)) * eig_vec
    log.debug("discrete fredholm equation of size {} solved [{:.2e}]".format(n, time.time() - t0))
    return eig_val, eig_vec

def align_eig_vec(eig_vec):
    for i in range(eig_vec.shape[1]):
        phase = np.exp(1j * np.arctan2(np.real(eig_vec[0,i]), np.imag(eig_vec[0,i])))
        eig_vec[:, i] /= phase

def opt_fredholm(kernel, lam, t, u, meth, ntimes):
    def fopt(x, p, k):
        n = len(x)//2 + 1
        lam = x[0]
        u = x[1:n] + 1j*x[n:]
        return np.log10((np.sum(np.abs(k.dot(u) - lam * u) ** p)) ** (1 / p))

    for i in range(ntimes):
        print("iter {}/{} lam {}".format(i+1, ntimes, lam))
        u_intp = tools.ComplexInterpolatedUnivariateSpline(t, u)
        ng = 2*(len(t)-1)+1
        t, w = meth(t[-1], ng)
        u_new = u_intp(t)
        k_new = kernel(t.reshape(-1,1)-t.reshape(1,-1))*w.reshape(-1,1)
        p = 5
        r = minimize(fopt, x0=np.hstack((lam, u_new.real, u_new.imag)), args=(p, k_new), method='CG')
        x = r.x
        n = len(x) // 2 + 1
        lam = x[0]
        u = x[1:n] + 1j * x[n:]
        print("func val {} ng {}".format(r.fun, ng))

    return lam, u, t





def get_mid_point_weights_times(t_max, num_grid_points):
    r"""Returns the time gridpoints and wiehgts for numeric integration via **mid point rule**.
        
    The idea is to use :math:`N_\mathrm{GP}` time grid points located in the middle
    each of the :math:`N_\mathrm{GP}` equally distributed subintervals of :math:`[0, t_\mathrm{max}]`.
    
    .. math:: t_i = \left(i + \frac{1}{2}\right)\frac{t_\mathrm{max}}{N_\mathrm{GP}} \qquad i = 0,1, ... N_\mathrm{GP} - 1 

    The corresponding trivial weights for integration are
    
    .. math:: w_i = \Delta t = \frac{t_\mathrm{max}}{N_\mathrm{GP}} \qquad i = 0,1, ... N_\mathrm{GP} - 1
    
    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
        (note: this would corespond to an integration from :math:`0-\Delta t / 2`
        to :math:`t_\mathrm{max}+\Delta t /2`)  
    :param num_grid_points: number of 
    """
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.ones(num_grid_points)*delta_t
    return t, w

def get_trapezoidal_weights_times(t_max, num_grid_points):
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.ones(num_grid_points)*delta_t
    w[0] /= 2
    w[-1] /= 2
    return t, w

def get_simpson_weights_times(t_max, num_grid_points):
    if num_grid_points % 2 != 1:
        raise RuntimeError("simpson weights needs grid points ng such that ng = 2*k+1, but git ng={}".format(num_grid_points))
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::2] = 2/3*delta_t
    w[1::2] = 4/3*delta_t
    w[0]    = 1/3*delta_t
    w[-1]   = 1/3*delta_t
    return t, w

def get_four_point_weights_times(t_max, num_grid_points):
    if num_grid_points % 4 != 1:
        raise RuntimeError("four point weights needs grid points ng such that ng = 4*k+1, but git ng={}".format(num_grid_points))
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::4] = 2 * 7  * 4 / 90 * delta_t
    w[1::4] = 1 * 32 * 4 / 90 * delta_t
    w[2::4] = 1 * 12 * 4 / 90 * delta_t
    w[3::4] = 1 * 32 * 4 / 90 * delta_t
    w[0]    = 1 * 7  * 4 / 90 * delta_t
    w[-1]   = 1 * 7  * 4 / 90 * delta_t
    return t, w

def get_gauss_legendre_weights_times(t_max, num_grid_points):
    return gquad.gauss_nodes_weights_legendre(n = num_grid_points, low=0, high=t_max)

def get_sinh_tanh_weights_times(t_max, num_grid_points):
    """
    inspired by Tanh-Sinh High-Precision Quadrature - David H. Bailey

    """
    def get_h_of_N(N):
        """returns the stepsize h for sinh tanh quad for a given number of points N
            such that the smallest weights are about 1e-14
        """
        a = 16.12087683080651
        b = -2.393599730652087
        c = 6.536936185577097
        d = -1.012504470475915
        if N < 4:
            raise ValueError("only tested for N >= 4")
        return a * N ** b + c * N ** d

    h = get_h_of_N(num_grid_points)
    if num_grid_points % 2 != 1:
        raise RuntimeError("sinh tanh weights needs grid points ng such that ng = 2*k+1, but git ng={}".format(num_grid_points))
    kmax = (num_grid_points - 1) / 2
    k = np.arange(0, kmax+1)
    w = h * np.pi / 2 * np.cosh(k * h) / np.cosh(np.pi / 2 * np.sinh(k * h)) ** 2
    w = np.hstack((w[-1:0:-1], w))*t_max/2

    tmp = np.pi/2*np.sinh(h*k)
    y_plus = 1/ (np.exp(tmp) * np.cosh(tmp))
    t = np.hstack((y_plus[-1:0:-1], (2-y_plus)))*t_max/2
    return t, w


_WC_ = 2
_S_ = 0.6
_GAMMA_S_PLUS_1 = gamma(_S_ + 1)
def lac(t):
    """lorenzian bath correlation function"""
    return np.exp(- np.abs(t) - 1j * _WC_ * t)

def oac(t):
    """ohmic bath correlation function"""
    return (1 + 1j*(t))**(-(_S_+1)) * _GAMMA_S_PLUS_1 / np.pi

def get_rel_diff(corr, t_max, ng, weights_times, ng_fac):
    t, w = weights_times(t_max, ng)
    r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
    _eig_val, _eig_vec = solve_hom_fredholm(r, w, eig_val_min=0)
    ng_fine = ng_fac * (ng - 1) + 1
    tfine = np.linspace(0, t_max, ng_fine)
    bcf_n_plus = corr(tfine - tfine[0])
    alpha_k = np.hstack((np.conj(bcf_n_plus[-1:0:-1]), bcf_n_plus))

    u_i_all_t = stocproc_c.eig_func_all_interp(delta_t_fac = ng_fac,
                                               time_axis   = t,
                                               alpha_k     = alpha_k,
                                               weights     = w,
                                               eigen_val   = _eig_val,
                                               eigen_vec   = _eig_vec)

    u_i_all_ast_s = np.conj(u_i_all_t)  # (N_gp, N_ev)
    num_ev = len(_eig_val)
    tmp = _eig_val.reshape(1, num_ev) * u_i_all_t  # (N_gp, N_ev)
    recs_bcf = np.tensordot(tmp, u_i_all_ast_s, axes=([1], [1]))

    refc_bcf = np.empty(shape=(ng_fine, ng_fine), dtype=np.complex128)
    for i in range(ng_fine):
        idx = ng_fine - 1 - i
        refc_bcf[:, i] = alpha_k[idx:idx + ng_fine]

    rd = np.abs(recs_bcf - refc_bcf) / np.abs(refc_bcf)
    return tfine, rd