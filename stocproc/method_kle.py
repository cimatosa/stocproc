import numpy as np
from scipy.linalg import eigh as scipy_eigh
import time

import logging
log = logging.getLogger(__name__)

def solve_hom_fredholm(r, w, eig_val_min):
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
    min_idx = sum(eig_val < eig_val_min)
    eig_val = eig_val[min_idx:][::-1]
    eig_vec = eig_vec[:, min_idx:][:, ::-1]
    log.debug("discrete fredholm equation of size {} solved [{:.2e}]".format(n, time.time()-t0))

    num_of_functions = len(eig_val)
    log.debug("use {} / {} eigenfunctions (sig_min = {})".format(num_of_functions, len(w), np.sqrt(eig_val_min)))
    eig_vec = np.reshape(1/w_sqrt, (n,1)) * eig_vec

    return eig_val, eig_vec

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
    if num_grid_points % 2 == 0:
        raise RuntimeError("simpson weight need odd number of grid points, but git ng={}".format(num_grid_points))
    # generate mid points
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    # equal weights for grid points
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::2] = 2/3*delta_t
    w[1::2] = 4/3*delta_t
    w[0]    = 1/3*delta_t
    w[-1]   = 1/3*delta_t
    return t, w