# -*- coding: utf8 -*-
from __future__ import print_function, division
"""
module to generate the weights and nodes for Guass quadrature

inspired by pyOrthpol (https://pypi.python.org/pypi/orthpol)

as well as the original fortran resource from

Gautschi, W. (1994). Algorithm 726: 
ORTHPOL–a package of routines for generating orthogonal polynomials
and Gauss-type quadrature rules. 
ACM Transactions on Mathematical Software (TOMS), 20(1), 21–62.
doi:10.1145/174603.174605

"""

import numpy as np
import numpy.polynomial as pln
from scipy.linalg import eig_banded
from scipy.special import gamma

 
def _recur_laguerre(n, al=0.):
    r"""Calculate the recursion coefficients leading to the 
    Laguerre polynomials motivated by the Gauss quadrature
    formula for integrals with exponential weights ~exp(-x)
    
    see Theodore Seio Chihara, 
    An Introduction to Orthogonal Polynomials, 1978, p.217
    """
    nrange = np.arange(n)
    a = 2*nrange + al + 1
    b = nrange*(nrange+al)
    b[0] = gamma(al + 1.)
    return (a, b)

def gauss_nodes_weights_laguerre(n, al=0.):
    r"""
        .. math::
            \int_0^\infty dx \; f(x) x^\alpha \exp(-x) ~= \sum_{i=1}^n w_i f(x_i)
    """
    a, b = _recur_laguerre(n, al)
    return _gauss_nodes_weights(a, b)


def _recur_legendre(n):
    nrange = np.arange(n, dtype = float)
    a = np.zeros(n)
    b = nrange**2 / ((2*nrange - 1)*(2*nrange + 1))
    b[0] = 2
    return (a, b)

def gauss_nodes_weights_legendre(n, low=-1, high=1):
    r"""
        .. math::
            \int_{-1}^{1} dx \; f(x) ~= \sum_{i=1}^n w_i f(x_i)
    """
    a, b = _recur_legendre(n)
    x, w= _gauss_nodes_weights(a, b)
    fac = (high-low)/2
    return (x + 1)*fac + low, fac*w


def _gauss_nodes_weights(a,b):
    r"""Calculate the nodes and weights for given 
    recursion coefficients assuming a normalized 
    weights functions.
    
    
    see Walter Gautschi, Algorithm 726: ORTHPOL; 
    a Package of Routines for Generating Orthogonal 
    Polynomials and Gauss-type Quadrature Rules, 1994 
    """
    assert len(a) == len(b)
    
    a_band = np.vstack((np.sqrt(b),a))
    w, v = eig_banded(a_band)
    
    nodes = w                  # eigenvalues
    weights = b[0] * v[0,:]**2 # first component of each eigenvector
                        # the prefactor b[0] from the original paper
                        # accounts for the weights of unnormalized weight functions
    return nodes, weights

def get_poly(a, b):   
    n = len(a)
    assert len(b) == n
    
    p = []
    
    p.append( 0 )
    p.append( pln.Polynomial(coef=(1,)) )
    
    x = pln.Polynomial(coef=(0,1))
    
    for i in range(n):
        p_i = (x - a[i]) * p[-1] - b[i] * p[-2]
        p.append( p_i )
        
    return p[1:]