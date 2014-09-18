import gquad
from scipy.integrate import quad
import numpy as np
import numpy.polynomial as pln
import pytest


def scp_laguerre(p1,p2):
    f = lambda x: p1(x)*p2(x)*np.exp(-x)
    return quad(f, 0, np.inf)

def scp_legendre(p1,p2):
    f = lambda x: p1(x)*p2(x)
    return quad(f, -1, 1)

def orthogonality(p, scp, tol_0, tol_1):
    n = len(p)
    
    for i in range(n):
        s = scp(p[i],p[i])
        p[i] /= np.sqrt(s[0])
    
    for i in range(n):
        for j in range(i,n):
            s = scp(p[i],p[j])
            print("test <p_{}|p_{}> = {:+.2e}".format(i,j,s[0]))
            if i == j:
                assert abs(s[0]-1) < tol_1, "error: {}".format(abs(s[0]-1))
            else: 
                assert abs(s[0]) < tol_0, "error: {}".format(abs(s[0]))

def test_orthogonality_laguerre():
    n = 12
    al = 0
    a,b = gquad._recur_laguerre(n, al)
    p = gquad.get_poly(a,b)
    orthogonality(p, scp=scp_laguerre, tol_0=1e-10, tol_1=1e-10)

def test_orthogonality_legendre():
    n = 12
    a,b = gquad._recur_legendre(n)
    p = gquad.get_poly(a,b)
    orthogonality(p, scp=scp_legendre, tol_0=1e-10, tol_1=1e-10)    


# due to the lack of python 3 compatible orthpol package
@pytest.mark.xfail
def test_compare_with_orthpol():
    n = 50
    ipoly = 7 # Laguerre
    al = 0
    be = 0 # not used
    
    a_op, b_op, ierr = op.drecur(n, ipoly, al, be)
    a, b = gquad._recur_laguerre(n, al)
    
    assert np.allclose(a, a_op)
    
    # note: the recur coef b[0] has no influence on the recursion formula,
    # because it is multiplied by the polynomial of index -1 which is defined to be zero
    # further more this coef does not occur when calculating the nodes and weights
    assert np.allclose(b[1:], b_op[1:])
    
    al = 1.2
    a_op, b_op, ierr = op.drecur(n, ipoly, al, be)
    a, b = gquad._recur_laguerre(n, al)
    assert np.allclose(a, a_op)
    assert np.allclose(b[1:], b_op[1:])
    
def test_integration_legendre():
    n = 12
    np.random.seed(0)
    num_samples = 10
    for tmp in range(num_samples):
        low = np.random.rand()
        high = np.random.rand()
        
        x, w = gquad.gauss_nodes_weights_legendre(n, low, high)
        
        coeff = 10*np.random.randn(2*n-1)
        
        p = pln.Polynomial(coef=coeff)
        a = 0.5
        p_a = p(a)
        
        p_a_ = 0
        for i, c in enumerate(coeff):
            p_a_ += coeff[i]* a**i
        assert abs(p_a - p_a_) < 1e-14, "error: {:.2e}".format(abs(p_a - p_a_))
        
        p_int = p.integ(m=1, lbnd=low)(high)
        p_int_gauss = np.sum(w*p(x))
        diff = abs(p_int - p_int_gauss)
        print("diff: {:.2e}".format(diff))
        assert diff < 1e-14
    
def test_compare_with_scipy_laguerre():
    n_list = [3,7,11,20,52,100]
    al = 0
    
    for n in n_list:
        x, w = gquad.gauss_nodes_weights_laguerre(n, al)
        x_, w_ = pln.laguerre.laggauss(deg=n)
        diff_x = np.abs(x-x_)
        diff_w = np.abs(w-w_)
        print("degree:", n)
        print("max diff x: {:.2e}".format(max(diff_x)))
        print("max diff w: {:.2e}".format(max(diff_w)))
        assert max(diff_x) < 1e-12
        assert max(diff_w) < 1e-12
    
def test_compare_with_scipy_legendre():
    n_list = [3,7,11,20,52,100,200,500]
    al = 0
    
    for n in n_list:
        x, w = gquad.gauss_nodes_weights_legendre(n)
        x_, w_ = pln.legendre.leggauss(deg=n)
        diff_x = np.abs(x-x_)
        diff_w = np.abs(w-w_)
        print("degree:", n)
        print("max diff x: {:.2e}".format(max(diff_x)))
        print("max diff w: {:.2e}".format(max(diff_w)))
        assert max(diff_x) < 1e-12
        assert max(diff_w) < 1e-12

if __name__ == "__main__":
#     test_orthogonality_laguerre()
#     test_orthogonality_legendre()
#     test_integration_legendre()
#     test_compare_with_scipy_laguerre()
    test_compare_with_scipy_legendre()
