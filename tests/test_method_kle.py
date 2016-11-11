import sys
import os

import numpy as np
from scipy.linalg import eig
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found -> any plotting will crash")


import pathlib
p = pathlib.PosixPath(os.path.abspath(__file__))
sys.path.insert(0, str(p.parent.parent))

from scipy.special import gamma
from scipy.integrate import quad
import stocproc as sp
from stocproc import tools
from stocproc import method_kle
from stocproc import stocproc_c
import logging

_S_ = 0.6
_GAMMA_S_PLUS_1 = gamma(_S_ + 1)
_WC_ = 2

def oac(tau):
    """ohmic bath correlation function"""
    return (1 + 1j * (tau)) ** (-(_S_ + 1)) * _GAMMA_S_PLUS_1 / np.pi
def osd(omega):
    return omega ** _S_ * np.exp(-omega)

def lac(t):
    """lorenzian bath correlation function"""
    return np.exp(- np.abs(t) - 1j * _WC_ * t)
def lsd(w):
    return 1 / (1 + (w - _WC_) ** 2)


def my_intp(ti, corr, w, t, u, lam):
    return np.sum(corr(ti - t) * w * u) / lam

def test_weights(plot=False):
    """
        here we check for the correct scaling of the various integration schemas
    """
    def f(x):
        return x/(1+x**2)

    tm = 10
    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times,
            method_kle.get_tanh_sinh_weights_times]
    cols = ['r', 'b', 'g', 'm', 'c', 'lime']
    errs = [2e-3, 6e-5, 2e-8, 7e-11, 5e-16, 8e-15]
    I_exact = np.log(tm**2 + 1)/2

    ng = 401
    for i, _meth in enumerate(meth):
        t, w = _meth(t_max=tm, num_grid_points=ng)
        err = abs(I_exact - np.sum(w * f(t)))
        print(_meth.__name__, err)
        assert err < errs[i]

    if plot:
        for i, _meth in enumerate(meth):
            xdata = []
            ydata = []
            for k in np.logspace(0, 2.5, 30):
                ng = 4 * int(k) + 1
                t, w = _meth(t_max=tm, num_grid_points=ng)
                I = np.sum(w*f(t))
                xdata.append(ng)
                ydata.append(abs(I - I_exact))
            plt.plot(xdata, ydata, marker='o', color=cols[i], label=_meth.__name__)

        x = np.logspace(1, 3, 50)
        plt.plot(x, 0.3 / x, color='0.5')
        plt.plot(x, 6 / x ** 2, color='0.5')
        plt.plot(x, 200 / x ** 4, color='0.5')
        plt.plot(x, 200000 / x ** 6, color='0.5')

        plt.legend(loc = 'lower left')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.tight_layout()
        plt.show()

def test_is_axis_equidistant():
    t, w = method_kle.get_mid_point_weights_times(1, 51)
    assert method_kle.is_axis_equidistant(t)

    t, w = method_kle.get_trapezoidal_weights_times(1, 51)
    assert method_kle.is_axis_equidistant(t)

    t, w = method_kle.get_simpson_weights_times(1, 51)
    assert method_kle.is_axis_equidistant(t)

    t, w = method_kle.get_four_point_weights_times(1, 53)
    assert method_kle.is_axis_equidistant(t)

    t, w = method_kle.get_gauss_legendre_weights_times(1, 51)
    assert not method_kle.is_axis_equidistant(t)

    t, w = method_kle.get_tanh_sinh_weights_times(1, 51)
    assert not method_kle.is_axis_equidistant(t)



def test_subdevide_axis():
    t = [0, 1, 3]
    tf1 = method_kle.subdevide_axis(t, ngfac=1)
    assert np.max(np.abs(tf1 - np.asarray([0, 1, 3]))) < 1e-15
    tf2 = method_kle.subdevide_axis(t, ngfac=2)
    assert np.max(np.abs(tf2 - np.asarray([0, 0.5, 1, 2, 3]))) < 1e-15
    tf3 = method_kle.subdevide_axis(t, ngfac=3)
    assert np.max(np.abs(tf3 - np.asarray([0, 1 / 3, 2 / 3, 1, 1 + 2 / 3, 1 + 4 / 3, 3]))) < 1e-15
    tf4 = method_kle.subdevide_axis(t, ngfac=4)
    assert np.max(np.abs(tf4 - np.asarray([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]))) < 1e-15


def test_analytic_lorentzian_eigenfunctions():
    tmax = 15
    gamma = 2.3
    w = 10.3
    num = 10
    corr = lambda x: np.exp(-gamma * np.abs(x) - 1j * w * x)
    lef = tools.LorentzianEigenFunctions(tmax, gamma, w, num)
    t = np.linspace(0, tmax, 55)
    for idx in range(num):
        u = lef.get_eigfunc(idx)

        u_kernel_re = np.asarray([quad(lambda s: np.real(corr(ti - s) * u(s)), 0, tmax)[0] for ti in t])
        u_kernel_im = np.asarray([quad(lambda s: np.imag(corr(ti - s) * u(s)), 0, tmax)[0] for ti in t])
        u_kernel = u_kernel_re + 1j * u_kernel_im
        c = 1 / lef.get_eigval(idx)
        md = np.max(np.abs(u(t) - c * u_kernel))
        assert md < 1e-6

        norm = quad(lambda s: np.abs(u(s)) ** 2, 0, tmax)[0]
        assert abs(1 - norm) < 1e-15

def test_solve_fredholm():
    """
        here we compare the fredholm eigenvalue problem solver
        (which maps the non hermetian eigenvalue problem to a hermetian one)
        with a straigt forward non hermetian eigenvalue solver
    """
    t_max = 15
    corr = lac
    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times]
    ng = 41
    for _meth in meth:
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        eval, evec = eig(r*w.reshape(1,-1))
        max_imag = np.max(np.abs(np.imag(eval)))
        assert max_imag < 1e-15                     # make sure the evals are real

        eval = np.real(eval)
        idx_sort = np.argsort(eval)[::-1]
        eval = eval[idx_sort]
        evec = evec[:,idx_sort]
        max_eval_diff = np.max(np.abs(eval - _eig_val))
        assert max_eval_diff < 1e-13                # compare evals with fredholm evals

        for i in range(ng-5):
            phase = np.arctan2(np.imag(_eig_vec[0,i]), np.real(_eig_vec[0,i]))
            _eig_vec[:, i] *= np.exp(-1j * phase)
            norm = np.sqrt(np.sum(w*np.abs(_eig_vec[:, i]) ** 2))
            assert abs(1-norm) < 1e-14              # fredholm should return vectors with integral norm 1

            phase = np.arctan2(np.imag(evec[0, i]), np.real(evec[0, i]))
            evec[:, i] *= np.exp(-1j * phase)
            norm = np.sqrt(np.sum(w* np.abs(evec[:, i])**2))
            evec[:, i] /= norm

            diff_evec = np.max(np.abs(_eig_vec[:,i] - evec[:,i]))
            assert diff_evec < 1e-10, "diff_evec {} = {} {}".format(i, diff_evec, _meth.__name__)

def test_cython_interpolation():
    """
    """
    t_max = 15
    corr = oac

    meth = method_kle.get_four_point_weights_times
    def my_intp(ti, corr, w, t, u, lam):
        return np.sum(u * corr(ti - t) * w) / lam

    k = 40
    ng = 4*k+1

    t, w = meth(t_max, ng)
    r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
    _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
    method_kle.align_eig_vec(_eig_vec)

    ngfac = 4
    tfine = np.linspace(0, t_max, (ng-1)*ngfac+1)

    bcf_n_plus = corr(tfine - tfine[0])
    alpha_k = np.hstack((np.conj(bcf_n_plus[-1:0:-1]), bcf_n_plus))
    for i in range(ng//2):
        evec = _eig_vec[:,i]
        sqrt_eval = np.sqrt(_eig_val[i])

        ui_fine = np.asarray([my_intp(ti, corr, w, t, evec, sqrt_eval) for ti in tfine])

        ui_fine2 = stocproc_c.eig_func_interp(delta_t_fac = ngfac,
                                              time_axis   = t,
                                              alpha_k     = alpha_k,
                                              weights     = w,
                                              eigen_val   = sqrt_eval,
                                              eigen_vec   = evec)
        assert np.max(np.abs(ui_fine - ui_fine2)) < 2e-11

def test_reconstr_ac():
    t_max = 15
    res = method_kle.auto_ng(corr=oac,
                             t_max=t_max,
                             ngfac=2,
                             meth=method_kle.get_mid_point_weights_times,
                             tol=1e-3,
                             diff_method='full',
                             dm_random_samples=10 ** 4)
    print(type(res))
    print(res.shape)

def test_solve_fredholm_reconstr_ac():
    """
        here we see that the reconstruction quality is independent of the integration weights

        differences occur when checking validity of the interpolated time continuous Fredholm equation
    """
    _WC_ = 2
    def lac(t):
        return np.exp(- np.abs(t) - 1j*_WC_*t)
    t_max = 10
    for ng in range(11,500,30):
        print(ng)
        t, w = sp.method_kle.get_mid_point_weights_times(t_max, ng)
        r = lac(t.reshape(-1,1)-t.reshape(1,-1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        _eig_vec_ast = np.conj(_eig_vec)  # (N_gp, N_ev)
        tmp = _eig_val.reshape(1, -1) * _eig_vec  # (N_gp, N_ev)
        recs_bcf = np.tensordot(tmp, _eig_vec_ast, axes=([1], [1]))
        rd = np.max(np.abs(recs_bcf - r) / np.abs(r))
        assert rd < 1e-10


        t, w = sp.method_kle.get_simpson_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        _eig_vec_ast = np.conj(_eig_vec)  # (N_gp, N_ev)
        tmp = _eig_val.reshape(1, -1) * _eig_vec  # (N_gp, N_ev)
        recs_bcf = np.tensordot(tmp, _eig_vec_ast, axes=([1], [1]))
        rd = np.max(np.abs(recs_bcf - r) / np.abs(r))
        assert rd < 1e-10


def test_auto_ng():
    corr = oac
    t_max = 8
    ng_fac = 1
    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times]
            #method_kle.get_gauss_legendre_weights_times]
            #method_kle.get_tanh_sinh_weights_times]


    for _meth in meth:
        ui = method_kle.auto_ng(corr, t_max, ngfac=ng_fac, meth = _meth)
        print(_meth.__name__, ui.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_weights(plot=True)
    # test_is_axis_equidistant()
    # test_subdevide_axis()
    # test_analytic_lorentzian_eigenfunctions()
    # test_solve_fredholm()
    # test_cython_interpolation()
    # test_reconstr_ac()
    # test_solve_fredholm()
    # test_solve_fredholm_reconstr_ac()
    # test_auto_ng()
    pass
