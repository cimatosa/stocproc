import sys
import os

import numpy as np
import math
from scipy.special import gamma as gamma_func
import scipy.integrate as sp_int
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
import pickle
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
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times]
            #method_kle.get_tanh_sinh_weights_times]


    for _meth in meth:
        ui = method_kle.auto_ng(corr, t_max, ngfac=ng_fac, meth = _meth)
        print(_meth.__name__, ui.shape)

def show_compare_weights_in_solve_fredholm_oac():
    """
        here we try to examine which integration weights perform best in order to
        calculate the eigenfunctions -> well it seems to depend on the situation

        although simpson and gauss-legendre perform well
    """
    t_max = 15
    corr = oac

    ng_ref = 3501

    _meth_ref = method_kle.get_simpson_weights_times
    t, w = _meth_ref(t_max, ng_ref)

    try:
        with open("test_fredholm_interpolation.dump", 'rb') as f:
            ref_data = pickle.load(f)
    except FileNotFoundError:
        ref_data = {}
    key = (tuple(t), tuple(w), corr.__name__)
    if key in ref_data:
        eigval_ref, evec_ref = ref_data[key]
    else:
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        eigval_ref, evec_ref = method_kle.solve_hom_fredholm(r, w)
        ref_data[key] = eigval_ref, evec_ref
        with open("test_fredholm_interpolation.dump", 'wb') as f:
            pickle.dump(ref_data, f)

    method_kle.align_eig_vec(evec_ref)

    ks = [20, 40, 80, 160]

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(16, 12))

    ax = ax.flatten()

    lines = []
    labels = []

    eigvec_ref = []
    for i in range(ng_ref):
        eigvec_ref.append(tools.ComplexInterpolatedUnivariateSpline(t, evec_ref[:, i]))

    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times,
            method_kle.get_tanh_sinh_weights_times]
    cols = ['r', 'b', 'g', 'm', 'c', 'lime']
    for j, k in enumerate(ks):
        axc = ax[j]
        ng = 4 * k + 1

        for i, _meth in enumerate(meth):
            t, w = _meth(t_max, ng)
            r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
            _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
            method_kle.align_eig_vec(_eig_vec)

            dx = []
            dy = []
            dy2 = []

            for l in range(len(_eig_val)):
                evr = eigvec_ref[l](t)
                diff = np.abs(_eig_vec[:, l] - evr)
                dx.append(l)
                dy.append(np.max(diff))
                dy2.append(abs(_eig_val[l] - eigval_ref[l]))

            p, = axc.plot(dx, dy, color=cols[i])
            axc.plot(dx, dy2, color=cols[i], ls='--')
            if j == 0:
                lines.append(p)
                labels.append(_meth.__name__)

        t, w = method_kle.get_simpson_weights_times(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        _eig_vec = _eig_vec[1::2, :]
        t = t[1::2]

        dx = []
        dy = []
        dy2 = []

        for l in range(len(_eig_val)):
            evr = eigvec_ref[l](t)
            diff = np.abs(_eig_vec[:, l] - evr)
            dx.append(l)
            dy.append(np.max(diff))
            dy2.append(abs(_eig_val[l] - eigval_ref[l]))

        p, = axc.plot(dx, dy, color='orange')
        p_lam, = axc.plot(list(range(ng)), eigval_ref[:ng], color='k')
        if j == 0:
            lines.append(p)
            labels.append('simp 2nd')
            lines.append(p_lam)
            labels.append('abs(lamnda_i)')

        axc.grid()
        axc.set_yscale('log')
        axc.set_xlim([0, 100])
        axc.set_ylim([1e-15, 10])
        axc.set_title("ng {}".format(ng))

    fig.suptitle("use ref with ng_ref {} and method '{}'".format(ng_ref, _meth_ref.__name__))
    fig.legend(handles=lines, labels=labels, ncol=3, loc='lower center')
    fig.subplots_adjust(bottom=0.15)
    plt.show()

def show_solve_fredholm_error_scaling_oac():
    """
    """
    t_max = 15
    corr = oac

    ng_ref = 3501

    _meth_ref = method_kle.get_simpson_weights_times
    t, w = _meth_ref(t_max, ng_ref)

    t_3501 = t

    try:
        with open("test_fredholm_interpolation.dump", 'rb') as f:
            ref_data = pickle.load(f)
    except FileNotFoundError:
        ref_data = {}
    key = (tuple(t), tuple(w), corr.__name__)
    if key in ref_data:
        eigval_ref, evec_ref = ref_data[key]
    else:
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        eigval_ref, evec_ref = method_kle.solve_hom_fredholm(r, w)
        ref_data[key] = eigval_ref, evec_ref
        with open("test_fredholm_interpolation.dump", 'wb') as f:
            pickle.dump(ref_data, f)

    method_kle.align_eig_vec(evec_ref)

    ks = np.logspace(0.7, 2.3, 15, dtype=np.int)

    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times,
            method_kle.get_tanh_sinh_weights_times]

    names = ['midp', 'trapz', 'simp', 'fp', 'gl', 'ts']
    idxs = [0,10,20]

    eigvec_ref = []
    for idx in idxs:
        eigvec_ref.append(tools.ComplexInterpolatedUnivariateSpline(t, evec_ref[:, idx]))

    data = np.empty(shape= (len(meth), len(ks), len(idxs)))
    data_spline = np.empty(shape=(len(meth), len(ks), len(idxs)))
    data_int = np.empty(shape=(len(meth), len(ks), len(idxs)))
    for j, k in enumerate(ks):
        print(j, len(ks))
        ng = 4 * k + 1
        for i, _meth in enumerate(meth):
            t, w = _meth(t_max, ng)
            r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
            _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
            method_kle.align_eig_vec(_eig_vec)
            for k, idx in enumerate(idxs):
                d = np.max(np.abs(_eig_vec[:,idx]-eigvec_ref[k](t)))
                data[i, j, k] = d

                uip = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:,idx])
                d = np.max(np.abs(uip(t_3501) - evec_ref[:, idx]))
                data_spline[i, j, k] = d

                uip = np.asarray([my_intp(ti, corr, w, t, _eig_vec[:, idx], _eig_val[idx]) for ti in t_3501])
                d = np.max(np.abs(uip - evec_ref[:, idx]))
                data_int[i, j, k] = d

    ng = 4*ks + 1
    for i in range(len(meth)):
        p, = plt.plot(ng, data[i, :, 0], marker='o', label="no intp {}".format(names[i]))
        c = p.get_color()
        plt.plot(ng, data_spline[i, :, 0], marker='.', color=c, label="spline {}".format(names[i]))
        plt.plot(ng, data_int[i, :, 0], marker='^', color=c, label="intp {}".format(names[i]))

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()

    plt.show()

def show_compare_weights_in_solve_fredholm_lac():
    """
        here we try to examine which integration weights perform best in order to
        calculate the eigenfunctions -> well it seems to depend on the situation

        although simpson and gauss-legendre perform well
    """
    t_max = 15
    corr = lac

    ng_ref = 3501

    _meth_ref = method_kle.get_simpson_weights_times
    t, w = _meth_ref(t_max, ng_ref)

    try:
        with open("test_fredholm_interpolation.dump", 'rb') as f:
            ref_data = pickle.load(f)
    except FileNotFoundError:
        ref_data = {}
    key = (tuple(t), tuple(w), corr.__name__)
    if key in ref_data:
        eigval_ref, evec_ref = ref_data[key]
    else:
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        eigval_ref, evec_ref = method_kle.solve_hom_fredholm(r, w)
        ref_data[key] = eigval_ref, evec_ref
        with open("test_fredholm_interpolation.dump", 'wb') as f:
            pickle.dump(ref_data, f)


    method_kle.align_eig_vec(evec_ref)

    ks = [20,40,80,160]

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(16,12))

    ax = ax.flatten()

    lines = []
    labels = []

    eigvec_ref = []
    for i in range(ng_ref):
        eigvec_ref.append(tools.ComplexInterpolatedUnivariateSpline(t, evec_ref[:, i]))

    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times,
            method_kle.get_sinh_tanh_weights_times]
    cols = ['r', 'b', 'g', 'm', 'c', 'lime']
    for j, k in enumerate(ks):
        axc = ax[j]
        ng = 4*k+1

        for i, _meth in enumerate(meth):
            t, w = _meth(t_max, ng)
            r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
            _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
            method_kle.align_eig_vec(_eig_vec)

            dx = []
            dy = []
            dy2 = []

            for l in range(len(_eig_val)):
                evr = eigvec_ref[l](t)
                diff = np.abs(_eig_vec[:,l] - evr)
                dx.append(l)
                dy.append(np.max(diff))
                dy2.append(abs(_eig_val[l] - eigval_ref[l]))

            p, = axc.plot(dx, dy, color=cols[i])
            axc.plot(dx, dy2, color=cols[i], ls='--')
            if j == 0:
                lines.append(p)
                labels.append(_meth.__name__)


        t, w = method_kle.get_simpson_weights_times(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
        method_kle.align_eig_vec(_eig_vec)
        _eig_vec = _eig_vec[1::2, :]
        t = t[1::2]

        dx = []
        dy = []
        dy2 = []

        for l in range(len(_eig_val)):
            evr = eigvec_ref[l](t)
            diff = np.abs(_eig_vec[:, l] - evr)
            dx.append(l)
            dy.append(np.max(diff))
            dy2.append(abs(_eig_val[l] - eigval_ref[l]))

        p, = axc.plot(dx, dy, color='lime')
        p_lam, = axc.plot(list(range(ng)), eigval_ref[:ng], color='k')
        if j == 0:
            lines.append(p)
            labels.append('simp 2nd')
            lines.append(p_lam)
            labels.append('abs(lamnda_i)')

        axc.grid()
        axc.set_yscale('log')
        axc.set_xlim([0,100])
        axc.set_ylim([1e-5, 10])
        axc.set_title("ng {}".format(ng))


    fig.suptitle("use ref with ng_ref {} and method '{}'".format(ng_ref, _meth_ref.__name__))
    fig.legend(handles=lines, labels=labels, ncol=3, loc='lower center')
    fig.subplots_adjust(bottom=0.15)
    plt.show()

def show_solve_fredholm_interp_eigenfunc():
    """
        here we take the discrete eigenfunctions of the Fredholm problem
        and use qubic interpolation to check the integral equality.

        the difference between the midpoint weights and simpson weights become
        visible. Although the simpson integration yields on average a better performance
        there are high fluctuation in the error.
    """
    _WC_ = 2
    def lac(t):
        return np.exp(- np.abs(t) - 1j*_WC_*t)

    t_max = 10
    ng = 81
    ngfac = 2
    tfine = np.linspace(0, t_max, (ng-1)*ngfac+1)

    lef = tools.LorentzianEigenFunctions(t_max=t_max, gamma=1, w=_WC_, num=5)


    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    ax = ax.flatten()

    for idx in range(4):
        u_exact = lef.get_eigfunc(idx)(tfine)
        method_kle.align_eig_vec(u_exact.reshape(-1,1))

        t, w = sp.method_kle.get_mid_point_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:,idx])

        err = np.abs(u0(tfine) - u_exact)
        axc = ax[idx]
        axc.plot(tfine, err, color='r', label='midp')

        t, w = sp.method_kle.get_trapezoidal_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])
        err = np.abs(u0(tfine) - u_exact)
        axc.plot(tfine, err, color='b', label='trapz')
        axc.plot(tfine[::ngfac], err[::ngfac], ls='', marker='x', color='b')

        t, w = sp.method_kle.get_simpson_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])
        err = np.abs(u0(tfine) - u_exact)
        axc.plot(tfine, err, color='k', label='simp')
        axc.plot(tfine[::ngfac], err[::ngfac], ls='', marker='x', color='k')

        axc.set_yscale('log')
        axc.set_title("eigen function # {}".format(idx))
        axc.grid()


    axc.set_yscale('log')
    fig.suptitle("np.abs(int R(t-s)u_i(s) - lam_i * u_i(t))")
    plt.show()

def show_fredholm_eigvec_interpolation():
    """
        for ohmic sd   : use 4 point and integral interpolation
        for lorentzian : use simpson and spline interpolation
    """
    t_max = 15
    corr = lac
    corr = oac

    ng_ref = 3501

    _meth_ref = method_kle.get_simpson_weights_times
    _meth_ref = method_kle.get_trapezoidal_weights_times
    _meth_ref = method_kle.get_four_point_weights_times

    t, w = _meth_ref(t_max, ng_ref)

    try:
        with open("test_fredholm_interpolation.dump", 'rb') as f:
            ref_data = pickle.load(f)
    except FileNotFoundError:
        ref_data = {}
    key = (tuple(t), tuple(w), corr.__name__)
    if key in ref_data:
        eigval_ref, evec_ref = ref_data[key]
    else:
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        eigval_ref, evec_ref = method_kle.solve_hom_fredholm(r, w)
        ref_data[key] = eigval_ref, evec_ref
        with open("test_fredholm_interpolation.dump", 'wb') as f:
            pickle.dump(ref_data, f)

    method_kle.align_eig_vec(evec_ref)
    t_ref = t

    eigvec_ref = []
    for l in range(ng_ref):
        eigvec_ref.append(tools.ComplexInterpolatedUnivariateSpline(t, evec_ref[:, l]))

    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times,
            method_kle.get_tanh_sinh_weights_times]
    cols = ['r', 'b', 'g', 'm', 'c', 'lime']



    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(16,12))
    ax = ax.flatten()

    ks = [10,14,18,26]

    lns, lbs = [], []

    for ik, k in enumerate(ks):
        axc = ax[ik]

        ng = 4*k+1
        for i, _meth in enumerate(meth):
            print(ik, i)
            t, w = _meth(t_max, ng)
            r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
            _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
            method_kle.align_eig_vec(_eig_vec)

            eigvec_intp = []
            for l in range(ng):
                eigvec_intp.append(tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, l]))

            ydata_fixed = []
            ydata_spline = []
            ydata_integr_intp = []
            xdata = np.arange(min(ng, 100))

            for idx in xdata:
                evr = eigvec_ref[idx](t)
                ydata_fixed.append(np.max(np.abs(_eig_vec[:,idx] - evr)))
                ydata_spline.append(np.max(np.abs(eigvec_intp[idx](t_ref) - evec_ref[:,idx])))
                uip = np.asarray([my_intp(ti, corr, w, t, _eig_vec[:,idx], _eig_val[idx]) for ti in t_ref])
                ydata_integr_intp.append(np.max(np.abs(uip - evec_ref[:,idx])))

            p1, = axc.plot(xdata, ydata_fixed, color=cols[i], label=_meth.__name__)
            p2, = axc.plot(xdata, ydata_spline, color=cols[i], ls='--')
            p3, = axc.plot(xdata, ydata_integr_intp, color=cols[i], alpha = 0.5)
            if ik == 0:
                lns.append(p1)
                lbs.append(_meth.__name__)

        if ik == 0:
            lines = [p1,p2,p3]
            labels = ['fixed', 'spline', 'integral interp']
        axc.set_yscale('log')
        axc.set_title("ng {}".format(ng))
        axc.set_xlim([0,100])
        axc.grid()
        axc.legend()


    fig.legend(lines, labels, loc = "lower right", ncol=3)
    fig.legend(lns, lbs, loc="lower left", ncol=2)
    plt.subplots_adjust(bottom = 0.15)

    plt.savefig("test_fredholm_eigvec_interpolation_{}_.pdf".format(corr.__name__))
    plt.show()



def show_reconstr_ac_interp():
    t_max = 25
    corr = lac
    #corr = oac

    meth = [method_kle.get_mid_point_weights_times,
            method_kle.get_trapezoidal_weights_times,
            method_kle.get_simpson_weights_times,
            method_kle.get_four_point_weights_times,
            method_kle.get_gauss_legendre_weights_times,
            method_kle.get_tanh_sinh_weights_times]

    cols = ['r', 'b', 'g', 'm', 'c']

    def my_intp(ti, corr, w, t, u, lam):
        return np.sum(corr(ti - t) * w * u) / lam

    fig, ax = plt.subplots(figsize=(16, 12))

    ks = [40]
    for i, k in enumerate(ks):
        axc = ax
        ng = 4 * k + 1
        for i, _meth in enumerate(meth):
            t, w = _meth(t_max, ng)
            r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
            _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)

            tf = method_kle.subdevide_axis(t, ngfac=3)
            tsf = method_kle.subdevide_axis(tf, ngfac=2)

            diff1 = - corr(t.reshape(-1,1) - t.reshape(1,-1))
            diff2 = - corr(tf.reshape(-1, 1) - tf.reshape(1, -1))
            diff3 = - corr(tsf.reshape(-1, 1) - tsf.reshape(1, -1))

            xdata = np.arange(ng)
            ydata1 = np.ones(ng)
            ydata2 = np.ones(ng)
            ydata3 = np.ones(ng)

            for idx in xdata:
                evec = _eig_vec[:, idx]
                if _eig_val[idx] < 0:
                    break
                sqrt_eval = np.sqrt(_eig_val[idx])

                uip = np.asarray([my_intp(ti, corr, w, t, evec, sqrt_eval) for ti in tf])
                uip_spl = tools.ComplexInterpolatedUnivariateSpline(tf, uip)
                uip_sf = uip_spl(tsf)
                diff1 += _eig_val[idx] * evec.reshape(-1, 1) * np.conj(evec.reshape(1, -1))
                diff2 += uip.reshape(-1, 1) * np.conj(uip.reshape(1, -1))
                diff3 += uip_sf.reshape(-1,1) * np.conj(uip_sf.reshape(1,-1))
                ydata1[idx] = np.max(np.abs(diff1))
                ydata2[idx] = np.max(np.abs(diff2))
                ydata3[idx] = np.max(np.abs(diff3))

            p, = axc.plot(xdata, ydata1, label=_meth.__name__, alpha = 0.5)
            axc.plot(xdata, ydata2, color=p.get_color(), ls='--')
            axc.plot(xdata, ydata3, color=p.get_color(), lw=2)

        axc.set_yscale('log')
        axc.set_title("ng: {}".format(ng))
        axc.grid()
        axc.legend()
    plt.show()







def show_lac_error_scaling():
    """
    """
    t_max = 15
    corr = lac

    idx = 0
    lef = tools.LorentzianEigenFunctions(t_max=t_max, gamma=1, w=_WC_, num=10)
    u = lef.get_eigfunc(idx)

    ngs = np.logspace(1, 2.5, 60, dtype=np.int)

    _meth = method_kle.get_mid_point_weights_times
    d = []
    for ng in ngs:
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        ut = u(t)
        method_kle.align_eig_vec(ut.reshape(-1,1))
        _d = np.abs(_eig_vec[:, idx]-ut)
        d.append(np.max(_d))
    plt.plot(ngs, d, label='midp', marker='o', ls='')


    _meth = method_kle.get_trapezoidal_weights_times
    d = []
    for ng in ngs:
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        ut = u(t)
        method_kle.align_eig_vec(ut.reshape(-1, 1))
        _d = np.abs(_eig_vec[:, idx] - ut)
        d.append(np.max(_d))
    plt.plot(ngs, d, label='trapz', marker='o', ls='')


    _meth = method_kle.get_simpson_weights_times
    d = []
    ng_used = []
    for ng in ngs:
        ng = 2*(ng//2)+1
        if ng in ng_used:
            continue
        ng_used.append(ng)
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        ut = u(t)
        method_kle.align_eig_vec(ut.reshape(-1, 1))
        _d = np.abs(_eig_vec[:, idx] - ut)
        d.append(np.max(_d))
    plt.plot(ng_used, d, label='simp', marker='o', ls='')

    x = np.logspace(1, 3, 50)
    plt.plot(x, 0.16 / x, color='0.5')
    plt.plot(x, 1 / x ** 2, color='0.5')
    plt.plot(x, 200 / x ** 4, color='0.5')
    plt.plot(x, 200000 / x ** 6, color='0.5')





    plt.yscale('log')
    plt.xscale('log')

    plt.legend()
    plt.grid()
    plt.show()

def show_oac_error_scaling():
    """
    """
    t_max = 15
    corr = lac

    idx = 0
    lef = tools.LorentzianEigenFunctions(t_max=t_max, gamma=1, w=_WC_, num=10)
    u = lef.get_eigfunc(idx)

    ngs = np.logspace(1, 2.5, 60, dtype=np.int)

    _meth = method_kle.get_mid_point_weights_times
    d = []
    for ng in ngs:
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        ut = u(t)
        method_kle.align_eig_vec(ut.reshape(-1,1))
        _d = np.abs(_eig_vec[:, idx]-ut)
        d.append(np.max(_d))
    plt.plot(ngs, d, label='midp', marker='o', ls='')


    _meth = method_kle.get_trapezoidal_weights_times
    d = []
    for ng in ngs:
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        ut = u(t)
        method_kle.align_eig_vec(ut.reshape(-1, 1))
        _d = np.abs(_eig_vec[:, idx] - ut)
        d.append(np.max(_d))
    plt.plot(ngs, d, label='trapz', marker='o', ls='')


    _meth = method_kle.get_simpson_weights_times
    d = []
    ng_used = []
    for ng in ngs:
        ng = 2*(ng//2)+1
        if ng in ng_used:
            continue
        ng_used.append(ng)
        t, w = _meth(t_max, ng)
        r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        ut = u(t)
        method_kle.align_eig_vec(ut.reshape(-1, 1))
        _d = np.abs(_eig_vec[:, idx] - ut)
        d.append(np.max(_d))
    plt.plot(ng_used, d, label='simp', marker='o', ls='')

    x = np.logspace(1, 3, 50)
    plt.plot(x, 0.16 / x, color='0.5')
    plt.plot(x, 1 / x ** 2, color='0.5')
    plt.plot(x, 200 / x ** 4, color='0.5')
    plt.plot(x, 200000 / x ** 6, color='0.5')





    plt.yscale('log')
    plt.xscale('log')

    plt.legend()
    plt.grid()
    plt.show()

def show_lac_simp_scaling():
    """
        even using the exact eigen functions yields bad error scaling due to the
        non smoothness of the correlation function at tau=0

        the problem is cured if the integral over s runs from 0 to t and then from
        t to t_max, this is only ensured for midpoint and trapezoidal weights.
    """
    t_max = 15
    corr = lac

    idx = 0
    lef = tools.LorentzianEigenFunctions(t_max=t_max, gamma=1, w=_WC_, num=10)
    u = lef.get_eigfunc(idx)
    c = 1/lef.get_eigval(idx)

    ngs = np.logspace(1, 2.5, 60, dtype=np.int)

    _meth = method_kle.get_simpson_weights_times
    d0 = []
    d1 = []
    ng_used = []
    for ng in ngs:
        ng = 2*(ng//2)+1
        if ng in ng_used:
            continue
        ng_used.append(ng)
        t, w = _meth(t_max, ng)
        ti = 0
        di = abs(u(ti) - c*np.sum(w*corr(ti-t)*u(t)))
        d0.append(di)
        ti = 1
        di = abs(u(ti) - c * np.sum(w * corr(ti - t) * u(t)))
        d1.append(di)

    plt.plot(ng_used, d0, marker='o', color='k', label="simps int corr(0-s) u(s)")
    plt.plot(ng_used, d1, marker='o', color='r', label="simps int corr(1-s) u(s)")

    ng_used = np.asanyarray(ng_used)
    plt.plot(ng_used, 1 / ng_used ** 2, label='1/ng**2')
    plt.plot(ng_used, 1 / ng_used ** 4, label='1/ng**4')

    plt.yscale('log')
    plt.xscale('log')


    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_weights(plot=True)
    test_is_axis_equidistant()
    # test_subdevide_axis()
    # test_analytic_lorentzian_eigenfunctions()
    # test_solve_fredholm()
    # test_cython_interpolation()
    # test_reconstr_ac()
    # test_solve_fredholm()
    # test_solve_fredholm_reconstr_ac()
    # test_auto_ng()


    # show_compare_weights_in_solve_fredholm_oac()
    # show_compare_weights_in_solve_fredholm_lac()
    # show_solve_fredholm_error_scaling_oac()
    # show_fredholm_eigvec_interpolation()
    # show_solve_fredholm_interp_eigenfunc()
    # show_reconstr_ac_interp()
    # show_lac_error_scaling()
    # show_lac_simp_scaling()
    pass
