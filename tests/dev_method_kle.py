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
import stocproc as sp
from stocproc import tools
from stocproc import method_kle
from stocproc import stocproc_c
import pickle

_S_ = 0.6
_GAMMA_S_PLUS_1 = gamma(_S_ + 1)
_WC_ = 2


def oac(tau):
    """ohmic bath correlation function"""
    return (1 + 1j * (tau)) ** (-(_S_ + 1)) * _GAMMA_S_PLUS_1 / np.pi


def osd(omega):
    return omega**_S_ * np.exp(-omega)


def lac(t):
    """lorenzian bath correlation function"""
    return np.exp(-np.abs(t) - 1j * _WC_ * t)


def lsd(w):
    return 1 / (1 + (w - _WC_) ** 2)


def my_intp(ti, corr, w, t, u, lam):
    return np.sum(corr(ti - t) * w * u) / lam


def show_auto_ng():
    corr = lac
    t_max = 15
    meth = [  # method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
    ]
    # method_kle.get_four_point_weights_times]
    # method_kle.get_gauss_legendre_weights_times]
    # method_kle.get_tanh_sinh_weights_times]

    ns = 10**4
    t_check = np.random.rand(ns) * t_max
    s_check = np.random.rand(ns) * t_max

    _meth = method_kle.get_trapezoidal_weights_times
    tol = 1e-2
    ng_fac = 1
    ui, t, ev = method_kle.auto_ng(
        corr, t_max, ngfac=ng_fac, meth=_meth, tol=tol, ret_eigvals=True
    )
    tsf = method_kle.subdivide_axis(t, 4)
    lef = tools.LorentzianEigenFunctions(t_max=t_max, gamma=1, w=_WC_, num=800)
    c_all = corr(t_check - s_check)

    c_allsf = corr(tsf.reshape(-1, 1) - tsf.reshape(1, -1))

    s = 200
    d_num = []
    d_exa = []
    c_num = np.zeros(shape=ns, dtype=np.complex128)
    c_exa = np.zeros(shape=ns, dtype=np.complex128)

    c_numsf = np.zeros(shape=(len(tsf), len(tsf)), dtype=np.complex128)
    c_exasf = np.zeros(shape=(len(tsf), len(tsf)), dtype=np.complex128)
    d_numsf = []
    d_exasf = []
    for i in range(s):
        u = tools.ComplexInterpolatedUnivariateSpline(t, ui[i])
        c_num += u(t_check) * np.conj(u(s_check))
        d_num.append(np.max(np.abs(c_all - c_num) / np.abs(c_all)))

        usf = u(tsf)
        c_numsf += usf.reshape(-1, 1) * np.conj(usf.reshape(1, -1))
        d_numsf.append(np.max(np.abs(c_allsf - c_numsf) / np.abs(c_allsf)))

        u = lef.get_eigfunc(i)
        c_exa += lef.get_eigval(i) * u(t_check) * np.conj(u(s_check))
        d_exa.append(np.max(np.abs(c_all - c_exa) / np.abs(c_all)))

        usf = u(tsf)
        c_exasf += lef.get_eigval(i) * usf.reshape(-1, 1) * np.conj(usf.reshape(1, -1))
        d_exasf.append(np.max(np.abs(c_allsf - c_exasf) / np.abs(c_allsf)))

    fig, ax = plt.subplots(ncols=1)

    ax.plot(np.arange(s), d_num, label="num_rnd")
    ax.plot(np.arange(s), d_exa, label="exa_rnd")

    ax.plot(np.arange(s), d_numsf, label="num super fine")
    ax.plot(np.arange(s), d_exasf, label="exa super fine")

    ng_fac = 1
    ui, t, ev = method_kle.auto_ng(
        corr, t_max, ngfac=ng_fac, meth=_meth, tol=tol, ret_eigvals=True
    )
    c_all = corr(t.reshape(-1, 1) - t.reshape((1, -1)))

    c0 = ui[0].reshape(-1, 1) * np.conj(ui[0]).reshape(1, -1)
    u = lef.get_eigfunc(0)
    ut = u(t)
    c0_exa = lef.get_eigval(0) * ut.reshape(-1, 1) * np.conj(ut.reshape(1, -1))
    d_num = [np.max(np.abs(c_all - c0))]
    d_exa = [np.max(np.abs(c_all - c0_exa))]
    for i in range(1, s):
        c0 += ui[i].reshape(-1, 1) * np.conj(ui[i]).reshape(1, -1)
        d_num.append(np.max(np.abs(c_all - c0) / np.abs(c_all)))

        u = lef.get_eigfunc(i)
        ut = u(t)
        c0_exa += lef.get_eigval(i) * ut.reshape(-1, 1) * np.conj(ut.reshape(1, -1))
        d_exa.append(np.max(np.abs(c_all - c0_exa) / np.abs(c_all)))

    print("re exa", np.max(np.abs(c_all.real - c0_exa.real)))
    print("im exa", np.max(np.abs(c_all.imag - c0_exa.imag)))
    print("re num", np.max(np.abs(c_all.real - c0.real)))
    print("im num", np.max(np.abs(c_all.imag - c0.imag)))

    ax.plot(np.arange(s), d_num, label="num")
    ax.plot(np.arange(s), d_exa, label="exa")

    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    plt.show()


def show_reconstr_ac():
    corr = lac

    t_max = 15
    meth = [
        method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
    ]
    # method_kle.get_four_point_weights_times,
    # method_kle.get_gauss_legendre_weights_times,
    # method_kle.get_tanh_sinh_weights_times]

    names = ["midp", "trapz", "simp", "four", "gl", "ts"]

    for _mi, _meth in enumerate(meth):
        ng_fac = 1
        ng = 401

        t, w = _meth(t_max=t_max, num_grid_points=ng)
        is_equi = method_kle.is_axis_equidistant(t)
        r = method_kle._calc_corr_matrix(t, corr, is_equi)
        _eig_vals, _eig_vecs = method_kle.solve_hom_fredholm(r, w)

        tfine = method_kle.subdivide_axis(t, ng_fac)  # setup fine
        tsfine = method_kle.subdivide_axis(tfine, 2)

        if is_equi:
            alpha_k = method_kle._calc_corr_min_t_plus_t(
                tfine, corr
            )  # from -tmax untill tmax on the fine grid

        num_ev = ng

        csf = -corr(tsfine.reshape(-1, 1) - tsfine.reshape(1, -1))
        abs_csf = np.abs(csf)

        cf = -corr(tfine.reshape(-1, 1) - tfine.reshape(1, -1))
        abs_cf = np.abs(cf)

        c = -corr(t.reshape(-1, 1) - t.reshape(1, -1))
        abs_c = np.abs(c)

        dsf = []
        df = []
        d = []
        dsfa = []
        dfa = []
        da = []
        for i in range(0, num_ev):
            evec = _eig_vecs[:, i]
            if _eig_vals[i] < 0:
                break
            sqrt_eval = np.sqrt(_eig_vals[i])
            if ng_fac != 1:
                if not is_equi:
                    sqrt_lambda_ui_fine = np.asarray(
                        [np.sum(corr(ti - t) * w * evec) / sqrt_eval for ti in tfine]
                    )
                else:
                    sqrt_lambda_ui_fine = stocproc_c.eig_func_interp(
                        delta_t_fac=ng_fac,
                        time_axis=t,
                        alpha_k=alpha_k,
                        weights=w,
                        eigen_val=sqrt_eval,
                        eigen_vec=evec,
                    )
            else:
                sqrt_lambda_ui_fine = evec * sqrt_eval

            sqrt_lambda_ui_spl = tools.ComplexInterpolatedUnivariateSpline(
                tfine, sqrt_lambda_ui_fine
            )
            ut = sqrt_lambda_ui_spl(tsfine)
            csf += ut.reshape(-1, 1) * np.conj(ut.reshape(1, -1))
            diff = np.abs(csf) / abs_csf
            rmidx = np.argmax(diff)
            rmidx = np.unravel_index(rmidx, diff.shape)
            dsf.append(diff[rmidx])

            diff = np.abs(csf)
            amidx = np.argmax(diff)
            amidx = np.unravel_index(amidx, diff.shape)
            dsfa.append(diff[amidx])

            if i == num_ev - 5:
                print(names[_mi], "rd max ", rmidx, "am", amidx)
                print("csf", np.abs(csf[rmidx]), np.abs(csf[amidx]))
                print("abs csf", abs_csf[rmidx], abs_csf[amidx])

            ut = sqrt_lambda_ui_fine
            cf += ut.reshape(-1, 1) * np.conj(ut.reshape(1, -1))
            df.append(np.max(np.abs(cf) / abs_cf))
            dfa.append(np.max(np.abs(cf)))

            ut = evec * sqrt_eval
            c += ut.reshape(-1, 1) * np.conj(ut.reshape(1, -1))
            d.append(np.max(np.abs(c) / abs_c))
            da.append(np.max(np.abs(c)))

        print(names[_mi], "rd max ", rmidx, "am", amidx)
        print("csf", np.abs(csf[rmidx]), np.abs(csf[amidx]))
        print("abs csf", abs_csf[rmidx], abs_csf[amidx])

        (p,) = plt.plot(np.arange(len(d)), d, label=names[_mi], ls="", marker=".")
        plt.plot(np.arange(len(df)), df, color=p.get_color(), ls="--")
        plt.plot(np.arange(len(dsf)), dsf, color=p.get_color())

        plt.plot(np.arange(len(d)), da, ls="", marker=".", color=p.get_color())
        plt.plot(np.arange(len(df)), dfa, color=p.get_color(), ls="--")
        plt.plot(np.arange(len(dsf)), dsfa, color=p.get_color())

    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


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
        with open("test_fredholm_interpolation.dump", "rb") as f:
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
        with open("test_fredholm_interpolation.dump", "wb") as f:
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

    meth = [
        method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
        method_kle.get_four_point_weights_times,
        method_kle.get_gauss_legendre_weights_times,
        method_kle.get_tanh_sinh_weights_times,
    ]
    cols = ["r", "b", "g", "m", "c", "lime"]
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

            (p,) = axc.plot(dx, dy, color=cols[i])
            axc.plot(dx, dy2, color=cols[i], ls="--")
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

        (p,) = axc.plot(dx, dy, color="orange")
        (p_lam,) = axc.plot(list(range(ng)), eigval_ref[:ng], color="k")
        if j == 0:
            lines.append(p)
            labels.append("simp 2nd")
            lines.append(p_lam)
            labels.append("abs(lamnda_i)")

        axc.grid()
        axc.set_yscale("log")
        axc.set_xlim([0, 100])
        axc.set_ylim([1e-15, 10])
        axc.set_title("ng {}".format(ng))

    fig.suptitle(
        "use ref with ng_ref {} and method '{}'".format(ng_ref, _meth_ref.__name__)
    )
    fig.legend(handles=lines, labels=labels, ncol=3, loc="lower center")
    fig.subplots_adjust(bottom=0.15)
    plt.show()


def show_solve_fredholm_error_scaling_oac():
    """ """
    t_max = 15
    corr = oac

    ng_ref = 3501

    _meth_ref = method_kle.get_simpson_weights_times
    t, w = _meth_ref(t_max, ng_ref)

    t_3501 = t

    try:
        with open("test_fredholm_interpolation.dump", "rb") as f:
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
        with open("test_fredholm_interpolation.dump", "wb") as f:
            pickle.dump(ref_data, f)

    method_kle.align_eig_vec(evec_ref)

    ks = np.logspace(0.7, 2.3, 15, dtype=np.int)

    meth = [
        method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
        method_kle.get_four_point_weights_times,
        method_kle.get_gauss_legendre_weights_times,
        method_kle.get_tanh_sinh_weights_times,
    ]

    names = ["midp", "trapz", "simp", "fp", "gl", "ts"]
    idxs = [0, 10, 20]

    eigvec_ref = []
    for idx in idxs:
        eigvec_ref.append(
            tools.ComplexInterpolatedUnivariateSpline(t, evec_ref[:, idx])
        )

    data = np.empty(shape=(len(meth), len(ks), len(idxs)))
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
                d = np.max(np.abs(_eig_vec[:, idx] - eigvec_ref[k](t)))
                data[i, j, k] = d

                uip = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])
                d = np.max(np.abs(uip(t_3501) - evec_ref[:, idx]))
                data_spline[i, j, k] = d

                uip = np.asarray(
                    [
                        my_intp(ti, corr, w, t, _eig_vec[:, idx], _eig_val[idx])
                        for ti in t_3501
                    ]
                )
                d = np.max(np.abs(uip - evec_ref[:, idx]))
                data_int[i, j, k] = d

    ng = 4 * ks + 1
    for i in range(len(meth)):
        (p,) = plt.plot(
            ng, data[i, :, 0], marker="o", label="no intp {}".format(names[i])
        )
        c = p.get_color()
        plt.plot(
            ng,
            data_spline[i, :, 0],
            marker=".",
            color=c,
            label="spline {}".format(names[i]),
        )
        plt.plot(
            ng, data_int[i, :, 0], marker="^", color=c, label="intp {}".format(names[i])
        )

    plt.yscale("log")
    plt.xscale("log")
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
        with open("test_fredholm_interpolation.dump", "rb") as f:
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
        with open("test_fredholm_interpolation.dump", "wb") as f:
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

    meth = [
        method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
        method_kle.get_four_point_weights_times,
        method_kle.get_gauss_legendre_weights_times,
        method_kle.get_sinh_tanh_weights_times,
    ]
    cols = ["r", "b", "g", "m", "c", "lime"]
    for j, k in enumerate(ks):
        axc = ax[j]
        ng = 4 * k + 1

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
                diff = np.abs(_eig_vec[:, l] - evr)
                dx.append(l)
                dy.append(np.max(diff))
                dy2.append(abs(_eig_val[l] - eigval_ref[l]))

            (p,) = axc.plot(dx, dy, color=cols[i])
            axc.plot(dx, dy2, color=cols[i], ls="--")
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

        (p,) = axc.plot(dx, dy, color="lime")
        (p_lam,) = axc.plot(list(range(ng)), eigval_ref[:ng], color="k")
        if j == 0:
            lines.append(p)
            labels.append("simp 2nd")
            lines.append(p_lam)
            labels.append("abs(lamnda_i)")

        axc.grid()
        axc.set_yscale("log")
        axc.set_xlim([0, 100])
        axc.set_ylim([1e-5, 10])
        axc.set_title("ng {}".format(ng))

    fig.suptitle(
        "use ref with ng_ref {} and method '{}'".format(ng_ref, _meth_ref.__name__)
    )
    fig.legend(handles=lines, labels=labels, ncol=3, loc="lower center")
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
        return np.exp(-np.abs(t) - 1j * _WC_ * t)

    t_max = 10
    ng = 81
    ngfac = 2
    tfine = np.linspace(0, t_max, (ng - 1) * ngfac + 1)

    lef = tools.LorentzianEigenFunctions(t_max=t_max, gamma=1, w=_WC_, num=5)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    ax = ax.flatten()

    for idx in range(4):
        u_exact = lef.get_eigfunc(idx)(tfine)
        method_kle.align_eig_vec(u_exact.reshape(-1, 1))

        t, w = sp.method_kle.get_mid_point_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])

        err = np.abs(u0(tfine) - u_exact)
        axc = ax[idx]
        axc.plot(tfine, err, color="r", label="midp")

        t, w = sp.method_kle.get_trapezoidal_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])
        err = np.abs(u0(tfine) - u_exact)
        axc.plot(tfine, err, color="b", label="trapz")
        axc.plot(tfine[::ngfac], err[::ngfac], ls="", marker="x", color="b")

        t, w = sp.method_kle.get_simpson_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w)
        method_kle.align_eig_vec(_eig_vec)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])
        err = np.abs(u0(tfine) - u_exact)
        axc.plot(tfine, err, color="k", label="simp")
        axc.plot(tfine[::ngfac], err[::ngfac], ls="", marker="x", color="k")

        axc.set_yscale("log")
        axc.set_title("eigen function # {}".format(idx))
        axc.grid()

    axc.set_yscale("log")
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
        with open("test_fredholm_interpolation.dump", "rb") as f:
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
        with open("test_fredholm_interpolation.dump", "wb") as f:
            pickle.dump(ref_data, f)

    method_kle.align_eig_vec(evec_ref)
    t_ref = t

    eigvec_ref = []
    for l in range(ng_ref):
        eigvec_ref.append(tools.ComplexInterpolatedUnivariateSpline(t, evec_ref[:, l]))

    meth = [
        method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
        method_kle.get_four_point_weights_times,
        method_kle.get_gauss_legendre_weights_times,
        method_kle.get_tanh_sinh_weights_times,
    ]
    cols = ["r", "b", "g", "m", "c", "lime"]

    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(16, 12))
    ax = ax.flatten()

    ks = [10, 14, 18, 26]

    lns, lbs = [], []

    for ik, k in enumerate(ks):
        axc = ax[ik]

        ng = 4 * k + 1
        for i, _meth in enumerate(meth):
            print(ik, i)
            t, w = _meth(t_max, ng)
            r = corr(t.reshape(-1, 1) - t.reshape(1, -1))
            _eig_val, _eig_vec = method_kle.solve_hom_fredholm(r, w)
            method_kle.align_eig_vec(_eig_vec)

            eigvec_intp = []
            for l in range(ng):
                eigvec_intp.append(
                    tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, l])
                )

            ydata_fixed = []
            ydata_spline = []
            ydata_integr_intp = []
            xdata = np.arange(min(ng, 100))

            for idx in xdata:
                evr = eigvec_ref[idx](t)
                ydata_fixed.append(np.max(np.abs(_eig_vec[:, idx] - evr)))
                ydata_spline.append(
                    np.max(np.abs(eigvec_intp[idx](t_ref) - evec_ref[:, idx]))
                )
                uip = np.asarray(
                    [
                        my_intp(ti, corr, w, t, _eig_vec[:, idx], _eig_val[idx])
                        for ti in t_ref
                    ]
                )
                ydata_integr_intp.append(np.max(np.abs(uip - evec_ref[:, idx])))

            (p1,) = axc.plot(xdata, ydata_fixed, color=cols[i], label=_meth.__name__)
            (p2,) = axc.plot(xdata, ydata_spline, color=cols[i], ls="--")
            (p3,) = axc.plot(xdata, ydata_integr_intp, color=cols[i], alpha=0.5)
            if ik == 0:
                lns.append(p1)
                lbs.append(_meth.__name__)

        if ik == 0:
            lines = [p1, p2, p3]
            labels = ["fixed", "spline", "integral interp"]
        axc.set_yscale("log")
        axc.set_title("ng {}".format(ng))
        axc.set_xlim([0, 100])
        axc.grid()
        axc.legend()

    fig.legend(lines, labels, loc="lower right", ncol=3)
    fig.legend(lns, lbs, loc="lower left", ncol=2)
    plt.subplots_adjust(bottom=0.15)

    plt.savefig("test_fredholm_eigvec_interpolation_{}_.pdf".format(corr.__name__))
    plt.show()


def show_reconstr_ac_interp():
    t_max = 25
    corr = lac
    # corr = oac

    meth = [
        method_kle.get_mid_point_weights_times,
        method_kle.get_trapezoidal_weights_times,
        method_kle.get_simpson_weights_times,
        method_kle.get_four_point_weights_times,
        method_kle.get_gauss_legendre_weights_times,
        method_kle.get_tanh_sinh_weights_times,
    ]

    cols = ["r", "b", "g", "m", "c"]

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

            tf = method_kle.subdivide_axis(t, ngfac=3)
            tsf = method_kle.subdivide_axis(tf, ngfac=2)

            diff1 = -corr(t.reshape(-1, 1) - t.reshape(1, -1))
            diff2 = -corr(tf.reshape(-1, 1) - tf.reshape(1, -1))
            diff3 = -corr(tsf.reshape(-1, 1) - tsf.reshape(1, -1))

            xdata = np.arange(ng)
            ydata1 = np.ones(ng)
            ydata2 = np.ones(ng)
            ydata3 = np.ones(ng)

            for idx in xdata:
                evec = _eig_vec[:, idx]
                if _eig_val[idx] < 0:
                    break
                sqrt_eval = np.sqrt(_eig_val[idx])

                uip = np.asarray(
                    [my_intp(ti, corr, w, t, evec, sqrt_eval) for ti in tf]
                )
                uip_spl = tools.ComplexInterpolatedUnivariateSpline(tf, uip)
                uip_sf = uip_spl(tsf)
                diff1 += (
                    _eig_val[idx] * evec.reshape(-1, 1) * np.conj(evec.reshape(1, -1))
                )
                diff2 += uip.reshape(-1, 1) * np.conj(uip.reshape(1, -1))
                diff3 += uip_sf.reshape(-1, 1) * np.conj(uip_sf.reshape(1, -1))
                ydata1[idx] = np.max(np.abs(diff1))
                ydata2[idx] = np.max(np.abs(diff2))
                ydata3[idx] = np.max(np.abs(diff3))

            (p,) = axc.plot(xdata, ydata1, label=_meth.__name__, alpha=0.5)
            axc.plot(xdata, ydata2, color=p.get_color(), ls="--")
            axc.plot(xdata, ydata3, color=p.get_color(), lw=2)

        axc.set_yscale("log")
        axc.set_title("ng: {}".format(ng))
        axc.grid()
        axc.legend()
    plt.show()


def show_lac_error_scaling():
    """ """
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
        method_kle.align_eig_vec(ut.reshape(-1, 1))
        _d = np.abs(_eig_vec[:, idx] - ut)
        d.append(np.max(_d))
    plt.plot(ngs, d, label="midp", marker="o", ls="")

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
    plt.plot(ngs, d, label="trapz", marker="o", ls="")

    _meth = method_kle.get_simpson_weights_times
    d = []
    ng_used = []
    for ng in ngs:
        ng = 2 * (ng // 2) + 1
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
    plt.plot(ng_used, d, label="simp", marker="o", ls="")

    x = np.logspace(1, 3, 50)
    plt.plot(x, 0.16 / x, color="0.5")
    plt.plot(x, 1 / x**2, color="0.5")
    plt.plot(x, 200 / x**4, color="0.5")
    plt.plot(x, 200000 / x**6, color="0.5")

    plt.yscale("log")
    plt.xscale("log")

    plt.legend()
    plt.grid()
    plt.show()


def show_oac_error_scaling():
    """ """
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
        method_kle.align_eig_vec(ut.reshape(-1, 1))
        _d = np.abs(_eig_vec[:, idx] - ut)
        d.append(np.max(_d))
    plt.plot(ngs, d, label="midp", marker="o", ls="")

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
    plt.plot(ngs, d, label="trapz", marker="o", ls="")

    _meth = method_kle.get_simpson_weights_times
    d = []
    ng_used = []
    for ng in ngs:
        ng = 2 * (ng // 2) + 1
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
    plt.plot(ng_used, d, label="simp", marker="o", ls="")

    x = np.logspace(1, 3, 50)
    plt.plot(x, 0.16 / x, color="0.5")
    plt.plot(x, 1 / x**2, color="0.5")
    plt.plot(x, 200 / x**4, color="0.5")
    plt.plot(x, 200000 / x**6, color="0.5")

    plt.yscale("log")
    plt.xscale("log")

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
    c = 1 / lef.get_eigval(idx)

    ngs = np.logspace(1, 2.5, 60, dtype=np.int)

    _meth = method_kle.get_simpson_weights_times
    d0 = []
    d1 = []
    ng_used = []
    for ng in ngs:
        ng = 2 * (ng // 2) + 1
        if ng in ng_used:
            continue
        ng_used.append(ng)
        t, w = _meth(t_max, ng)
        ti = 0
        di = abs(u(ti) - c * np.sum(w * corr(ti - t) * u(t)))
        d0.append(di)
        ti = 1
        di = abs(u(ti) - c * np.sum(w * corr(ti - t) * u(t)))
        d1.append(di)

    plt.plot(ng_used, d0, marker="o", color="k", label="simps int corr(0-s) u(s)")
    plt.plot(ng_used, d1, marker="o", color="r", label="simps int corr(1-s) u(s)")

    ng_used = np.asanyarray(ng_used)
    plt.plot(ng_used, 1 / ng_used**2, label="1/ng**2")
    plt.plot(ng_used, 1 / ng_used**4, label="1/ng**4")

    plt.yscale("log")
    plt.xscale("log")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # show_auto_ng()
    show_reconstr_ac()
    # show_compare_weights_in_solve_fredholm_oac()
    # show_compare_weights_in_solve_fredholm_lac()
    # show_solve_fredholm_error_scaling_oac()
    # show_fredholm_eigvec_interpolation()
    # show_solve_fredholm_interp_eigenfunc()
    # show_reconstr_ac_interp()
    # show_lac_error_scaling()
    # show_lac_simp_scaling()
    pass
