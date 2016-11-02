import sys
import os

import numpy as np
import math
from scipy.special import gamma as gamma_func
import scipy.integrate as sp_int
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found -> any plotting will crash")


import pathlib
p = pathlib.PosixPath(os.path.abspath(__file__))
sys.path.insert(0, str(p.parent.parent))

import stocproc as sp
from stocproc import tools
from stocproc import stocproc_c
from scipy.integrate import quad
import logging

def test_solve_fredholm():
    _WC_ = 2
    def lac(t):
        return np.exp(- np.abs(t) - 1j*_WC_*t)

    t_max = 10

    for ng in range(11, 450, 30):
        t, w = sp.method_kle.get_mid_point_weights_times(t_max, ng)
        r = lac(t.reshape(-1,1)-t.reshape(1,-1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w, eig_val_min=0)

        ng_fac = 4
        ng_fine = ng_fac * (ng - 1) + 1
        tfine = np.linspace(0, t_max, ng_fine)
        bcf_n_plus = lac(tfine - tfine[0])
        alpha_k = np.hstack((np.conj(bcf_n_plus[-1:0:-1]), bcf_n_plus))

        u_i_all_t = stocproc_c.eig_func_all_interp(delta_t_fac=ng_fac,
                                                   time_axis=t,
                                                   alpha_k=alpha_k,
                                                   weights=w,
                                                   eigen_val=_eig_val,
                                                   eigen_vec=_eig_vec)

        u_i_all_ast_s = np.conj(u_i_all_t)  # (N_gp, N_ev)
        num_ev = len(_eig_val)
        tmp = _eig_val.reshape(1, num_ev) * u_i_all_t  # (N_gp, N_ev)
        recs_bcf = np.tensordot(tmp, u_i_all_ast_s, axes=([1], [1]))

        refc_bcf = np.empty(shape=(ng_fine, ng_fine), dtype=np.complex128)
        for i in range(ng_fine):
            idx = ng_fine - 1 - i
            refc_bcf[:, i] = alpha_k[idx:idx + ng_fine]

        err = np.max(np.abs(recs_bcf - refc_bcf) / np.abs(refc_bcf))
        plt.plot(ng, err, marker='o', color='r')

        ng += 2
        t, w = sp.method_kle.get_simpson_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val2, _eig_vec2 = sp.method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
        #print(np.max(np.abs(_eig_val - _eig_val2)), np.max(np.abs(_eig_vec - _eig_vec2)))
        _eig_val, _eig_vec = _eig_val2, _eig_vec2

        ng_fac = 4
        ng_fine = ng_fac * (ng - 1) + 1
        tfine = np.linspace(0, t_max, ng_fine)
        bcf_n_plus = lac(tfine - tfine[0])
        alpha_k = np.hstack((np.conj(bcf_n_plus[-1:0:-1]), bcf_n_plus))

        u_i_all_t2 = stocproc_c.eig_func_all_interp(delta_t_fac=ng_fac,
                                                   time_axis=t,
                                                   alpha_k=alpha_k,
                                                   weights=w,
                                                   eigen_val=_eig_val,
                                                   eigen_vec=_eig_vec)
        #print(np.max(np.abs(u_i_all_t - u_i_all_t2)))
        u_i_all_t = u_i_all_t2
        #print()


        u_i_all_ast_s = np.conj(u_i_all_t)  # (N_gp, N_ev)
        num_ev = len(_eig_val)
        tmp = _eig_val.reshape(1, num_ev) * u_i_all_t  # (N_gp, N_ev)
        recs_bcf = np.tensordot(tmp, u_i_all_ast_s, axes=([1], [1]))

        refc_bcf = np.empty(shape=(ng_fine, ng_fine), dtype=np.complex128)
        for i in range(ng_fine):
            idx = ng_fine - 1 - i
            refc_bcf[:, i] = alpha_k[idx:idx + ng_fine]

        err2 = np.max(np.abs(recs_bcf - refc_bcf) / np.abs(refc_bcf))
        plt.plot(ng, err2, marker='o', color='k')
        print(err, err2)

    plt.yscale('log')
    plt.grid()
    plt.show()

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
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
        _eig_vec_ast = np.conj(_eig_vec)  # (N_gp, N_ev)
        tmp = _eig_val.reshape(1, -1) * _eig_vec  # (N_gp, N_ev)
        recs_bcf = np.tensordot(tmp, _eig_vec_ast, axes=([1], [1]))
        rd = np.max(np.abs(recs_bcf - r) / np.abs(r))
        assert rd < 1e-10


        t, w = sp.method_kle.get_simpson_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
        _eig_vec_ast = np.conj(_eig_vec)  # (N_gp, N_ev)
        tmp = _eig_val.reshape(1, -1) * _eig_vec  # (N_gp, N_ev)
        recs_bcf = np.tensordot(tmp, _eig_vec_ast, axes=([1], [1]))
        rd = np.max(np.abs(recs_bcf - r) / np.abs(r))
        assert rd < 1e-10

def test_solve_fredholm_interp_eigenfunc(plot=False):
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

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
        ax = ax.flatten()

    for idx in range(4):
        t, w = sp.method_kle.get_mid_point_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:,idx])
        lam0 = _eig_val[idx]
        err_mp = []
        for ti in tfine:
            I = tools.complex_quad(lambda s: lac(ti-s)*u0(s), 0, t_max, limit=500)
            err_mp.append(np.abs(I - lam0*u0(ti)))
        if plot:
            axc = ax[idx]
            axc.plot(tfine, err_mp, color='r', label='midp')

        t, w = sp.method_kle.get_simpson_weights_times(t_max, ng)
        r = lac(t.reshape(-1, 1) - t.reshape(1, -1))
        _eig_val, _eig_vec = sp.method_kle.solve_hom_fredholm(r, w, eig_val_min=0)
        u0 = tools.ComplexInterpolatedUnivariateSpline(t, _eig_vec[:, idx])
        lam0 = _eig_val[idx]
        err_sp = []
        for ti in tfine:
            I = tools.complex_quad(lambda s: lac(ti - s) * u0(s), 0, t_max, limit=500)
            err_sp.append(np.abs(I - lam0 * u0(ti)))
        if plot:
            axc.plot(tfine, err_sp, color='k', label='simp')
            axc.set_yscale('log')
            axc.plot(tfine[::ngfac], err_sp[::ngfac], ls='', marker='x', color='k')
            axc.set_title("eigen function # {}".format(idx))

        assert max(err_mp) > max(err_sp)

    if plot:
        fig.suptitle("np.abs(int R(t-s)u_i(s) - lam_i * u_i(t))")
        plt.show()


if __name__ == "__main__":
    # test_solve_fredholm()
    # test_solve_fredholm_reconstr_ac()
    test_solve_fredholm_interp_eigenfunc(plot=True)