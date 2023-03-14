"""
Tests related to the submodule stocproc.method_fft
"""

# python imports
import logging
from functools import partial

# third party imports
import fcSpline
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.integrate as sp_int
from scipy.special import gamma as gamma_func

# stocproc module imports
import stocproc as sp
from stocproc import method_ft


sp.logging_setup(
    sh_level=logging.DEBUG,
    smpl_log_level=logging.DEBUG,
    kle_log_level=logging.DEBUG,
    ft_log_level=logging.DEBUG,
)


def test_find_integral_boundary():
    def f(x):
        return np.exp(-(x**2))

    tol = 1e-10

    with pytest.raises(ValueError):
        sp.method_ft.find_integral_boundary(integrand=f, direction="somethings")

    a = sp.method_ft.find_integral_boundary(integrand=f, direction="left")
    b = sp.method_ft.find_integral_boundary(integrand=f, direction="right")
    assert a < b
    assert abs(f(a) - tol) < 1e-14
    assert abs(f(b) - tol) < 1e-14

    a2 = sp.method_ft.find_integral_boundary(integrand=f, direction="Left")
    b2 = sp.method_ft.find_integral_boundary(integrand=f, direction="righT")
    assert a == a2
    assert b == b2

    with pytest.raises(RuntimeError):
        sp.method_ft.find_integral_boundary(
            integrand=f, tol=tol, ref_val=a - 1, direction="left"
        )
    with pytest.raises(RuntimeError):
        sp.method_ft.find_integral_boundary(
            integrand=f, tol=tol, ref_val=b + 1, direction="right"
        )

    def f2(x):
        return np.exp(-(x**2)) * x**2

    tol = 1e-10
    b = sp.method_ft.find_integral_boundary(integrand=f2, direction="right")
    a = sp.method_ft.find_integral_boundary(integrand=f2, direction="left")
    assert a < b
    assert abs(f2(a) - tol) < 1e-14
    assert abs(f2(b) - tol) < 1e-14

    def f3(x):
        return np.exp(-((x - 5) ** 2)) * x**2

    tol = 1e-10
    b = sp.method_ft.find_integral_boundary(integrand=f3, direction="right", ref_val=5)
    a = sp.method_ft.find_integral_boundary(integrand=f3, direction="left", ref_val=5)
    assert a < b
    assert abs(f3(a) - tol) < 1e-14
    assert abs(f3(b) - tol) < 1e-14

    # the shifted case
    def f(x):
        return 0 if x < 5 else np.exp(-x)

    b = sp.method_ft.find_integral_boundary(integrand=f, direction="right")
    assert abs(f(b) - tol) < 1e-14

    with pytest.raises(RuntimeError):
        sp.method_ft.find_integral_boundary(integrand=f, direction="left")


def test_simpson_even_quad_and_midpoint():
    def f(x):
        return np.cos(x) ** 2 * (x + 1) ** 2

    i_analyt = np.pi / 6 * (8 * np.pi**2 + 12 * np.pi + 9)

    for odd in [0, 1]:
        data_simpson = []
        data_mid = []
        for n in range(8, 13):
            n = 2**n + odd

            # simpson integration
            x, dx = np.linspace(0, 2 * np.pi, n, retstep=True)
            f_x = f(x)
            w = method_ft.simpson_weights(n)
            i = np.sum(f_x * w) * dx
            d = abs(i - i_analyt)
            data_simpson.append([dx, d])

            # midpoint
            x, dx = np.linspace(0, 2 * np.pi, n, retstep=True, endpoint=False)
            f_x = f(x + dx / 2)
            i = np.sum(f_x) * dx
            d = abs(i - i_analyt)
            data_mid.append([dx, d])

        data_simpson = np.asarray(data_simpson)
        data_mid = np.asarray(data_mid)

        is_odd = odd == 1

        plt.title(f"odd number of points {is_odd}")

        # plt.plot(data_simpson[:, 0], data_simpson[:, 1], marker='o', label='simpson')
        # plt.plot(data_mid[:, 0], data_mid[:, 1], marker='o', label='midpoint')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        # dx_scaling = np.logspace(-4, 0, 25)
        # plt.plot(dx_scaling, dx_scaling ** 2, color='k', ls=':')
        # plt.plot(dx_scaling, dx_scaling ** 3, color='k', ls=':')
        # plt.plot(dx_scaling, dx_scaling ** 4, color='k', ls=':')
        # plt.show()

        delta_y = np.log(data_simpson[1:, 1]) - np.log(data_simpson[:-1, 1])
        delta_x = np.log(data_simpson[1:, 0]) - np.log(data_simpson[:-1, 0])
        slope = delta_y / delta_x
        print(f"slope simpson (odd number of points {is_odd}, should approx 4)", slope)
        assert np.all(slope > 3.9)

        delta_y = np.log(data_mid[1:, 1]) - np.log(data_mid[:-1, 1])
        delta_x = np.log(data_mid[1:, 0]) - np.log(data_mid[:-1, 0])
        slope = delta_y / delta_x
        print(f"slope midpoint (odd number of points {is_odd}, should approx 2)", slope)
        assert np.all(slope > 1.9)


def test_fourier_integral_finite_boundary():
    intg = lambda x: x**2
    a = -1.23
    b = 4.87

    def ft_ref(k, a, b):
        """
        analytic solution of int_a^b dx x^2 e^{-i k x}
        """
        r = np.empty_like(k, dtype=np.complex128)
        idx_zero = np.where(k == 0)
        idx_not_zero = np.where(k != 0)
        r[idx_zero] = 1 / 3 * (b**3 - a**3)
        k_nz = k[idx_not_zero]
        r[idx_not_zero] = (
            np.exp(-1j * a * k_nz) * (2j - a * k_nz * (2 + 1j * a * k_nz))
            + np.exp(-1j * b * k_nz) * (-2j + b * k_nz * (2 + 1j * b * k_nz))
        ) / k_nz**3
        return r

    # check midpoint with analytic result
    # --------------------------
    N = 2**18
    N_test = 100
    tau, ft_n = sp.method_ft.fourier_integral_midpoint_fft(intg, a, b, N)
    ft_ref_n = ft_ref(tau, a, b)
    ft_n = ft_n[1:N_test]
    ft_ref_n = ft_ref_n[1:N_test]
    rd = np.max(np.abs(ft_n - ft_ref_n) / np.abs(ft_ref_n))
    assert rd < 4e-6, f"rd = {rd}"

    N = 1024
    tau, ft_n = sp.method_ft.fourier_integral_midpoint_fft(intg, a, b, N)
    ft_ref_n = ft_ref(tau, a, b)
    rd = np.abs(ft_ref_n - ft_n) / np.abs(ft_ref_n)
    idx = np.where(np.logical_and(tau < 75, np.isfinite(rd)))
    rd = rd[idx]
    mrd_midp = np.max(rd)
    assert mrd_midp < 9e-3, f"mrd_midp = {mrd_midp}"

    # check simpson
    # --------------------------
    N = 513
    tau, ft_n = sp.method_ft.fourier_integral_simps_fft(intg, a, b, N)
    ft_ref_n = ft_ref(tau, a, b)
    rd = np.abs(ft_ref_n - ft_n) / np.abs(ft_ref_n)
    idx = np.where(np.logical_and(tau < 75, np.isfinite(rd)))
    rd = rd[idx]
    # plt.plot(rd)
    # plt.show()
    mrd_simps = np.max(rd)
    assert mrd_simps < 4e-3, f"mrd_simps = {mrd_simps}"
    assert (
        mrd_simps < mrd_midp
    ), f"mrd_simps ({mrd_simps:.3e}) >= mrd_trapz ({mrd_midp:.3e})"

    # check tanh-sinh
    # --------------------------
    N = 512
    ft_n = sp.method_ft.fourier_integral_tanhsinh(intg, a, b, N, tau)
    ft_ref_n = ft_ref(tau, a, b)
    rd = np.abs(ft_ref_n - ft_n) / np.abs(ft_ref_n)
    idx = np.where(np.logical_and(tau < 75, np.isfinite(rd)))
    rd = rd[idx]
    # plt.plot(rd)
    # plt.show()
    mrd_ts = np.max(rd)
    assert mrd_ts < 1e-10, f"mrd_ts = {mrd_ts}"


def osd(w, s, wc):
    if not isinstance(w, np.ndarray):
        if w < 0:
            return 0
        else:
            return w**s * np.exp(-w / wc)
    else:
        res = np.zeros(shape=w.shape)

        w_flat = w.flatten()
        idx_pos = np.where(w_flat > 0)
        fv_res = res.flat
        fv_res[idx_pos] = w_flat[idx_pos] ** s * np.exp(-w_flat[idx_pos] / wc)

        return res


def obcf(t, s, wc):
    return gamma_func(s + 1) * wc ** (s + 1) * (1 + 1j * wc * t) ** (-(s + 1))


def test_fourier_integral_infinite_boundary():
    _PLOT_ = False
    s = 0.5
    wc = 4
    intg = lambda x: osd(x, s, wc)
    bcf_ref = (
        lambda t: gamma_func(s + 1) * wc ** (s + 1) * (1 + 1j * wc * t) ** (-(s + 1))
    )

    a, b = sp.method_ft.find_integral_boundary_auto(
        integrand=intg, tol=1e-12, ref_val=1
    )
    errs = [9e-5, 2e-5, 1.3e-6]

    for i, n in enumerate([2**16, 2**18, 2**20]):
        tau, bcf_n = sp.method_ft.fourier_integral_midpoint_fft(intg, a, b, n=n)
        bcf_ref_n = bcf_ref(tau)

        tau_max = 5
        idx = np.where(tau <= tau_max)
        tau = tau[idx]
        bcf_n = bcf_n[idx]
        bcf_ref_n = bcf_ref_n[idx]

        rd_mp = np.abs(bcf_ref_n - bcf_n) / np.abs(bcf_ref_n)
        if _PLOT_:
            (p,) = plt.plot(tau, rd_mp, label="trapz n {}".format(n))

        tau, bcf_n = sp.method_ft.fourier_integral_simps_fft(intg, a, b=b, n=n - 1)
        bcf_ref_n = bcf_ref(tau)

        idx = np.where(tau <= tau_max)

        tau = tau[idx]
        bcf_n = bcf_n[idx]
        bcf_ref_n = bcf_ref_n[idx]

        rd_sm = np.abs(bcf_ref_n - bcf_n) / np.abs(bcf_ref_n)
        if _PLOT_:
            plt.plot(
                tau, rd_sm, label="simps n {}".format(n), color=p.get_color(), ls="--"
            )

        t_ = 3

        x_simps, dx = np.linspace(a, b, n - 1, endpoint=True, retstep=True)
        I = sp_int.simps(intg(x_simps) * np.exp(-1j * x_simps * t_), dx=dx)
        err = np.abs(I - bcf_ref(t_)) / np.abs(bcf_ref(t_))
        assert np.max(rd_mp) < errs[i], "np.max(rd_mp) = {} >= {}".format(
            np.max(rd_mp), errs[i]
        )
        assert np.max(rd_sm) < errs[i], "np.max(rd_sm) = {} >= {}".format(
            np.max(rd_sm), errs[i]
        )
        if _PLOT_:
            plt.plot(t_, err, marker="o", color="g")

    if _PLOT_:
        plt.legend(loc="lower right")
        plt.grid()
        plt.yscale("log")
        plt.show()


def test_get_suitable_a_b_n_for_fourier_integral():
    _WC_ = 2
    intg = lambda w: 1 / (1 + (w - _WC_) ** 2) / np.pi
    bcf_ref = lambda t: np.exp(-np.abs(t) - 1j * _WC_ * t)
    tol = 1e-2

    a, b = sp.method_ft.find_integral_boundary_auto(
        integrand=intg, tol=tol, ref_val=_WC_
    )
    assert abs(abs(intg(a)) - tol) < 1e-8
    assert abs(abs(intg(b)) - tol) < 1e-8

    a, b, n = sp.method_ft.get_suitable_a_b_n_for_fourier_integral(
        intg,
        k_max=50,
        tol=tol,
        ft_ref=bcf_ref,
        opt_b_only=False,
        diff_method=method_ft._abs_diff,
    )
    t_i, ft_i = method_ft.fourier_integral_midpoint_fft(integrand=intg, a=a, b=b, n=n)
    assert np.all(np.abs(bcf_ref(t_i) - ft_i) < tol)


def test_get_suitable_a_b_n_for_fourier_integral_b_only():
    s = 0.5
    wc = 4
    intg = lambda x: osd(x, s, wc)
    bcf_ref = (
        lambda t: gamma_func(s + 1) * wc ** (s + 1) * (1 + 1j * wc * t) ** (-(s + 1))
    )

    tol = 1e-5
    t_max = 15
    a, b, n = sp.method_ft.get_suitable_a_b_n_for_fourier_integral(
        integrand=intg,
        k_max=t_max,
        tol=tol,
        ft_ref=bcf_ref,
        opt_b_only=True,
        diff_method=method_ft._abs_diff,
    )
    assert a == 0
    t_i, ft_i = method_ft.fourier_integral_midpoint_fft(integrand=intg, a=0, b=b, n=n)
    idx = np.where(t_i <= t_max)
    assert np.max(np.abs(bcf_ref(t_i[idx]) - ft_i[idx])) < tol


def test_get_dt_for_accurate_interpolation():
    s = 0.5
    wc = 4
    tol = 1e-4
    bcf_ref = partial(obcf, s=s, wc=wc)
    dt = sp.method_ft.get_dt_for_accurate_interpolation(
        t_max=40, tol=tol, ft_ref=bcf_ref
    )
    t = np.arange(0, 2, dt)
    bcf_t = bcf_ref(t)
    bcf_fcs = fcSpline.FCS(x_low=t[0], x_high=t[-1], y=bcf_t)
    t_fine = np.linspace(0, 2, len(t * 3) + 7)

    assert np.max(np.abs(bcf_ref(t_fine) - bcf_fcs(t_fine))) < tol


def test_calc_abn():
    def testing(intg, bcf_ref, tol, tmax):
        diff_method = method_ft._abs_diff

        a, b, n, dx, dt = sp.method_ft.calc_ab_n_dx_dt(
            integrand=intg,
            intgr_tol=tol,
            intpl_tol=tol,
            t_max=tmax,
            ft_ref=bcf_ref,
            opt_b_only=True,
            diff_method=diff_method,
        )

        tau, ft_tau = sp.method_ft.fourier_integral_midpoint_fft(intg, a, b, n)
        idx = np.where(tau <= tmax)
        ft_ref_tau = bcf_ref(tau[idx])
        rd = diff_method(ft_tau[idx], ft_ref_tau)
        assert np.max(rd) < tol

        ft_intp = fcSpline.FCS(x_low=0, x_high=tau[idx][-1], y=ft_tau[idx])
        tau_fine = np.linspace(0, tmax, 1500)
        ft_ref_n = bcf_ref(tau_fine)
        ft_intp_n = ft_intp(tau_fine)
        d = diff_method(ft_intp_n, ft_ref_n)
        assert np.max(d) < tol
        assert (np.abs(dx * dt * n - np.pi * 2)) < 1e-15

    s = 0.5
    wc = 4
    intg = partial(osd, s=s, wc=wc)
    bcf_ref = partial(obcf, s=s, wc=wc)

    tol = 1e-3
    tmax = 40
    testing(intg, bcf_ref, tol, tmax)

    s = 0.5
    wc = 40
    intg = partial(osd, s=s, wc=wc)
    bcf_ref = partial(obcf, s=s, wc=wc)

    tol = 1e-3
    tmax = 40
    testing(intg, bcf_ref, tol, tmax)
