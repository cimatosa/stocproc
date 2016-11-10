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

def solve_hom_fredholm(r, w):
    r"""Solves the discrete homogeneous Fredholm equation of the second kind
    
    .. math:: \int_0^{t_\mathrm{max}} \mathrm{d}s R(t-s) u(s) = \lambda u(t)
    
    Quadrature approximation of the integral gives a discrete representation
    which leads to a regular eigenvalue problem.
    
    .. math:: \sum_i w_i R(t_j-s_i) u(s_i) = \lambda u(t_j) \equiv \mathrm{diag(w_i)} \cdot R \cdot u = \lambda u 
    
    Note: If :math:`t_i = s_i \forall i` the matrix :math:`R(t_j-s_i)` is 
    a hermitian matrix. In order to preserve hermiticity for arbitrary :math:`w_i` 
    one defines the diagonal matrix :math:`D = \mathrm{diag(\sqrt{w_i})}` 
    with leads to the equivalent expression:
    
    .. math:: D \cdot R \cdot D \cdot D \cdot u = \lambda D \cdot u \equiv \tilde R \tilde u = \lambda \tilde u
    
    where :math:`\tilde R` is hermitian and :math:`u = D^{-1}\tilde u`.
    
    :param r: correlation matrix :math:`R(t_j-s_i)`
    :param w: integrations weights :math:`w_i` 
        (they have to correspond to the discrete time :math:`t_i`)

    :return: eigenvalues, eigenvectos (eigenvectos are stored in the normal numpy fashion, ordered in decreasing order)

    .. seealso::
        There are various convenient functions to calculate the integration weights and times
        to approximate the integral over the interval [0, t_max] using ng points.

        :py:func:`get_mid_point_weights_times`,
        :py:func:`get_trapezoidal_weights_times`,
        :py:func:`get_simpson_weights_times`,
        :py:func:`get_four_point_weights_times`,
        :py:func:`get_gauss_legendre_weights_times`,
        :py:func:`get_tanh_sinh_weights_times`

    .. note::
        It has been noticed that the performance of the various weights depends on the auto correlation
        function. As default one should use the simpson weights. 'four point', 'gauss legendre' and 'tanh sinh'
        might perform better for auto correlation function that decay slowly. Their advantage becomes evident
        for a large numbers of grid points only. So if one cares about relative differences below 1e-4
        the more sophisticated weights are suitable.
    """
    t0 = time.time()
    # weighted matrix r due to quadrature weights
    n = len(w)
    w_sqrt = np.sqrt(w)
    r = w_sqrt.reshape(n,1) * r * w_sqrt.reshape(1,n)
    eig_val, eig_vec = scipy_eigh(r, overwrite_a=True)   # eig_vals in ascending

    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]
    eig_vec = np.reshape(1/w_sqrt, (n,1)) * eig_vec
    log.debug("discrete fredholm equation of size {} solved [{:.2e}s]".format(n, time.time() - t0))
    return eig_val, eig_vec

def align_eig_vec(eig_vec):
    for i in range(eig_vec.shape[1]):
        phase = np.exp(1j * np.arctan2(np.real(eig_vec[0,i]), np.imag(eig_vec[0,i])))
        eig_vec[:, i] /= phase


def _calc_corr_min_t_plus_t(s, bcf):
    bcf_n_plus = bcf(s - s[0])
    #    [bcf(-3)    , bcf(-2)    , bcf(-1)    , bcf(0), bcf(1), bcf(2), bcf(3)]
    # == [bcf(3)^\ast, bcf(2)^\ast, bcf(1)^\ast, bcf(0), bcf(1), bcf(2), bcf(3)]
    return np.hstack((np.conj(bcf_n_plus[-1:0:-1]), bcf_n_plus))


def _calc_corr_matrix(s, bcf):
    """calculates the matrix alpha_ij = bcf(t_i-s_j)

    calls bcf only for s-s_0 and reconstructs the rest
    """
    n_ = len(s)
    bcf_n = _calc_corr_min_t_plus_t(s, bcf)
    # we want
    # r = bcf(0) bcf(-1), bcf(-2)
    #     bcf(1) bcf( 0), bcf(-1)
    #     bcf(2) bcf( 1), bcf( 0)
    r = np.empty(shape=(n_, n_), dtype=np.complex128)
    for i in range(n_):
        idx = n_ - 1 - i
        r[:, i] = bcf_n[idx:idx + n_]
    return r


def get_mid_point_weights_times(t_max, num_grid_points):
    r"""Returns the weights and grid points for numeric integration via **mid point rule**.

    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: number of grid points N
    :return: location of the grid points, corresponding weights

    Because this function is intended to provide the weights to be used in :py:func:`solve_hom_fredholm`
    is stretches the homogeneous weights over the grid points starting from 0 up to t_max, so the
    term min_point is somewhat miss leading. This is possible because we want to simulate
    stationary stochastic processes which allows :math:`\alpha(t_i+\Delta - (s_j+\Delta)) = \alpha(t_i-s_j)`.
        
    The N grid points are located at
    
    .. math:: t_i = i \frac{t_\mathrm{max}}{N-1} \qquad i = 0,1, ... N - 1

    and the corresponding weights are
    
    .. math:: w_i = \Delta t = \frac{t_\mathrm{max}}{N-1} \qquad i = 0,1, ... N - 1
    """
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    w = np.ones(num_grid_points)*delta_t
    return t, w

def get_trapezoidal_weights_times(t_max, num_grid_points):
    r"""Returns the weights and grid points for numeric integration via **trapezoidal rule**.

    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: number of grid points N
    :return: location of the grid points, corresponding weights

    The N grid points are located at

    .. math:: t_i = i \frac{t_\mathrm{max}}{N-1} \qquad i = 0,1, ... N - 1

    and the corresponding weights are

    .. math:: w_0 = w_{N-1} = \Delta t /2 \qquad w_i = \Delta t \quad  0 < i < N - 1
              \qquad \Delta t = \frac{t_\mathrm{max}}{N-1}
    """
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    w = np.ones(num_grid_points)*delta_t
    w[0] /= 2
    w[-1] /= 2
    return t, w

def get_simpson_weights_times(t_max, num_grid_points):
    r"""Returns the weights and grid points for numeric integration via **simpson rule**.

    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: number of grid points N (needs to be odd)
    :return: location of the grid points, corresponding weights

    The N grid points are located at

    .. math:: t_i = i \frac{t_\mathrm{max}}{N-1} \qquad i = 0,1, ... N - 1

    and the corresponding weights are

    .. math:: w_0 = w_{N-1} = 1/3 \Delta t \qquad w_{2i} = 2/3 \Delta t \quad 0 < i < (N - 1)/2
    .. math:: \qquad w_{2i+1} = 4/3 \Delta t \quad  0 \leq i < (N - 1)/2 \qquad \Delta t = \frac{t_\mathrm{max}}{N-1}
    """
    if num_grid_points % 2 != 1:
        raise RuntimeError("simpson weights needs grid points ng such that ng = 2*k+1, but git ng={}".format(num_grid_points))
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::2] = 2/3*delta_t
    w[1::2] = 4/3*delta_t
    w[0]    = 1/3*delta_t
    w[-1]   = 1/3*delta_t
    return t, w

def get_four_point_weights_times(t_max, num_grid_points):
    r"""Returns the weights and grid points for numeric integration via **four-point Newton-Cotes rule**.

    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: number of grid points N (needs to be (4k+1) where k is an integer greater 0)
    :return: location of the grid points, corresponding weights

    The N grid points are located at

    .. math:: t_i = i \frac{t_\mathrm{max}}{N-1} \qquad i = 0,1, ... N - 1

    and the corresponding weights are

    .. math:: w_0 = w_{N-1} = 28/90 \Delta t
    .. math:: w_{4i+1} = w_{4i+3} = 128/90 \Delta t \quad 0 \leq i < (N - 1)/4
    .. math:: w_{4i+2} = 48/90 \Delta t \quad 0 \leq i < (N - 1)/4
    .. math:: w_{4i}   = 56/90 \Delta t \quad 0 < i < (N - 1)/4
    .. math:: \Delta t = \frac{t_\mathrm{max}}{N-1}
    """
    if num_grid_points % 4 != 1:
        raise RuntimeError("four point weights needs grid points ng such that ng = 4*k+1, but git ng={}".format(num_grid_points))
    t, delta_t = np.linspace(0, t_max, num_grid_points, retstep = True)
    w = np.empty(num_grid_points, dtype=np.float64)
    w[0::4] = 56  / 90 * delta_t
    w[1::4] = 128 / 90 * delta_t
    w[2::4] = 48  / 90 * delta_t
    w[3::4] = 128 / 90 * delta_t
    w[0]    = 28  / 90 * delta_t
    w[-1]   = 28  / 90 * delta_t
    return t, w

def get_gauss_legendre_weights_times(t_max, num_grid_points):
    """Returns the weights and grid points for numeric integration via **Gauss integration**
    by expanding the function in terms of Legendre Polynomials.

    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: number of grid points N
    :return: location of the grid points, corresponding weights
    """
    return gquad.gauss_nodes_weights_legendre(n = num_grid_points, low=0, high=t_max)

def get_tanh_sinh_weights_times(t_max, num_grid_points):
    r"""Returns the weights and grid points for numeric integration via **Tanh-Sinh integration**. The idea is to
    transform the integral over a finite interval :math:`x \in [-1, 1]` via the variable transformation

    .. math :: x = \tanh(\pi/2 \sinh(t))

    to a integral over the entire real axis :math:`t \in [-\infty,\infty]` but where the new
    transformed integrand decay rapidly such that a simply midpoint rule performs very well.

    inspired by 'Tanh-Sinh High-Precision Quadrature - David H. Bailey'

    :param t_max: end of the interval for the time grid :math:`[0,t_\mathrm{max}]`
    :param num_grid_points: number of grid points N (needs to be odd)
    :return: location of the grid points, corresponding weights

    For a fixed small parameter h the location of the grid points read

    .. math :: x_i = \tanh(\pi/2 \sinh(ih)

    with corresponding weights

    .. math :: w_i = \frac{\pi/2 \cosh(ih)}{\cosh^2(\pi/2 \sinh(ih))}

    where i can be any integer. For a given number of grid points N, h is chosen such that
    :math:`w_{(N-1)/2} < 10^{-14}` which implies odd N. With that particular h :math:`x_i` and
    :math:`w_i` are calculated for :math:`-(N-1)/2 \leq i \leq (N-1)/2`. Afterwards the :math:`x_i` are linearly
    scaled such that :math:`x_{-(N-1)/2} = 0` and :math:`x_{(N-1)/2} = t_\mathrm{max}`.
    """
    def get_h_of_N(N):
        r"""returns the stepsize h for sinh tanh quad for a given number of points N
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

def subdevide_axis(t, ngfac):
    ng = len(t)
    if not isinstance(t, np.ndarray):
        t = np.asarray(t)
    tfine = np.empty(shape=((ng - 1) * ngfac + 1))
    tfine[::ngfac] = t
    for l in range(1, ngfac):
        tfine[l::ngfac] = (l * t[1:] + (ngfac - l) * t[:-1]) / ngfac
    return tfine

def auto_ng(corr, t_max, ngfac=2, meth=get_mid_point_weights_times, tol=1e-3, diff_method='full', dm_random_samples=10**4):
    r"""increase the number of gridpoints until the desired accuracy is met

    This function increases the number of grid points of the discrete Fredholm equation exponentially until
    a given accuracy is met. The accuracy is determined from the deviation of the approximated
    auto correlation of the Karhunen-Loève expansion from the given reference auto correlation.

    .. math::

        \Delta(n) = \max_{t,s \in [0,t_\mathrm{max}]}\left( \Big | \alpha(t-s) - \sum_{i=1}^n \lambda_i u_i(t) u_i^\ast(s) \Big | \right )

    :param corr: the auto correlation function
    :param t_max: specifies the interval [0, t_max] for which the stochastic process can be evaluated
    :param ngfac: specifies the fine grid to use for the spline interpolation, the intermediate points are
        calculated using integral interpolation
    :param meth: the method for calculation integration weights and times, a callable or one of the following strings
        'midpoint' ('midp'), 'trapezoidal' ('trapz'), 'simpson' ('simp'), 'fourpoint' ('fp'),
        'gauss_legendre' ('gl'), 'tanh_sinh' ('ts')
    :param tol: defines the success criterion max(abs(corr_exact - corr_reconstr)) < tol
    :param diff_method: either 'full' or 'random', determines the points where the above success criterion is evaluated,
        'full': full grid in between the fine grid, such that the spline interpolation error is expected to be maximal
        'random': pick a fixed number of random times t and s within the interval [0, t_max]
    :param dm_random_samples: the number of random times used for diff_method 'random'
    :return: an array containing the necessary eigenfunctions of the Karhunen-Loève expansion for sampling the
        stochastic processes (shape=(num_eigen_functions, num_grid_points)

    The procedure works as follows:
        1) Solve the discrete Fredholm equation on a grid with ng points.
           This gives ng eigenvalues/vectors where each ng-dimensional vector approximates the continuous eigenfunction.
           (:math:`t, u_i(t) \leftrightarrow t_k, u_{ik}` where the :math:`t_k` depend on the integration weights
           method)
        2) Approximate the eigenfunction on a finer, equidistant grid
           (:math:`ng_\mathrm{fine} = ng_\mathrm{fac}(ng-1)+1`) using

           .. math::

               u_i(t) = \frac{1}{\lambda_i} \int_0^{t_\mathrm{max}} \mathrm{d}s \; \alpha(t-s) u_i(s)
               \approx \frac{1}{\lambda_i} \sum_k w_k \alpha(t-s_k) u_{ik}

           According to the Numerical Recipes [1] this interpolation should perform better that simple
           spline interpolation. However it turns that this is not the case in general (e.g. for exponential
           auto correlation functions the spline interpolation performs better). For that reason it might be
           usefull to set ngfac to 1 which will skip the integral interpolation
        3) Use the eigenfunction on the fine grid to setup a cubic spline interpolation.
        4) Use the spline interpolation to estimate the deviation :math:`\Delta(n)`. When using diff_method = 'full'
           the maximization is performed over all :math:`t'_i, s'_j` where :math:`t'_i = (t_i + t_{i+1})/2` and
           :math:`s'_i = (s_i + s_{i+1})/2` with :math:`i,j = 0, \, ...\, , ng_\mathrm{fine}-2`. It is expected that
           the interpolation error is maximal when beeing in between the reference points.
        5) Now calculate the deviation :math:`\Delta(n)` for sequential n starting at n=0. Stop if
           :math:`\Delta(n) < tol`. If the deviation does not drop below tol for all :math:`0 \leq n < ng-1` increase
           ng as follows :math:`ng = 2*ng-1` and start over at 1). (This update schema for ng asured that ng is odd
           which is needed for the 'simpson' and 'fourpoint' integration weights)



    [1] Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P.,
    2007. Numerical Recipes 3rd Edition: The Art of Scientific Computing,
    Auflage: 3. ed. Cambridge University Press, Cambridge, UK ; New York. (pp. 990)
    """
    time_start = time.time()
    if diff_method == 'full':
        pass
    elif diff_method == 'random':
        t_rand = np.random.rand(dm_random_samples) * t_max
        s_rand = np.random.rand(dm_random_samples) * t_max
        alpha_ref = corr(t_rand - s_rand)
    else:
        raise ValueError("unknown diff_method '{}', use 'full' or 'random'".format(diff_method))
    alpha_0 = np.abs(corr(0))
    log.debug("diff_method: {}".format(diff_method))

    time_fredholm = 0
    time_calc_ac = 0
    time_integr_intp = 0
    time_spline = 0
    time_calc_diff = 0

    if isinstance(meth, str):
        meth = str_meth_to_meth(meth)

    k = 4
    while True:
        k += 1
        ng = 2 ** k + 1
        log.info("check {} grid points".format(ng))
        t, w = meth(t_max, ng)

        is_equi = is_axis_equidistant(t)

        t0 = time.time()                                      # efficient way to construct the
        r = _calc_corr_matrix(t, corr)                        # auto correlation matrix r
        time_calc_ac += (time.time() - t0)

        t0 = time.time()                                      # solve the dicrete fredholm equation
        _eig_val, _eig_vec = solve_hom_fredholm(r, w)         # using integration weights w
        time_fredholm += (time.time() - t0)

        tfine = subdevide_axis(t, ngfac)                      # setup fine
        tsfine = subdevide_axis(tfine, 2)[1::2]               # and super fine time grid

        if is_equi:
            t0 = time.time()                                  # efficient way to calculate the auto correlation
            alpha_k = _calc_corr_min_t_plus_t(tfine, corr)    # from -tmax untill tmax on the fine grid
            time_calc_ac += (time.time() - t0)                # needed for integral interpolation

        if diff_method == 'full':
            if not is_equi:
                alpha_ref = corr(tsfine.reshape(-1,1) - tsfine.reshape(1,-1))
            else:
                ng_sfine = len(tsfine)
                alpha_ref = np.empty(shape=(ng_sfine, ng_sfine), dtype=np.complex128)
                for i in range(ng_sfine):
                    idx = ng_sfine - i
                    alpha_ref[:, i] = alpha_k[idx:idx + ng_sfine]  # note we can use alpha_k as
                                                                   # alpha(ti+dt/2 - (tj+dt/2)) = alpha(ti - tj)

        diff = -alpha_ref
        old_md = np.inf
        sqrt_lambda_ui_fine_all = []
        for i in range(ng):
            evec = _eig_vec[:, i]
            if _eig_val[i] < 0:
                print(ng, i)
                break
            sqrt_eval = np.sqrt(_eig_val[i])
            if ngfac != 1:
                t0 = time.time()
                # when using sqrt_lambda instead of lambda we get sqrt_lamda time u
                # which is the quantity needed for the stochastic process generation
                if not is_equi:
                    sqrt_lambda_ui_fine = np.asarray([np.sum(corr(ti - t) * w * evec) / sqrt_eval for ti in tfine])
                else:
                    sqrt_lambda_ui_fine = stocproc_c.eig_func_interp(delta_t_fac=ngfac,
                                                                     time_axis=t,
                                                                     alpha_k=alpha_k,
                                                                     weights=w,
                                                                     eigen_val=sqrt_eval,
                                                                     eigen_vec=evec)

                time_integr_intp += (time.time() - t0)
            else:
                sqrt_lambda_ui_fine = evec*sqrt_eval

            sqrt_lambda_ui_fine_all.append(sqrt_lambda_ui_fine)

            # setup cubic spline interpolator
            t0 = time.time()
            sqrt_lambda_ui_spl = tools.ComplexInterpolatedUnivariateSpline(tfine, sqrt_lambda_ui_fine)
            time_spline += (time.time() - t0)

            # calculate the max deviation
            t0 = time.time()
            if diff_method == 'random':
                ui_t = sqrt_lambda_ui_spl(t_rand)
                ui_s = sqrt_lambda_ui_spl(s_rand)
                diff += ui_t * np.conj(ui_s)
            elif diff_method == 'full':
                ui_super_fine = sqrt_lambda_ui_spl(tsfine)
                diff += ui_super_fine.reshape(-1, 1) * np.conj(ui_super_fine.reshape(1, -1))
            md = np.max(np.abs(diff)) / alpha_0
            time_calc_diff += (time.time() - t0)

            log.debug("num evec {} -> max diff {:.3e}".format(i+1, md))
            #if old_md < md:
            #    log.info("max diff increased -> break, use higher ng")
            #    break
            #old_md = md
            if md < tol:
                time_total = time_calc_diff + time_spline + time_integr_intp + time_calc_ac + time_fredholm
                time_overall = time.time() - time_start
                time_rest = time_overall - time_total

                log.info("calc_ac {:.3%}, fredholm {:.3%}, integr_intp {:.3%}, spline {:.3%}, calc_diff {:.3%}, rest {:.3%}".format(
                          time_calc_ac/time_overall, time_fredholm/time_overall, time_integr_intp/time_overall,
                          time_spline/time_overall, time_calc_diff/time_overall, time_rest/time_overall))
                log.info("auto ng SUCCESSFUL max diff {:.3e} < tol {:.3e} ng {} num evec {}".format(md, tol, ng, i+1))
                return np.asarray(sqrt_lambda_ui_fine_all)
        log.info("ng {} yields md {:.3e}".format(ng, md))

def is_axis_equidistant(ax):
    ax = np.asarray(ax)
    d = ax[1:]-ax[:-1]
    return np.max(np.abs(d - d[0])) < 1e-15


def str_meth_to_meth(meth):
    if (meth == 'midpoint') or (meth == 'midp'):
        return get_mid_point_weights_times
    elif (meth == 'trapezoidal') or (meth == 'trapz'):
        return get_trapezoidal_weights_times
    elif (meth == 'simpson') or (meth == 'simp'):
        return get_simpson_weights_times
    elif (meth == 'fourpoint') or (meth == 'fp'):
        return get_four_point_weights_times
    elif (meth == 'gauss_legendre') or (meth == 'gl'):
        return get_gauss_legendre_weights_times
    elif (meth == 'tanh_sinh') or (meth == 'ts'):
        return get_tanh_sinh_weights_times
    else:
        raise ValueError("unknown method to get integration weights '{}'".format(meth))