from numpy import zeros
from numpy import empty
from numpy import conj
from numpy import complex128, float64
cimport numpy as cnp
from cpython cimport bool

DTYPE_CPLX = complex128
ctypedef cnp.complex128_t DTYPE_CPLX_t

DTYPE_DBL = float64
ctypedef cnp.float64_t DTYPE_DBL_t

cpdef cnp.ndarray[DTYPE_CPLX_t, ndim=1] eig_func_interp(unsigned int                     delta_t_fac,
                                                        cnp.ndarray[DTYPE_DBL_t,  ndim=1] time_axis,
                                                        cnp.ndarray[DTYPE_CPLX_t, ndim=1] alpha_k, 
                                                        cnp.ndarray[DTYPE_DBL_t,  ndim=1] weights,
                                                        double                           eigen_val,
                                                        cnp.ndarray[DTYPE_CPLX_t, ndim=1] eigen_vec):

    cdef unsigned int N1
    N1 = len(time_axis)
    
    cdef unsigned int N2
    N2 = delta_t_fac * (N1 - 1) + 1
    
    cdef cnp.ndarray[DTYPE_CPLX_t, ndim=1] u_res 
    u_res = zeros(shape=N2, dtype=DTYPE_CPLX)
    
    cdef unsigned int j
    cdef unsigned int l
    cdef unsigned int k
    
    for j in range(N2):
        for l in range(N1):
            k = j - delta_t_fac*l + N2-1
            u_res[j] = u_res[j] + weights[l] * alpha_k[k] * eigen_vec[l]      

    return u_res / eigen_val      



cpdef cnp.ndarray[DTYPE_CPLX_t, ndim=2] eig_func_all_interp(unsigned int                      delta_t_fac,
                                                            cnp.ndarray[DTYPE_DBL_t,  ndim=1] time_axis,
                                                            cnp.ndarray[DTYPE_CPLX_t, ndim=1] alpha_k, 
                                                            cnp.ndarray[DTYPE_DBL_t,  ndim=1] weights,
                                                            cnp.ndarray[DTYPE_DBL_t,  ndim=1] eigen_val,
                                                            cnp.ndarray[DTYPE_CPLX_t, ndim=2] eigen_vec):

    cdef unsigned int N1
    N1 = len(time_axis)
    
    cdef unsigned int N2
    N2 = delta_t_fac * (N1 - 1) + 1
    
    cdef unsigned int num_ev
    num_ev = len(eigen_val)
    
    cdef  cnp.ndarray[DTYPE_CPLX_t, ndim=2] u_res 
    u_res = zeros(shape=(N2,num_ev), dtype=DTYPE_CPLX)
    
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int l
    cdef unsigned int k
    
    for i in range(num_ev):
        for j in range(N2):
            for l in range(N1):
                k = j - delta_t_fac*l + N2-1
                u_res[j, i] = u_res[j, i] + weights[l] * alpha_k[k] * eigen_vec[l, i]
                      
            u_res[j, i] = u_res[j, i]/eigen_val[i]
            
    return u_res   



cpdef cnp.ndarray[DTYPE_CPLX_t, ndim=1] z_t(unsigned int                      delta_t_fac,
                                            unsigned int                      N1,
                                            cnp.ndarray[DTYPE_CPLX_t, ndim=1] alpha_k,
                                            cnp.ndarray[DTYPE_CPLX_t, ndim=1] a_tmp,
                                            bool                              kahanSum):
    cdef unsigned int N2
    N2 = delta_t_fac * (N1 - 1) + 1
       
    cdef cnp.ndarray[DTYPE_CPLX_t, ndim=1] z_t_res 
    z_t_res = empty(shape=N2, dtype=DTYPE_CPLX)
 
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int a
    cdef unsigned int k
    
    cdef DTYPE_CPLX_t s
    cdef DTYPE_CPLX_t c
    cdef DTYPE_CPLX_t y
    cdef DTYPE_CPLX_t t
     
    if kahanSum:
        for j in range(N2):
            s = 0.0
            c = 0.0
            for i in range(N1):
                k = j - delta_t_fac*i + N2-1
                
                y = alpha_k[k] * a_tmp[i] - c          # the summand with come correction
                t = s + y                              # do the summation, and store in tmp var 't'
                c = (t - s) - y                        # see what got lost
                s = t                                  # update sum
     
            z_t_res[j] = s
        return z_t_res
    else:
        for j in range(N2):
            s = 0.0
            for i in range(N1):
                t = 0.0
                k = j - delta_t_fac*i + N2-1
                s += alpha_k[k] * a_tmp[i]
            z_t_res[j] = s
        return z_t_res

# cpdef cnp.ndarray[DTYPE_CPLX_t, ndim=2] auto_correlation(cnp.ndarray[DTYPE_CPLX_t, ndim=2] x):
def auto_correlation(cnp.ndarray[DTYPE_CPLX_t, ndim=2] x):
    r"""Computes the auto correlation function for a set of wide-sense stationary stochastic processes
    
    Computes the auto correlation function for the given set :math:`{X_i(t)}` of stochastic processes:
    
    .. math:: \alpha(s, t) = \langle X(t)X^\ast(s) \rangle
    
    For wide-sense stationary processes :math:`\alpha` is independent of :math:`s`.
    
    :param x: 2D array of the shape (num_samples, num_time_points) containing the set of stochastic processes where each row represents one process
    
    :return: 2D array containing the correlation function as function of :math:`s, t` 
    """
            
    cdef unsigned int num_samples = x.shape[0]
    cdef unsigned int num_time_points = x.shape[1]
    
    cdef cnp.ndarray[DTYPE_CPLX_t, ndim=2] ac_res
    ac_res = empty(shape=(num_time_points, num_time_points), dtype=DTYPE_CPLX)
    cdef cnp.ndarray[DTYPE_CPLX_t, ndim=2] ac_res_prime
    ac_res_prime = empty(shape=(num_time_points, num_time_points), dtype=DTYPE_CPLX)
    
    cdef cnp.ndarray[DTYPE_CPLX_t, ndim=2] x_conj = conj(x)
    
    cdef DTYPE_CPLX_t tmp
    cdef DTYPE_CPLX_t tmp_prime
    
    cdef unsigned int time_idx_1
    cdef unsigned int time_idx_2
    cdef unsigned int i
    
    
    for time_idx_1 in range(num_time_points):
        for time_idx_2 in range(num_time_points):
            tmp = 0
            tmp_prime = 0
            for i in range(num_samples):
                tmp += x[i, time_idx_1] * x_conj[i, time_idx_2]
                tmp_prime += x[i, time_idx_1] * x[i, time_idx_2]

            ac_res[time_idx_1, time_idx_2] = tmp / num_samples
            ac_res_prime[time_idx_1, time_idx_2] = tmp_prime / num_samples
    
    return ac_res, ac_res_prime
  