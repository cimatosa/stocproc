import numpy as np
cimport numpy as np

DTYPE_CPLX = np.complex128
ctypedef np.complex128_t DTYPE_CPLX_t

DTYPE_DBL = np.float64
ctypedef np.float64_t DTYPE_DBL_t

cpdef np.ndarray[DTYPE_CPLX_t, ndim=1] eig_func_interp(unsigned int                     delta_t_fac,
                                                       np.ndarray[DTYPE_DBL_t,  ndim=1] time_axis,
                                                       np.ndarray[DTYPE_CPLX_t, ndim=1] alpha_k, 
                                                       np.ndarray[DTYPE_DBL_t,  ndim=1] weights,
                                                       double                           eigen_val,
                                                       np.ndarray[DTYPE_CPLX_t, ndim=1] eigen_vec):

    cdef unsigned int N1
    N1 = len(time_axis)
    
    cdef unsigned int N2
    N2 = delta_t_fac * (N1 - 1) + 1
    
    cdef np.ndarray[DTYPE_CPLX_t, ndim=1] u_res 
    u_res = np.zeros(shape=N2, dtype=DTYPE_CPLX)
    
    cdef unsigned int j
    cdef unsigned int l
    cdef unsigned int k
    for j in range(N2):
        for l in range(N1):
            k = j - delta_t_fac*l + N2-1
            u_res[j] = u_res[j] + weights[l] * alpha_k[k] * eigen_vec[l]      

    return u_res / eigen_val      



cpdef np.ndarray[DTYPE_CPLX_t, ndim=2] eig_func_all_interp(unsigned int                     delta_t_fac,
                                                           np.ndarray[DTYPE_DBL_t,  ndim=1] time_axis,
                                                           np.ndarray[DTYPE_CPLX_t, ndim=1] alpha_k, 
                                                           np.ndarray[DTYPE_DBL_t,  ndim=1] weights,
                                                           np.ndarray[DTYPE_DBL_t,  ndim=1] eigen_val,
                                                           np.ndarray[DTYPE_CPLX_t, ndim=2] eigen_vec):

    cdef unsigned int N1
    N1 = len(time_axis)
    
    cdef unsigned int N2
    N2 = delta_t_fac * (N1 - 1) + 1
    
    cdef unsigned int num_ev
    num_ev = len(eigen_val)
    
    cdef np.ndarray[DTYPE_CPLX_t, ndim=2] u_res 
    u_res = np.zeros(shape=(N2,num_ev), dtype=DTYPE_CPLX)
    
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



cpdef np.ndarray[DTYPE_CPLX_t, ndim=1] z_t(unsigned int                     delta_t_fac,
                                           np.ndarray[DTYPE_DBL_t,  ndim=1] time_axis,
                                           np.ndarray[DTYPE_CPLX_t, ndim=1] alpha_k,
                                           np.ndarray[DTYPE_CPLX_t, ndim=1] Y,
                                           np.ndarray[DTYPE_CPLX_t, ndim=2] A):
    
    cdef unsigned int N1
    N1 = len(time_axis)
    
    cdef unsigned int N2
    N2 = delta_t_fac * (N1 - 1) + 1
    
    cdef unsigned int num_ev
    num_ev = len(Y)
    
    cdef np.ndarray[DTYPE_CPLX_t, ndim=1] z_t_res 
    z_t_res = np.zeros(shape=N2, dtype=DTYPE_CPLX)
 
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int a
    
    for j in range(N2):
        for i in range(N1):
            k = j - delta_t_fac*i + N2-1
            for a in range(num_ev):
                z_t_res[j] = z_t_res[j] + Y[a]*alpha_k[k]*A[i,a]
    
    return z_t_res