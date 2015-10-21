import numpy as np 

def get_param_single_lorentz(tmax, dw_max, eta, gamma, wc, x=1e-4, verbose=0):
    d = gamma * np.sqrt(1/x - 1)
    w_min = wc - d
    w_max = wc + d
    
    if verbose > 0:
        print('w_min :{:.3}'.format(w_min))
        print('w_max :{:.3}'.format(w_max))
    
    C = (w_max - w_min)*tmax / 2 / np.pi
    
    N = int(np.ceil((2 + C)/2 + np.sqrt( (2+C)**2 / 4 - 1)))
    dw = w_max - w_min
    if verbose > 0:
        print('N: {}'.format(N))
        print('-> dw: {:.3}'.format(dw))
    
    if dw <= dw_max:
        print('dw <= dw_max: {:.3}'.format(dw_max))
        return N, w_min, tmax
    else:
        print('dw > dw_max: {:.3}'.format(dw_max))
        print('adjust tmax and N to fulfill dw <= dw_max')
        N = int(np.ceil((w_max - w_min) / dw_max)) - 1
        dt = 2*np.pi / (dw_max*N)
        tmax_ = dt*N
        print('N: {}'.format(N))
        print('-> tmax: {:.3}'.format(tmax_))
        assert tmax_ > tmax
        return N, w_min, tmax
