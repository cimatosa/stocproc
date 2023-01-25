from functools import partial
import stocproc as sp


def alpha(t, wc):
    """Ohmic correlation function, wc: cutoff frequency"""
    return (wc/(1+1j*wc*t))**2


al = partial(alpha, wc=5)
t_max = 3
my_sp = sp.StocProc_KLE(alpha=al, t_max=t_max)
my_sp.new_process()
print(my_sp(2.3))

print(my_sp.get_num_y())

my_sp = sp.StocProc_KLE(alpha=al, t_max=t_max, tol=1e-3)
print(my_sp.get_num_y())
