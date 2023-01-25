## Introduction to stochastic processes

While it is well established to generate (pseudo) random numbers, it is slightly more
challenging to sample stochastic processes.
In essence, such processes are random "functions" of a single continuous parameter $z(t)$, usually called time.
For a fixed realization of such a process $z_1$, the values returned by that function, i.e., the values of
the stochastic process, are fixed ($z_1(t)$ is deterministic for all $t$).
On the other hand, for a fixed time $t$, the "value" of the stochastic process $z(t)$ is a random variable.
A stochastic process is thus a family of random variables.

The [`stocproc`][stocproc] module allows to sample a certain class of stochastic processes with the following properties

- complex valued: $z(t) \in \mathbb{Z}$
- Gaussian distributed: Any finite set of random variables $(z(t_1),\dots z(t_n)) is a multivariavte
Gaussian random variable. This implies that such a process is fully characterized by its first and second moments.
- zero average: $\langle z(t)\rangle = 0$
- zero pseudo-covariance: $\langle z(t)z(s)\rangle = 0$
- wide-sense stationary: $\langle z(t)z^\ast(s)\rangle = \alpha(t-s)$

Thus, such a process is fully characterized by its auto-correlation function $\alpha(\tau)$.