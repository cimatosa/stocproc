## How to sample stochastic processes

Note that, in general, such a stochastic proces is not Markovian.
Which is equivalent to the fact that the stochastic process cannot be generated
by deducing its value $z(u)$ for $u > t$ from the value $z(t)$ only.
Instead, the algorithms used in this module generate a sample of the stochastic process
"at once". This does not mean that Markovian processes, i.e.
[Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process),
cannot be generated. It just means, that they might be generated more easily using other approaches.

!!! todo
    Implement special generators for Markovian processes.

Two distinct classes of algorithms have been considered (see
R. Hartmann [2022](https://nbn-resolving.org/urn:nbn:de:bsz:14-qucosa2-785576)
Sec. 3.1 for a detailed discussion).


### Karhunen–Loève expansion

Using the
[Karhunen–Loève expansion](https://en.wikipedia.org/wiki/Kosambi%E2%80%93Karhunen%E2%80%93Lo%C3%A8ve_theorem),
a stochastic process over the interval $t\in[a, b]$ can be written as

$$
    z(t) =\sum_{k=1}^\infty g_k Y_k e_k(t)
$$

where $Y_k$ are independent complex valued Gaussian random variables with variance 1,
i.e. $\langle Y_k Y^\ast_l \rangle=\delta_{kl}$.
The coefficients $g_k$ and the $L^2[a, b]$-orthogonal functions $e_k(t)$ follow from the solutions
of the homogeneous Fredholm equation

$$
    \int_a^b \mathrm{d}s\alpha(t-s) e_k(s) = |g_k|^2 e_k(t)
$$

with the auto-correlation function $\alpha(t-s)$ serves as integral kernel.

For a purely numerical treatment the integral is discretized.
The resulting equation amounts to a common eigenvalue problem.
Since the dimension of that system of equations originates from the number of
nodes used for the integral discretization, the interval $[a,b]$ must not be too large
to obtain solution in considerable time.

[`KarhunenLoeve`][stocproc.samplers.KarhunenLoeve] implements that procedure ensuring a given degree of accuracy.

!!! ToDo
    Exploit the Toeplitz-structure of the integral kernel to reduce the complexity of the Eigenvalue problem to $O(n^2)$
    (see Trench, F.W., 1989. Numerical Solution of the Eigenvalue Problem for Hermitian Toeplitz Matrices.
    SIAM J. Matrix Anal. Appl. 10, 135–146. [https://doi.org/10.1137/0610010](https://doi.org/10.1137/0610010)).

### Fourier method

It follows from the [Wiener–Khinchin theorem](https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem)
that the auto-correlation function $\alpha(\tau)$ can be expressed as Fourier transform of the non-negative
power spectral density $S(\omega) \geq 0$, i.e.,

$$
    \alpha(\tau) =\int_{-\infty}^\infty\mathrm{d}\omega S(\omega) e^{-i\omega\tau} \; .
$$

It follows that the stochastic process

$$
    z_N(t) = \sum_{k=1}^N Y_k \sqrt{m_k J(\omega_k)} e^{-i \omega_k t}
$$

with $Y_k$ being independent complex valued Gaussian random variables with variance 1,
quadrature weights $m_k$ and quadrature nodes $\omega_k$ approximately fulfills the desired statistics, i.e.

$$
      \langle z_N(t) z_N^\ast(s) \rangle
    = \sum_k m_k J(\omega_k) e^{-i \omega_k (t-s)}
    \approx \alpha(t-s)\; .
$$

Depending on the spectral density $J(\omega)$, the quadrature weights and nodes can be determined
such that a given accuracy is met.

#### Fast Fourier Transform
In many cases simple midpoint quadrature in combinations with the
[fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) algorithm
yields great results (implemented by [`FastFourier`][stocproc.samplers.FastFourier]).

#### TanhSinh quadrature
To efficiently treat spectral densities with singularities an algorithm based on
[Tanh-Sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) is implemented in
[`TanhSinh`][stocproc.samplers.TanhSinh]).

### Cholesky decomposition

see [`Cholesky`][stocproc.samplers.Cholesky]

!!! warning
    details still missing