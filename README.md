# 2D Burgers' Equation: AD vs. FD JFNK Solver
A optimized, fully implicit solver for the 2D Burgers' equation using a Jacobian-Free Newton-Krylov (JFNK) method. This script benchmarks Automatic Differentiation (AD) from JAX against Finite Differences (FD) for Jacobian-vector products.

**Burgers' equation:**

$$
\frac{\partial \mathcal{U}}{\partial t} + (\mathcal{U} \cdot \nabla)\mathcal{U} = \nu \nabla^2 \mathcal{U}
$$

Where $\mathcal{U} = (u, v)$ is the velocity field, and $\nu$ is the kinematic viscosity.

**Key Features:**
* Combines JIT-compiled JAX math with Optimized GMRES solvers (SciPy for CPU version, CuPy for GPU version).
* Includes Backtracking Line Search to stabilize highly non-linear shocks.
* Test cases: Taylor-Green Vortex (TGV), Double Shear Layer (DSL), 4-Vortex Collision (4VC).

# Taylor-Green Vortex (TGV)
![TGV](./output/examplesBurgers/taylorGreenVortex_.gif)

**Initial Conditions:**

$$u(x, y) = \sin(x) \ \cos(y)$$
$$v(x, y) = -\cos(x) \ \sin(y)$$

# Double Shear Layer (DSL)
![DSL](./output/examplesBurgers/doubleShearLayer_.gif)

**Initial Conditions:**
Given the steepness parameter $\rho = 30.0$ and the perturbation amplitude $\delta = 0.05$, the initial velocity field is defined as a piecewise function:

$$u(x, y) = \begin{cases} \tanh\left(\rho \left(y - \frac{\pi}{2}\right)\right) & \text{if } y \le \pi \\ \tanh\left(\rho \left(\frac{3\pi}{2} - y\right)\right) & \text{if } y > \pi \end{cases}$$
$$v(x, y) = \delta \sin(x)$$

$$u(x, y) = \begin{cases} \tanh\left(\rho \left(y - \frac{\pi}{2}\right)\right) & \text{if } y \le \pi \\ \tanh\left(\rho \left(\frac{3\pi}{2} - y\right)\right) & \text{if } y > \pi \end{cases}$$
$$v(x, y) = \delta \sin(x)$$

# 4-Vortex Collision (4VC)
![4VC](./output/examplesBurgers/fourVortexColl_.gif)

**Initial Conditions:**
The field is defined by a superposition of four Gaussian vortices. Given the vortex radius $R = 0.5$, and a set of four vortices with centers $C_i = (c_{x,i}, c_{y,i})$ and circulation strengths $\Gamma_i$:
* $C_1 = (\pi - 0.8, \pi - 0.8)$ with $\Gamma_1 = 1.0$
* $C_2 = (\pi + 0.8, \pi + 0.8)$ with $\Gamma_2 = 1.0$
* $C_3 = (\pi - 0.8, \pi + 0.8)$ with $\Gamma_3 = -1.0$
* $C_4 = (\pi + 0.8, \pi - 0.8)$ with $\Gamma_4 = -1.0$

Let the squared distance to each center be $r_i^2 = (x - c_{x,i})^2 + (y - c_{y,i})^2$. The velocity components are:

$$u(x, y) = \sum_{i=1}^{4} -\Gamma_i (y - c_{y,i}) \ e^ {- \frac{r_i^2}{R^2} }$$
$$v(x, y) = \sum_{i=1}^{4} \Gamma_i (x - c_{x,i}) \ e^ { - \frac{r_i^2}{R^2}  }$$
