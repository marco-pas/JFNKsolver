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

<p align="center">
  <img src="./output/examplesBurgers/taylorGreenVortex_.gif" alt="TGV" width="100%">
  <br>
  <em>Fig. 1: Simulation of the Taylor-Green Vortex evolution over time.</em>
</p>


**Initial Conditions:**

$$u(x, y) = \sin(x) \ \cos(y)$$
$$v(x, y) = -\cos(x) \ \sin(y)$$

**Energy Dissipation:**
<p align="center">
  <img src="./output/examplesBurgers/taylorGreenVortex_energy_.png" alt="TGV" width="50%">
  <br>
  <em>Fig. 2: Kinetic energy exponential decay for the Taylor-Green Vortex.</em>
</p>

# Double Shear Layer (DSL)

<p align="center">
  <img src="./output/examplesBurgers/doubleShearLayer_.gif" alt="TGV" width="100%">
  <br>
  <em>Fig. 3: Simulation of the Double Shear Layer evolution over time.</em>
</p>

**Initial Conditions:**
Given the steepness parameter $\rho = 30.0$ and the perturbation amplitude $\delta = 0.05$, the initial velocity field is defined as a piecewise function:

$$u(x, y) = \begin{cases} \tanh\left(\rho \left(y - \frac{\pi}{2}\right)\right) & \text{if } y \le \pi \\ \tanh\left(\rho \left(\frac{3\pi}{2} - y\right)\right) & \text{if } y > \pi \end{cases}$$
$$v(x, y) = \delta \sin(x)$$

$$u(x, y) = \begin{cases} \tanh\left(\rho \left(y - \frac{\pi}{2}\right)\right) & \text{if } y \le \pi \\ \tanh\left(\rho \left(\frac{3\pi}{2} - y\right)\right) & \text{if } y > \pi \end{cases}$$
$$v(x, y) = \delta \sin(x)$$

**Energy Dissipation:**
<p align="center">
  <img src="./output/examplesBurgers/doubleShearLayer_energy_.png" alt="TGV" width="50%">
  <br>
  <em>Fig. 4: Kinetic energy exponential decay for the Double Shear Layer.</em>
</p>

# 4-Vortex Collision (4VC)


<p align="center">
  <img src="./output/examplesBurgers/fourVortexColl_.gif" alt="TGV" width="100%">
  <br>
  <em>Fig. 5: Simulation of the 4-Vortex Collision evolution over time.</em>
</p>

**Initial Conditions:**
The field is defined by a superposition of four Gaussian vortices. Given the vortex radius $R = 0.5$, and a set of four vortices with centers $C_i = (c_{x,i}, c_{y,i})$ and circulation strengths $\Gamma_i$:
* $C_1 = (\pi - 0.8, \pi - 0.8)$ with $\Gamma_1 = 1.0$
* $C_2 = (\pi + 0.8, \pi + 0.8)$ with $\Gamma_2 = 1.0$
* $C_3 = (\pi - 0.8, \pi + 0.8)$ with $\Gamma_3 = -1.0$
* $C_4 = (\pi + 0.8, \pi - 0.8)$ with $\Gamma_4 = -1.0$

Let the squared distance to each center be $r_i^2 = (x - c_{x,i})^2 + (y - c_{y,i})^2$. The velocity components are:

$$u(x, y) = \sum_{i=1}^{4} -\Gamma_i (y - c_{y,i}) \ e^ {- \frac{r_i^2}{R^2} }$$
$$v(x, y) = \sum_{i=1}^{4} \Gamma_i (x - c_{x,i}) \ e^ { - \frac{r_i^2}{R^2}  }$$

**Energy Dissipation:**
<p align="center">
  <img src="./output/examplesBurgers/fourVortexColl_energy_.png" alt="TGV" width="50%">
  <br>
  <em>Fig. 6: Kinetic energy exponential decay for the 4-Vortex Collision.</em>
</p>
