# AD vs. FD Jacobian-Free Newton-Krylov (JFNK) Solver

An optimized solver for nonlinear PDEs using the **Jacobian-Free Newton-Krylov (JFNK)** method. The Jacobian-vector products required at each inner linear iteration are computed via **Automatic Differentiation (AD)** from JAX and **Finite Differences (FD)**. The solver is applied to various nonlinear PDE problems.

