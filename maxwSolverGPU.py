import os
import time
from functools import partial as ft_partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_spla
from jax import dlpack as jax_dlpack

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 12
})

# ============================================================
#  Differential operators  (JIT compiled)
# ============================================================
@jax.jit
def Dxx_op(f, dx):
    d2 = jnp.zeros_like(f)
    return d2.at[1:-1, :].set((f[2:, :] + f[:-2, :] - 2*f[1:-1, :]) / dx**2)

@jax.jit
def Dyy_op(f, dy):
    d2 = jnp.zeros_like(f)
    return d2.at[:, 1:-1].set((f[:, 2:] + f[:, :-2] - 2*f[:, 1:-1]) / dy**2)

@jax.jit
def Dxy_op(f, dx, dy):
    dxy = jnp.zeros_like(f)
    return dxy.at[1:-1, 1:-1].set(
        (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4*dx*dy)
    )

# ============================================================
#  Nonlinear permittivity (With damping and NaN fix)
# ============================================================
@jax.jit
def eps_func(Ex, Ey, eps0):
    # Add offset to prevent NaN gradients at zero
    offset = jnp.finfo(Ex.real.dtype).eps
    # Add a tiny imaginary loss to help with Helmholtz resonance
    loss_factor = 1.0 - 0.05j 
    return eps0 * loss_factor * (1.0 + jnp.sqrt(jnp.abs(Ex)**2 + jnp.abs(Ey)**2 + offset))

# ============================================================
#  Source distributions
# ============================================================
def make_source(X, Y, source_type, dtype):
    Nx, Ny = X.shape
    Jx = jnp.zeros((Nx, Ny), dtype=dtype)
    Jy = jnp.zeros((Nx, Ny), dtype=dtype)

    if source_type == 'gaussian_center':
        x0 = float(X.mean());  y0 = float(Y.mean())
        sigma = 0.05 * float(X.max() - X.min())
        G = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2))
        Jx = G.astype(dtype)
    elif source_type == 'point':
        Jx = Jx.at[Nx//2, Ny//2].set(1.0)
    elif source_type == 'line_x':
        Jx = Jx.at[:, Ny//2].set(1.0)
    elif source_type == 'dipole':
        Jx = Jx.at[Nx//2,   Ny//3].set( 1.0)
        Jx = Jx.at[Nx//2, 2*Ny//3].set(-1.0)
    else:
        raise ValueError(f"Unknown source type '{source_type}'.")

    return Jx, Jy

# ============================================================
#  TE residual F(u) = 0 (Real-Valued Wrapper)
# ============================================================
@ft_partial(jax.jit, static_argnums=(8, 9))
def residual_TE_real(state_real, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    N  = Nx * Ny
    # Reconstruct complex fields from the 4N real vector
    Ex = state_real[:N].reshape(Nx, Ny) + 1j * state_real[N:2*N].reshape(Nx, Ny)
    Ey = state_real[2*N:3*N].reshape(Nx, Ny) + 1j * state_real[3*N:4*N].reshape(Nx, Ny)

    eps = eps_func(Ex, Ey, eps0)

    Fx = (  Dxy_op(Ey, dx, dy)
           - Dyy_op(Ex, dy)
           - omega**2 * mu0 * eps * Ex
           - 1j * omega * mu0 * Jx )

    Fy = (  Dxy_op(Ex, dx, dy)
           - Dxx_op(Ey, dx)
           - omega**2 * mu0 * eps * Ey
           - 1j * omega * mu0 * Jy )

    Fx = Fx.at[:, 0 ].set(Ex[:, 0 ])   
    Fx = Fx.at[:, -1].set(Ex[:, -1])   
    Fy = Fy.at[0,  :].set(Ey[0,  :])   
    Fy = Fy.at[-1, :].set(Ey[-1, :])   

    # Return purely real vector of size 4N
    return jnp.concatenate([Fx.real.ravel(), Fx.imag.ravel(), 
                            Fy.real.ravel(), Fy.imag.ravel()])

# ============================================================
#  Jacobian-vector products (JIT compiled)
# ============================================================
@ft_partial(jax.jit, static_argnums=(9, 10))
def JacobianActionAD_jit(state, perturb, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    def res_fn(s):
        return residual_TE_real(s, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
    _, jvp_result = jax.jvp(res_fn, (state,), (perturb,))
    return jvp_result

@ft_partial(jax.jit, static_argnums=(10, 11))
def JacobianActionFD_jit(state, F0, perturb, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    mach_eps = jnp.finfo(state.dtype).eps
    b = jnp.sqrt(mach_eps)
    state_norm = jnp.linalg.norm(state)
    
    eps = b * jnp.maximum(1.0, state_norm)
    eps_max = jnp.sqrt(b)
    eps = jnp.clip(eps, b, eps_max)
    eps = jnp.where(jnp.isfinite(eps), eps, b)

    F_pert = residual_TE_real(state + eps * perturb, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
    return (F_pert - F0) / eps

# ============================================================
#  Main simulation
# ============================================================
def runSimulation(PRECISION, SIMULATION_J, useAD, verbose,
                  mu0, eps0, omega_start, omega_stop, omega_steps,
                  Nx, Ny, KrylovTol, KrylovIter, NewtonTol, NewtonIter,
                  maxBackTrackingIter, figFolder):

    t_wall_start = time.perf_counter()
    
    # Configure Datatypes
    if PRECISION == 'float64':
        jax.config.update("jax_enable_x64", True)
        real_dtype = jnp.float64
        complex_dtype = jnp.complex128
    elif PRECISION == 'float32':
        real_dtype = jnp.float32
        complex_dtype = jnp.complex64
    else:
        raise ValueError(f"Unknown precision '{PRECISION}'.")

    os.makedirs(figFolder, exist_ok=True)
    
    print(f"Precision: {PRECISION} | Linearization: {'AD' if useAD else 'FD'}")

    Lx, Ly = 1.0, 1.0
    x  = jnp.linspace(0, Lx, Nx)
    y  = jnp.linspace(0, Ly, Ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Source is complex
    Jx, Jy = make_source(X, Y, SIMULATION_J, complex_dtype)
    omegas = jnp.linspace(omega_start, omega_stop, omega_steps)
    response = [] 

    # State is purely REAL and 4x the grid size
    state = jnp.zeros(4 * Nx * Ny, dtype=real_dtype) 

    newton_iters_per_step = []
    time_per_newton_iter = []
    time_per_omega_step = []

    print("Starting Frequency Sweep")
    for i, omega in enumerate(omegas):
        omega_val = float(omega)
        print(f"\nStep {i+1}/{omega_steps} | ω = {omega_val:.4e} rad/s")
        omega_start_time = time.perf_counter()

        # --- Newton Loop ---
        for k in range(NewtonIter):
            newton_start_time = time.perf_counter()

            F = residual_TE_real(state, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
            res_norm = float(jnp.linalg.norm(F))

            print(f"  Newton iter {k}: ||F|| = {res_norm:.3e}")

            if res_norm < NewtonTol:
                newton_end_time = time.perf_counter()
                time_per_newton_iter.append(newton_end_time - newton_start_time)
                newton_iters_per_step.append(k + 1)
                break

            # 1. Define JAX matvec
            if useAD:
                def A_matvec_jax(vec):
                    return JacobianActionAD_jit(state, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
            else:
                def A_matvec_jax(vec):
                    return JacobianActionFD_jit(state, F, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)

            # 2. Bridge CuPy and JAX 
            def A_matvec_cp(vec_cp):
                vec_jax = jax_dlpack.from_dlpack(vec_cp)
                res_jax = A_matvec_jax(vec_jax)
                return cp.from_dlpack(res_jax)

            JLinearOp = cupy_spla.LinearOperator(
                (4*Nx*Ny, 4*Nx*Ny),
                matvec=A_matvec_cp,
                dtype=np.float64 if PRECISION == 'float64' else np.float32
            )

            # 3. Setup Jacobi Preconditioner
            N_pts = Nx * Ny
            Ex_c = state[:N_pts].reshape(Nx, Ny) + 1j * state[N_pts:2*N_pts].reshape(Nx, Ny)
            Ey_c = state[2*N_pts:3*N_pts].reshape(Nx, Ny) + 1j * state[3*N_pts:4*N_pts].reshape(Nx, Ny)
            eps_current = eps_func(Ex_c, Ey_c, eps0)

            # Take absolute magnitude of the diagonal for stability to make M purely real
            diag_Ex_real = jnp.abs(2.0 / dy**2 - omega_val**2 * mu0 * eps_current)
            diag_Ex_real = diag_Ex_real.at[:, 0].set(1.0)
            diag_Ex_real = diag_Ex_real.at[:, -1].set(1.0)

            diag_Ey_real = jnp.abs(2.0 / dx**2 - omega_val**2 * mu0 * eps_current)
            diag_Ey_real = diag_Ey_real.at[0, :].set(1.0)
            diag_Ey_real = diag_Ey_real.at[-1, :].set(1.0)

            diag_full = jnp.concatenate([
                diag_Ex_real.ravel(), diag_Ex_real.ravel(), 
                diag_Ey_real.ravel(), diag_Ey_real.ravel()
            ])
            # Prevent div-by-zero
            diag_full = jnp.maximum(diag_full, 1e-12)

            inv_diag_cp = 1.0 / cp.from_dlpack(diag_full)
            def M_matvec_cp(vec_cp):
                return vec_cp * inv_diag_cp

            M_LinearOp = cupy_spla.LinearOperator(
                (4*N_pts, 4*N_pts),
                matvec=M_matvec_cp,
                dtype=np.float64 if PRECISION == 'float64' else np.float32
            )

            # 4. Solve natively on GPU
            F_vec_cp = cp.from_dlpack(-F)
            delta_cp, info = cupy_spla.gmres(
                JLinearOp, 
                F_vec_cp, 
                M=M_LinearOp,
                rtol=KrylovTol, 
                restart=100,
                maxiter=10
            )
            delta = jax_dlpack.from_dlpack(delta_cp)

            if verbose and info > 0:
                print(f"  CuPy GMRES failed to converge: info={info}")

            # --- Backtracking Line Search ---
            alpha = 1.0
            for _ in range(maxBackTrackingIter):
                state_try = state + alpha * delta
                F_try = residual_TE_real(state_try, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
                F_try_norm = float(jnp.linalg.norm(F_try))
                
                if F_try_norm < res_norm:
                    if verbose:
                        print(f"    Line search accepted alpha={alpha:.3f}, new ||F||={F_try_norm:.3e}")
                    state = state_try
                    break
                alpha *= 0.5
            else:
                if verbose:
                    print("    Line search failed to find descent. Forcing update.")
                state = state_try

            newton_end_time = time.perf_counter()
            time_per_newton_iter.append(newton_end_time - newton_start_time)
            
        else:
             newton_iters_per_step.append(NewtonIter)

        omega_end_time = time.perf_counter()
        time_per_omega_step.append(omega_end_time - omega_start_time)

        # Record Max E-Field magnitude from 4N real array
        N_pts  = Nx * Ny
        Ex_real = state[:N_pts].reshape(Nx, Ny)
        Ex_imag = state[N_pts:2*N_pts].reshape(Nx, Ny)
        Ey_real = state[2*N_pts:3*N_pts].reshape(Nx, Ny)
        Ey_imag = state[3*N_pts:4*N_pts].reshape(Nx, Ny)
        
        E_mag = jnp.sqrt(Ex_real**2 + Ex_imag**2 + Ey_real**2 + Ey_imag**2)
        response.append(float(jnp.max(E_mag)))

    t_wall_end = time.perf_counter()
    print(f"\nDone! Frequency sweep completed in {(t_wall_end - t_wall_start):.2f} s")

    # ---- Solver Summary Text Block ----
    arr_newton_iters = np.array(newton_iters_per_step)
    arr_time_newton = np.array(time_per_newton_iter)
    arr_time_omega = np.array(time_per_omega_step)
    lin_type = "Automatic Differentiation" if useAD else "Finite Difference"

    summary_text = f"""
{"="*50}
 SOLVER PERFORMANCE SUMMARY
{"="*50}
--- Simulation Options 
  Linearization : {lin_type}
  Precision     : {PRECISION}
  Source Type   : {SIMULATION_J}
  Grid          : ({Nx}, {Ny})
  Frequency pts : {omega_steps}
  Newton tol    : {NewtonTol}
  Krylov tol    : {KrylovTol}

--- Newton Iterations per Omega Step
  Average : {np.mean(arr_newton_iters):.2f}
  Max     : {np.max(arr_newton_iters)}
  Min     : {np.min(arr_newton_iters)}

--- Time per Newton Iteration, s
  Average : {np.mean(arr_time_newton):.4f}
  Max     : {np.max(arr_time_newton):.4f}

--- Time per Outer Omega Step, s
  Average : {np.mean(arr_time_omega):.4f}

--- Overall Time, s
  Total Wall Time : {(t_wall_end - t_wall_start):.4f}
{"="*50}
"""
    print(summary_text)

    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(np.array(omegas), response, color='#4361EE', linewidth=2)
    ax.set_xlabel(r'$\omega$ (rad/s)')
    ax.set_ylabel(r'$\max\,|\mathbf{E}|$')
    ax.set_title(r'TE Maxwell — Frequency Response  ($\varepsilon = \varepsilon_0(1+|\mathbf{E}|)$)')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(figFolder, 'maxwell_TE_response.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(np.array(E_mag).T, origin='lower', cmap='magma', extent=[0, Lx, 0, Ly])
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_title(rf'$|\mathbf{{E}}|$ at $\omega={float(omegas[-1]):.2e}$ rad/s')
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig2.savefig(os.path.join(figFolder, 'maxwell_TE_field.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)


if __name__ == "__main__":
    SIMULATION_J    = 'gaussian_center'
    PRECISION       = 'float64'

    mu0             = 1.25663706127e-6      
    eps0            = 8.8541878188e-12      

    omega_start     = 1e8
    omega_stop      = 1e10
    omega_steps     = 3

    Nx, Ny          = 64, 64    

    useAD           = True      
    verbose         = True      
    maxBackTrackingIter = 20

    if PRECISION == 'float64':
        KrylovTol, KrylovIter   = 1e-6, 100
        NewtonTol, NewtonIter   = 1e-6,  50
    else:
        KrylovTol, KrylovIter   = 1e-3, 100
        NewtonTol, NewtonIter   = 1e-3,  50

    figFolder = "output"

    runSimulation(
        PRECISION       = PRECISION,
        SIMULATION_J    = SIMULATION_J,
        useAD           = useAD,
        verbose         = verbose,
        mu0             = mu0,
        eps0            = eps0,
        omega_start     = omega_start,
        omega_stop      = omega_stop,
        omega_steps     = omega_steps,
        Nx              = Nx,
        Ny              = Ny,
        KrylovTol       = KrylovTol,
        KrylovIter      = KrylovIter,
        NewtonTol       = NewtonTol,
        NewtonIter      = NewtonIter,
        maxBackTrackingIter = maxBackTrackingIter,
        figFolder       = figFolder,
    )