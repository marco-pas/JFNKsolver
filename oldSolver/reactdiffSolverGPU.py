import glob
import io
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial as ft_partial
from PIL import Image

import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_spla
from jax import dlpack as jax_dlpack

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 12
})

'''
NONLINEAR REACTION-DIFFUSION SOLVER using Jacobian-Free Newton-Krylov (JFNK)
Both Finite Difference (FD) and Automatic Differentiation (AD) Jacobian-vector products
Inner linear solver: Conjugate Gradient (CG) — valid because the Jacobian is SPD.

PDE:
  d/dt u = D * nabla^2 u - u^3        (scalar field u on a 2D domain)

The cubic sink f(u) = u^3 satisfies f'(u) = 3u^2 >= 0 everywhere, which guarantees
that the backward Euler Jacobian:

  J = I - dt * D * L + dt * 3 * (u^k)^2 * I

is Symmetric Positive Definite (SPD) for all dt > 0 and all Newton iterates u^k.
This is the minimal structural requirement for CG to be a valid linear solver, making
this PDE the canonical benchmark for comparing CG vs GMRES in a JFNK context.

Key contrast with Burgers / Maxwell:
  - Burgers: advection makes J non-symmetric  -> GMRES required
  - Maxwell: complex, curl-curl, Kerr loss    -> GMRES required
  - This PDE: monotone cubic sink, real, symmetric diffusion -> CG valid, CG optimal

Backward Euler residual at Newton iteration k:
  F(u^k) = u^k - u^n - dt * [ D * L(u^k) - (u^k)^3 ] = 0

Boundary condition flags (set in __main__):
  BC_X : 'dirichlet' | 'periodic'
  BC_Y : 'dirichlet' | 'periodic'

Precision flag (set in __main__):
  PRECISION : 'float32' | 'float64'
  For float64, JAX x64 mode is enabled automatically.

GPU backend:
  All Jacobian-vector products are computed in JAX (AD or FD) and passed to CuPy CG
  via zero-copy DLPack. The CG iteration runs entirely on the GPU — no host transfers
  inside the Newton loop.
'''

DIRICHLET = 'dirichlet'
PERIODIC  = 'periodic'


# -------- precision helper --------
def configure_precision(precision: str):
    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        return jnp.float64
    elif precision == 'float32':
        return jnp.float32
    else:
        raise ValueError(f"Unknown precision '{precision}'. Use 'float32' or 'float64'.")


# -------- increment GIF path --------
def next_gif_path(base: str = 'rcd_evolution') -> str:
    existing = glob.glob(f'{base}_???.gif')
    used = set()
    for f in existing:
        stem = f[len(base) + 1:-4]
        if stem.isdigit():
            used.add(int(stem))
    for n in range(1, 1000):
        if n not in used:
            return f'{base}_{n:03d}.gif'
    raise RuntimeError("All GIF slots _001-_999 are taken. Clean up old files.")


# ---------- ICs ----------
def get_initial_conditions(X, Y, dtype, sim_type):
    if sim_type == 'gaussian':
        # Single Gaussian pulse centred at the domain origin.
        # The cubic sink immediately attacks the peak while diffusion spreads it.
        # Good for watching the nonlinear vs linear regime competition.
        x0, y0  = float(jnp.mean(X)), float(jnp.mean(Y))
        sigma   = 0.15 * float(jnp.max(X) - jnp.min(X))
        u = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2))
    elif sim_type == 'multi_gaussian':
        # Four Gaussian blobs at symmetric off-center positions.
        # Tests whether the solver handles multiple interacting decay fronts.
        Lx = float(jnp.max(X) - jnp.min(X))
        sigma = 0.08 * Lx
        centers = [( 0.25,  0.25), (-0.25,  0.25),
                   ( 0.25, -0.25), (-0.25, -0.25)]
        u = jnp.zeros_like(X)
        for cx, cy in centers:
            u = u + jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))
    elif sim_type == 'sinusoidal':
        # Low-frequency sine wave — smooth IC that stresses the coupling between
        # diffusion (which damps it) and the cubic sink (which attacks the peaks).
        # With periodic BCs this IC is spectrally clean: one Fourier mode each way.
        kx, ky = 2.0, 2.0
        u = jnp.sin(kx * jnp.pi * X) * jnp.sin(ky * jnp.pi * Y)
        # Shift to be positive so f'(u) = 3u^2 is nontrivial from the start
        u = jnp.abs(u)
    else:
        raise ValueError(f"Unknown IC type: {sim_type}. Use 'gaussian', 'multi_gaussian', or 'sinusoidal'.")
    return u.astype(dtype)


def apply_BC(field, bc_x, bc_y):
    if bc_x == DIRICHLET:
        field = field.at[0,  :].set(0.0)
        field = field.at[-1, :].set(0.0)
    if bc_y == DIRICHLET:
        field = field.at[:,  0].set(0.0)
        field = field.at[:, -1].set(0.0)
    return field


# -------- PDE components --------
def calc_dt(dx, dy, D, C=0.7):
    # Diffusion stability: dt ~ dx^2 / D.
    # Backward Euler is unconditionally stable so C >> 1 is fine for accuracy tests;
    # keep C <= 1 for temporally accurate runs.
    return C * min(dx, dy)**2 / D

def laplacian(f, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
    if bc_x == PERIODIC:
        d2x = (jnp.roll(f, -1, axis=0) + jnp.roll(f, 1, axis=0) - 2*f) / dx**2
    else:
        d2x = jnp.zeros_like(f)
        d2x = d2x.at[1:-1, :].set((f[2:, :] + f[:-2, :] - 2*f[1:-1, :]) / dx**2)

    if bc_y == PERIODIC:
        d2y = (jnp.roll(f, -1, axis=1) + jnp.roll(f, 1, axis=1) - 2*f) / dy**2
    else:
        d2y = jnp.zeros_like(f)
        d2y = d2y.at[:, 1:-1].set((f[:, 2:] + f[:, :-2] - 2*f[:, 1:-1]) / dy**2)

    lap = d2x + d2y

    if bc_x == DIRICHLET:
        lap = lap.at[0,  :].set(0.0)
        lap = lap.at[-1, :].set(0.0)
    if bc_y == DIRICHLET:
        lap = lap.at[:,  0].set(0.0)
        lap = lap.at[:, -1].set(0.0)

    return lap

def constructF(u_k, u_old, lap_u_k, dt, D):
    # Backward Euler residual for  du/dt = D*lap(u) - u^3
    # F(u^k) = u^k - u^old - dt * [ D * lap(u^k) - (u^k)^3 ] = 0
    # The cubic sink is the only source of nonlinearity.
    return u_k - u_old - dt * (D * lap_u_k - u_k**3)

def concatenateJnp(vec1, vec2=None):
    # Kept as single-field; no concatenation needed. Wrapper for ravel only.
    return vec1.ravel()

def flattenJnp(state_vec, Nx, Ny):
    return state_vec.reshape((Nx, Ny))


# ------- Diagnostics -------
def l2_norm(u, dx, dy):
    # Integrated L2 norm: ||u||_2 = sqrt( integral u^2 dx dy )
    return float(jnp.sqrt(jnp.sum(u**2) * dx * dy))

def plot_norm(norm_history, dt_history, gif_path):
    times = [0.0]
    for dt in dt_history:
        times.append(times[-1] + dt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, norm_history, color='royalblue', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\|u\|_{L^2}$')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    norm_path = gif_path.replace('.gif', '_norm.png')
    fig.savefig(norm_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Norm plot saved -> {norm_path}")


# ------------ Jacobian-vector products (FD and AD) ---------------

def JacobianActionFD(u_k, F_k, Nx, Ny, perturb, dt, D, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
    du = perturb.reshape((Nx, Ny))

    # @@ Gateaux derivative of the residual F(u_k + eps*du) - F(u_k)) / eps.
    # @@ The Jacobian is SPD, so CG is valid as the linear solver.
    # @@ For SPD problems the FD eps choice matters less than for non-symmetric
    # @@ systems, but near-singularity of J can still amplify cancellation error.
    # @@
    # @@ FIX: eps independent of perturb norm — keeps the matvec strictly linear
    # @@ in the input vector, which CG requires to maintain A-conjugacy of its
    # @@ search directions. Any perturb-norm dependence breaks this and causes
    # @@ CG to drift from the true Krylov subspace.

    mach_eps = jnp.finfo(u_k.dtype).eps
    b        = jnp.sqrt(mach_eps)

    state_norm = jnp.linalg.norm(u_k)
    eps = b * jnp.maximum(1.0, state_norm)
    eps = jnp.clip(eps, b, jnp.sqrt(b))
    eps = jnp.where(jnp.isfinite(eps), eps, b)

    u_pert   = u_k + eps * du
    lap_pert = laplacian(u_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_pert   = constructF(u_pert, u_k, lap_pert, dt, D)   # u_old cancels in difference

    Jv = (F_pert.ravel() - F_k.ravel()) / eps
    return Jv

# Flat residual evaluator for the AD graph
def residual_flat(state, u_old, dt, D, dx, dy, bc_x, bc_y, Nx, Ny):
    u_   = state.reshape((Nx, Ny))
    lap_ = laplacian(u_, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_   = constructF(u_, u_old, lap_, dt, D)
    return F_.ravel()

# Jitted AD Jacobian-Vector Product
@ft_partial(jax.jit, static_argnums=(7, 8, 9, 10))
def JacobianActionAD_jit(u_k, u_old, perturb, dt, D, dx, dy, bc_x, bc_y, Nx, Ny):
    state_k = u_k.ravel()
    def F_flat(s):
        return residual_flat(s, u_old, dt, D, dx, dy, bc_x, bc_y, Nx, Ny)
    _, jvp_result = jax.jvp(F_flat, (state_k,), (perturb,))
    return jvp_result


# ------ Plotting ------
def init_plot(u0, xi, xf, yi, yf, bc_x, bc_y, sim_type):
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    extent  = [xi, xf, yi, yf]

    vmax = max(float(jnp.max(jnp.abs(u0))), 1e-6)

    bc_label = f'BC: x={bc_x[0].upper()}  y={bc_y[0].upper()}'

    img = ax.imshow(np.array(u0).T, cmap='inferno', extent=extent,
                    origin='lower', vmin=0.0, vmax=vmax)
    ax.set_title(f'u(x,y) | Step 0  IC: {sim_type}\n{bc_label}')
    ax.axis('off')
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return fig, img, ax, bc_label, vmax

def update_plot(img, ax, u, step, bc_label, vmax, displayPlot=True):
    img.set_data(np.array(u).T)
    ax.set_title(f'u(x,y) | Step {step}\n{bc_label}')
    if displayPlot:
        plt.pause(0.01)

def capture_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    return Image.open(buf).copy()

def save_gif(frames, path='rcd_evolution_001.gif', fps=6):
    if not frames:
        print("No frames captured — nothing to save.")
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=int(1000 / fps),
        loop=0
    )
    print(f"GIF saved -> {path}  ({len(frames)} frames @ {fps} fps)")


# Function to run all the simulation
def runSimulation(PRECISION, BC_X, BC_Y, SIMULATION_IC, verbose, useAD, maxBackTrackingIter,
                  D, steps, Nx, Ny, Courant,
                  KrylovTol, KrylovIter, NewtonNonlinTol, NewtonIter,
                  plot_steps, gif_fps, displayPlot, figFolder, save_steps, dataFolder):

    t_wall_start = time.perf_counter()

    dtype = configure_precision(PRECISION)
    print(f"Precision: {PRECISION}  (dtype={dtype})")
    print(f"Linear solver: CuPy CG  (valid because J is SPD for all Newton iterates)")

    os.makedirs(figFolder, exist_ok=True)
    gif_path = next_gif_path(f'{figFolder}/rcd_evolution')
    print(f"Output GIF: {gif_path}")

    # ---- grids ----
    xi, xf = -0.5, 0.5
    yi, yf = -0.5, 0.5

    x  = jnp.linspace(xi, xf, Nx, endpoint=(BC_X == DIRICHLET)).astype(dtype)
    y  = jnp.linspace(yi, yf, Ny, endpoint=(BC_Y == DIRICHLET)).astype(dtype)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # ---- ICs / BCs ----
    u0_raw = get_initial_conditions(X, Y, dtype, SIMULATION_IC)
    u0     = apply_BC(u0_raw, BC_X, BC_Y)
    u      = u0

    # ---- JIT compiled components ----
    laplacian_jit  = jax.jit(ft_partial(laplacian,   bc_x=BC_X, bc_y=BC_Y))
    constructF_jit = jax.jit(constructF)
    JacobianActionFD_jit = jax.jit(
        ft_partial(JacobianActionFD, bc_x=BC_X, bc_y=BC_Y),
        static_argnums=(2, 3)   # Nx, Ny
    )

    print(f"BC configuration: x={BC_X}  y={BC_Y}")
    print(f"Grid: {Nx}x{Ny}   dx={dx:.4f}  dy={dy:.4f}")
    print(f"Diffusion coeff D={D}  Courant={Courant}")

    fig, img, ax, bc_label, vmax = init_plot(u0, xi, xf, yi, yf, BC_X, BC_Y, SIMULATION_IC)
    gif_frames = [capture_frame(fig)]

    norm_history          = [l2_norm(u, dx, dy)]
    dt_history            = []
    newton_iters_per_step = []
    time_per_newton_iter  = []
    time_per_time_step    = []

    # ----------- TIME LOOP -----------
    print("Starting Time Loop")
    for step in range(steps):
        print(f"\nStep {step+1}/{steps}")
        step_start_time = time.perf_counter()

        dt = calc_dt(dx, dy, D, Courant)
        dt_history.append(float(dt))

        u_old = u
        u_k   = u

        # -------------------- Newton loop ------------------------------
        for k in range(NewtonIter):
            newton_start_time = time.perf_counter()

            lap_u_k = laplacian_jit(u_k, dx, dy)
            F_k     = constructF_jit(u_k, u_old, lap_u_k, dt, D)
            F_vec   = F_k.ravel()

            res_norm = float(jnp.linalg.norm(F_vec))
            print(f"  Newton iter {k}: ||F|| = {res_norm:.3e}")

            if res_norm <= NewtonNonlinTol:
                newton_end_time = time.perf_counter()
                time_per_newton_iter.append(newton_end_time - newton_start_time)
                newton_iters_per_step.append(k + 1)
                break

            # --- JAX matvec closures capturing the current Newton state ---
            # The Jacobian J = I - dt*D*L + dt*3*u_k^2*I is SPD for all u_k,
            # so CG is the optimal Krylov solver — no restart needed, no
            # non-symmetric handling, O(N) fixed memory per iteration.
            if useAD:
                def A_matvec_jax(vec):
                    return JacobianActionAD_jit(u_k, u_old, vec, dt, D, dx, dy, BC_X, BC_Y, Nx, Ny)
            else:
                def A_matvec_jax(vec):
                    return JacobianActionFD_jit(u_k, F_k, Nx, Ny, vec, dt, D, dx, dy)

            # Wrapper to bridge CuPy and JAX — zero-copy via DLPack
            def A_matvec_cp(vec_cp):
                # 1. CuPy -> JAX (Zero-copy)
                vec_jax = jax_dlpack.from_dlpack(vec_cp)

                # 2. Run the chosen JIT-compiled SPD matvec
                res_jax = A_matvec_jax(vec_jax)

                # 3. JAX -> CuPy (Zero-copy)
                return cp.from_dlpack(res_jax)

            # Define the Linear Operator for CuPy.
            # Shape is (Nx*Ny, Nx*Ny) — single scalar field, unlike Burgers (2*Nx*Ny).
            JLinearOp = cupy_spla.LinearOperator(
                (Nx*Ny, Nx*Ny),
                matvec=A_matvec_cp,
                dtype=np.float32 if PRECISION == 'float32' else np.float64
            )

            # Convert the RHS (-F_k) to CuPy directly — zero-copy
            F_vec_cp = cp.from_dlpack(-F_vec)

            # ------ CuPy CG — runs entirely on the GPU ------
            # CG is valid here because J is SPD. It uses short recurrences
            # (fixed O(N) memory) and minimizes the A-norm of the error,
            # which is the natural norm for SPD systems.
            # No preconditioner is used here; the diagonal dominance of
            # (I + dt*3*u_k^2*I) already provides good conditioning.
            delta_cp, info = cupy_spla.cg(
                JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter
            )

            # Convert the CG solution back to JAX — zero-copy
            delta_flat = jax_dlpack.from_dlpack(delta_cp)

            if verbose and info > 0:
                print(f"  CuPy CG failed to converge: info={info}")
            elif verbose and info == 0:
                print(f"  CuPy CG converged successfully")

            # ---- Backtracking line search ----
            # Flatten natively on the GPU array — no host copies
            delta_u = flattenJnp(delta_flat, Nx, Ny)
            alpha   = 1.0

            for ls in range(maxBackTrackingIter):
                u_try = u_k + alpha * delta_u
                u_try = apply_BC(u_try, BC_X, BC_Y)

                lap_try = laplacian_jit(u_try, dx, dy)
                F_try   = constructF_jit(u_try, u_old, lap_try, dt, D)

                F_try_norm = float(jnp.linalg.norm(F_try.ravel()))

                if F_try_norm < res_norm:
                    if verbose:
                        print(f"    Line search accepted alpha={alpha:.3f}, new ||F||={F_try_norm:.3e}")
                    u_k = u_try
                    break
                else:
                    alpha *= 0.5
            else:
                if verbose:
                    print(f"    Line search failed to find a descent step. Forcing update.")
                u_k = u_try

            newton_end_time = time.perf_counter()
            time_per_newton_iter.append(newton_end_time - newton_start_time)

        else:
            newton_iters_per_step.append(NewtonIter)

        u = u_k
        norm_history.append(l2_norm(u, dx, dy))

        step_end_time = time.perf_counter()
        time_per_time_step.append(step_end_time - step_start_time)

        if (step + 1) % plot_steps == 0:
            update_plot(img, ax, u, step + 1, bc_label, vmax, displayPlot)
            gif_frames.append(capture_frame(fig))

        # Save field snapshots for post-processing / restart
        if save_steps > 0 and (step + 1) % save_steps == 0:
            os.makedirs(dataFolder, exist_ok=True)

            u_np   = np.array(u)
            t_curr = sum(dt_history)

            u_path = f"{dataFolder}/u_step{step + 1:04d}_t{t_curr:.6f}.npy"
            np.save(u_path, u_np)

            if verbose:
                print(f"  Saved field to {u_path}")

    print("\nDone!")
    t_wall_end = time.perf_counter()

    arr_newton_iters = np.array(newton_iters_per_step)
    arr_time_newton  = np.array(time_per_newton_iter)
    arr_time_step    = np.array(time_per_time_step)

    lin_type   = "Automatic Differentiation" if useAD else "Finite Difference"
    krylov_str = "CuPy CG (SPD — exact for this problem)"

    summary_text = f"""
{"="*50}
 SOLVER PERFORMANCE SUMMARY
{"="*50}
--- Simulation Options
  Linearization : {lin_type}
  Krylov solver : {krylov_str}
  Precision     : {PRECISION}
  BC on x       : {BC_X}
  BC on y       : {BC_Y}
  Simulation IC : {SIMULATION_IC}
  Grid          : ({Nx}, {Ny})
  Diffusion D   : {D}
  Courant       : {Courant}
  Time steps    : {steps}
  Newton tol    : {NewtonNonlinTol}
  Krylov tol    : {KrylovTol}
  Max BT iters  : {maxBackTrackingIter}

--- Newton Iterations per Time Step
  Average : {np.mean(arr_newton_iters):.2f}
  Std Dev : {np.std(arr_newton_iters):.2f} ({np.std(arr_newton_iters) / np.mean(arr_newton_iters):.4f}%)
  Max     : {np.max(arr_newton_iters)}
  Min     : {np.min(arr_newton_iters)}

--- Time per Newton Iteration, s
  Average : {np.mean(arr_time_newton):.4f}
  Std Dev : {np.std(arr_time_newton):.4f} ({np.std(arr_time_newton) / np.mean(arr_time_newton):.4f}%)
  Max     : {np.max(arr_time_newton):.4f}
  Min     : {np.min(arr_time_newton):.4f}

--- Time per Outer Time Step, s
  Average : {np.mean(arr_time_step):.4f}
  Std Dev : {np.std(arr_time_step):.4f} ({np.std(arr_time_step) / np.mean(arr_time_step):.4f}%)
  Max     : {np.max(arr_time_step):.4f}
  Min     : {np.min(arr_time_step):.4f}

--- Overall Time, s
  Total Solver Time       : {np.sum(arr_time_step):.4f}
  Total Wall Time         : {(t_wall_end - t_wall_start):.4f}
  Physical Time Simulated : {np.sum(np.array(dt_history)):.4f}
{"="*50}
"""
    print(summary_text)

    txt_path = gif_path.replace('.gif', '_summary.txt')
    with open(txt_path, "w") as f:
        f.write(summary_text)

    print("Saving norm plot")
    plot_norm(norm_history, dt_history, gif_path)

    print("Saving the GIF")
    save_gif(gif_frames, path=gif_path, fps=gif_fps)

    plt.ioff()
    if displayPlot:
        plt.show()


# Launch simulation
if __name__ == "__main__":

    # ---- Simulation Configuration ---- #
    SIMULATION_IC   = 'gaussian'   # 'gaussian' | 'multi_gaussian' | 'sinusoidal'
    PRECISION       = 'float32'
    BC_X            = DIRICHLET
    BC_Y            = DIRICHLET

    # ---- Physical Parameters ---- #
    # D controls how fast diffusion spreads the solution.
    # Larger D -> faster spread, earlier equilibration.
    # The cubic sink u^3 always wins at large amplitudes; diffusion wins at small ones.
    D               = 0.01
    steps           = 500
    Nx, Ny          = 256, 256
    Courant         = 0.7           # keep <= 1 for accurate time integration

    # ---- Solver ---- #
    useAD               = False      # True -> AD JVP;  False -> FD JVP
    verbose             = False
    maxBackTrackingIter = 15

    # ---- Solver Tolerances ---- #
    if PRECISION == 'float32':
        KrylovTol       = 1e-3
        KrylovIter      = int(1e3)
        NewtonNonlinTol = 1e-3
        NewtonIter      = int(15)
    elif PRECISION == 'float64':
        KrylovTol       = 1e-6
        KrylovIter      = int(1e3)
        NewtonNonlinTol = 1e-6
        NewtonIter      = int(15)
    else:
        raise ValueError('Choose different Precision')

    # ---- Plotting + I/O ---- #
    plot_steps      = 25
    save_steps      = int(1e7)
    gif_fps         = 15
    displayPlot     = True
    figFolder       = "output/rcd"
    dataFolder      = f"data/rcd_{SIMULATION_IC}_{Nx}_{Ny}_{'AD' if useAD else 'FD'}_{PRECISION}"

    runSimulation(
        PRECISION=PRECISION,
        BC_X=BC_X,
        BC_Y=BC_Y,
        SIMULATION_IC=SIMULATION_IC,
        verbose=verbose,
        useAD=useAD,
        maxBackTrackingIter=maxBackTrackingIter,
        D=D,
        steps=steps,
        Nx=Nx,
        Ny=Ny,
        Courant=Courant,
        KrylovTol=KrylovTol,
        KrylovIter=KrylovIter,
        NewtonNonlinTol=NewtonNonlinTol,
        NewtonIter=NewtonIter,
        plot_steps=plot_steps,
        gif_fps=gif_fps,
        displayPlot=displayPlot,
        figFolder=figFolder,
        save_steps=save_steps,
        dataFolder=dataFolder
    )
