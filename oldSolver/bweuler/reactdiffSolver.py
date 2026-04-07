import glob                      # File pattern matching "*.txt"
import io                        # Input/output
import os                        # Operating system utilities
import time                      # Timing and performance measurement
import argparse                  # Command-line argument parsing
import matplotlib.pyplot as plt  # Plotting
import numpy as np               # Numerical
from numpy.linalg import norm    # Compute norm
from PIL import Image            # Image loading and processing (.gif)

import jax                                   # JAX main library (autodiff + GPU/TPU computing)
import jax.numpy as jnp                      # JAX version of NumPy
from functools import partial as ft_partial  # For partially applied functions "g = partial(f, a=1, b=2)"

# --- CPU Sparse Solvers ---
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator   # Matrix-free linear operator
from scipy.sparse.linalg import cg as cg_scipy                          # CG from SciPy
from scipy.sparse.linalg import gmres as gmres_scipy                    # GMRES from SciPy
from scipy.sparse.linalg import bicgstab as bicgstab_scipy              # BiCG-Stab from SciPy

# --- GPU Sparse Solvers (Conditional Import) ---
try:
    import cupy as cp                                # NumPy-like GPU array library
    import cupyx.scipy.sparse.linalg as cupy_spla    # GPU sparse linear algebra (SciPy-like)
    from jax import dlpack as jax_dlpack             # Zero-copy data transfer between frameworks
    from bicgstabCuPy import bicgstab                # Custom CuPy-ish GPU BiCGSTAB solver
    HAS_GPU_LIBS = True
except ImportError:
    HAS_GPU_LIBS = False



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
Inner linear solver options: CG, GMRES, BiCGSTAB.

PDE:
  d/dt u = D * nabla^2 u - u^3        (scalar field u on a 2D domain)

The cubic sink f(u) = u^3 satisfies f'(u) = 3u^2 >= 0 everywhere, which guarantees
that the backward Euler Jacobian:

  J = I - dt * D * L + dt * 3 * (u^k)^2 * I

is Symmetric Positive Definite (SPD) for all dt > 0 and all Newton iterates u^k.
This is the minimal structural requirement for CG to be a valid linear solver, making
this PDE the canonical benchmark for comparing CG vs GMRES in a JFNK context.

Hardware targets:
  --device cpu : Runs pure JAX on CPU and uses scipy.sparse.linalg solvers.
  --device gpu : Runs JAX on GPU, passes zero-copy DLPack to CuPy solvers.
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
def next_gif_path(base: str = 'reactdiff') -> str:
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
        x0, y0  = float(jnp.mean(X)), float(jnp.mean(Y))
        sigma   = 0.15 * float(jnp.max(X) - jnp.min(X))
        u = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2))
    elif sim_type == 'multi_gaussian':
        Lx = float(jnp.max(X) - jnp.min(X))
        sigma = 0.08 * Lx
        centers = [( 0.25,  0.25), (-0.25,  0.25),
                   ( 0.25, -0.25), (-0.25, -0.25)]
        u = jnp.zeros_like(X)
        for cx, cy in centers:
            u = u + jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))
    elif sim_type == 'sinusoidal':
        kx, ky = 2.0, 2.0
        u = jnp.sin(kx * jnp.pi * X) * jnp.sin(ky * jnp.pi * Y)
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
    return u_k - u_old - dt * (D * lap_u_k - u_k**3)

def flattenJnp(state_vec, Nx, Ny):
    return state_vec.reshape((Nx, Ny))


# ------- Diagnostics -------
def l2_norm(u, dx, dy):
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

# FORMULA 1: Strict Linearity for Conjugate Gradient (CG)
def JacobianActionFD_CG(u_k, F_k, Nx, Ny, perturb, dt, D, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
    du = perturb.reshape((Nx, Ny))

    # @@ CG requires strictly linear matvec operations.
    # @@ We remove the dependence on perturb_norm to ensure A-conjugacy is preserved.
    mach_eps = jnp.finfo(u_k.dtype).eps
    b        = jnp.sqrt(mach_eps)

    state_norm = jnp.linalg.norm(u_k)
    eps = b * jnp.maximum(1.0, state_norm)
    eps = jnp.clip(eps, b, jnp.sqrt(b))
    eps = jnp.where(jnp.isfinite(eps), eps, b)

    u_pert   = u_k + eps * du
    lap_pert = laplacian(u_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_pert   = constructF(u_pert, u_k, lap_pert, dt, D)

    Jv = (F_pert.ravel() - F_k.ravel()) / eps
    return Jv

# FORMULA 2: General Stability for GMRES / BiCGSTAB
# def JacobianActionFD_General(u_k, F_k, Nx, Ny, perturb, dt, D, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
def JacobianActionFD_General(u_k, u_old, F_k, Nx, Ny, perturb, dt, D, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
    du = perturb.reshape((Nx, Ny))

    # @@ Standard Brown/Saad directional derivative formula.
    # @@ Optimal for floating-point stability on general/non-symmetric matrices.
    mach_eps = jnp.finfo(u_k.dtype).eps
    b        = jnp.sqrt(mach_eps)

    state_norm   = jnp.linalg.norm(u_k)
    perturb_norm = jnp.linalg.norm(perturb)
    safe_perturb_norm = jnp.where(perturb_norm > 0.0, perturb_norm, 1.0)
    
    eps = b * jnp.maximum(1.0, state_norm) / safe_perturb_norm
    eps_max = jnp.sqrt(b)
    eps = jnp.clip(eps, b, eps_max)
    eps = jnp.where(jnp.isfinite(eps), eps, b)

    u_pert   = u_k + eps * du
    lap_pert = laplacian(u_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    # F_pert   = constructF(u_pert, u_k, lap_pert, dt, D)
    F_pert   = constructF(u_pert, u_old, lap_pert, dt, D)

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


class KrylovCounter:
    """A simple callback to count Krylov iterations for SciPy and CuPy."""
    def __init__(self):
        self.niter = 0
    def __call__(self, *args, **kwargs):
        self.niter += 1


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

def save_gif(frames, path='reactdiff_001.gif', fps=6):
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
def runSimulation(device, PRECISION, BC_X, BC_Y, SIMULATION_IC, verbose, useAD, maxBackTrackingIter,
                  D, steps, Nx, Ny, Courant, KrylovSolver,
                  KrylovTol, KrylovIter, NewtonNonlinTol, NewtonIter,
                  plot_steps, gif_fps, displayPlot, figFolder, save_steps, dataFolder):

    t_wall_start = time.perf_counter()

    dtype = configure_precision(PRECISION)
    print(f"Hardware Target: {device.upper()}")
    print(f"Precision: {PRECISION}  (dtype={dtype})")

    os.makedirs(figFolder, exist_ok=True)
    gif_path = next_gif_path(f'{figFolder}/reactdiff')
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

    # ---- DYNAMICALLY SELECT FD FUNCTION BASED ON SOLVER ----
    if KrylovSolver.lower() == 'cg':
        fd_function = ft_partial(JacobianActionFD_CG, bc_x=BC_X, bc_y=BC_Y)
        print("FD Mode: Strict Linearity (CG optimal)")
    else:
        fd_function = ft_partial(JacobianActionFD_General, bc_x=BC_X, bc_y=BC_Y)
        print(f"FD Mode: Brown/Saad Formulation ({KrylovSolver.upper()} optimal)")

    # ---- JIT compiled components ----
    laplacian_jit  = jax.jit(ft_partial(laplacian,   bc_x=BC_X, bc_y=BC_Y))
    constructF_jit = jax.jit(constructF)
    JacobianActionFD_jit = jax.jit(fd_function, static_argnums=(2, 3)) # Nx, Ny

    print(f"BC configuration: x={BC_X}  y={BC_Y}")
    print(f"Grid: {Nx}x{Ny}   dx={dx:.4f}  dy={dy:.4f}")
    print(f"Diffusion coeff D={D}  Courant={Courant}")

    fig, img, ax, bc_label, vmax = init_plot(u0, xi, xf, yi, yf, BC_X, BC_Y, SIMULATION_IC)
    gif_frames = [capture_frame(fig)]

    norm_history            = [l2_norm(u, dx, dy)]
    dt_history              = []
    newton_iters_per_step   = []
    krylov_iters_per_newton = []
    time_per_newton_iter    = []
    time_per_time_step      = []
    step_success_history    = []
    final_residual_history  = []

    # ----------- TIME LOOP -----------
    print("Starting Time Loop")
    for step in range(steps):
        print(f"\nStep {step+1}/{steps}")
        step_start_time = time.perf_counter()

        dt = calc_dt(dx, dy, D, Courant)
        dt_history.append(float(dt))

        u_old = u
        u_k   = u

        # --- track convergence for Perf Analysis ---
        step_converged = False
        final_res = 0.0

        # -------------------- Newton loop ------------------------------
        for k in range(NewtonIter):
            newton_start_time = time.perf_counter()

            lap_u_k = laplacian_jit(u_k, dx, dy)
            F_k     = constructF_jit(u_k, u_old, lap_u_k, dt, D)
            F_vec   = F_k.ravel()

            res_norm = float(jnp.linalg.norm(F_vec))
            final_res = res_norm
            print(f"  Newton iter {k}: ||F|| = {res_norm:.3e}")

            if res_norm <= NewtonNonlinTol:
                step_converged = True
                newton_end_time = time.perf_counter()
                time_per_newton_iter.append(newton_end_time - newton_start_time)
                newton_iters_per_step.append(k + 1)
                break

            # --- JAX matvec closures capturing the current Newton state ---
            if useAD:
                def A_matvec_jax(vec):
                    return JacobianActionAD_jit(u_k, u_old, vec, dt, D, dx, dy, BC_X, BC_Y, Nx, Ny)
            else:
                def A_matvec_jax(vec):
                    # return JacobianActionFD_jit(u_k, F_k, Nx, Ny, vec, dt, D, dx, dy)
                    return JacobianActionFD_jit(u_k, u_old, F_k, Nx, Ny, vec, dt, D, dx, dy)

            krylov_counter = KrylovCounter()

            # --- HARDWARE BRANCHING ---
            if device == 'cpu':
                def A_matvec_scipy(vec_np):
                    vec_jax = jnp.array(vec_np, dtype=dtype)
                    res_jax = A_matvec_jax(vec_jax)
                    return np.asarray(res_jax).copy()

                JLinearOp = ScipyLinearOperator(
                    (Nx*Ny, Nx*Ny),
                    matvec=A_matvec_scipy,
                    dtype=np.float32 if PRECISION == 'float32' else np.float64
                )

                if KrylovSolver.lower() == 'cg':
                    delta_state_np, info = cg_scipy(
                        JLinearOp, -np.asarray(F_vec).copy(),
                        rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter
                    )
                elif KrylovSolver.lower() == 'gmres':
                    delta_state_np, info = gmres_scipy(
                        JLinearOp, -np.asarray(F_vec).copy(),
                        rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter, callback_type='pr_norm'
                    )
                elif KrylovSolver.lower() == 'bicgstab':
                    delta_state_np, info = bicgstab_scipy(
                        JLinearOp, -np.asarray(F_vec).copy(), 
                        rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter
                    )
                else:
                    raise ValueError("CPU KrylovSolver must be 'cg', 'gmres', or 'bicgstab'")
                    
                delta_state = jnp.array(delta_state_np, dtype=dtype)
                
                if verbose:
                    status = "OK" if info == 0 else f"FAILED (info={info})"
                    print(f"  SciPy {KrylovSolver.upper()} [{status}] iters={krylov_counter.niter}")

            elif device == 'gpu':
                # Wrapper to bridge CuPy and JAX — zero-copy via DLPack
                def A_matvec_cp(vec_cp):
                    vec_jax = jax_dlpack.from_dlpack(vec_cp)
                    res_jax = A_matvec_jax(vec_jax)
                    return cp.from_dlpack(res_jax)

                # Define the Linear Operator for CuPy. Shape is (Nx*Ny, Nx*Ny).
                JLinearOp = cupy_spla.LinearOperator(
                    (Nx*Ny, Nx*Ny),
                    matvec=A_matvec_cp,
                    dtype=np.float32 if PRECISION == 'float32' else np.float64
                )

                # Convert the RHS (-F_k) to CuPy directly — zero-copy
                F_vec_cp = cp.from_dlpack(-F_vec)

                # Execute chosen Krylov solver natively on the GPU via CuPy
                if KrylovSolver.lower() == 'cg':
                    delta_state_cp, info = cupy_spla.cg(
                        JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter
                    )
                elif KrylovSolver.lower() == 'gmres':
                    delta_state_cp, info = cupy_spla.gmres(
                        JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter
                    )
                elif KrylovSolver.lower() == 'bicgstab':
                    delta_state_cp, info = bicgstab(
                        JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter
                    )
                elif KrylovSolver.lower() == 'cgs':
                    delta_state_cp, info = cupy_spla.cgs(
                        JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter
                    )
                else:
                    raise ValueError("GPU KrylovSolver must be 'cg', 'gmres', 'bicgstab', or 'cgs'")
                
                # Convert the resulting perturbation vector back to JAX directly
                delta_state = jax_dlpack.from_dlpack(delta_state_cp)

                if verbose:
                    status = "OK" if info == 0 else f"FAILED (info={info})"
                    print(f"  CuPy {KrylovSolver.upper()} [{status}] iters={krylov_counter.niter}")

            krylov_iters_per_newton.append(krylov_counter.niter)

            # ---- Backtracking line search ----
            # Flatten natively on the hardware array — no host copies
            delta_u = flattenJnp(delta_state, Nx, Ny)
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

        # --- store Newton Success and residual ---
        step_success_history.append(1 if step_converged else 0)
        final_residual_history.append(final_res)

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
    arr_krylov_iters = np.array(krylov_iters_per_newton)
    arr_time_newton  = np.array(time_per_newton_iter)
    arr_time_step    = np.array(time_per_time_step)

    arr_success = np.array(step_success_history)
    arr_final_res = np.array(final_residual_history)
    total_successes = np.sum(arr_success)
    total_failures = len(arr_success) - total_successes

    lin_type = "Automatic Differentiation" if useAD else "Finite Difference"

    summary_text = f"""
{"="*50}
 SOLVER PERFORMANCE SUMMARY
{"="*50}
--- Simulation Options 
  Hardware      : {device.upper()}
  Linearization : {lin_type}
  Precision     : {PRECISION}
  Outer loop    : time
  Outer steps   : {steps}
  BC on x       : {BC_X}
  BC on y       : {BC_Y}
  Simulation    : {SIMULATION_IC}
  Grid          : ({Nx}, {Ny})
  Krylov solver : {KrylovSolver.upper()}
  Newton tol    : {NewtonNonlinTol}
  Krylov tol    : {KrylovTol}
  Newton MaxIt  : {NewtonIter}
  Krylov MaxIt  : {KrylovIter}
  Max BT iters  : {maxBackTrackingIter}

--- Convergence Robustness
  Total Successes : {total_successes}
  Total Failures  : {total_failures}
  Win Rate        : {(total_successes/steps)*100:.2f}%

--- Newton Iters per Outer Step
  Average : {np.mean(arr_newton_iters):.2f}
  Std Dev : {np.std(arr_newton_iters):.2f} ({np.std(arr_newton_iters) / np.mean(arr_newton_iters):.4f}%)
  Max     : {np.max(arr_newton_iters)}
  Min     : {np.min(arr_newton_iters)}

--- Krylov Iters per Newton Step
  Average : {np.mean(arr_krylov_iters):.2f}
  Std Dev : {np.std(arr_krylov_iters):.2f} ({np.std(arr_krylov_iters) / np.mean(arr_krylov_iters):.4f}%)
  Max     : {np.max(arr_krylov_iters)}
  Min     : {np.min(arr_krylov_iters)}

--- Time per Newton Iter, s
  Average : {np.mean(arr_time_newton):.4f}
  Std Dev : {np.std(arr_time_newton):.4f} ({np.std(arr_time_newton) / np.mean(arr_time_newton):.4f}%)
  Max     : {np.max(arr_time_newton):.4f}
  Min     : {np.min(arr_time_newton):.4f}

--- Time per Outer Step, s
  Average : {np.mean(arr_time_step):.4f}
  Std Dev : {np.std(arr_time_step):.4f} ({np.std(arr_time_step) / np.mean(arr_time_step):.4f}%)
  Max     : {np.max(arr_time_step):.4f}
  Min     : {np.min(arr_time_step):.4f}

--- Overall Time, s
  Total Solver Time       : {np.sum(arr_time_step):.4f}
  Total Wall Time         : {(t_wall_end - t_wall_start):.4f}
{"="*50}

DATA ARRAYS FOR CSV PARSING:
ARRAY_SUCCESS_FLAGS: {arr_success.tolist()}
ARRAY_FINAL_RESIDUALS: {arr_final_res.tolist()}
ARRAY_NEWTON_ITERS: {arr_newton_iters.tolist()}
ARRAY_STEP_TIMES: {arr_time_step.tolist()}
"""
    # print(summary_text)

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

    # ---- Command Line Argument Parsing ---- #
    parser = argparse.ArgumentParser(description="JFNK Reaction-Diffusion Solver")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help="Target hardware: 'cpu' or 'gpu' (default: cpu)")
    args = parser.parse_args()

    # Hardware enforcement
    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')
    elif args.device == 'gpu' and not HAS_GPU_LIBS:
        raise ImportError("You requested '--device gpu', but CuPy or jax.dlpack could not be loaded.")

    # ---- Simulation Configuration ---- #
    SIMULATION_IC   = 'multi_gaussian'   # 'gaussian' | 'multi_gaussian' | 'sinusoidal'
    PRECISION       = 'float32'
    BC_X            = DIRICHLET
    BC_Y            = DIRICHLET

    # ---- Physical Parameters ---- #
    D               = 0.01
    steps           = 5000
    Nx, Ny          = 256, 256
    Courant         = 0.7           # keep <= 1 for accurate time integration

    # ---- Solver ---- #
    KrylovSolver        = 'cg'       # 'cg', 'gmres', 'bicgstab', or 'cgs'
    useAD               = True      # True -> AD JVP;  False -> FD JVP
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
    plot_steps      = 100
    save_steps      = int(1e7)
    gif_fps         = 15
    displayPlot     = True
    figFolder       = "output/reactdiff"
    dataFolder      = f"data/reactdiff_{SIMULATION_IC}_{args.device.upper()}_{Nx}_{Ny}_{'AD' if useAD else 'FD'}_{PRECISION}"

    runSimulation(
        device=args.device,
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
        KrylovSolver=KrylovSolver,
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