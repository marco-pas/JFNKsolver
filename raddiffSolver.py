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
SU-OLSON SOLVER using Jacobian-Free Newton-Krylov (JFNK)
Both Finite Difference (FD) and Automatic Differentiation (AD) Jacobian-vector products

Dimensionless non-equilibrium radiation diffusion system (Su & Olson 1996):

  d/dt U =  (1/3) * nabla^2 U  -  (U - V)  +  Q          [radiation energy]
  d/dt V =  epsilon * (U - V)                            [material energy]

with EOS  e = alpha * T^4  => epsilon = alpha / (4a).

State vector w = [U, V]^T — identical 2-component structure to the Burgers (u, v) system.

Backward Euler residual (all RHS at n+1):
  F1 = U^{n+1} - U^n - dt * [ (1/3)*lap(U^{n+1}) - (U^{n+1} - V^{n+1}) + Q^{n+1} ] = 0
  F2 = V^{n+1} - V^n - dt * epsilon * (U^{n+1} - V^{n+1}) = 0

Coupling notes:
  - F1 contains the diffusion operator (1/3)*lap and the emission/absorption term (U-V).
  - F2 is purely local (no spatial operator); V couples back into F1 through (U-V).
  - The Jacobian off-diagonal blocks are O(dt), so block-diagonal preconditioning
    is effective at small dt and degrades as dt grows (not implemented here).

Boundary condition flags (set in __main__):
  BC_X : 'dirichlet' | 'periodic'   (vacuum BCs in x for Su-Olson)
  BC_Y : 'dirichlet' | 'periodic'   (periodic in y; problem is 1D in x)

Precision flag (set in __main__):
  PRECISION : 'float32' | 'float64'
  For float64, JAX x64 mode is enabled automatically.
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
def next_gif_path(base: str = 'suolson') -> str:
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
    if sim_type == 'SO':
        # Su-Olson cold start: U = V = 0 everywhere.
        # Energy is injected entirely by the external source Q.
        U = jnp.zeros_like(X)
        V = jnp.zeros_like(Y)
    elif sim_type == 'checkerboard':
        # 2D sine wave grid. Modulate frequency with kx, ky.
        kx, ky = 2.0, 2.0
        
        # Absolute value makes sure energy is positive. 
        # Gives a beautiful "egg carton" shape.
        U = jnp.abs(jnp.sin(kx * X) * jnp.cos(ky * Y)) 
        
        # Material starts cold, creating immediate complex local absorption
        V = jnp.zeros_like(Y)
    elif sim_type == 'rings':
        # Center of the rings
        r = jnp.sqrt(X**2 + Y**2)
        
        # Create an initial pulse structured as a decaying ring
        # Max energy at r=2, decaying outward and inward
        ring_width = 0.25
        U = jnp.exp(-((r - 2.0)**2) / ring_width)
        
        # Add a central hot spot
        U += 0.3 * jnp.exp(-(r**2) / 0.2)
        
        V = jnp.zeros_like(Y)
    else:
        raise ValueError(f"Unknown IC type: {sim_type}. Use 'SO', 'checkerboard', or 'rings'.")
    return U.astype(dtype), V.astype(dtype)

def get_source_term(source, X, Y, tau, Q0, x_src, tau_src, x_center, dtype):
    if source == "central":
        # Box source in space and time:
        #   Q = Q0  for |x - x_center| <= x_src  AND  tau <= tau_src
        #   Q = 0   otherwise
        # Called once per time step from Python — no JIT needed.
        in_space = jnp.abs(X - x_center) <= x_src
        in_time  = float(tau) <= float(tau_src)   # Python-level branch; avoids JAX trace issues
        if in_time:
            Q = jnp.where(in_space, jnp.array(Q0, dtype=dtype), jnp.zeros((), dtype=dtype))
        else:
            Q = jnp.zeros_like(X)
        return Q
    elif source == "pulsar":
        # --- Orbiting Source (Continuous) ---
        R = 2.5                  # Radius of the orbit
        omega = 1.75             # Orbit speed
        sigma = 0.4              # Width of the Gaussian spot

        # Calculate moving center coordinates
        x_c = R * jnp.cos(omega * tau)
        y_c = R * jnp.sin(omega * tau)

        # Constant amplitude (no flickering)
        amp = Q0 

        # 2D Gaussian profile
        r2 = (X - x_c)**2 + (Y - y_c)**2
        Q = amp * jnp.exp(-r2 / (2 * sigma**2))
        return Q.astype(dtype)
        
    else:
        raise ValueError(f"Unknown source type: {source}")


def apply_BC(field, bc_x, bc_y):
    if bc_x == DIRICHLET:
        field = field.at[0,  :].set(0.0)
        field = field.at[-1, :].set(0.0)
    if bc_y == DIRICHLET:
        field = field.at[:,  0].set(0.0)
        field = field.at[:, -1].set(0.0)
    return field


# -------- PDE components --------
def calc_dt(dx, dy, epsilon, C=0.7, D=1.0/3.0, offset=1e-12):
    # Backward Euler is unconditionally stable, but accuracy still requires
    # resolving both the diffusion timescale and the coupling timescale.
    dt_diff   = C * min(dx, dy)**2 / D        # diffusion accuracy: dt ~ dx^2 / D
    dt_couple = C / (epsilon + offset)        # coupling accuracy:  dt ~ 1 / epsilon
    return min(dt_diff, dt_couple)

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

def constructF_rad(U_k, U_old, V_k, lap_U_k, dt, Q):
    # Radiation energy residual  F1 = 0
    # Backward Euler: U_k - U_old - dt*[ (1/3)*lap(U_k) - (U_k - V_k) + Q ] = 0
    # Diffusion coefficient D = 1/3 is built in (dimensionless Su-Olson form).
    return U_k - U_old - dt * (1.0/3.0 * lap_U_k - (U_k - V_k) + Q)

def constructF_mat(V_k, V_old, U_k, dt, epsilon):
    # Material energy residual  F2 = 0  — no spatial operator, purely local coupling.
    # Backward Euler: V_k - V_old - dt * epsilon * (U_k - V_k) = 0
    return V_k - V_old - dt * epsilon * (U_k - V_k)

def concatenateJnp(vec1, vec2):
    return jnp.concatenate([vec1.ravel(), vec2.ravel()])

def flattenJnp(state_vec, Nx, Ny):
    NxNy = Nx * Ny
    return state_vec[:NxNy].reshape((Nx, Ny)), state_vec[NxNy:].reshape((Nx, Ny))


# ------- Total energy -------
def total_energy(U, V, dx, dy):
    # Integral of total energy density (radiation + material) over the domain
    return float(jnp.sum(U + V)) * dx * dy

def plot_energy(energy_history, dt_history, gif_path):
    times = [0.0]
    for dt in dt_history:
        times.append(times[-1] + dt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, energy_history, color='red', linewidth=2)
    ax.set_xlabel(r'$\tau$ (dimensionless time)')
    ax.set_ylabel(r'Total Energy $\int (U + V)\,dx\,dy$')
    ax.set_ylim([0.0, 1.1 * max(energy_history) if max(energy_history) > 0 else 1.0])
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    energy_path = gif_path.replace('.gif', '_energy.png')
    fig.savefig(energy_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Energy plot saved -> {energy_path}")


# ------------ Jacobian-vector products (FD and AD) ---------------

def JacobianActionFD(U_k, V_k, F_U, F_V, Nx, Ny, perturb, dt, epsilon, Q, dx, dy, bc_x=DIRICHLET, bc_y=PERIODIC):
    NxNy = Nx * Ny
    dU = perturb[:NxNy].reshape((Nx, Ny))
    dV = perturb[NxNy:].reshape((Nx, Ny))

    # @@ Gateaux derivative approximation of the Jacobian-vector product:
    # @@   J(w_k) * p  ~=  [ F(w_k + eps*p) - F(w_k) ] / eps
    # @@ where w_k = (U_k, V_k) is the current Newton iterate.
    # @@ U_old / V_old are fixed constants and cancel exactly in the difference,
    # @@ so they do not need to be passed here.
    # @@ Only difficulty: choosing eps to balance truncation and cancellation error.

    mach_eps = jnp.finfo(U_k.dtype).eps
    b = jnp.sqrt(mach_eps)

    state_norm   = jnp.linalg.norm(jnp.concatenate([U_k.ravel(), V_k.ravel()]))
    perturb_norm = jnp.linalg.norm(perturb)
    safe_perturb_norm = jnp.where(perturb_norm > 0.0, perturb_norm, 1.0)
    eps = b * jnp.maximum(1.0, state_norm) / safe_perturb_norm
    eps_max = jnp.sqrt(b)
    eps = jnp.clip(eps, b, eps_max)
    eps = jnp.where(jnp.isfinite(eps), eps, b)

    U_pert = U_k + eps * dU
    V_pert = V_k + eps * dV

    # Perturbed residuals (U_old = U_k, V_old = V_k — cancel in the difference)
    lap_U_pert = laplacian(U_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_U_pert   = constructF_rad(U_pert, U_k, V_pert, lap_U_pert, dt, Q)
    F_V_pert   = constructF_mat(V_pert, V_k, U_pert, dt, epsilon)

    Jv_U = (F_U_pert - F_U) / eps
    Jv_V = (F_V_pert - F_V) / eps

    return concatenateJnp(Jv_U, Jv_V)

# Flat residual evaluator for the AD graph
def residual_flat(state, U_old, V_old, dt, epsilon, Q, dx, dy, bc_x, bc_y, Nx, Ny):
    U_ = state[:Nx*Ny].reshape((Nx, Ny))
    V_ = state[Nx*Ny:].reshape((Nx, Ny))

    lap_U_ = laplacian(U_, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_U_   = constructF_rad(U_, U_old, V_, lap_U_, dt, Q)
    F_V_   = constructF_mat(V_, V_old, U_, dt, epsilon)

    return concatenateJnp(F_U_, F_V_)

# Jitted AD Jacobian-Vector Product
@ft_partial(jax.jit, static_argnums=(10, 11, 12, 13))
def JacobianActionAD_jit(U_k, V_k, U_old, V_old, perturb, dt, epsilon, Q, dx, dy, bc_x, bc_y, Nx, Ny):
    state_k = concatenateJnp(U_k, V_k)
    def F_flat(s):
        return residual_flat(s, U_old, V_old, dt, epsilon, Q, dx, dy, bc_x, bc_y, Nx, Ny)
    _, jvp_result = jax.jvp(F_flat, (state_k,), (perturb,))
    return jvp_result

class KrylovCounter:
    """A simple callback to count Krylov iterations for SciPy and CuPy."""
    def __init__(self):
        self.niter = 0
    def __call__(self, *args, **kwargs):
        self.niter += 1

# ------ Plotting -------------
def init_plot(U0, V0, xi, xf, yi, yf, bc_x, bc_y):
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_U, ax_V, ax_tot = axes
    extent = [xi, xf, yi, yf]
    tot0   = U0 + V0

    # Cold start: all zeros. Use a small positive vmax so colorbar is valid.
    U_vmax  = 0.5
    V_vmax = 0.35
    tot_vmax = 0.7

    bc_label = f'BC: x={bc_x[0].upper()}  y={bc_y[0].upper()}'

    img_U = ax_U.imshow(np.array(U0), cmap='inferno', extent=extent,
                        origin='lower', vmin=0.0, vmax=U_vmax)
    ax_U.set_title(r'$U$ (radiation) | Step 0' + f'\n{bc_label}')
    ax_U.axis('off')
    fig.colorbar(img_U,   ax=ax_U,   fraction=0.046, pad=0.04)

    img_V = ax_V.imshow(np.array(V0), cmap='plasma', extent=extent,
                        origin='lower', vmin=0.0, vmax=V_vmax)
    ax_V.set_title(r'$V$ (material) | Step 0' + f'\n{bc_label}')
    ax_V.axis('off')
    fig.colorbar(img_V,   ax=ax_V,   fraction=0.046, pad=0.04)

    img_tot = ax_tot.imshow(np.array(tot0), cmap='viridis', extent=extent,
                            origin='lower', vmin=0.0, vmax=tot_vmax)
    ax_tot.set_title(r'$U+V$ (total) | Step 0' + f'\n{bc_label}')
    ax_tot.axis('off')
    fig.colorbar(img_tot, ax=ax_tot, fraction=0.046, pad=0.04)

    return fig, img_U, img_V, img_tot, ax_U, ax_V, ax_tot, bc_label

def update_plot(img_U, img_V, img_tot, ax_U, ax_V, ax_tot, U, V, step, bc_label, tau, displayPlot=True):
    tot = U + V

    U_vmax  = 0.5
    V_vmax = 0.35
    tot_vmax = 0.7

    img_U.set_data(np.array(U))
    img_U.set_clim(vmin=0.0, vmax=U_vmax)
    ax_U.set_title(r'$U$ | Step ' + f'{step}')

    img_V.set_data(np.array(V))
    img_V.set_clim(vmin=0.0, vmax=V_vmax)
    ax_V.set_title(r'$V$ | Step ' + f'{step}')

    img_tot.set_data(np.array(tot))
    img_tot.set_clim(vmin=0.0, vmax=tot_vmax)
    ax_tot.set_title(r'$U+V$ | Step ' + f'{step}')

    if displayPlot:
        plt.pause(0.01)

def capture_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    return Image.open(buf).copy()

def save_gif(frames, path='raddiff_001.gif', fps=6):
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
def runSimulation(device,
                  PRECISION,
                  BC_X,
                  BC_Y,
                  SIMULATION_TYPE,
                  SIMULATION_IC,
                  SOURCE_TYPE,
                  verbose,
                  useAD,
                  maxBackTrackingIter,
                  epsilon,
                  Q0,
                  x_src,
                  tau_src,
                  Courant,
                  steps,
                  Nx,
                  Ny,
                  KrylovSolver,
                  KrylovTol,
                  KrylovIter,
                  NewtonNonlinTol,
                  NewtonIter,
                  plot_steps,
                  gif_fps,
                  displayPlot,
                  figFolder,
                  save_steps,
                  dataFolder):

    t_wall_start = time.perf_counter()

    dtype = configure_precision(PRECISION)
    print(f"Hardware Target: {device.upper()}")
    print(f"Precision: {PRECISION}  (dtype={dtype})")

    # auto-increment filename and save into figFolder
    os.makedirs(figFolder, exist_ok=True)
    gif_path = next_gif_path(f'{figFolder}/raddiff')
    print(f"Output GIF: {gif_path}")

    # ---- grids --------------------------------------------------------
    xi, xf   = -5.0, 5.0
    yi, yf   = -5.0, 5.0
    x_center =  0.0          # source centered at origin

    x = jnp.linspace(xi, xf, Nx, endpoint=(BC_X == DIRICHLET)).astype(dtype)
    y = jnp.linspace(yi, yf, Ny, endpoint=(BC_Y == DIRICHLET)).astype(dtype)

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # ---- ICs / BCs ----------------------------------------------------
    U0_raw, V0_raw = get_initial_conditions(X, Y, dtype, SIMULATION_IC)
    U0 = apply_BC(U0_raw, BC_X, BC_Y)
    V0 = apply_BC(V0_raw, BC_X, BC_Y)
    U, V = U0, V0

    # ---- bake BC flags into spatial operators; JIT-compile ------------
    laplacian_jit      = jax.jit(ft_partial(laplacian,      bc_x=BC_X, bc_y=BC_Y))
    constructF_rad_jit = jax.jit(constructF_rad)
    constructF_mat_jit = jax.jit(constructF_mat)

    # static_argnums: Nx(4), Ny(5), bc_x(12), bc_y(13)
    JacobianActionFD_jit = jax.jit(JacobianActionFD, static_argnums=(4, 5, 12, 13))

    def _scipy_matvec_fn(U_k, V_k, Nx, Ny, p, dt, epsilon, Q, dx, dy, bc_x, bc_y):
        lap_U_b = laplacian(U_k, dx, dy, bc_x=bc_x, bc_y=bc_y)
        F_U_b   = constructF_rad(U_k, U_k, V_k, lap_U_b, dt, Q)
        F_V_b   = constructF_mat(V_k, V_k, U_k, dt, epsilon)
        return JacobianActionFD(U_k, V_k, F_U_b, F_V_b, Nx, Ny, p, dt, epsilon, Q, dx, dy, bc_x=bc_x, bc_y=bc_y)

    # static_argnums: Nx(2), Ny(3), bc_x(10), bc_y(11)
    JacobianActionScipy_jit = jax.jit(_scipy_matvec_fn, static_argnums=(2, 3, 10, 11))

    print(f"BC configuration: x={BC_X}  y={BC_Y}")
    print(f"Grid: {Nx}x{Ny}   dx={dx:.4f}  dy={dy:.4f}")
    print(f"Params: epsilon={epsilon}  Q0={Q0}  x_src={x_src}  tau_src={tau_src}  Courant={Courant}")

    # ---- init plot; capture frame 0 -----------------------------------
    fig, img_U, img_V, img_tot, ax_U, ax_V, ax_tot, bc_label = init_plot(
        U0, V0, xi, xf, yi, yf, BC_X, BC_Y
    )
    gif_frames = [capture_frame(fig)]

    # ---- tracking variables -------------------------------------------
    energy_history          = [total_energy(U, V, dx, dy)]
    dt_history              = []
    tau                     = 0.0   # current dimensionless time

    newton_iters_per_step   = []
    krylov_iters_per_newton = []
    time_per_newton_iter    = []
    time_per_time_step      = []
    step_success_history    = []
    final_residual_history  = []

    # ----------- TIME LOOP -----------
    print("Starting Time Loop")
    for step in range(steps):
        print(f"\nStep {step+1}/{steps}  tau={tau:.4f}")

        step_start_time = time.perf_counter()

        dt = calc_dt(dx, dy, epsilon, Courant)
        dt_history.append(float(dt))

        # Source evaluated at tau^{n+1} (consistent with backward Euler)
        Q = get_source_term(SOURCE_TYPE, X, Y, tau + dt, Q0, x_src, tau_src, x_center, dtype)

        U_old, V_old = U, V
        U_k,   V_k   = U, V

        # --- track convergence for Perf Analysis ---
        step_converged = False
        final_res = 0.0

        # -------------------- Newton loop ------------------------------
        for k in range(NewtonIter):
            newton_start_time = time.perf_counter()

            lap_U_k = laplacian_jit(U_k, dx, dy)

            F_U_k = constructF_rad_jit(U_k, U_old, V_k, lap_U_k, dt, Q)
            F_V_k = constructF_mat_jit(V_k, V_old, U_k, dt, epsilon)
            F_vec = concatenateJnp(F_U_k, F_V_k)

            res_norm = float(jnp.linalg.norm(F_vec))
            final_res = res_norm
            print(f"  Newton iter {k}: ||F|| = {res_norm:.3e}")

            if res_norm <= NewtonNonlinTol:
                step_converged = True
                newton_end_time = time.perf_counter()
                time_per_newton_iter.append(newton_end_time - newton_start_time)
                newton_iters_per_step.append(k + 1)
                break

            # Matvec closures capture the current nonlinear state locally
            if useAD:
                def A_matvec_jax(vec):
                    return JacobianActionAD_jit(U_k, V_k, U_old, V_old, vec, dt, epsilon, Q, dx, dy, BC_X, BC_Y, Nx, Ny)
            else:
                def A_matvec_jax(vec):
                    # Passing U_k and V_k as the 'old' terms for the FD evaluation
                    return JacobianActionScipy_jit(U_k, V_k, Nx, Ny, vec, dt, epsilon, Q, dx, dy, BC_X, BC_Y)

            krylov_counter = KrylovCounter()

            # --- HARDWARE BRANCHING ---
            if device == 'cpu':
                # Convert JAX vector to SciPy vector using standard NumPy mapping
                def A_matvec_scipy(vec_np):
                    vec_jax = jnp.array(vec_np, dtype=dtype)
                    res_jax = A_matvec_jax(vec_jax)
                    return np.asarray(res_jax).copy()

                JLinearOp = ScipyLinearOperator(
                    (2*Nx*Ny, 2*Nx*Ny),
                    matvec=A_matvec_scipy,
                    dtype=np.float32 if PRECISION == 'float32' else np.float64
                )

                if KrylovSolver.lower() == 'gmres':
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
                    raise ValueError("CPU KrylovSolver must be 'gmres' or 'bicgstab'")
                    
                delta_state = jnp.array(delta_state_np, dtype=dtype)
                
                if verbose:
                    status = "OK" if info == 0 else f"FAILED (info={info})"
                    print(f"  SciPy {KrylovSolver.upper()} [{status}] iters={krylov_counter.niter}")

            elif device == 'gpu':
                # Wrapper to bridge CuPy and JAX
                def A_matvec_cp(vec_cp):
                    vec_jax = jax_dlpack.from_dlpack(vec_cp)
                    res_jax = A_matvec_jax(vec_jax)
                    return cp.from_dlpack(res_jax)

                # Define the Linear Operator for CuPy
                JLinearOp = cupy_spla.LinearOperator(
                    (2*Nx*Ny, 2*Nx*Ny),
                    matvec=A_matvec_cp,
                    dtype=np.float32 if PRECISION == 'float32' else np.float64
                )

                # Convert the RHS base residual (-F_k) to CuPy directly
                F_vec_cp = cp.from_dlpack(-F_vec)

                # Execute chosen Krylov solver natively on the GPU via CuPy
                if KrylovSolver.lower() == 'gmres':
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
                    raise ValueError("GPU KrylovSolver must be 'gmres', 'bicgstab', or 'cgs'")
                
                # Convert the resulting perturbation vector back to JAX directly
                delta_state = jax_dlpack.from_dlpack(delta_state_cp)

                if verbose:
                    status = "OK" if info == 0 else f"FAILED (info={info})"
                    print(f"  CuPy {KrylovSolver.upper()} [{status}] iters={krylov_counter.niter}")

            krylov_iters_per_newton.append(krylov_counter.niter)

            # ---- Backtracking line search to stabilise Newton near strong gradients -----
            delta_U, delta_V = flattenJnp(delta_state, Nx, Ny)
            alpha = 1.0

            for ls in range(maxBackTrackingIter):
                U_try = U_k + alpha * delta_U
                V_try = V_k + alpha * delta_V

                # Apply vacuum BCs
                U_try = apply_BC(U_try, BC_X, BC_Y)
                V_try = apply_BC(V_try, BC_X, BC_Y)

                lap_U_try = laplacian_jit(U_try, dx, dy)
                F_U_try   = constructF_rad_jit(U_try, U_old, V_try, lap_U_try, dt, Q)
                F_V_try   = constructF_mat_jit(V_try, V_old, U_try, dt, epsilon)

                F_try_norm = float(jnp.linalg.norm(concatenateJnp(F_U_try, F_V_try)))

                if F_try_norm < res_norm:
                    if verbose:
                        print(f"    Line search accepted alpha={alpha:.3f}, new ||F||={F_try_norm:.3e}")
                    U_k, V_k = U_try, V_try
                    break
                else:
                    alpha *= 0.5
            else:
                if verbose:
                    print(f"    Line search failed to find a descent step. Forcing update.")
                U_k, V_k = U_try, V_try

            newton_end_time = time.perf_counter()
            time_per_newton_iter.append(newton_end_time - newton_start_time)

        else:
            newton_iters_per_step.append(NewtonIter)

        # --- store Newton Success and residual ---
        step_success_history.append(1 if step_converged else 0)
        final_residual_history.append(final_res)

        U, V  = U_k, V_k
        tau  += dt
        energy_history.append(total_energy(U, V, dx, dy))

        step_end_time = time.perf_counter()
        time_per_time_step.append(step_end_time - step_start_time)

        if (step + 1) % plot_steps == 0:
            update_plot(
                img_U, img_V, img_tot,
                ax_U, ax_V, ax_tot,
                U, V, step + 1, bc_label, tau, displayPlot
            )
            gif_frames.append(capture_frame(fig))

        # @@ save for comparison
        if save_steps > 0 and (step + 1) % save_steps == 0:
            os.makedirs(dataFolder, exist_ok=True)
            
            # Convert to standard CPU NumPy arrays for portable saving
            U_np = np.array(U)
            V_np = np.array(V)
            
            # Save files
            U_path = f"{dataFolder}/U_step{step + 1:04d}_t{tau:.4f}.npy"
            V_path = f"{dataFolder}/V_step{step + 1:04d}_t{tau:.4f}.npy"
            
            np.save(U_path, U_np)
            np.save(V_path, V_np)
            
            if verbose:
                print(f"  Saved fields to {U_path} and {V_path}")
        # @@ here

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
  Simulation    : {SIMULATION_TYPE}
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

    print("Saving energy plot")
    plot_energy(energy_history, dt_history, gif_path)

    print("Saving the GIF")
    save_gif(gif_frames, path=gif_path, fps=gif_fps)

    plt.ioff()
    if displayPlot:
        plt.show()


# Launch simulation
if __name__ == "__main__":

    # ---- Command Line Argument Parsing ---- #
    parser = argparse.ArgumentParser(description="JFNK Su-Olson Radiative Diffusion Solver")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help="Target hardware: 'cpu' or 'gpu' (default: cpu)")
    args = parser.parse_args()

    # Hardware enforcement
    if args.device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')
    elif args.device == 'gpu' and not HAS_GPU_LIBS:
        raise ImportError("You requested '--device gpu', but CuPy or jax.dlpack could not be loaded.")

    # ---- Simulation Configuration Type ---- #
    # Set to 'CLASSIC_SU_OLSON' or 'DYNAMIC'
    SIMULATION_TYPE = 'CLASSIC_SU_OLSON'

    # ---- Apply the chosen Profile Configurations ---- #
    if SIMULATION_TYPE == 'CLASSIC_SU_OLSON':
        SIMULATION_IC   = 'SO'            # Su-Olson: cold start, source-driven
        SOURCE_TYPE     = 'central'       # Central boxed source
        BC_X            = DIRICHLET       # vacuum BCs in x  (U = 0 at x = ±L)
        BC_Y            = PERIODIC        # homogeneous in y  (1D problem on 2D grid)
    elif SIMULATION_TYPE == 'DYNAMIC':
        SIMULATION_IC   = 'SO'            # Try 'rings' 
        SOURCE_TYPE     = 'pulsar'        # Orbiting and pulsating Gaussian
        BC_X            = DIRICHLET       # Closed box, waves crash into the walls
        BC_Y            = DIRICHLET       # Closed box, waves crash into the walls
    else:
        raise ValueError(f"Unknown SIMULATION_TYPE: {SIMULATION_TYPE}")

    # Precision
    PRECISION       = 'float64'

    # ---- Physical Parameters ---- #
    # epsilon: ratio of material to radiation heat capacity
    #   epsilon = 1.0  -> equal heat capacities, moderate coupling
    #   epsilon << 1   -> stiff regime, large contrast in timescales
    epsilon         = 0.1
    Q0              = 1.0             # source amplitude  (dimensionless)
    x_src           = 0.5            # source half-width  (dimensionless)
    tau_src         = float('inf')   # source duration; inf = always on
    Courant         = 3           # fixed timestep (backward Euler: unconditionally stable)
    steps           = 1000             # tau_final = steps * dt_fixed = 3.0 (lowered slightly for rendering test)

    # ---- Grid ---- #
    # Ny small: problem is 1D in x; keep Ny >= 4 for 2D operator correctness checks
    Nx, Ny          = 128, 128

    # ---- Su-Olson Solver ---- #
    KrylovSolver        = 'bicgstab'  # 'gmres', 'bicgstab', or 'cgs' (cgs is GPU only)
    useAD               = True        # True -> AD Jacobian-vector product; False -> FD
    verbose             = False
    maxBackTrackingIter = 15

    # ---- Solver Tolerances ---- #
    if PRECISION == 'float32':
        KrylovTol       = 1e-3
        KrylovIter      = int(1e2)
        NewtonNonlinTol = 1e-3
        NewtonIter      = int(15)
    elif PRECISION == 'float64':
        KrylovTol       = 1e-5
        KrylovIter      = int(1e2)
        NewtonNonlinTol = 1e-5
        NewtonIter      = int(15)
    else:
        raise ValueError('Choose different Precision')

    # ---- Plotting + I/O ---- #
    plot_steps      = 5
    save_steps      = 1e7   # saves .npy data every "save_steps" steps
    gif_fps         = 10
    displayPlot     = True
    figFolder       = "output/raddiff"
    dataFolder      = f"data/raddiff_{SIMULATION_IC}_{args.device.upper()}_{Nx}_{Ny}_{'AD' if useAD else 'FD'}_{PRECISION}"

    # Run simulation
    runSimulation(
        device=args.device,
        PRECISION=PRECISION,
        BC_X=BC_X,
        BC_Y=BC_Y,
        SIMULATION_TYPE=SIMULATION_TYPE,
        SIMULATION_IC=SIMULATION_IC,
        SOURCE_TYPE=SOURCE_TYPE,
        verbose=verbose,
        useAD=useAD,
        maxBackTrackingIter=maxBackTrackingIter,
        epsilon=epsilon,
        Q0=Q0,
        x_src=x_src,
        tau_src=tau_src,
        Courant=Courant,
        steps=steps,
        Nx=Nx,
        Ny=Ny,
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
