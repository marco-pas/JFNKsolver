import glob
import io
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial as ft_partial
from numpy.linalg import norm
from PIL import Image

import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_spla
from jax import dlpack as jax_dlpack

# Use JAX's native sparse solvers to keep everything on the GPU
import jax.scipy.sparse.linalg as jax_spla

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],   
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 12
})

DIRICHLET = 'dirichlet'
PERIODIC  = 'periodic'

# -------- precision helper ------
def configure_precision(precision: str):
    if precision == 'float64':
        jax.config.update("jax_enable_x64", True)
        return jnp.float64
    elif precision == 'float32':
        return jnp.float32
    else:
        raise ValueError(f"Unknown precision '{precision}'. Use 'float32' or 'float64'.")

# -------- increment GIF path --------
def next_gif_path(base: str = 'burgers_evolution') -> str:
    existing = glob.glob(f'{base}_???.gif')
    used = set()
    for f in existing:
        stem = f[len(base) + 1:-4]
        if stem.isdigit():
            used.add(int(stem))
    for n in range(1, 1000):
        if n not in used:
            return f'{base}_{n:03d}.gif'
    raise RuntimeError("All GIF slots _001–_999 are taken. Clean up old files.")

#  ---------- ICs / BCs  --------
def get_initial_conditions(X, Y, dtype, sim_type):
    if sim_type == 'TGV':
        u = jnp.sin(X) * jnp.cos(Y)
        v = -jnp.cos(X) * jnp.sin(Y)
    elif sim_type == 'DSL':
        rho = 30.0
        delta = 0.05
        u = jnp.where(Y <= jnp.pi, 
                      jnp.tanh(rho * (Y - jnp.pi/2)), 
                      jnp.tanh(rho * (3*jnp.pi/2 - Y)))
        v = delta * jnp.sin(X)
    elif sim_type == '4VC':
        R = 0.5
        centers = [
            (jnp.pi - 0.8, jnp.pi - 0.8,  1.0),
            (jnp.pi + 0.8, jnp.pi + 0.8,  1.0),
            (jnp.pi - 0.8, jnp.pi + 0.8, -1.0),
            (jnp.pi + 0.8, jnp.pi - 0.8, -1.0)
        ]
        u = jnp.zeros_like(X)
        v = jnp.zeros_like(Y)
        for cx, cy, gamma in centers:
            r2 = (X - cx)**2 + (Y - cy)**2
            u += -gamma * (Y - cy) * jnp.exp(-r2 / R**2)
            v +=  gamma * (X - cx) * jnp.exp(-r2 / R**2)
    else:
        raise ValueError(f"Unknown IC type: {sim_type}. Use 'TGV', 'DSL', or '4VC'.")
        
    return u.astype(dtype), v.astype(dtype)

def apply_BC(field, bc_x, bc_y):
    if bc_x == DIRICHLET:
        field = field.at[0,  :].set(0.0)
        field = field.at[-1, :].set(0.0)
    if bc_y == DIRICHLET:
        field = field.at[:,  0].set(0.0)
        field = field.at[:, -1].set(0.0)
    return field

#  -------- PDE components ------------
def calc_dt(vec1, vec2, dx1, dx2, nu, C=0.7, offset=1e-12):
    dt = C * min([
        dx1 / (jnp.max(abs(vec1)) + offset),
        dx2 / (jnp.max(abs(vec2)) + offset),
        1 / (2 * nu * (1 / dx1**2 + 1 / dx2**2))
    ])
    return dt

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

def advection(f1, f2, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
    if bc_x == PERIODIC:
        df1_dx = (jnp.roll(f1, -1, axis=0) - jnp.roll(f1, 1, axis=0)) / (2*dx)
    else:
        df1_dx = jnp.zeros_like(f1)
        df1_dx = df1_dx.at[1:-1, :].set((f1[2:, :] - f1[:-2, :]) / (2*dx))

    if bc_y == PERIODIC:
        df1_dy = (jnp.roll(f1, -1, axis=1) - jnp.roll(f1, 1, axis=1)) / (2*dy)
    else:
        df1_dy = jnp.zeros_like(f1)
        df1_dy = df1_dy.at[:, 1:-1].set((f1[:, 2:] - f1[:, :-2]) / (2*dy))

    adv = f1 * df1_dx + f2 * df1_dy

    if bc_x == DIRICHLET:
        adv = adv.at[0,  :].set(0.0)
        adv = adv.at[-1, :].set(0.0)
    if bc_y == DIRICHLET:
        adv = adv.at[:,  0].set(0.0)
        adv = adv.at[:, -1].set(0.0)

    return adv

def constructF(vec1, vec2, adv, lap, dt, nu):
    return vec1 - vec2 + dt * (adv - nu * lap)

def concatenateJnp(vec1, vec2):
    return jnp.concatenate([vec1.ravel(), vec2.ravel()])

def flattenJnp(vel_vec, Nx, Ny):
    NxNy = Nx * Ny
    return vel_vec[:NxNy].reshape((Nx, Ny)), vel_vec[NxNy:].reshape((Nx, Ny))


# ------- Kinetic energy plot -------
def kinetic_energy(u, v, dx, dy):
    return 0.5 * float(jnp.sum(u**2 + v**2)) * dx * dy

def plot_energy(energy_history, dt_history, gif_path):
    times = [0.0]
    for dt in dt_history:
        times.append(times[-1] + dt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, energy_history, color='red', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Kinetic energy  E(t)')
    ax.set_ylim([0.0, 1.1*max(energy_history)])
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    energy_path = gif_path.replace('.gif', '_energy.png')
    fig.savefig(energy_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Energy plot saved -> {energy_path}")

#  ------------ Jacobian-vector products (FD and AD) ---------------
def JacobianActionFD(u, v, F_u, F_v, Nx, Ny, perturb, dt, nu, dx, dy, bc_x=DIRICHLET, bc_y=DIRICHLET):
    NxNy = Nx * Ny
    du = perturb[:NxNy].reshape((Nx, Ny))
    dv = perturb[NxNy:].reshape((Nx, Ny))

    mach_eps = jnp.finfo(u.dtype).eps
    b = jnp.sqrt(mach_eps)

    state_norm = jnp.linalg.norm(jnp.concatenate([u.ravel(), v.ravel()]))
    
    # FIX: Remove perturb_norm. GMRES internal vectors are normalized to ~1.0 anyway.
    # This keeps eps constant relative to the input vector, ensuring strict JAX linearity.
    eps = b * jnp.maximum(1.0, state_norm) 
    
    eps_max = jnp.sqrt(b)
    eps = jnp.clip(eps, b, eps_max)
    eps = jnp.where(jnp.isfinite(eps), eps, b)

    u_pert = u + eps * du
    v_pert = v + eps * dv

    lap_u_pert = laplacian(u_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    adv_u_pert = advection(u_pert, v_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_u_pert   = constructF(u_pert, u, adv_u_pert, lap_u_pert, dt, nu)

    lap_v_pert = laplacian(v_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    adv_v_pert = advection(v_pert, u_pert, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_v_pert   = constructF(v_pert, v, adv_v_pert, lap_v_pert, dt, nu)

    Jv_u = (F_u_pert - F_u) / eps
    Jv_v = (F_v_pert - F_v) / eps

    return concatenateJnp(Jv_u, Jv_v)

def residual_flat(state, u_old, v_old, dt, nu, dx, dy, bc_x, bc_y, Nx, Ny):
    u_ = state[:Nx*Ny].reshape((Nx, Ny))
    v_ = state[Nx*Ny:].reshape((Nx, Ny))
    
    lap_u_ = laplacian(u_, dx, dy, bc_x=bc_x, bc_y=bc_y)
    adv_u_ = advection(u_, v_, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_u_   = constructF(u_, u_old, adv_u_, lap_u_, dt, nu)
    
    lap_v_ = laplacian(v_, dx, dy, bc_x=bc_x, bc_y=bc_y)
    adv_v_ = advection(v_, u_, dx, dy, bc_x=bc_x, bc_y=bc_y)
    F_v_   = constructF(v_, v_old, adv_v_, lap_v_, dt, nu)
    
    return concatenateJnp(F_u_, F_v_)

@ft_partial(jax.jit, static_argnums=(9, 10, 11, 12))
def JacobianActionAD_jit(u_k, v_k, u_old, v_old, perturb, dt, nu, dx, dy, bc_x, bc_y, Nx, Ny):
    state_k = concatenateJnp(u_k, v_k)
    def F_flat(s):
        return residual_flat(s, u_old, v_old, dt, nu, dx, dy, bc_x, bc_y, Nx, Ny)
    _, jvp_result = jax.jvp(F_flat, (state_k,), (perturb,))
    return jvp_result


# ------ Plotting -------------             
def vel_magnitude(u, v):
    return jnp.sqrt(u**2 + v**2)

def init_plot(u0, v0, xi, xf, yi, yf, bc_x, bc_y):
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_u, ax_v, ax_mag = axes
    extent = [xi, xf, yi, yf]
    mag0   = vel_magnitude(u0, v0)

    u_vmin, u_vmax     = float(jnp.min(u0)),   float(jnp.max(u0))
    v_vmin, v_vmax     = float(jnp.min(v0)),   float(jnp.max(v0))
    mag_vmin, mag_vmax = 0.0,                   float(jnp.max(mag0))

    bc_label = f'BC: x={bc_x[0].upper()}  y={bc_y[0].upper()}'

    img_u = ax_u.imshow(np.array(u0),    cmap='magma',   extent=extent,
                        origin='lower', vmin=u_vmin,   vmax=u_vmax)
    ax_u.set_title(f'u(x,y) | Step 0\n{bc_label}')
    ax_u.axis('off')
    fig.colorbar(img_u,   ax=ax_u,   fraction=0.046, pad=0.04)

    img_v = ax_v.imshow(np.array(v0),    cmap='plasma',  extent=extent,
                        origin='lower', vmin=v_vmin,   vmax=v_vmax)
    ax_v.set_title(f'v(x,y) | Step 0\n{bc_label}')
    ax_v.axis('off')
    fig.colorbar(img_v,   ax=ax_v,   fraction=0.046, pad=0.04)

    img_mag = ax_mag.imshow(np.array(mag0), cmap='viridis', extent=extent,
                            origin='lower', vmin=mag_vmin, vmax=mag_vmax)
    ax_mag.set_title(f'||vel|| | Step 0\n{bc_label}')
    ax_mag.axis('off')
    fig.colorbar(img_mag, ax=ax_mag, fraction=0.046, pad=0.04)

    plt.tight_layout()

    clims = dict(
        u_vmin=u_vmin,     u_vmax=u_vmax,
        v_vmin=v_vmin,     v_vmax=v_vmax,
        mag_vmin=mag_vmin, mag_vmax=mag_vmax,
    )
    return fig, img_u, img_v, img_mag, ax_u, ax_v, ax_mag, clims, bc_label

def update_plot(img_u, img_v, img_mag, ax_u, ax_v, ax_mag, u, v, step, clims, bc_label, displayPlot=True):
    mag = vel_magnitude(u, v)
    img_u.set_data(np.array(u))
    ax_u.set_title(f'u(x,y) | Step {step}\n{bc_label}')
    img_v.set_data(np.array(v))
    ax_v.set_title(f'v(x,y) | Step {step}\n{bc_label}')
    img_mag.set_data(np.array(mag))
    ax_mag.set_title(f'||vel|| | Step {step}\n{bc_label}')
    plt.tight_layout()
    if displayPlot:
        plt.pause(0.01)

def capture_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf).copy()

def save_gif(frames, path='burgers_evolution_001.gif', fps=6):
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
                  nu, steps, Nx, Ny, Courant, KrylovTol, KrylovIter, NewtonNonlinTol, NewtonIter,
                  plot_steps, gif_fps, displayPlot, figFolder):
    
    t_wall_start = time.perf_counter()

    dtype = configure_precision(PRECISION)
    print(f"Precision: {PRECISION}  (dtype={dtype})")

    os.makedirs(figFolder, exist_ok=True)
    gif_path = next_gif_path(f'{figFolder}/burgers_evolution')
    print(f"Output GIF: {gif_path}")

    # ---- grids ----
    xi, xf = 0.0, 2*np.pi
    yi, yf = 0.0, 2*np.pi
    x = jnp.linspace(xi, xf, Nx, endpoint=(BC_X == DIRICHLET)).astype(dtype)
    y = jnp.linspace(yi, yf, Ny, endpoint=(BC_Y == DIRICHLET)).astype(dtype)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # ---- ICs / BCs ----
    u0_raw, v0_raw = get_initial_conditions(X, Y, dtype, SIMULATION_IC)
    u0 = apply_BC(u0_raw, BC_X, BC_Y)
    v0 = apply_BC(v0_raw, BC_X, BC_Y)
    u, v = u0, v0

    # ---- JIT compiled components ----
    laplacian_jit  = jax.jit(ft_partial(laplacian,  bc_x=BC_X, bc_y=BC_Y))
    advection_jit  = jax.jit(ft_partial(advection,  bc_x=BC_X, bc_y=BC_Y))
    constructF_jit = jax.jit(constructF)
    JacobianActionFD_jit = jax.jit(JacobianActionFD, static_argnums=(4, 5, 11, 12))

    print(f"BC configuration: x={BC_X}  y={BC_Y}")
    print(f"Grid: {Nx}x{Ny}   dx={dx:.4f}  dy={dy:.4f}")

    fig, img_u, img_v, img_mag, ax_u, ax_v, ax_mag, clims, bc_label = init_plot(
        u0, v0, xi, xf, yi, yf, BC_X, BC_Y
    )
    gif_frames = [capture_frame(fig)]

    energy_history = [kinetic_energy(u, v, dx, dy)]
    dt_history     = []
    newton_iters_per_step = []
    time_per_newton_iter = []
    time_per_time_step = []

    # -----------TIME LOOP -----------
    print("Starting Time Loop")
    for step in range(steps):
        print(f"\nStep {step+1}/{steps}")
        step_start_time = time.perf_counter()

        dt = calc_dt(u, v, dx, dy, nu, Courant)
        dt_history.append(float(dt))

        u_old, v_old = u, v
        u_k,   v_k   = u, v

        # -------------------- Newton loop ------------------------------
        for k in range(NewtonIter):
            newton_start_time = time.perf_counter()

            lap_u_k = laplacian_jit(u_k, dx, dy)
            lap_v_k = laplacian_jit(v_k, dx, dy)
            adv_u_k = advection_jit(u_k, v_k, dx, dy)          
            adv_v_k = advection_jit(v_k, u_k, dx, dy)          

            F_u_k = constructF_jit(u_k, u_old, adv_u_k, lap_u_k, dt, nu)
            F_v_k = constructF_jit(v_k, v_old, adv_v_k, lap_v_k, dt, nu)
            F_vec = concatenateJnp(F_u_k, F_v_k)

            res_norm = float(jnp.linalg.norm(F_vec))
            print(f"  Newton iter {k}: ||F|| = {res_norm:.3e}")

            if res_norm <= NewtonNonlinTol:
                newton_end_time = time.perf_counter()
                time_per_newton_iter.append(newton_end_time - newton_start_time)
                newton_iters_per_step.append(k + 1)
                break

            # --- JAX Native GMRES ---
            # Matvec closures capture the current nonlinear state locally
            if useAD:
                def A_matvec_jax(vec):
                    return JacobianActionAD_jit(u_k, v_k, u_old, v_old, vec, dt, nu, dx, dy, BC_X, BC_Y, Nx, Ny)
            else:
                def A_matvec_jax(vec):
                    return JacobianActionFD_jit(u_k, v_k, F_u_k, F_v_k, Nx, Ny, vec, dt, nu, dx, dy, BC_X, BC_Y)

            # Wrapper to bridge CuPy and JAX
            def A_matvec_cp(vec_cp):
                # 1. CuPy -> JAX (Zero-copy)
                vec_jax = jax_dlpack.from_dlpack(vec_cp)
                
                # 2. Run the chosen JIT-compiled operator
                res_jax = A_matvec_jax(vec_jax)
                
                # 3. JAX -> CuPy (Zero-copy)
                return cp.from_dlpack(res_jax)

            # Define the Linear Operator for CuPy
            JLinearOp = cupy_spla.LinearOperator(
                (2*Nx*Ny, 2*Nx*Ny),
                matvec=A_matvec_cp,
                dtype=np.float32 if PRECISION == 'float32' else np.float64
            )

            # Convert the RHS base residual (-F_k) to CuPy directly
            F_vec_cp = cp.from_dlpack(-F_vec)
            
            # Execute GMRES natively on the GPU via CuPy
            delta_vel_cp, info = cupy_spla.gmres(
                JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter
            )
            
            # Convert the resulting perturbation vector back to JAX directly
            delta_vel = jax_dlpack.from_dlpack(delta_vel_cp)

            if verbose and info > 0:
                print(f"  CuPy GMRES failed to converge: info={info}")

            # ---- start the Backtracking -----
            # Flatten natively on the GPU array (no np.array conversions!)
            delta_u, delta_v = flattenJnp(delta_vel, Nx, Ny)
            alpha = 1.0
            
            for ls in range(maxBackTrackingIter):
                u_try = u_k + alpha * delta_u
                v_try = v_k + alpha * delta_v
                
                u_try = apply_BC(u_try, BC_X, BC_Y)
                v_try = apply_BC(v_try, BC_X, BC_Y)
                
                lap_u_try = laplacian_jit(u_try, dx, dy)
                lap_v_try = laplacian_jit(v_try, dx, dy)
                adv_u_try = advection_jit(u_try, v_try, dx, dy)
                adv_v_try = advection_jit(v_try, u_try, dx, dy)
                
                F_u_try = constructF_jit(u_try, u_old, adv_u_try, lap_u_try, dt, nu)
                F_v_try = constructF_jit(v_try, v_old, adv_v_try, lap_v_try, dt, nu)
                
                F_try_norm = float(jnp.linalg.norm(concatenateJnp(F_u_try, F_v_try)))
                
                if F_try_norm < res_norm:
                    if verbose:
                        print(f"    Line search accepted alpha={alpha:.3f}, new ||F||={F_try_norm:.3e}")
                    u_k, v_k = u_try, v_try
                    break
                else:
                    alpha *= 0.5
            else:
                if verbose:
                    print(f"    Line search failed to find a descent step. Forcing update.")
                u_k, v_k = u_try, v_try

            newton_end_time = time.perf_counter()
            time_per_newton_iter.append(newton_end_time - newton_start_time)
            
        else:
            newton_iters_per_step.append(NewtonIter)

        u, v = u_k, v_k
        energy_history.append(kinetic_energy(u, v, dx, dy))
        
        step_end_time = time.perf_counter()
        time_per_time_step.append(step_end_time - step_start_time)

        if (step + 1) % plot_steps == 0:
            update_plot(img_u, img_v, img_mag, ax_u, ax_v, ax_mag, u, v, step + 1, clims, bc_label, displayPlot)
            gif_frames.append(capture_frame(fig))

    print("\nDone!")
    t_wall_end = time.perf_counter()

    arr_newton_iters = np.array(newton_iters_per_step)
    arr_time_newton = np.array(time_per_newton_iter)
    arr_time_step = np.array(time_per_time_step)

    lin_type = "Automatic Differentiation" if useAD else "Finite Difference"

    summary_text = f"""
{"="*50}
 SOLVER PERFORMANCE SUMMARY
{"="*50}
--- Simulation Options 
  Linearization : {lin_type}
  Precision     : {PRECISION}
  BC on x       : {BC_X}
  BC on y       : {BC_Y}
  Simulation    : {SIMULATION_IC}
  Grid          : ({Nx}, {Ny})
  Viscosity     : {nu}
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

    print("Saving energy plot")
    plot_energy(energy_history, dt_history, gif_path)

    print("Saving the GIF")
    save_gif(gif_frames, path=gif_path, fps=gif_fps)

    plt.ioff()
    if displayPlot:
        plt.show()

# Launch simulation 
if __name__ == "__main__":
    
    # ---- Simulation Configuration ---- #
    SIMULATION_IC   = 'DSL'           # 'TGV' (Taylor-Green Vortex), 'DSL' (Double Shear Layer), or '4VC' (4-Vortex Collision)      
    PRECISION       = 'float32'
    BC_X            = PERIODIC
    BC_Y            = PERIODIC
    
    # ---- Physical Parameters ---- #
    nu              = 0.05
    steps           = 400
    Nx, Ny          = 512, 512
    Courant         = 0.7

    # ---- Burgers Solver ---- #
    useAD               = False                  # True -> AD Jacobian, False -> FD Jacobian
    verbose             = False
    maxBackTrackingIter = 15                
    
    # ---- Solver Tolerances ---- #  
    if PRECISION == 'float32':
        KrylovTol       = 1e-3
        KrylovIter      = int(1e2)
        NewtonNonlinTol = 1e-3
        NewtonIter      = int(15)
    elif PRECISION == 'float64':  
        KrylovTol       = 1e-6
        KrylovIter      = int(1e2)
        NewtonNonlinTol = 1e-6
        NewtonIter      = int(15)
    else:
        raise ValueError('Choose different Precision')
    
    # ---- Plotting + I/O ---- #
    plot_steps      = 20
    gif_fps         = 10
    displayPlot     = True
    figFolder       = "output"


    runSimulation(PRECISION, 
                  BC_X, 
                  BC_Y, 
                  SIMULATION_IC, 
                  verbose, 
                  useAD, 
                  maxBackTrackingIter,
                  nu, 
                  steps, 
                  Nx, 
                  Ny, 
                  Courant, 
                  KrylovTol, 
                  KrylovIter, 
                  NewtonNonlinTol, 
                  NewtonIter,
                  plot_steps, 
                  gif_fps, 
                  displayPlot, 
                  figFolder)
