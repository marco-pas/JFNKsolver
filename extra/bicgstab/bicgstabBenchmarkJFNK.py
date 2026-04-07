import os
import sys
import time
import argparse
import json
import subprocess
import numpy as np

# --- JAX & Solvers (Imported safely based on worker/orchestrator mode) ---
try:
    import jax
    import jax.numpy as jnp
    from functools import partial as ft_partial
    from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
    from scipy.sparse.linalg import bicgstab as bicgstab_scipy
except ImportError:
    pass # Let it fail gracefully if running in a weird env

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cupy_spla
    from jax import dlpack as jax_dlpack
    from bicgstabCuPy import bicgstab as bicgstab_cupy
    HAS_GPU_LIBS = True
except ImportError:
    HAS_GPU_LIBS = False

# --- Constants & Helpers ---
DIRICHLET = 'dirichlet'
PERIODIC  = 'periodic'

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
    return u.astype(dtype), v.astype(dtype)

def apply_BC(field, bc_x, bc_y):
    if bc_x == DIRICHLET:
        field = field.at[0,  :].set(0.0)
        field = field.at[-1, :].set(0.0)
    if bc_y == DIRICHLET:
        field = field.at[:,  0].set(0.0)
        field = field.at[:, -1].set(0.0)
    return field

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

class KrylovCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, *args, **kwargs):
        self.niter += 1

# --- Worker Function ---
def run_worker(device, Nx, Ny, sim_type):
    if device == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')

    PRECISION = 'float32'
    dtype = jnp.float32
    BC_X, BC_Y = PERIODIC, PERIODIC
    nu = 0.05
    steps = 10
    Courant = 0.7
    NewtonNonlinTol = 1e-3
    NewtonIter = 15
    KrylovTol = 1e-3
    KrylovIter = 100
    maxBackTrackingIter = 15

    xi, xf, yi, yf = 0.0, 2*np.pi, 0.0, 2*np.pi
    x = jnp.linspace(xi, xf, Nx, endpoint=False).astype(dtype)
    y = jnp.linspace(yi, yf, Ny, endpoint=False).astype(dtype)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    u0_raw, v0_raw = get_initial_conditions(X, Y, dtype, sim_type)
    u, v = apply_BC(u0_raw, BC_X, BC_Y), apply_BC(v0_raw, BC_X, BC_Y)

    laplacian_jit  = jax.jit(ft_partial(laplacian,  bc_x=BC_X, bc_y=BC_Y))
    advection_jit  = jax.jit(ft_partial(advection,  bc_x=BC_X, bc_y=BC_Y))
    constructF_jit = jax.jit(constructF)

    time_per_time_step = []
    krylov_iters_per_newton = []

    t_wall_start = time.perf_counter()

    for step in range(steps):
        step_start_time = time.perf_counter()
        dt = calc_dt(u, v, dx, dy, nu, Courant)
        u_old, v_old = u, v
        u_k, v_k = u, v

        for k in range(NewtonIter):
            lap_u_k = laplacian_jit(u_k, dx, dy)
            lap_v_k = laplacian_jit(v_k, dx, dy)
            adv_u_k = advection_jit(u_k, v_k, dx, dy)          
            adv_v_k = advection_jit(v_k, u_k, dx, dy)          

            F_u_k = constructF_jit(u_k, u_old, adv_u_k, lap_u_k, dt, nu)
            F_v_k = constructF_jit(v_k, v_old, adv_v_k, lap_v_k, dt, nu)
            F_vec = concatenateJnp(F_u_k, F_v_k)

            res_norm = float(jnp.linalg.norm(F_vec))
            if res_norm <= NewtonNonlinTol:
                break

            def A_matvec_jax(vec):
                return JacobianActionAD_jit(u_k, v_k, u_old, v_old, vec, dt, nu, dx, dy, BC_X, BC_Y, Nx, Ny)

            krylov_counter = KrylovCounter()

            if device == 'cpu':
                def A_matvec_scipy(vec_np):
                    vec_jax = jnp.array(vec_np, dtype=dtype)
                    return np.asarray(A_matvec_jax(vec_jax)).copy()

                JLinearOp = ScipyLinearOperator((2*Nx*Ny, 2*Nx*Ny), matvec=A_matvec_scipy, dtype=np.float32)
                delta_vel_np, info = bicgstab_scipy(JLinearOp, -np.asarray(F_vec).copy(), rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter)
                delta_vel = jnp.array(delta_vel_np, dtype=dtype)

            elif device == 'gpu':
                def A_matvec_cp(vec_cp):
                    vec_jax = jax_dlpack.from_dlpack(vec_cp)
                    return cp.from_dlpack(A_matvec_jax(vec_jax))

                JLinearOp = cupy_spla.LinearOperator((2*Nx*Ny, 2*Nx*Ny), matvec=A_matvec_cp, dtype=np.float32)
                F_vec_cp = cp.from_dlpack(-F_vec)
                delta_vel_cp, info = bicgstab_cupy(JLinearOp, F_vec_cp, rtol=KrylovTol, maxiter=KrylovIter, callback=krylov_counter)
                delta_vel = jax_dlpack.from_dlpack(delta_vel_cp)

            krylov_iters_per_newton.append(krylov_counter.niter)

            # Backtracking
            delta_u, delta_v = flattenJnp(delta_vel, Nx, Ny)
            alpha = 1.0
            
            for ls in range(maxBackTrackingIter):
                u_try = apply_BC(u_k + alpha * delta_u, BC_X, BC_Y)
                v_try = apply_BC(v_k + alpha * delta_v, BC_X, BC_Y)
                
                lap_u_try = laplacian_jit(u_try, dx, dy)
                lap_v_try = laplacian_jit(v_try, dx, dy)
                adv_u_try = advection_jit(u_try, v_try, dx, dy)
                adv_v_try = advection_jit(v_try, u_try, dx, dy)
                
                F_u_try = constructF_jit(u_try, u_old, adv_u_try, lap_u_try, dt, nu)
                F_v_try = constructF_jit(v_try, v_old, adv_v_try, lap_v_try, dt, nu)
                
                if float(jnp.linalg.norm(concatenateJnp(F_u_try, F_v_try))) < res_norm:
                    u_k, v_k = u_try, v_try
                    break
                alpha *= 0.5
            else:
                u_k, v_k = u_try, v_try

        u, v = u_k, v_k
        time_per_time_step.append(time.perf_counter() - step_start_time)

    t_wall_end = time.perf_counter()

    # Calculate metrics, ignore step 0 (JIT compilation overhead)
    avg_step_time = float(np.mean(time_per_time_step[1:])) if len(time_per_time_step) > 1 else 0.0
    avg_krylov = float(np.mean(krylov_iters_per_newton)) if krylov_iters_per_newton else 0.0

    result = {
        "device": device,
        "Nx": Nx,
        "sim_type": sim_type,
        "wall_time": float(t_wall_end - t_wall_start),
        "avg_step_time": avg_step_time, # Excludes JIT compile
        "avg_krylov_iters": avg_krylov
    }
    
    # Must prefix with BENCHMARK_RESULT so orchestrator can find it
    print(f"BENCHMARK_RESULT:{json.dumps(result)}")


# --- Orchestrator ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', help="Internal flag to run as worker")
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--ic', type=str, default='TGV')
    args = parser.parse_args()

    if args.worker:
        run_worker(args.device, args.N, args.N, args.ic)
        sys.exit(0)

    # Orchestrator Loop
    grids = [128, 256, 1024]
    ics = ['TGV', '4VC', 'DSL']
    devices = ['cpu', 'gpu']
    
    results_db = []

    print(f"{'='*65}")
    print(f" JFNK BiCGSTAB Benchmark (AD only, 10 steps, float32)")
    print(f"{'='*65}")
    print(f"{'Grid':<10} | {'Case':<5} | {'CPU Wall (s)':<12} | {'GPU Wall (s)':<12} | {'Speedup':<8}")
    print("-" * 65)

    for N in grids:
        for ic in ics:
            pair_results = {}
            for dev in devices:
                cmd = [sys.executable, __file__, '--worker', '--device', dev, '--N', str(N), '--ic', ic]
                res = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse output
                for line in res.stdout.split('\n'):
                    if line.startswith("BENCHMARK_RESULT:"):
                        data = json.loads(line.replace("BENCHMARK_RESULT:", ""))
                        pair_results[dev] = data

                if res.returncode != 0:
                    print(f"Error running {dev} {N}x{N} {ic}:\n{res.stderr}")

            if 'cpu' in pair_results and 'gpu' in pair_results:
                cpu_time = pair_results['cpu']['avg_step_time']
                gpu_time = pair_results['gpu']['avg_step_time']
                
                # We compare avg_step_time because it excludes the JAX JIT compile time
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                print(f"{N}x{N:<10} | {ic:<5} | {cpu_time:<12.4f} | {gpu_time:<12.4f} | {speedup:.2f}x")
            
    print(f"{'='*65}")
    print("Note: Times reported are the Average Time per Outer Step (excluding Step 1 JIT Compilation).")
