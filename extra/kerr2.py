import os
import time
import csv
from functools import partial as ft_partial

import jax
import jax.numpy as jnp
import numpy as np

import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_spla
from jax import dlpack as jax_dlpack

# ============================================================
#  Differential operators
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
#  Nonlinear permittivity
# ============================================================
@jax.jit
def eps_func(Ex, Ey, eps0):
    offset = jnp.finfo(Ex.real.dtype).eps
    loss_factor = 1.0 - 0.05j
    chi = 0.05
    return eps0 * loss_factor * (1.0 + chi * jnp.sqrt(jnp.abs(Ex)**2 + jnp.abs(Ey)**2 + offset))

# ============================================================
#  Source distributions
# ============================================================
def make_source(X, Y, source_type, dtype, amplitude=1.0):
    Nx, Ny = X.shape
    Jx = jnp.zeros((Nx, Ny), dtype=dtype)
    Jy = jnp.zeros((Nx, Ny), dtype=dtype)

    if source_type == 'dipole':
        x0 = float(X.mean())
        y1, y2 = float(Y.max())/3.0, 2.0*float(Y.max())/3.0
        sigma = 2.0 * (float(X.max() - X.min()) / Nx)
        G1 = jnp.exp(-((X - x0)**2 + (Y - y1)**2) / (2*sigma**2))
        G2 = jnp.exp(-((X - x0)**2 + (Y - y2)**2) / (2*sigma**2))
        Jx = amplitude * (G1 - G2).astype(dtype) 
    else:
        raise ValueError(f"Unknown source type '{source_type}'.")
    return Jx, Jy

# ============================================================
#  Residual and Jacobian
# ============================================================
@ft_partial(jax.jit, static_argnums=(8, 9))
def residual_TE(state, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    N  = Nx * Ny
    Ex = state[:N].reshape(Nx, Ny)
    Ey = state[N:].reshape(Nx, Ny)

    eps = eps_func(Ex, Ey, eps0)

    Fx = Dxy_op(Ey, dx, dy) - Dyy_op(Ex, dy) - omega**2 * mu0 * eps * Ex - 1j * omega * mu0 * Jx 
    Fy = Dxy_op(Ex, dx, dy) - Dxx_op(Ey, dx) - omega**2 * mu0 * eps * Ey - 1j * omega * mu0 * Jy 

    Fx = Fx.at[:, 0 ].set(Ex[:, 0 ]); Fx = Fx.at[:, -1].set(Ex[:, -1])   
    Fy = Fy.at[0,  :].set(Ey[0,  :]); Fy = Fy.at[-1, :].set(Ey[-1, :])   

    return jnp.concatenate([Fx.ravel(), Fy.ravel()])

@ft_partial(jax.jit, static_argnums=(9, 10))
def JacobianActionAD_jit(state, perturb, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    def res_fn(s): return residual_TE(s, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
    _, jvp_result = jax.jvp(res_fn, (state,), (perturb,))
    return jvp_result

# ============================================================
#  Simulation Engine
# ============================================================
def run_single_sweep(J_amplitude, omega_start, omega_stop, omega_steps, Nx, Ny):
    
    mu0, eps0 = 1.0, 1.0
    complex_dtype = jnp.complex128
    cupy_dtype    = cp.complex128

    Lx, Ly = 1.0, 1.0
    x, y  = jnp.linspace(0, Lx, Nx), jnp.linspace(0, Ly, Ny)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    Jx, Jy  = make_source(X, Y, 'dipole', complex_dtype, amplitude=J_amplitude)
    omegas  = jnp.linspace(omega_start, omega_stop, omega_steps)
    response = []

    N_pts = Nx * Ny
    c_coupling = 1.0 / (4.0 * dx * dy)

    is_bnd_Ex = jnp.zeros((Nx, Ny), dtype=bool).at[:,  0].set(True).at[:, -1].set(True)
    is_bnd_Ey = jnp.zeros((Nx, Ny), dtype=bool).at[0,  :].set(True).at[-1, :].set(True)
    bnd_mask_cp = cp.asarray(np.array((is_bnd_Ex | is_bnd_Ey).ravel()).astype(np.float64))

    for i, omega in enumerate(omegas):
        print(f"Step {i+1}/{omega_steps}, omega = {omega:.4f}")

        omega_val = float(omega)
        state = jnp.zeros(2 * Nx * Ny, dtype=complex_dtype)

        # Linear initial guess
        F_lin = residual_TE(state, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
        
        def A_matvec_jax_lin(vec): return JacobianActionAD_jit(state, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
        def A_matvec_cp_lin(vec_cp): return cp.from_dlpack(A_matvec_jax_lin(jax_dlpack.from_dlpack(vec_cp)))
        JLinearOp_lin = cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=A_matvec_cp_lin, dtype=cupy_dtype)

        eps_c_lin = eps_func(state[:N_pts].reshape(Nx, Ny), state[N_pts:].reshape(Nx, Ny), eps0)
        eps_precond_lin = eps_c_lin * (1.0 - 0.5j)

        diag_Ex_lin = jnp.where(is_bnd_Ex, 1.0+0j, (2.0/dy**2) - omega_val**2 * mu0 * eps_precond_lin)
        diag_Ey_lin = jnp.where(is_bnd_Ey, 1.0+0j, (2.0/dx**2) - omega_val**2 * mu0 * eps_precond_lin)

        d_Ex_cp_lin, d_Ey_cp_lin = cp.from_dlpack(diag_Ex_lin.ravel()), cp.from_dlpack(diag_Ey_lin.ravel())
        c_cp_lin = cp.full(N_pts, c_coupling, dtype=cupy_dtype) * (1.0 - bnd_mask_cp)

        det_cp_lin = cp.where(cp.abs(d_Ex_cp_lin * d_Ey_cp_lin - c_cp_lin**2) < 1e-12, 1e-12+0j, d_Ex_cp_lin * d_Ey_cp_lin - c_cp_lin**2)
        M_11_lin, M_12_lin, M_22_lin = d_Ey_cp_lin / det_cp_lin, -c_cp_lin / det_cp_lin, d_Ex_cp_lin / det_cp_lin

        def M_matvec_cp_lin(vec_cp):
            return cp.concatenate([M_11_lin * vec_cp[:N_pts] + M_12_lin * vec_cp[N_pts:], M_12_lin * vec_cp[:N_pts] + M_22_lin * vec_cp[N_pts:]])

        delta_cp_lin, _ = cupy_spla.gmres(JLinearOp_lin, cp.from_dlpack(-F_lin), M=cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=M_matvec_cp_lin, dtype=cupy_dtype), rtol=1e-4, restart=200, maxiter=20)
        state = jax_dlpack.from_dlpack(delta_cp_lin)

        # Newton Loop
        for k in range(40):
            F = residual_TE(state, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
            res_norm = float(jnp.linalg.norm(F))
            
            if k == 0: initial_res_norm = res_norm + 1e-30
            if res_norm <= 1e-5 * initial_res_norm + 1e-14: break

            def A_matvec_jax(vec): return JacobianActionAD_jit(state, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
            def A_matvec_cp(vec_cp): return cp.from_dlpack(A_matvec_jax(jax_dlpack.from_dlpack(vec_cp)))
            JLinearOp = cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=A_matvec_cp, dtype=cupy_dtype)

            eps_c = eps_func(state[:N_pts].reshape(Nx, Ny), state[N_pts:].reshape(Nx, Ny), eps0)
            eps_precond = eps_c * (1.0 - 0.5j)

            diag_Ex = jnp.where(is_bnd_Ex, 1.0+0j, (2.0/dy**2) - omega_val**2 * mu0 * eps_precond)
            diag_Ey = jnp.where(is_bnd_Ey, 1.0+0j, (2.0/dx**2) - omega_val**2 * mu0 * eps_precond)

            d_Ex_cp, d_Ey_cp = cp.from_dlpack(diag_Ex.ravel()), cp.from_dlpack(diag_Ey.ravel())
            det_cp = cp.where(cp.abs(d_Ex_cp * d_Ey_cp - c_cp_lin**2) < 1e-12, 1e-12+0j, d_Ex_cp * d_Ey_cp - c_cp_lin**2)
            M_11, M_12, M_22 = d_Ey_cp / det_cp, -c_cp_lin / det_cp, d_Ex_cp / det_cp

            def M_matvec_cp(vec_cp):
                return cp.concatenate([M_11 * vec_cp[:N_pts] + M_12 * vec_cp[N_pts:], M_12 * vec_cp[:N_pts] + M_22 * vec_cp[N_pts:]])

            krylov_tol = max(0.1 * (res_norm / initial_res_norm), 1e-5)
            delta_cp, _ = cupy_spla.gmres(JLinearOp, cp.from_dlpack(-F), M=cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=M_matvec_cp, dtype=cupy_dtype), rtol=krylov_tol, restart=200, maxiter=20)

            # Line search
            delta = jax_dlpack.from_dlpack(delta_cp)
            alpha = 1.0
            for _ in range(15):
                state_try = state + alpha * delta
                if float(jnp.linalg.norm(residual_TE(state_try, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny))) <= (1.0 - 1e-4 * alpha) * res_norm:
                    state = state_try; break
                alpha *= 0.5
            else: state = state_try

        # --- EXTRACT REQUESTED DATA METRICS ---
        Ex = state[:N_pts]
        Ey = state[N_pts:]

        # 1. Max absolute value of the electric field
        E_mag = jnp.sqrt(jnp.abs(Ex)**2 + jnp.abs(Ey)**2)
        max_abs_E = float(jnp.max(E_mag))

        # 2. Max absolute value of the real/imaginary parts on X
        max_abs_real_Ex = float(jnp.max(jnp.abs(jnp.real(Ex))))
        max_abs_imag_Ex = float(jnp.max(jnp.abs(jnp.imag(Ex))))

        # 3. Max absolute value of the real/imaginary parts on Y
        max_abs_real_Ey = float(jnp.max(jnp.abs(jnp.real(Ey))))
        max_abs_imag_Ey = float(jnp.max(jnp.abs(jnp.imag(Ey))))

        # Store the metrics for this frequency
        response.append([
            omega_val, 
            max_abs_E, 
            max_abs_real_Ex, 
            max_abs_imag_Ex, 
            max_abs_real_Ey, 
            max_abs_imag_Ey
        ])

    return response

# ============================================================
#  Data Collection & CSV Export
# ============================================================
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    
    # Simulation Parameters
    omega_start, omega_stop, omega_steps = 0.3, 2.99, 150  # Note: 1 step means it only tests 4.0. Adjust omega_steps for a full sweep.
    Nx, Ny = 128, 128
    amplitudes = [0.1]
    
    # File setup
    os.makedirs('output', exist_ok=True)
    csv_filename = 'output/kerr_sweep_data.csv'

    print("Starting Data Extraction Protocol...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write CSV Header
        writer.writerow([
            'Source_Amplitude', 
            'Frequency_omega', 
            'Max_Abs_E', 
            'Max_Abs_Real_Ex', 
            'Max_Abs_Imag_Ex', 
            'Max_Abs_Real_Ey', 
            'Max_Abs_Imag_Ey'
        ])
        
        for amp in amplitudes:
            print(f"\nRunning sweep for Source Amplitude J = {amp}...")
            sweep_data = run_single_sweep(amp, omega_start, omega_stop, omega_steps, Nx, Ny)
            
            # Write data row by row, prepending the current amplitude
            for row in sweep_data:
                writer.writerow([amp] + row)

    print(f"\nSUCCESS! CSV exported successfully to {csv_filename}")
