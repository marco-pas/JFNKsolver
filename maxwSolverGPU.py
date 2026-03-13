import os
import glob
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

# ----------------- File I/O Helper ----------------
def get_next_prefix(folder: str, base: str = "maxwell") -> str:
    os.makedirs(folder, exist_ok=True)
    existing = glob.glob(os.path.join(folder, f"{base}_*_summary.txt"))
    used = set()
    for f in existing:
        basename = os.path.basename(f)
        try:
            stem = basename.split('_')[1]
            used.add(int(stem))
        except (IndexError, ValueError):
            pass
    for n in range(1, 1000):
        if n not in used:
            return os.path.join(folder, f"{base}_{n:03d}")
    return os.path.join(folder, f"{base}_999")

# -------------- Differential operators (JIT compiled) ---------------
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

# ---------------- Nonlinear permittivity --------------
@jax.jit
def eps_func(Ex, Ey, eps0, chi=0.05):
    offset = jnp.finfo(Ex.real.dtype).eps
    loss_factor = 1.0 - 0.05j
    return eps0 * loss_factor * (1.0 + chi * jnp.sqrt(jnp.abs(Ex)**2 + jnp.abs(Ey)**2 + offset))

# --------------- Source distributions ----------------
def make_source(X, Y, source_type, dtype):
    Nx, Ny = X.shape
    Jx = jnp.zeros((Nx, Ny), dtype=dtype)
    Jy = jnp.zeros((Nx, Ny), dtype=dtype)

    if source_type == 'gaussian_center':
        x0 = float(X.mean()); y0 = float(Y.mean())
        sigma = 0.05 * float(X.max() - X.min())
        G = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2))
        Jx = G.astype(dtype)
    elif source_type == 'dipole':
        x0 = float(X.mean())
        y1, y2 = float(Y.max())/3.0, 2.0*float(Y.max())/3.0
        sigma = 2.0 * (float(X.max() - X.min()) / Nx)
        G1 = jnp.exp(-((X - x0)**2 + (Y - y1)**2) / (2*sigma**2))
        G2 = jnp.exp(-((X - x0)**2 + (Y - y2)**2) / (2*sigma**2))
        Jx = (G1 - G2).astype(dtype)
    else:
        raise ValueError(f"Unknown source type '{source_type}'.")
    return Jx, Jy


# ------------ Complex 2N residual — TRUE PEC BOUNDARIES --------------
@ft_partial(jax.jit, static_argnums=(8, 9))
def residual_TE(state, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    N  = Nx * Ny
    Ex = state[:N].reshape(Nx, Ny)
    Ey = state[N:].reshape(Nx, Ny)

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

    return jnp.concatenate([Fx.ravel(), Fy.ravel()])

# -------- Jacobian-vector products ----------------
@ft_partial(jax.jit, static_argnums=(9, 10))
def JacobianActionAD_jit(state, perturb, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    def res_fn(s):
        return residual_TE(s, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
    _, jvp_result = jax.jvp(res_fn, (state,), (perturb,))
    return jvp_result

@ft_partial(jax.jit, static_argnums=(10, 11))
def JacobianActionFD_jit(state, F0, perturb, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny):
    mach_eps = jnp.finfo(state.real.dtype).eps
    perturb_norm = jnp.linalg.norm(perturb)
    
    # machine-aware tiny from JAX
    safe_norm    = jnp.maximum(perturb_norm, jnp.finfo(state.real.dtype).tiny)
    perturb_unit = perturb / safe_norm

    h = jnp.sqrt(mach_eps) * (1.0 + jnp.linalg.norm(state))

    F_pert = residual_TE(state + h * perturb_unit,
                         omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
    return ((F_pert - F0) / h) * perturb_norm

#  ------------------- Main simulation ---------------
def runSimulation(PRECISION, 
                  SIMULATION_J, 
                  useAD, 
                  verbose,
                  mu0, 
                  eps0, 
                  omega_start, 
                  omega_stop, 
                  omega_steps,
                  Nx, 
                  Ny, 
                  KrylovTol, 
                  KrylovIter,
                  NewtonTol, 
                  NewtonIter,
                  maxBackTrackingIter, 
                  figFolder, 
                  save_field_pic):

    t_wall_start = time.perf_counter()

    if PRECISION == 'float64':
        jax.config.update("jax_enable_x64", True)
        complex_dtype = jnp.complex128
        cupy_dtype    = cp.complex128
    elif PRECISION == 'float32':
        complex_dtype = jnp.complex64
        cupy_dtype    = cp.complex64

    prefix = get_next_prefix(figFolder, "maxwell")
    print(f"Precision: {PRECISION} | Linearization: {'AD' if useAD else 'FD'}")
    print(f"Output files will be saved with prefix: {prefix}")

    Lx, Ly = 1.0, 1.0
    x  = jnp.linspace(0, Lx, Nx)
    y  = jnp.linspace(0, Ly, Ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    Jx, Jy  = make_source(X, Y, SIMULATION_J, complex_dtype)
    omegas  = jnp.linspace(omega_start, omega_stop, omega_steps)
    response = []

    N_pts = Nx * Ny
    c_coupling = 1.0 / (4.0 * dx * dy)

    is_bnd_Ex = jnp.zeros((Nx, Ny), dtype=bool).at[:,  0].set(True).at[:, -1].set(True)
    is_bnd_Ey = jnp.zeros((Nx, Ny), dtype=bool).at[0,  :].set(True).at[-1, :].set(True)
    bnd_mask_cp = cp.asarray(np.array((is_bnd_Ex | is_bnd_Ey).ravel()).astype(
                             np.float64 if PRECISION == 'float64' else np.float32))

    newton_iters_per_step = []
    time_per_newton_iter  = []
    time_per_omega_step   = []
    final_res_norms       = []

    # ----------- FREQUENCY LOOP -----------
    print("\nStarting Frequency Sweep")
    for i, omega in enumerate(omegas):
        omega_val = float(omega)
        print(f"\n{'='*60}")
        print(f"Step {i+1}/{omega_steps} | omega = {omega_val:.4e}")
        omega_start_time = time.perf_counter()

        state = jnp.zeros(2 * Nx * Ny, dtype=complex_dtype)

        # ----------- LINEARIZED INITIAL GUESS (Born Approximation) ----------
        print("  --> Computing linear initial guess...")
        F_lin = residual_TE(state, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
        
        if useAD:
            def A_matvec_jax_lin(vec): return JacobianActionAD_jit(state, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
        else:
            def A_matvec_jax_lin(vec): return JacobianActionFD_jit(state, F_lin, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)

        def A_matvec_cp_lin(vec_cp):
            vec_jax = jax_dlpack.from_dlpack(vec_cp)
            res_jax = A_matvec_jax_lin(vec_jax)
            return cp.from_dlpack(res_jax)

        JLinearOp_lin = cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=A_matvec_cp_lin, dtype=cupy_dtype)

        # ----- Preconditioner evaluated at E=0 --------
        Ex_c_lin  = state[:N_pts].reshape(Nx, Ny)
        Ey_c_lin  = state[N_pts:].reshape(Nx, Ny)
        eps_c_lin = eps_func(Ex_c_lin, Ey_c_lin, eps0)
        cslp_shift = 1.0 - 0.5j
        eps_precond_lin = eps_c_lin * cslp_shift

        diag_Ex_lin = (jnp.full((Nx, Ny), 2.0/dy**2, dtype=complex_dtype) - omega_val**2 * mu0 * eps_precond_lin)
        diag_Ey_lin = (jnp.full((Nx, Ny), 2.0/dx**2, dtype=complex_dtype) - omega_val**2 * mu0 * eps_precond_lin)

        diag_Ex_lin = jnp.where(is_bnd_Ex, 1.0 + 0j, diag_Ex_lin)
        diag_Ey_lin = jnp.where(is_bnd_Ey, 1.0 + 0j, diag_Ey_lin)

        d_Ex_cp_lin = cp.from_dlpack(diag_Ex_lin.ravel())
        d_Ey_cp_lin = cp.from_dlpack(diag_Ey_lin.ravel())
        c_cp_lin    = cp.full(N_pts, c_coupling, dtype=cupy_dtype) * (1.0 - bnd_mask_cp)

        det_cp_lin = d_Ex_cp_lin * d_Ey_cp_lin - c_cp_lin**2
        det_cp_lin = cp.where(cp.abs(det_cp_lin) < 1e-12, 1e-12 + 0j, det_cp_lin)

        # Optimization: Pre-compute inverse elements to minimize GPU kernel calls inside loop
        M_11_lin = d_Ey_cp_lin / det_cp_lin
        M_12_lin = -c_cp_lin / det_cp_lin
        M_22_lin = d_Ex_cp_lin / det_cp_lin

        def M_matvec_cp_lin(vec_cp):
            v_Ex = vec_cp[:N_pts]
            v_Ey = vec_cp[N_pts:]
            out_Ex = M_11_lin * v_Ex + M_12_lin * v_Ey
            out_Ey = M_12_lin * v_Ex + M_22_lin * v_Ey
            return cp.concatenate([out_Ex, out_Ey])

        M_LinearOp_lin = cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=M_matvec_cp_lin, dtype=cupy_dtype)
        F_rhs_cp_lin = cp.from_dlpack(-F_lin)
        
        lin_krylov_tol = max(KrylovTol, 1e-4)
        delta_cp_lin, info_lin = cupy_spla.gmres(JLinearOp_lin, F_rhs_cp_lin, M=M_LinearOp_lin, 
                                                 rtol=lin_krylov_tol, restart=200, maxiter=30)
        
        state = jax_dlpack.from_dlpack(delta_cp_lin)

        initial_res_norm = None
        c1 = 1e-4  

        # -------------- Newton loop -------------
        for k in range(NewtonIter):
            t0_iter = time.perf_counter()

            F        = residual_TE(state, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
            res_norm = float(jnp.linalg.norm(F))

            if initial_res_norm is None:
                initial_res_norm = res_norm + 1e-30

            rel_res = res_norm / initial_res_norm
            print(f"  Newton {k:3d}: ||F|| = {res_norm:.4e}  (rel = {rel_res:.4e})")

            if res_norm <= NewtonTol * initial_res_norm + 1e-14:
                print(f"    Converged in {k} Newton iterations")
                newton_iters_per_step.append(k)
                time_per_newton_iter.append(time.perf_counter() - t0_iter)
                break

            if useAD:
                def A_matvec_jax(vec): return JacobianActionAD_jit(state, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
            else:
                def A_matvec_jax(vec): return JacobianActionFD_jit(state, F, vec, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)

            def A_matvec_cp(vec_cp):
                vec_jax = jax_dlpack.from_dlpack(vec_cp)
                res_jax = A_matvec_jax(vec_jax)
                return cp.from_dlpack(res_jax)

            JLinearOp = cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=A_matvec_cp, dtype=cupy_dtype)

            Ex_c  = state[:N_pts].reshape(Nx, Ny)
            Ey_c  = state[N_pts:].reshape(Nx, Ny)
            eps_c = eps_func(Ex_c, Ey_c, eps0)
            cslp_shift = 1.0 - 0.5j
            eps_precond = eps_c * cslp_shift

            diag_Ex = (jnp.full((Nx, Ny), 2.0/dy**2, dtype=complex_dtype) - omega_val**2 * mu0 * eps_precond)
            diag_Ey = (jnp.full((Nx, Ny), 2.0/dx**2, dtype=complex_dtype) - omega_val**2 * mu0 * eps_precond)

            diag_Ex = jnp.where(is_bnd_Ex, 1.0 + 0j, diag_Ex)
            diag_Ey = jnp.where(is_bnd_Ey, 1.0 + 0j, diag_Ey)

            d_Ex_cp = cp.from_dlpack(diag_Ex.ravel())
            d_Ey_cp = cp.from_dlpack(diag_Ey.ravel())
            c_cp  = cp.full(N_pts, c_coupling, dtype=cupy_dtype) * (1.0 - bnd_mask_cp)

            det_cp = d_Ex_cp * d_Ey_cp - c_cp**2
            det_cp = cp.where(cp.abs(det_cp) < 1e-12, 1e-12 + 0j, det_cp)

            # Optimization: Pre-compute inverse elements to minimize GPU kernel calls inside loop
            M_11 = d_Ey_cp / det_cp
            M_12 = -c_cp / det_cp
            M_22 = d_Ex_cp / det_cp

            def M_matvec_cp(vec_cp):
                v_Ex = vec_cp[:N_pts]
                v_Ey = vec_cp[N_pts:]
                out_Ex = M_11 * v_Ex + M_12 * v_Ey
                out_Ey = M_12 * v_Ex + M_22 * v_Ey
                return cp.concatenate([out_Ex, out_Ey])

            M_LinearOp = cupy_spla.LinearOperator((2*N_pts, 2*N_pts), matvec=M_matvec_cp, dtype=cupy_dtype)

            krylov_tol = max(0.1 * rel_res, KrylovTol)
            F_rhs_cp = cp.from_dlpack(-F)
            delta_cp, info = cupy_spla.gmres(JLinearOp, F_rhs_cp, M=M_LinearOp, rtol=krylov_tol, restart=200, maxiter=30)

            if verbose:
                status = "OK" if info == 0 else f"WARN info={info}"
                print(f"    GMRES [{status}] tol={krylov_tol:.2e}")

            delta = jax_dlpack.from_dlpack(delta_cp)

            # Line search
            alpha = 1.0
            for _ in range(maxBackTrackingIter):
                state_try  = state + alpha * delta
                F_try      = residual_TE(state_try, omega_val, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)
                F_try_norm = float(jnp.linalg.norm(F_try))
                if F_try_norm <= (1.0 - c1 * alpha) * res_norm:
                    state = state_try
                    break
                alpha *= 0.5
            else:
                state = state_try
                
            time_per_newton_iter.append(time.perf_counter() - t0_iter)

        else:
            newton_iters_per_step.append(NewtonIter)
            print(f"    Reached max Newton iterations ({NewtonIter})")

        time_per_omega_step.append(time.perf_counter() - omega_start_time)
        final_res_norms.append(res_norm) 

        Ex_c  = state[:N_pts].reshape(Nx, Ny)
        Ey_c  = state[N_pts:].reshape(Nx, Ny)
        E_mag = jnp.sqrt(jnp.abs(Ex_c)**2 + jnp.abs(Ey_c)**2)
        response.append(float(jnp.max(E_mag)))

        if save_field_pic > 0 and (i + 1) % save_field_pic == 0:
            fig_fields, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            ax_Ex, ax_Ey, ax_f = axes

            # (J: {SIMULATION_J})

            im_Ex = ax_Ex.imshow(np.abs(np.array(Ex_c)).T, origin='lower', cmap='plasma', extent=[0, Lx, 0, Ly])
            im_Ex = ax_Ex.imshow(np.abs(np.array(Ex_c)).T, origin='lower', cmap='plasma', extent=[0, Lx, 0, Ly], vmin=0.0, vmax=0.1)
            # ax_Ex.set_xlabel('x')
            # ax_Ex.set_ylabel('y')
            ax_Ex.axis('off')
            ax_Ex.set_title(rf'$E_x$ at $\omega={omega_val:.2f}$' + '\n' + 'BC: x=PEC y=PEC')
            fig_fields.colorbar(im_Ex, ax=ax_Ex, fraction=0.046, pad=0.04)

            im_Ey = ax_Ey.imshow(np.abs(np.array(Ey_c)).T, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
            # ax_Ey.set_xlabel('x')
            # ax_Ey.set_ylabel('y')
            ax_Ey.axis('off')
            ax_Ey.set_title(rf'$E_y$ at $\omega={omega_val:.2f}$' + '\n' + 'BC: x=PEC y=PEC')
            fig_fields.colorbar(im_Ey, ax=ax_Ey, fraction=0.046, pad=0.04)

            im_f = ax_f.imshow(np.array(E_mag).T, origin='lower', cmap='magma', extent=[0, Lx, 0, Ly])
            # ax_f.set_xlabel('x')
            # ax_f.set_ylabel('y')
            ax_f.axis('off')
            ax_f.set_title(rf'$|\mathbf{{E}}|$ at $\omega={omega_val:.2f}$' + '\n' + 'BC: x=PEC y=PEC')
            fig_fields.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)

            plt.tight_layout()
            pic_path = f"{prefix}_w{i+1}_fields.png"
            fig_fields.savefig(pic_path, dpi=150, bbox_inches='tight')
            plt.close(fig_fields)

            if verbose:
                print(f"    Saved combined field images -> {pic_path}")
    t_wall_end = time.perf_counter()

    # ------- Summary Generation & Saving -------
    arr_newton_iters = np.array(newton_iters_per_step)
    arr_time_newton = np.array(time_per_newton_iter)
    arr_time_omega = np.array(time_per_omega_step)
    arr_res_norms = np.array(final_res_norms)
    
    lin_type = "Automatic Differentiation" if useAD else "Finite Difference"
    
    def safe_pct(std, mean):
        return (std / mean * 100) if mean > 0 else 0.0

    summary_text = f"""
{"="*50}
 SOLVER PERFORMANCE SUMMARY
{"="*50}
--- Simulation Options 
  Linearization : {lin_type}
  Precision     : {PRECISION}
  BC on x       : PEC
  BC on y       : PEC
  Source Type   : {SIMULATION_J}
  Grid          : ({Nx}, {Ny})
  mu0           : {mu0}
  eps0          : {eps0}
  Omega steps   : {omega_steps}
  Newton tol    : {NewtonTol}
  Krylov tol    : {KrylovTol}
  Max BT iters  : {maxBackTrackingIter}

--- Newton Iterations per Omega Step
  Average : {np.mean(arr_newton_iters):.2f}
  Std Dev : {np.std(arr_newton_iters):.2f} ({safe_pct(np.std(arr_newton_iters), np.mean(arr_newton_iters)):.4f}%)
  Max     : {np.max(arr_newton_iters)}
  Min     : {np.min(arr_newton_iters)}

--- Final Residual Norms (||F||)
  Average : {np.mean(arr_res_norms):.4e}
  Std Dev : {np.std(arr_res_norms):.4e} ({safe_pct(np.std(arr_res_norms), np.mean(arr_res_norms)):.4f}%)
  Max     : {np.max(arr_res_norms):.4e}
  Min     : {np.min(arr_res_norms):.4e}

--- Time per Newton Iteration, s
  Average : {np.mean(arr_time_newton):.4f}
  Std Dev : {np.std(arr_time_newton):.4f} ({safe_pct(np.std(arr_time_newton), np.mean(arr_time_newton)):.4f}%)
  Max     : {np.max(arr_time_newton):.4f}
  Min     : {np.min(arr_time_newton):.4f}

--- Time per Outer Omega Step, s
  Average : {np.mean(arr_time_omega):.4f}
  Std Dev : {np.std(arr_time_omega):.4f} ({safe_pct(np.std(arr_time_omega), np.mean(arr_time_omega)):.4f}%)
  Max     : {np.max(arr_time_omega):.4f}
  Min     : {np.min(arr_time_omega):.4f}

--- Overall Time, s
  Total Solver Time       : {np.sum(arr_time_omega):.4f}
  Total Wall Time         : {(t_wall_end - t_wall_start):.4f}
  Frequency Sweep Range   : {omega_start} to {omega_stop}
{"="*50}
"""
    print(summary_text)

    txt_path = f"{prefix}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)
    print(f"Summary saved -> {txt_path}")

    # -------- Final Plots --------
    theoretical_resonances = set()
    max_idx = int(omega_stop / np.pi) + 2
    for m in range(max_idx):
        for n in range(max_idx):
            if m == 0 and n == 0:
                continue
            w_mn = np.pi * np.sqrt(m**2 + n**2)
            if omega_start <= w_mn <= omega_stop:
                theoretical_resonances.add(w_mn)
                
    theoretical_resonances = sorted(list(theoretical_resonances))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.array(omegas), response, color='green', linewidth=2, label='Simulated Response')
    
    for w_res in theoretical_resonances:
        ax.axvline(x=w_res, color='red', linestyle=':', alpha=0.4)
    ax.plot([], [], color='red', linestyle=':', alpha=0.4, label='Vacuum Resonances')
    
    ax.set_xlabel(r'$\omega$ (dimensionless)')
    ax.set_ylabel(r'$\max\,|\mathbf{E}|$')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc="best")
    plt.tight_layout()
    
    resp_path = f"{prefix}_response.png"
    fig.savefig(resp_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Response plot saved -> {resp_path}")

# Launch simulation 
if __name__ == "__main__":

    # ---- Simulation Configuration ---- #
    SIMULATION_J        = 'gaussian_center'      # source term: 'dipole' or 'gaussian_center'
    PRECISION           = 'float64'

    # ---- Physical Parameters ---- #
    mu0                 = 1.0
    eps0                = 1.0
    omega_start         = 4
    omega_stop          = 20
    omega_steps         = 17
    Nx, Ny              = 256, 256

    # ---- Maxwell Solver ---- #
    useAD               = False
    verbose             = False
    maxBackTrackingIter = 15
    
    # ---- Solver Tolerances ---- #  
    if PRECISION == 'float64':
        KrylovTol, KrylovIter = 1e-5, 100
        NewtonTol, NewtonIter = 1e-5, 100
    elif PRECISION == 'float32':
        KrylovTol, KrylovIter = 1e-2, 100
        NewtonTol, NewtonIter = 1e-2, 100
    else:
        raise ValueError('Choose different Precision')
    
     # ---- Plotting + I/O ---- #
    save_field_pic      = 1
    figFolder = "output/maxw"


    runSimulation(
        PRECISION           = PRECISION,
        SIMULATION_J        = SIMULATION_J,
        useAD               = useAD,
        verbose             = verbose,
        mu0                 = mu0,
        eps0                = eps0,
        omega_start         = omega_start,
        omega_stop          = omega_stop,
        omega_steps         = omega_steps,
        Nx                  = Nx,
        Ny                  = Ny,
        KrylovTol           = KrylovTol,
        KrylovIter          = KrylovIter,
        NewtonTol           = NewtonTol,
        NewtonIter          = NewtonIter,
        maxBackTrackingIter = maxBackTrackingIter,
        figFolder           = figFolder,
        save_field_pic      = save_field_pic
    )