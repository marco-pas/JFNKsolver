import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- Configuration ----
SIMULATION_IC = 'TGV'
BASE_NX, BASE_NY = 1024, 1024
BASE_DIR = f"data/{SIMULATION_IC}_{BASE_NX}_{BASE_NY}_AD_float64"

COARSE_GRIDS = [128, 256, 512]
LIN_TYPES = ['AD', 'FD']
PRECISIONS = ['float32', 'float64']

# Configure plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})

# ---- 1. Discover all time steps dynamically ----
base_u_files = sorted(glob.glob(os.path.join(BASE_DIR, "u_*.npy")))

if not base_u_files:
    raise FileNotFoundError(f"Could not find baseline files in {BASE_DIR}")

step_ids = [os.path.basename(f).replace('.npy', '').split('_')[-1] for f in base_u_files]
num_steps = len(step_ids)
print(f"Found {num_steps} time steps to analyze.")

# ---- 2. Initialize Data Storage ----
errors = {}
for Nx in COARSE_GRIDS:
    errors[Nx] = {}
    for lin in LIN_TYPES:
        for prec in PRECISIONS:
            config_name = f"{lin}_{prec}"
            errors[Nx][config_name] = {
                'L2': np.zeros(num_steps),
                'Linf': np.zeros(num_steps)
            }

# ---- 3. Main Data Processing Loop ----
for t_idx, step_id in enumerate(step_ids):
    print(f"Processing step {step_id} ({t_idx+1}/{num_steps})...")
    
    # Robustly load baseline arrays
    base_u_path = os.path.join(BASE_DIR, f"u_{step_id}.npy")
    if not os.path.exists(base_u_path):
        base_u_path = os.path.join(BASE_DIR, f"u_{BASE_NX}_{BASE_NY}_{step_id}.npy")
        
    base_v_path = os.path.join(BASE_DIR, f"v_{step_id}.npy")
    if not os.path.exists(base_v_path):
        base_v_path = os.path.join(BASE_DIR, f"v_{BASE_NX}_{BASE_NY}_{step_id}.npy")
        
    u_base = np.load(base_u_path)
    v_base = np.load(base_v_path)
    
    for Nx in COARSE_GRIDS:
        Ny = Nx 
        factor = BASE_NX // Nx  # Downsampling factor
        
        # Exact pointwise sub-sampling of the baseline grid
        u_base_sub = u_base[::factor, ::factor]
        v_base_sub = v_base[::factor, ::factor]
        
        for lin in LIN_TYPES:
            for prec in PRECISIONS:
                config_name = f"{lin}_{prec}"
                target_dir = f"data/{SIMULATION_IC}_{Nx}_{Ny}_{lin}_{prec}"
                
                # Robustly load the coarse simulation files
                coarse_u_path = os.path.join(target_dir, f"u_{step_id}.npy")
                coarse_v_path = os.path.join(target_dir, f"v_{step_id}.npy")
                
                if not os.path.exists(coarse_u_path):
                    coarse_u_path = os.path.join(target_dir, f"u_{Nx}_{Ny}_{step_id}.npy")
                    coarse_v_path = os.path.join(target_dir, f"v_{Nx}_{Ny}_{step_id}.npy")

                try:
                    u_coarse = np.load(coarse_u_path)
                    v_coarse = np.load(coarse_v_path)
                except FileNotFoundError:
                    print(f"    [WARNING] Missing file! Looked for: {coarse_u_path}")
                    errors[Nx][config_name]['L2'][t_idx] = np.nan
                    errors[Nx][config_name]['Linf'][t_idx] = np.nan
                    continue
                
                # Compute Error Fields
                e_u = u_coarse - u_base_sub
                e_v = v_coarse - v_base_sub
                
                # Compute Norms
                linf = float(max(np.max(np.abs(e_u)), np.max(np.abs(e_v))))
                l2 = float(np.sqrt(np.sum(e_u**2 + e_v**2) / (Nx * Ny)))
                
                # Store Error Values
                errors[Nx][config_name]['Linf'][t_idx] = linf
                errors[Nx][config_name]['L2'][t_idx] = l2


# ---- 4. Plotting All Grids Together ----
time_axis = np.arange(1, num_steps + 1)

# Color by grid size
grid_colors = {128: '#4361EE', 256: '#F72585', 512: '#7209B7'}

# Linestyle by configuration
config_styles = {
    'AD_float64': '-',    # Solid
    'FD_float64': '--',   # Dashed
    'AD_float32': ':',    # Dotted
    'FD_float32': '-.'    # Dash-dot
}

os.makedirs("data", exist_ok=True)
print("\nGenerating combined plot...")

# Create a single figure with 1 row, 2 columns
fig, (ax_l2, ax_linf) = plt.subplots(1, 2, figsize=(12, 4))
# fig.suptitle("Error Norms vs Baseline (1024x1024) for All Grids", fontsize=16, fontweight='bold')

for Nx in COARSE_GRIDS:
    for lin in LIN_TYPES:
        for prec in PRECISIONS:
            config_name = f"{lin}_{prec}"
            
            # Label format: "128x128 AD float64"
            label_name = f"{Nx}x{Nx} {lin} {prec}"
            
            # Plot L2
            ax_l2.plot(50 * time_axis, errors[Nx][config_name]['L2'], 
                       linestyle=config_styles[config_name], 
                       color=grid_colors[Nx], 
                       label=label_name, linewidth=2, alpha=0.8)
            
            # Plot Linf
            ax_linf.plot(50 * time_axis, errors[Nx][config_name]['Linf'], 
                         linestyle=config_styles[config_name], 
                         color=grid_colors[Nx], 
                         label=label_name, linewidth=2, alpha=0.8)

# Formatting L2 subplot
ax_l2.set_title("$L_2$ Error Norm")
ax_l2.set_xlabel("Time Steps")
ax_l2.set_ylabel("$L_2$ Error")
ax_l2.set_ylim(0.7e-2, 1.15)
ax_l2.set_yscale("log")
# Put legend outside the plot box to avoid overlapping the lines
# ax_l2.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))

# Formatting Linf subplot
ax_linf.set_title("$L_\\infty$ Error Norm")
ax_linf.set_xlabel("Time Steps")
ax_linf.set_ylabel("$L_\\infty$ Error")
ax_linf.set_yscale("log")
ax_linf.set_ylim(0.7e-2, 1.15)
ax_linf.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))

# Adjust layout to make room for the legends on the right
plt.tight_layout()

# Save the combined figure
save_path = "data/error_convergence_all_grids.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close(fig) 

print(f"  -> Saved: {save_path}")
print("Analysis complete.")
