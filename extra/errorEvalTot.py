import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- Configuration ----
SIMULATION_IC = 'TGV'
BASE_NX, BASE_NY = 1024, 1024
BASE_DIR = f"data/{SIMULATION_IC}_{BASE_NX}_{BASE_NY}_AD_float64"

# Added 64 here
COARSE_GRIDS = [64, 128, 256, 512]
LIN_TYPES = ['AD', 'FD']
PRECISIONS = ['float64', 'float32']

# Configure plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})

# ---- 1. Discover all time steps dynamically ----
base_u_files = glob.glob(os.path.join(BASE_DIR, "u_*.npy"))

if not base_u_files:
    raise FileNotFoundError(f"Could not find baseline files in {BASE_DIR}")

# Extract integer step IDs and sort them numerically
step_ids = []
for f in base_u_files:
    step_num = int(os.path.basename(f).replace('.npy', '').replace('u_', ''))
    step_ids.append(step_num)

step_ids = sorted(step_ids)
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
        factor = BASE_NX // Nx
        
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
                
                # --- Compute Relative Error Fields ---
                e_u = u_coarse - u_base_sub
                e_v = v_coarse - v_base_sub
                
                # 1. Compute Norm of the Absolute Error
                abs_l2_num = np.sum(e_u**2 + e_v**2)
                abs_linf_num = max(np.max(np.abs(e_u)), np.max(np.abs(e_v)))
                
                # 2. Compute Norm of the Baseline
                base_l2_den = np.sum(u_base_sub**2 + v_base_sub**2)
                base_linf_den = max(np.max(np.abs(u_base_sub)), np.max(np.abs(v_base_sub)))
                
                # 3. Compute Relative Norms
                rel_l2 = float(np.sqrt(abs_l2_num / (base_l2_den + 1e-16)))
                rel_linf = float(abs_linf_num / (base_linf_den + 1e-16))
                
                # Store Error Values
                errors[Nx][config_name]['Linf'][t_idx] = rel_linf
                errors[Nx][config_name]['L2'][t_idx] = rel_l2

# ---- 4. Plotting All Grids Together (Time Evolution) ----
time_axis = np.array([
    0.03373399941778589, 0.06746799883557178, 0.10120199825335767,
    0.13493599767114356, 0.16866999708892944, 0.2024039965067153,
    0.2361379959245012, 0.26987199534228706, 0.30360599476007294,
    0.3373399941778588, 0.3710739935956447, 0.40480799301343057,
    0.43854199243121644, 0.4722759918490023, 0.5060099912667883,
    0.5397439906845741, 0.57347799010236, 0.6072119895201459,
    0.6409459889379318, 0.6746799883557176, 0.7084139877735035,
    0.7421479871912894, 0.7758819866090753, 0.8096159860268611,
    0.843349985444647
])

# TWEAK: Added 64 grid size mapping
grid_colors = {
    512: {'float64': '#4361EE', 'float32': '#4CC9F0'}, 
    256: {'float64': '#F72585', 'float32': '#FF99C8'}, 
    128: {'float64': '#7209B7', 'float32': '#9D4EDD'},  
    64:  {'float64': '#F8961E', 'float32': '#F9C74F'}   # Dark Orange vs Yellow
}

config_styles = {
    'AD_float64': {'ls': '-',  'marker': '^'}, 
    'FD_float64': {'ls': '-.', 'marker': 'o'}, 
    'AD_float32': {'ls': '-',  'marker': '^'}, 
    'FD_float32': {'ls': '-',  'marker': 'o'}  
}

os.makedirs("data", exist_ok=True)
print("\nGenerating time evolution plot...")

fig, (ax_l2, ax_linf) = plt.subplots(1, 2, figsize=(12, 4))

for Nx in COARSE_GRIDS:
    for lin in LIN_TYPES:
        for prec in PRECISIONS:
            config_name = f"{lin}_{prec}"
            label_name = f"{Nx}x{Nx} {lin} {prec}"
            
            style = config_styles[config_name]
            current_color = grid_colors[Nx][prec]
            
            ax_l2.plot(time_axis * 1e3, errors[Nx][config_name]['L2'], 
                       linestyle=style['ls'], marker=style['marker'],
                       markersize=6, markevery=2, color=current_color, 
                       label=label_name, linewidth=2.5, alpha=0.8)
            
            ax_linf.plot(time_axis * 1e3, errors[Nx][config_name]['Linf'], 
                         linestyle=style['ls'], marker=style['marker'],
                         markersize=6, markevery=2, color=current_color, 
                         label=label_name, linewidth=2.5, alpha=0.8)

# Formatting L2 subplot
# ax_l2.set_title("Relative $L_2$ Error Over Time")
ax_l2.set_xlabel("Physical Time, ms")
ax_l2.set_ylabel("Relative $L_2$ Error")
ax_l2.set_yscale("log")
ax_l2.set_xscale("log")

# Formatting Linf subplot
# ax_linf.set_title("Relative $L_\\infty$ Error Over Time")
ax_linf.set_xlabel("Physical Time, ms")
ax_linf.set_ylabel("Relative $L_\\infty$ Error")
ax_linf.set_yscale("log")
ax_linf.set_xscale("log")

ax_linf.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()

save_path_time = "data/timeErrorBurgers.pdf"
plt.savefig(save_path_time, dpi=150, bbox_inches='tight')
plt.close(fig) 
print(f"  -> Saved: {save_path_time}")


# ---- 5. Plotting Spatial Grid Convergence (Final Time Step) ----

print("\nGenerating spatial convergence plot...")

config_convergence_colors = {
    'AD_float64': "#002FFF", # Blue
    'FD_float64': '#F72585', # Pink
    'AD_float32': "#FADD00", # Yellow
    'FD_float32': "#13A709"  # Green
}

# TWEAK: Define descending sizes and alternating hollow/filled markers
# zorder ensures the thinnest/smallest lines are drawn last (on top)
config_plot_props = {
    'AD_float64': {'lw': 4.5, 'ms': 12, 'mfc': 'none', 'zorder': 1}, # Thickest, largest, hollow
    'FD_float64': {'lw': 3.0, 'ms': 10, 'mfc': 'auto', 'zorder': 2}, # Medium, filled
    'AD_float32': {'lw': 1.5, 'ms': 8,  'mfc': 'none', 'zorder': 3}, # Thinner, hollow
    'FD_float32': {'lw': 0.8, 'ms': 5,  'mfc': 'auto', 'zorder': 4}  # Thinnest, smallest, filled
}

fig2, (ax2_l2, ax2_linf) = plt.subplots(1, 2, figsize=(12, 4))
nx_array = np.array(COARSE_GRIDS)

# Create the custom labels list (e.g., ["64x64", "128x128", "256x256"])
grid_labels = [rf"{nx}$\times${nx}" for nx in COARSE_GRIDS]

for lin in LIN_TYPES:
    for prec in PRECISIONS:
        config_name = f"{lin}_{prec}"
        
        style = config_styles[config_name]
        props = config_plot_props[config_name]
        current_color = config_convergence_colors.get(config_name, 'k')
        label_name = f"{lin} {prec}"
        
        l2_final = [errors[Nx][config_name]['L2'][-1] for Nx in COARSE_GRIDS]
        linf_final = [errors[Nx][config_name]['Linf'][-1] for Nx in COARSE_GRIDS]
        
        # Plot L2 Spatial Convergence
        ax2_l2.plot(nx_array, l2_final, 
                    linestyle=style['ls'], 
                    marker=style['marker'],
                    markersize=props['ms'], 
                    markerfacecolor=props['mfc'], # Hollow or filled
                    linewidth=props['lw'], 
                    color=current_color, 
                    zorder=props['zorder'],       # Draw order
                    label=label_name, 
                    alpha=0.85)                   # Increased alpha so colors pop
        
        # Plot Linf Spatial Convergence
        ax2_linf.plot(nx_array, linf_final, 
                      linestyle=style['ls'], 
                      marker=style['marker'],
                      markersize=props['ms'], 
                      markerfacecolor=props['mfc'],
                      linewidth=props['lw'], 
                      color=current_color, 
                      zorder=props['zorder'],
                      label=label_name, 
                      alpha=0.85)

# Formatting L2 Spatial Subplot
ax2_l2.set_xlabel("Grid Resolution")
ax2_l2.set_ylabel("Final Relative $L_2$ Error")
ax2_l2.set_xscale("log", base=2)
ax2_l2.set_yscale("log")
ax2_l2.set_xticks(COARSE_GRIDS)
ax2_l2.set_xticklabels(grid_labels) # <-- Changed this line

# Formatting Linf Spatial Subplot
ax2_linf.set_xlabel("Grid Resolution")
ax2_linf.set_ylabel("Final Relative $L_\\infty$ Error")
ax2_linf.set_xscale("log", base=2)
ax2_linf.set_yscale("log")
ax2_linf.set_xticks(COARSE_GRIDS)
ax2_linf.set_xticklabels(grid_labels) # <-- Changed this line

# Move legend out of the way
ax2_linf.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()

save_path_spatial = "data/spaceErrorBurgers.pdf"
plt.savefig(save_path_spatial, dpi=150, bbox_inches='tight')
plt.close(fig2)

print(f"  -> Saved: {save_path_spatial}")
print("Analysis complete.")