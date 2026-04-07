import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---- Configuration ----
SIMULATION_IC = 'TGV'
BASE_NX, BASE_NY = 1024, 1024
BASE_DIR = f"output/burgers_benchmark/{SIMULATION_IC}_{BASE_NX}_{BASE_NY}_AD_float64"

COARSE_GRIDS = [64, 128, 256, 512]
LIN_TYPES = ['AD', 'FD']
PRECISIONS = ['float64', 'float32']

# Configure plot styling suitable for a 2-column article scale-down
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", 
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False, 
    "font.size": 14,             # Larger base font to survive scaling
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4
})

# =============================================================================
# 1. PROCESS BURGERS DATA (TIME & SPACE CONVERGENCE)
# =============================================================================
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
print(f"Found {num_steps} time steps to analyze in Burgers dataset.")

errors = {Nx: {f"{lin}_{prec}": {'L2': np.zeros(num_steps), 'Linf': np.zeros(num_steps)} 
               for lin in LIN_TYPES for prec in PRECISIONS} for Nx in COARSE_GRIDS}

for t_idx, step_id in enumerate(step_ids):
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
                target_dir = f"output/burgers_benchmark/{SIMULATION_IC}_{Nx}_{Ny}_{lin}_{prec}"
                
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
                    errors[Nx][config_name]['L2'][t_idx] = np.nan
                    errors[Nx][config_name]['Linf'][t_idx] = np.nan
                    continue
                
                # --- Compute Relative Error Fields ---
                e_u = u_coarse - u_base_sub
                e_v = v_coarse - v_base_sub
                
                # 1. Absolute Error
                abs_l2_num = np.sum(e_u**2 + e_v**2)
                abs_linf_num = max(np.max(np.abs(e_u)), np.max(np.abs(e_v)))
                
                # 2. Baseline Norm
                base_l2_den = np.sum(u_base_sub**2 + v_base_sub**2)
                base_linf_den = max(np.max(np.abs(u_base_sub)), np.max(np.abs(v_base_sub)))
                
                # 3. Relative Norms
                errors[Nx][config_name]['L2'][t_idx] = float(np.sqrt(abs_l2_num / (base_l2_den + 1e-16)))
                errors[Nx][config_name]['Linf'][t_idx] = float(abs_linf_num / (base_linf_den + 1e-16))

time_axis = np.array([
    0.04819143042179147,
0.09638286084358294,
0.1445742912653744,
0.19276572168716588,
0.24095715210895735,
0.2891485825307488,
0.3373400129525403,
0.38553144337433176,
0.43372287379612323,
0.4819143042179147,
0.5301057346397062,
0.5782971650614976,
0.6264885954832891,
0.6746800259050805,
0.7228714563268719,
0.7710628867486633,
0.8192543171704547,
0.8674457475922461,
0.9156371780140375,
0.963828608435829,
1.0120200388576204,
1.0602114692794118,
1.1084028997012032,
1.1565943301229946,
1.204785760544786,
])

# =============================================================================
# 2. PROCESS MAXWELL NEWTON LOGS
# =============================================================================
logs = {
    r"$\omega = 15$":  {"color":"#c0392b","ls":"-","data":"Newton   0: ||F|| = 1.5536e+01\nNewton   1: ||F|| = 1.0701e+01\nNewton   2: ||F|| = 8.2515e+00\nNewton   3: ||F|| = 6.4593e+00\nNewton   4: ||F|| = 5.0713e+00\nNewton   5: ||F|| = 3.9766e+00\nNewton   6: ||F|| = 3.1125e+00\nNewton   7: ||F|| = 2.4399e+00\nNewton   8: ||F|| = 1.9186e+00\nNewton   9: ||F|| = 1.5104e+00\nNewton  10: ||F|| = 1.1935e+00\nNewton  11: ||F|| = 9.4422e-01\nNewton  12: ||F|| = 7.4627e-01\nNewton  13: ||F|| = 5.8803e-01\nNewton  14: ||F|| = 4.6208e-01\nNewton  15: ||F|| = 3.6229e-01\nNewton  16: ||F|| = 2.8406e-01\nNewton  17: ||F|| = 2.2340e-01\nNewton  18: ||F|| = 1.7609e-01\nNewton  19: ||F|| = 1.3919e-01\nNewton  20: ||F|| = 1.1009e-01\nNewton  21: ||F|| = 8.7140e-02\nNewton  22: ||F|| = 6.8868e-02\nNewton  23: ||F|| = 5.4372e-02\nNewton  24: ||F|| = 4.2815e-02\nNewton  25: ||F|| = 3.3687e-02\nNewton  26: ||F|| = 2.6418e-02\nNewton  27: ||F|| = 2.0777e-02\nNewton  28: ||F|| = 1.6415e-02\nNewton  29: ||F|| = 1.2976e-02\nNewton  30: ||F|| = 1.0263e-02\nNewton  31: ||F|| = 8.1124e-03\nNewton  32: ||F|| = 6.4129e-03\nNewton  33: ||F|| = 5.0545e-03\nNewton  34: ||F|| = 3.9874e-03\nNewton  35: ||F|| = 3.1327e-03\nNewton  36: ||F|| = 2.4644e-03\nNewton  37: ||F|| = 1.9423e-03\nNewton  38: ||F|| = 1.5352e-03\nNewton  39: ||F|| = 1.2160e-03\nNewton  40: ||F|| = 9.6523e-04\nNewton  41: ||F|| = 7.6346e-04\nNewton  42: ||F|| = 6.0307e-04\nNewton  43: ||F|| = 4.7641e-04\nNewton  44: ||F|| = 3.7598e-04\nNewton  45: ||F|| = 2.9663e-04\nNewton  46: ||F|| = 2.3394e-04\nNewton  47: ||F|| = 1.8452e-04\nNewton  48: ||F|| = 1.4550e-04\nNewton  49: ||F|| = 1.1519e-04\nNewton  50: ||F|| = 9.1435e-05\nNewton  51: ||F|| = 7.2508e-05\nNewton  52: ||F|| = 5.7369e-05\nNewton  53: ||F|| = 4.5392e-05\nNewton  54: ||F|| = 3.5875e-05\nNewton  55: ||F|| = 2.8286e-05\nNewton  56: ||F|| = 2.2317e-05\nNewton  57: ||F|| = 1.7577e-05\nNewton  58: ||F|| = 1.3850e-05\nNewton  59: ||F|| = 1.0954e-05\nNewton  60: ||F|| = 8.6800e-06\nNewton  61: ||F|| = 6.8903e-06\nNewton  62: ||F|| = 5.4602e-06\nNewton  63: ||F|| = 4.3174e-06\nNewton  64: ||F|| = 3.4084e-06\nNewton  65: ||F|| = 2.6847e-06\nNewton  66: ||F|| = 2.1138e-06\nNewton  67: ||F|| = 1.6609e-06\nNewton  68: ||F|| = 1.3082e-06\nNewton  69: ||F|| = 1.0335e-06\nNewton  70: ||F|| = 8.1764e-07\nNewton  71: ||F|| = 6.4871e-07\nNewton  72: ||F|| = 5.1342e-07\nNewton  73: ||F|| = 4.0542e-07\nNewton  74: ||F|| = 3.2014e-07\nNewton  75: ||F|| = 2.5227e-07\nNewton  76: ||F|| = 1.9824e-07\nNewton  77: ||F|| = 1.5582e-07\nNewton  78: ||F|| = 1.2298e-07"},
    r"$\omega = 30$":  {"color":"#de8c45","ls":"-","data":"Newton   0: ||F|| = 1.9286e+01\nNewton   1: ||F|| = 1.1569e+01\nNewton   2: ||F|| = 7.3377e+00\nNewton   3: ||F|| = 4.6782e+00\nNewton   4: ||F|| = 2.9807e+00\nNewton   5: ||F|| = 1.9112e+00\nNewton   6: ||F|| = 1.2255e+00\nNewton   7: ||F|| = 7.9119e-01\nNewton   8: ||F|| = 5.1151e-01\nNewton   9: ||F|| = 3.2970e-01\nNewton  10: ||F|| = 2.1203e-01\nNewton  11: ||F|| = 1.3475e-01\nNewton  12: ||F|| = 8.6326e-02\nNewton  13: ||F|| = 5.5374e-02\nNewton  14: ||F|| = 3.5754e-02\nNewton  15: ||F|| = 2.3123e-02\nNewton  16: ||F|| = 1.4930e-02\nNewton  17: ||F|| = 9.6001e-03\nNewton  18: ||F|| = 6.1264e-03\nNewton  19: ||F|| = 3.9127e-03\nNewton  20: ||F|| = 2.5077e-03\nNewton  21: ||F|| = 1.6241e-03\nNewton  22: ||F|| = 1.0516e-03\nNewton  23: ||F|| = 6.7897e-04\nNewton  24: ||F|| = 4.3772e-04\nNewton  25: ||F|| = 2.8027e-04\nNewton  26: ||F|| = 1.7958e-04\nNewton  27: ||F|| = 1.1574e-04\nNewton  28: ||F|| = 7.4993e-05\nNewton  29: ||F|| = 4.8692e-05\nNewton  30: ||F|| = 3.1671e-05\nNewton  31: ||F|| = 2.0525e-05\nNewton  32: ||F|| = 1.3235e-05\nNewton  33: ||F|| = 8.4980e-06\nNewton  34: ||F|| = 5.4636e-06\nNewton  35: ||F|| = 3.5383e-06\nNewton  36: ||F|| = 2.3001e-06\nNewton  37: ||F|| = 1.5007e-06\nNewton  38: ||F|| = 9.7681e-07\nNewton  39: ||F|| = 6.3202e-07\nNewton  40: ||F|| = 4.0754e-07\nNewton  41: ||F|| = 2.6226e-07\nNewton  42: ||F|| = 1.6901e-07"},
    r"$\omega = 200$": {"color":"#3db46f","ls":"-","data":"Newton   0: ||F|| = 8.1775e+00\nNewton   1: ||F|| = 2.8309e-01\nNewton   2: ||F|| = 1.2645e-02\nNewton   3: ||F|| = 4.9895e-04\nNewton   4: ||F|| = 1.9578e-05\nNewton   5: ||F|| = 9.1023e-07\nNewton   6: ||F|| = 3.3818e-08"},   
    r"$\omega = 400$": {"color":"#2532c7","ls":"-","data":"Newton   0: ||F|| = 2.1789e-01\nNewton   1: ||F|| = 5.4403e-05\nNewton   2: ||F|| = 5.5383e-08\nNewton   3: ||F|| = 1.0423e-10"},
}

pattern = r"Newton\s+(\d+):\s+\|\|F\|\|\s+=\s+([\d\.e\+\-]+)"
parsed = {}
for label, cfg in logs.items():
    iters, res = [], []
    for line in cfg["data"].split("\n"):
        m = re.search(pattern, line)
        if m:
            iters.append(int(m.group(1)))
            res.append(float(m.group(2)))
    parsed[label] = {"iters": np.array(iters), "res": np.array(res),
                     "color": cfg["color"], "ls": cfg["ls"]}


# =============================================================================
# 3. BUILD UNIFIED FIGURE — 2-column layout (70% | 30%)
# =============================================================================
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18, 7.5))

gs = GridSpec(
    2, 3,                          # 2 rows, 3 logical columns
    figure=fig,
    width_ratios=[0.3, 0.3, 0.40],  # col0+col1 = left 60%, col2 = right 40%
    height_ratios=[1, 1],
    hspace=0.2,
    wspace=0.17,
)
gs2 = GridSpec(
    2, 3,                          # 2 rows, 3 logical columns
    figure=fig,
    width_ratios=[0.3, 0.3, 0.40],  # col0+col1 = left 60%, col2 = right 40%
    height_ratios=[1, 1],
    hspace=0.2,
    wspace=0.325,
)

ax_tl2   = fig.add_subplot(gs[0, 0])          # row 0, col 0 — Time L2
ax_tlinf = fig.add_subplot(gs[1, 0])          # row 0, col 1 — Time Linf
ax_sl2   = fig.add_subplot(gs[0, 1])          # row 1, col 0 — Space L2
ax_slinf = fig.add_subplot(gs[1, 1])          # row 1, col 1 — Space Linf
ax_newt  = fig.add_subplot(gs2[:, 2])          # both rows, col 2 — Newton

# ── shared y-label for each left-block row ──────────────────────────────────
ax_tl2.set_ylabel(r"Relative $L_2$ error", fontsize = 19)
ax_tlinf.set_ylabel(r"Relative $L_\infty$ error", fontsize = 19)

# =============================================================================
# PLOTS A & B  —  Time evolution
# =============================================================================
grid_colors = {
    512: '#4361EE',
    256: '#F72585',
    128: '#7209B7',
    64:  '#F8961E',
}
prec_lw  = {'float64': 2.4, 'float32': 1.0}
lin_marker = {
    'AD': {'marker': '^', 'ms': 6, 'mfc': 'auto'},
    'FD': {'marker': 'o', 'ms': 6, 'mfc': 'none'},
}
MARKEVERY = 3

for Nx in COARSE_GRIDS:
    for lin in LIN_TYPES:
        for prec in PRECISIONS:
            config_name = f"{lin}_{prec}"
            clr = grid_colors[Nx]
            lw  = prec_lw[prec]
            mp  = lin_marker[lin]
            kw = dict(
                linestyle='-', linewidth=lw,
                marker=mp['marker'], markersize=mp['ms'],
                markerfacecolor=clr if mp['mfc'] == 'auto' else 'none',
                markeredgecolor=clr,
                markevery=MARKEVERY, color=clr, alpha=0.9,
            )
            ax_tl2.plot(time_axis * 1e3,   errors[Nx][config_name]['L2'],   **kw)
            ax_tlinf.plot(time_axis * 1e3,  errors[Nx][config_name]['Linf'], **kw)

# ax_tl2.set_title(r"Time Evolution ($L_2$)")
# ax_tl2.set_xlabel("Time, ms)")
ax_tl2.set_yscale("log")

# ax_tlinf.set_title(r"Time Evolution ($L_\infty$)")
ax_tlinf.set_xlabel("Time, ms", fontsize = 19)
ax_tlinf.set_yscale("log")

# ── Legend on ax_tlinf — 3 compact groups in 3 columns ─────────────────────
import matplotlib.lines as mlines

leg_grid = [
    mlines.Line2D([0],[0], color=grid_colors[64],  lw=2.5, label=r'$64\times64$'),
    mlines.Line2D([0],[0], color=grid_colors[128], lw=2.5, label=r'$128\times128$'),
    mlines.Line2D([0],[0], color=grid_colors[256], lw=2.5, label=r'$256\times256$'),
    mlines.Line2D([0],[0], color=grid_colors[512], lw=2.5, label=r'$512\times512$'),
]
leg_prec = [
    mlines.Line2D([0],[0], color='k', lw=2.4, label='float64'),
    mlines.Line2D([0],[0], color='k', lw=1.0, label='float32'),
]
leg_lin = [
    mlines.Line2D([0],[0], color='k', lw=0, marker='^', ms=7, mfc='k',    label='AD'),
    mlines.Line2D([0],[0], color='k', lw=0, marker='o', ms=7, mfc='none', label='FD'),
]

# Place below the two time panels using figure-level bbox
ax_tlinf.legend(
    handles=[*leg_grid, *leg_prec, *leg_lin],
    fontsize=10,
    loc='best',
    framealpha=0.9,
    handlelength=1.5,
    labelspacing=0.1,
    borderpad=0.2,
    ncol=2,
)

# =============================================================================
# PLOTS C & D  —  Spatial convergence
# =============================================================================
config_styles_sp = {
    'AD_float64': {'ls': '-',  'marker': '^'},
    'FD_float64': {'ls': '--', 'marker': 'o'},
    'AD_float32': {'ls': '-',  'marker': '^'},
    'FD_float32': {'ls': '--', 'marker': 'o'},
}
config_convergence_colors = {
    'AD_float64': '#002FFF', 'FD_float64': '#F72585',
    'AD_float32': '#FADD00', 'FD_float32': '#13A709',
}
config_plot_props = {
    'AD_float64': {'lw': 3.5, 'ms': 9, 'mfc': 'none', 'zorder': 1},
    'FD_float64': {'lw': 2.5, 'ms': 7, 'mfc': 'auto', 'zorder': 2},
    'AD_float32': {'lw': 1.5, 'ms': 6, 'mfc': 'none', 'zorder': 3},
    'FD_float32': {'lw': 1.0, 'ms': 4, 'mfc': 'auto', 'zorder': 4},
}

nx_array   = np.array(COARSE_GRIDS)
grid_labels = [rf"{nx}$\times${nx}" for nx in COARSE_GRIDS]

for lin in LIN_TYPES:
    for prec in PRECISIONS:
        config_name = f"{lin}_{prec}"
        style = config_styles_sp[config_name]
        props = config_plot_props[config_name]
        clr   = config_convergence_colors[config_name]
        lbl   = f"{lin} {prec}"

        l2_final   = [errors[Nx][config_name]['L2'][-1]   for Nx in COARSE_GRIDS]
        linf_final = [errors[Nx][config_name]['Linf'][-1] for Nx in COARSE_GRIDS]

        ax_sl2.plot(nx_array,   l2_final,   linestyle=style['ls'], marker=style['marker'],
                    markersize=props['ms'], markerfacecolor=props['mfc'],
                    linewidth=props['lw'], color=clr, zorder=props['zorder'],
                    label=lbl, alpha=0.9)
        ax_slinf.plot(nx_array, linf_final, linestyle=style['ls'], marker=style['marker'],
                      markersize=props['ms'], markerfacecolor=props['mfc'],
                      linewidth=props['lw'], color=clr, zorder=props['zorder'],
                      label=lbl, alpha=0.9)

ref_base_l2   = errors[64]['AD_float64']['L2'][-1]   * 0.8
ref_base_linf = errors[64]['AD_float64']['Linf'][-1] * 0.8
ax_sl2.plot(nx_array,   ref_base_l2   * (nx_array[0]/nx_array)**2,
            'k--', lw=2, zorder=5, label=r'$\mathcal{O}(\Delta x^2)$')
ax_slinf.plot(nx_array, ref_base_linf * (nx_array[0]/nx_array)**2,
              'k--', lw=2, zorder=5, label=r'$\mathcal{O}(\Delta x^2)$')

for ax in (ax_sl2, ax_slinf):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(COARSE_GRIDS)
    ax.set_xticklabels(grid_labels)
    if ax == ax_slinf:
        ax.set_xlabel("Grid Size", fontsize = 19)

# ax_sl2.set_title(r"Space Convergence ($L_2$)")
ax_sl2.legend(fontsize=9, loc='lower left')

# ax_slinf.set_title(r"Space Convergence ($L_\infty$)")

# =============================================================================
# PLOT E  —  Newton convergence (right column, full height)
# =============================================================================
for label, d in parsed.items():
    rel = d["res"] / d["res"][0]
    ax_newt.semilogy(d["iters"], rel, color=d["color"], ls=d["ls"], lw=2.0, label=label)

# ax_newt.set_title("Newton Convergence (Maxwell)")
# ax_newt.set_title(r"$\|F_k\| \, / \, \|F_0\|$")
ax_newt.set_xlabel(r"Iteration $k$", fontsize = 19)
ax_newt.set_ylabel(r"$\|F_k\| \, / \, \|F_0\|$ convergence", fontsize = 19)
ax_newt.set_xlim(left=0)
ax_newt.set_ylim(1e-8, 1.3)
ax_newt.legend(fontsize=14, loc="upper right")

# =============================================================================
# 4. FINALIZE & SAVE
# =============================================================================
os.makedirs("data", exist_ok=True)
plt.savefig("data/combined_plots_new.pdf", dpi=200, bbox_inches='tight')
plt.close(fig)
print("Done → data/combined_plots_new.pdf")