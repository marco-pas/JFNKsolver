import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. Style Setup
# ==========================================
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset":   "cm",
    "text.usetex":        False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.unicode_minus": False,
    "font.size":          10,
    "axes.labelsize":     10.5,
    "axes.titlesize":     11.5,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "axes.linewidth":     0.9,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2.5,
    "ytick.minor.size":   2.5,
    "xtick.minor.visible":True,
    "ytick.minor.visible":True,
    "axes.grid":          True,
    "grid.linestyle":     ":",
    "grid.linewidth":     0.45,
    "grid.alpha":         0.55,
    "grid.color":         "#aaaaaa",
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ==========================================
# 1. Synthetic data mirroring real outputs
# ==========================================
np.random.seed(42)
COARSE_GRIDS = [64, 128, 256, 512]
LIN_TYPES    = ['AD', 'FD']
PRECISIONS   = ['float64', 'float32']
NUM_STEPS    = 40
time_axis    = np.linspace(0.0337, 0.8433, NUM_STEPS) * 1e3   # ms

def make_error_curve(nx, lin, prec, norm):
    base    = (1.2 / nx) ** 1.0
    noise_s = 0.06 if prec == 'float32' else 0.02
    offset  = 0.55 if prec == 'float32' else 0.0
    fd_bump = 1.20 if lin == 'FD' else 1.0
    growth  = 0.58 if norm == 'L2' else 0.62
    t       = np.linspace(0, 1, NUM_STEPS)
    curve   = base * fd_bump * (1 + offset) * (1 + t**growth * 1.8)
    curve  *= 1 + noise_s * np.cumsum(np.random.randn(NUM_STEPS)) / NUM_STEPS
    return np.clip(curve, 1e-6, None)

errors = {
    Nx: {
        f"{lin}_{prec}": {
            'L2':   make_error_curve(Nx, lin, prec, 'L2'),
            'Linf': make_error_curve(Nx, lin, prec, 'Linf') * 3,
        }
        for lin in LIN_TYPES for prec in PRECISIONS
    }
    for Nx in COARSE_GRIDS
}

# Newton data
logs = {
    r"$\omega = 15$":  {"color":"#c0392b","data":"Newton 0: ||F|| = 1.5536e+01\nNewton 1: ||F|| = 1.0701e+01\nNewton 10: ||F|| = 1.1935e+00\nNewton 20: ||F|| = 1.1009e-01\nNewton 30: ||F|| = 1.0263e-02\nNewton 40: ||F|| = 9.6523e-04\nNewton 50: ||F|| = 9.1435e-05\nNewton 60: ||F|| = 8.6800e-06\nNewton 70: ||F|| = 8.1764e-07\nNewton 78: ||F|| = 1.2298e-07"},
    r"$\omega = 30$":  {"color":"#e07b39","data":"Newton 0: ||F|| = 1.9286e+01\nNewton 10: ||F|| = 2.1203e-01\nNewton 20: ||F|| = 2.5077e-03\nNewton 30: ||F|| = 3.1671e-05\nNewton 40: ||F|| = 4.0754e-07\nNewton 42: ||F|| = 1.6901e-07"},
    r"$\omega = 200$": {"color":"#27a96c","data":"Newton 0: ||F|| = 8.1775e+00\nNewton 2: ||F|| = 1.2645e-02\nNewton 4: ||F|| = 1.9578e-05\nNewton 6: ||F|| = 3.3818e-08"},
    r"$\omega = 400$": {"color":"#1a3ab5","data":"Newton 0: ||F|| = 2.1789e-01\nNewton 1: ||F|| = 5.4403e-05\nNewton 3: ||F|| = 1.0423e-10"},
}
pattern = r"Newton\s+(\d+):\s+\|\|F\|\|\s+=\s+([\d\.e\+\-]+)"
parsed_newton = {}
for label, cfg in logs.items():
    iters, res = [], []
    for line in cfg["data"].split("\n"):
        m = re.search(pattern, line)
        if m:
            iters.append(int(m.group(1)))
            res.append(float(m.group(2)))
    parsed_newton[label] = {"iters": np.array(iters), "res": np.array(res), "color": cfg["color"]}

# ==========================================
# 2. Colour / style palettes
# ==========================================
GRID_COLORS = {
    64:  "#e07b39",   # amber
    128: "#c53b8f",   # raspberry
    256: "#6640c4",   # violet
    512: "#1a6eb5",   # steel blue
}

CONV_COLORS  = {
    "AD_float64": "#162d6b",
    "FD_float64": "#c0392b",
    "AD_float32": "#e07b39",
    "FD_float32": "#27a96c",
}
CONV_MARKERS = {
    "AD_float64": "^", "FD_float64": "o",
    "AD_float32": "s", "FD_float32": "D",
}
CONV_LW = {
    "AD_float64": 2.4, "FD_float64": 1.8,
    "AD_float32": 1.4, "FD_float32": 1.0,
}

# ==========================================
# 3. Figure layout (Clustered SubGridSpecs)
# ==========================================
fig = plt.figure(figsize=(18, 4.2))

# Create 3 main compartments: Time, Space, Newton
gs_main = GridSpec(1, 3, figure=fig, width_ratios=[2.1, 2.1, 1], 
                   wspace=0.30, left=0.04, right=0.98, bottom=0.15, top=0.88)

# Subdivide Time and Space into 2 plots each, tightly coupled
gs_time  = gs_main[0].subgridspec(1, 2, wspace=0.22)
gs_space = gs_main[1].subgridspec(1, 2, wspace=0.22)

ax_tl2   = fig.add_subplot(gs_time[0])
ax_tlinf = fig.add_subplot(gs_time[1])
ax_sl2   = fig.add_subplot(gs_space[0])
ax_slinf = fig.add_subplot(gs_space[1])
ax_n     = fig.add_subplot(gs_main[2])
axes = [ax_tl2, ax_tlinf, ax_sl2, ax_slinf, ax_n]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]
PANEL_TITLES = [
    r"Time Evolution — $L_2$", r"Time Evolution — $L_\infty$",
    r"Spatial Conv. — $L_2$", r"Spatial Conv. — $L_\infty$",
    "Newton Conv. (Maxwell)",
]
for ax, lbl, ttl in zip(axes, PANEL_LABELS, PANEL_TITLES):
    ax.set_title(ttl, fontweight="semibold", pad=8)
    ax.text(0.03, 0.97, lbl, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontstyle="italic", color="#444444")

# ==========================================
# 4. Panels 1 & 2 — Time Evolution
# ==========================================
for Nx in COARSE_GRIDS:
    for lin in LIN_TYPES:
        for prec in PRECISIONS:
            cname  = f"{lin}_{prec}"
            color  = GRID_COLORS[Nx]
            
            # TWEAK: Distinct line styles for AD vs FD
            ls = "-" if lin == 'AD' else "--"
            
            # TWEAK: Distinct alpha/width for precision
            alpha = 0.95 if prec == 'float64' else 0.55
            lw    = 1.8  if prec == 'float64' else 1.0
            
            style  = dict(color=color, lw=lw, ls=ls, alpha=alpha, zorder=3)
            ax_tl2.plot(time_axis,   errors[Nx][cname]['L2'],   **style)
            ax_tlinf.plot(time_axis, errors[Nx][cname]['Linf'], **style)

for ax in (ax_tl2, ax_tlinf):
    ax.set_yscale("log")
    ax.set_xlabel("Physical time  (ms)")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))

ax_tl2.set_ylabel(r"Relative $L_2$ error")
ax_tlinf.set_ylabel(r"Relative $L_\infty$ error")

# TWEAK: Comprehensive legend placed in empty bottom right area
leg_handles = (
    [Line2D([0],[0], color=GRID_COLORS[nx], lw=1.8, label=rf"$N={nx}$") for nx in COARSE_GRIDS] +
    [Line2D([0],[0], color="k", lw=1.5, ls="-", label="AD"),
     Line2D([0],[0], color="k", lw=1.5, ls="--", label="FD")] +
    [Line2D([0],[0], color="k", lw=1.8, alpha=0.95, label="float64"),
     Line2D([0],[0], color="k", lw=1.0, alpha=0.55, label="float32")]
)
ax_tlinf.legend(
    handles=leg_handles,
    loc="lower right", ncol=2,
    framealpha=0.92, edgecolor="#cccccc", fontsize=8.0,
    handlelength=2.0, labelspacing=0.28, borderpad=0.5,
    title="Grid / Method / Prec.", title_fontsize=8.0,
)

# ==========================================
# 5. Panels 3 & 4 — Spatial Convergence
# ==========================================
nx_arr = np.array(COARSE_GRIDS, dtype=float)
grid_labels = [rf"$64$", r"$128$", r"$256$", r"$512$"]

for lin in LIN_TYPES:
    for prec in PRECISIONS:
        cname      = f"{lin}_{prec}"
        l2_final   = np.array([errors[Nx][cname]['L2'][-1]   for Nx in COARSE_GRIDS])
        linf_final = np.array([errors[Nx][cname]['Linf'][-1] for Nx in COARSE_GRIDS])
        kw = dict(
            color=CONV_COLORS[cname], marker=CONV_MARKERS[cname],
            lw=CONV_LW[cname], ms=6, mfc='white', mew=1.3,
            zorder=4, alpha=0.9,
        )
        ax_sl2.loglog(nx_arr,   l2_final,   **kw)
        ax_slinf.loglog(nx_arr, linf_final, **kw, label=f"{lin} / {prec}")

# Reference slopes with differing line styles
def ref_slope(ax, nx_arr, vals, order, label, color, ls, x_frac=0.55):
    x0, y0 = nx_arr[-1], vals[-1]
    xs = np.array([nx_arr[0], nx_arr[-1]])
    ys = y0 * (xs / x0)**(-order)
    scale = 2.0
    ax.loglog(xs, ys * scale, ls=ls, lw=1.4, color=color, zorder=1)
    # Keep text placement just in case
    xi = xs[0] * (xs[1]/xs[0])**x_frac
    yi = ys[0] * (ys[1]/ys[0])**x_frac * scale
    ax.text(xi, yi * 0.7, label, fontsize=8, color=color, va="top",
            ha="center", rotation=-np.degrees(np.arctan(order))*0.45)

for ax, norm_key in [(ax_sl2, 'L2'), (ax_slinf, 'Linf')]:
    ref_vals = np.array([errors[nx]['AD_float64'][norm_key][-1] for nx in COARSE_GRIDS])
    # TWEAK: Dotted for O(h1), Dashed for O(h2)
    ref_slope(ax, nx_arr, ref_vals, 1, r"$\mathcal{O}(h^1)$", "#aaaaaa", ls=":", x_frac=0.3)
    ref_slope(ax, nx_arr, ref_vals, 2, r"$\mathcal{O}(h^2)$", "#555555", ls="--", x_frac=0.3)

for ax in (ax_sl2, ax_slinf):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(nx_arr)
    ax.set_xticklabels(grid_labels)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Grid resolution  $N$")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))

ax_sl2.set_ylabel(r"Final rel. $L_2$ error")
ax_slinf.set_ylabel(r"Final rel. $L_\infty$ error")

# TWEAK: Add explicit reference lines to the legend
conv_handles = [
    Line2D([0],[0], color=CONV_COLORS[k], marker=CONV_MARKERS[k], lw=CONV_LW[k], ms=5, mfc='white', mew=1.2,
           label=k.replace("_", r" / ").replace("float", "f"))
    for k in ["AD_float64","FD_float64","AD_float32","FD_float32"]
]
ref_handles = [
    Line2D([0],[0], color="#aaaaaa", ls=":",  lw=1.5, label=r"$\mathcal{O}(h^1)$ ref."),
    Line2D([0],[0], color="#555555", ls="--", lw=1.5, label=r"$\mathcal{O}(h^2)$ ref.")
]
ax_slinf.legend(
    handles=conv_handles + ref_handles, loc="upper right",
    framealpha=0.92, edgecolor="#cccccc", fontsize=8.0,
    handlelength=2.2, labelspacing=0.28, borderpad=0.5,
    title="Method / Prec / Refs", title_fontsize=8.0,
)

# ==========================================
# 6. Panel 5 — Newton Convergence
# ==========================================
newton_handles = []
for label, d in parsed_newton.items():
    if len(d["res"]) > 0:
        rel = d["res"] / d["res"][0]
        line, = ax_n.semilogy(d["iters"], rel, color=d["color"], lw=2.0,
                              marker="o", ms=4.5, mfc='white', mew=1.3,
                              solid_capstyle='round', label=label)
        newton_handles.append(line)

ax_n.set_yscale("log")
ax_n.set_ylim(5e-11, 2.0)
ax_n.set_xlim(left=-1)
ax_n.set_xlabel(r"Newton iteration  $k$")
ax_n.set_ylabel(r"$\|F_k\| \,/\, \|F_0\|$")
ax_n.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False))
ax_n.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
ax_n.legend(
    handles=newton_handles, loc="upper right",
    framealpha=0.92, edgecolor="#cccccc", fontsize=8.5,
    handlelength=2.0, labelspacing=0.32, borderpad=0.55,
    title=r"Frequency $\omega$", title_fontsize=8.5,
)

# ==========================================
# 7. Uniform spine / tick polish
# ==========================================
SPINE_COLOR = "#222222"
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
        spine.set_edgecolor(SPINE_COLOR)
    ax.tick_params(which='both', color=SPINE_COLOR)
    ax.tick_params(which='major', width=0.85, length=4.0)
    ax.tick_params(which='minor', width=0.55, length=2.5)

# ==========================================
# 8. Save
# ==========================================
import os; os.makedirs("data", exist_ok=True)
out = "data/combined_convergence_plots_pro.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)