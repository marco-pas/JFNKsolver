import re, numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "axes.unicode_minus": False, "font.size": 12
})

logs = {
    r"$\omega = 15$":      {"color":"#c0392b","ls":"-","data":"Newton   0: ||F|| = 1.5536e+01\nNewton   1: ||F|| = 1.0701e+01\nNewton   2: ||F|| = 8.2515e+00\nNewton   3: ||F|| = 6.4593e+00\nNewton   4: ||F|| = 5.0713e+00\nNewton   5: ||F|| = 3.9766e+00\nNewton   6: ||F|| = 3.1125e+00\nNewton   7: ||F|| = 2.4399e+00\nNewton   8: ||F|| = 1.9186e+00\nNewton   9: ||F|| = 1.5104e+00\nNewton  10: ||F|| = 1.1935e+00\nNewton  11: ||F|| = 9.4422e-01\nNewton  12: ||F|| = 7.4627e-01\nNewton  13: ||F|| = 5.8803e-01\nNewton  14: ||F|| = 4.6208e-01\nNewton  15: ||F|| = 3.6229e-01\nNewton  16: ||F|| = 2.8406e-01\nNewton  17: ||F|| = 2.2340e-01\nNewton  18: ||F|| = 1.7609e-01\nNewton  19: ||F|| = 1.3919e-01\nNewton  20: ||F|| = 1.1009e-01\nNewton  21: ||F|| = 8.7140e-02\nNewton  22: ||F|| = 6.8868e-02\nNewton  23: ||F|| = 5.4372e-02\nNewton  24: ||F|| = 4.2815e-02\nNewton  25: ||F|| = 3.3687e-02\nNewton  26: ||F|| = 2.6418e-02\nNewton  27: ||F|| = 2.0777e-02\nNewton  28: ||F|| = 1.6415e-02\nNewton  29: ||F|| = 1.2976e-02\nNewton  30: ||F|| = 1.0263e-02\nNewton  31: ||F|| = 8.1124e-03\nNewton  32: ||F|| = 6.4129e-03\nNewton  33: ||F|| = 5.0545e-03\nNewton  34: ||F|| = 3.9874e-03\nNewton  35: ||F|| = 3.1327e-03\nNewton  36: ||F|| = 2.4644e-03\nNewton  37: ||F|| = 1.9423e-03\nNewton  38: ||F|| = 1.5352e-03\nNewton  39: ||F|| = 1.2160e-03\nNewton  40: ||F|| = 9.6523e-04\nNewton  41: ||F|| = 7.6346e-04\nNewton  42: ||F|| = 6.0307e-04\nNewton  43: ||F|| = 4.7641e-04\nNewton  44: ||F|| = 3.7598e-04\nNewton  45: ||F|| = 2.9663e-04\nNewton  46: ||F|| = 2.3394e-04\nNewton  47: ||F|| = 1.8452e-04\nNewton  48: ||F|| = 1.4550e-04\nNewton  49: ||F|| = 1.1519e-04\nNewton  50: ||F|| = 9.1435e-05\nNewton  51: ||F|| = 7.2508e-05\nNewton  52: ||F|| = 5.7369e-05\nNewton  53: ||F|| = 4.5392e-05\nNewton  54: ||F|| = 3.5875e-05\nNewton  55: ||F|| = 2.8286e-05\nNewton  56: ||F|| = 2.2317e-05\nNewton  57: ||F|| = 1.7577e-05\nNewton  58: ||F|| = 1.3850e-05\nNewton  59: ||F|| = 1.0954e-05\nNewton  60: ||F|| = 8.6800e-06\nNewton  61: ||F|| = 6.8903e-06\nNewton  62: ||F|| = 5.4602e-06\nNewton  63: ||F|| = 4.3174e-06\nNewton  64: ||F|| = 3.4084e-06\nNewton  65: ||F|| = 2.6847e-06\nNewton  66: ||F|| = 2.1138e-06\nNewton  67: ||F|| = 1.6609e-06\nNewton  68: ||F|| = 1.3082e-06\nNewton  69: ||F|| = 1.0335e-06\nNewton  70: ||F|| = 8.1764e-07\nNewton  71: ||F|| = 6.4871e-07\nNewton  72: ||F|| = 5.1342e-07\nNewton  73: ||F|| = 4.0542e-07\nNewton  74: ||F|| = 3.2014e-07\nNewton  75: ||F|| = 2.5227e-07\nNewton  76: ||F|| = 1.9824e-07\nNewton  77: ||F|| = 1.5582e-07\nNewton  78: ||F|| = 1.2298e-07"},
    r"$\omega = 30$":                         {"color":"#de8c45","ls":"-","data":"Newton   0: ||F|| = 1.9286e+01\nNewton   1: ||F|| = 1.1569e+01\nNewton   2: ||F|| = 7.3377e+00\nNewton   3: ||F|| = 4.6782e+00\nNewton   4: ||F|| = 2.9807e+00\nNewton   5: ||F|| = 1.9112e+00\nNewton   6: ||F|| = 1.2255e+00\nNewton   7: ||F|| = 7.9119e-01\nNewton   8: ||F|| = 5.1151e-01\nNewton   9: ||F|| = 3.2970e-01\nNewton  10: ||F|| = 2.1203e-01\nNewton  11: ||F|| = 1.3475e-01\nNewton  12: ||F|| = 8.6326e-02\nNewton  13: ||F|| = 5.5374e-02\nNewton  14: ||F|| = 3.5754e-02\nNewton  15: ||F|| = 2.3123e-02\nNewton  16: ||F|| = 1.4930e-02\nNewton  17: ||F|| = 9.6001e-03\nNewton  18: ||F|| = 6.1264e-03\nNewton  19: ||F|| = 3.9127e-03\nNewton  20: ||F|| = 2.5077e-03\nNewton  21: ||F|| = 1.6241e-03\nNewton  22: ||F|| = 1.0516e-03\nNewton  23: ||F|| = 6.7897e-04\nNewton  24: ||F|| = 4.3772e-04\nNewton  25: ||F|| = 2.8027e-04\nNewton  26: ||F|| = 1.7958e-04\nNewton  27: ||F|| = 1.1574e-04\nNewton  28: ||F|| = 7.4993e-05\nNewton  29: ||F|| = 4.8692e-05\nNewton  30: ||F|| = 3.1671e-05\nNewton  31: ||F|| = 2.0525e-05\nNewton  32: ||F|| = 1.3235e-05\nNewton  33: ||F|| = 8.4980e-06\nNewton  34: ||F|| = 5.4636e-06\nNewton  35: ||F|| = 3.5383e-06\nNewton  36: ||F|| = 2.3001e-06\nNewton  37: ||F|| = 1.5007e-06\nNewton  38: ||F|| = 9.7681e-07\nNewton  39: ||F|| = 6.3202e-07\nNewton  40: ||F|| = 4.0754e-07\nNewton  41: ||F|| = 2.6226e-07\nNewton  42: ||F|| = 1.6901e-07"},
    r"$\omega = 200$":  {"color":"#3db46f","ls":"-","data":"Newton   0: ||F|| = 8.1775e+00\nNewton   1: ||F|| = 2.8309e-01\nNewton   2: ||F|| = 1.2645e-02\nNewton   3: ||F|| = 4.9895e-04\nNewton   4: ||F|| = 1.9578e-05\nNewton   5: ||F|| = 9.1023e-07\nNewton   6: ||F|| = 3.3818e-08"},   
    r"$\omega = 400$":  {"color":"#2532c7","ls":"-","data":"Newton   0: ||F|| = 2.1789e-01\nNewton   1: ||F|| = 5.4403e-05\nNewton   2: ||F|| = 5.5383e-08\nNewton   3: ||F|| = 1.0423e-10"},
}

pattern = r"Newton\s+(\d+):\s+\|\|F\|\|\s+=\s+([\d\.e\+\-]+)"
parsed = {}
for label, cfg in logs.items():
    iters, res = [], []
    for line in cfg["data"].split("\n"):
        m = re.search(pattern, line)
        if m:
            iters.append(int(m.group(1))); res.append(float(m.group(2)))
    parsed[label] = {"iters": np.array(iters), "res": np.array(res),
                     "color": cfg["color"], "ls": cfg["ls"]}

def mean_contraction(res):
    ratios = res[1:] / res[:-1]
    return float(np.exp(np.mean(np.log(ratios))))

fig, ax = plt.subplots(figsize=(8, 4))

for label, d in parsed.items():
    rel = d["res"] / d["res"][0]
    ax.semilogy(d["iters"], rel, color=d["color"], ls=d["ls"], lw=2.0, label=label)

k_lin  = np.linspace(1, 78, 300)
r_near = mean_contraction(parsed[r"$\omega = 15$"]["res"])
r_far  = mean_contraction(parsed[r"$\omega = 200$"]["res"])
# ax.semilogy(k_lin,      r_near**k_lin,      color="gray",  ls=":", lw=1.2, label=rf"Linear ref  $r={r_near:.3f}$")
# ax.semilogy(k_lin[:10], r_far**k_lin[:10],  color="silver",ls=":", lw=1.2, label=rf"Linear ref  $r={r_far:.3f}$")

ax.set_xlabel("Newton iteration  $k$", fontsize=13)
ax.set_ylabel(r"$\|F_k\| \, / \, \|F_0\|$", fontsize=13)
ax.set_title("Newton convergence: near-resonance vs. far-from-resonance\n"
             r"Nonlinear Kerr-media Maxwell solver  ($\chi=0.05$, PEC BC, $256\times256$)", fontsize=12)
ax.set_xlim(left=0)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.set_ylim(1e-8, 1.3)
plt.tight_layout()
fig.savefig("data/newtonConv.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved")