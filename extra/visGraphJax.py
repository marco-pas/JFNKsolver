import jax
import sys
import jax.numpy as jnp
import graphviz
from functools import partial as ft_partial


sys.path.insert(0, ".")          # make sure your solver file is on the path
from maxwSolverGPU1 import (            # ← replace "maxwell" with your actual filename (without .py)
    residual_TE,
    eps_func,
    Dxx_op, Dyy_op, Dxy_op
)

jax.config.update("jax_enable_x64", True)

# ── minimal stubs so we can trace without a full grid ────────────────
Nx, Ny = 8, 8          # tiny grid — graph topology is the same as 256x256
N = Nx * Ny
dtype = jnp.complex128

# dummy inputs matching your signature
state  = jnp.zeros(2 * N, dtype=dtype)
perturb = jnp.ones(2 * N, dtype=dtype)
omega  = 15.0
mu0, eps0 = 1.0, 1.0
dx, dy = 1.0/(Nx-1), 1.0/(Ny-1)
Jx = jnp.zeros((Nx, Ny), dtype=dtype)
Jy = jnp.zeros((Nx, Ny), dtype=dtype)

# ── trace both functions ─────────────────────────────────────────────
def fwd(s):
    return residual_TE(s, omega, mu0, eps0, Jx, Jy, dx, dy, Nx, Ny)

def jvp_fn(s, v):
    return jax.jvp(fwd, (s,), (v,))

jaxpr_fwd = jax.make_jaxpr(fwd)(state)
jaxpr_jvp = jax.make_jaxpr(jvp_fn)(state, perturb)

# ── render JAXpr as a Graphviz dot graph ─────────────────────────────
def jaxpr_to_dot(jaxpr, title="JAXpr"):
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir="TB", label=title, fontsize="14")

    # input nodes
    for i, var in enumerate(jaxpr.jaxpr.invars):
        dot.node(str(id(var)), label=f"in_{i}\n{var.aval.shape}\n{var.aval.dtype}",
                 shape="ellipse", style="filled", fillcolor="lightblue")

    # equation nodes
    for eq in jaxpr.jaxpr.eqns:
        eq_id = str(id(eq))
        prim_name = eq.primitive.name
        dot.node(eq_id, label=prim_name, shape="box",
                 style="filled", fillcolor="lightyellow")

        for invar in eq.invars:
            if hasattr(invar, 'aval'):   # skip literals
                dot.edge(str(id(invar)), eq_id)

        for outvar in eq.outvars:
            out_label = f"{outvar.aval.shape}\n{outvar.aval.dtype}"
            dot.node(str(id(outvar)), label=out_label,
                     shape="ellipse", style="filled", fillcolor="lightgreen")
            dot.edge(eq_id, str(id(outvar)))

    # output nodes
    for var in jaxpr.jaxpr.outvars:
        dot.node(str(id(var)) + "_out", label="OUT",
                 shape="doublecircle", style="filled", fillcolor="salmon")
        dot.edge(str(id(var)), str(id(var)) + "_out")

    return dot

dot_fwd = jaxpr_to_dot(jaxpr_fwd, title="residual_TE  (forward)")
dot_jvp = jaxpr_to_dot(jaxpr_jvp, title="residual_TE  (JVP / AD)")

dot_fwd.render("graph_forward", format="pdf", cleanup=True)
dot_jvp.render("graph_jvp",     format="pdf", cleanup=True)
print("Saved graph_forward.pdf and graph_jvp.pdf")

# ── also print primitive counts — tells you AD overhead ──────────────
from collections import Counter

def count_primitives(jaxpr):
    return Counter(eq.primitive.name for eq in jaxpr.jaxpr.eqns)

fwd_counts = count_primitives(jaxpr_fwd)
jvp_counts = count_primitives(jaxpr_jvp)

print("\n--- Forward primitive counts ---")
for k, v in sorted(fwd_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:30s} {v}")

print("\n--- JVP primitive counts ---")
for k, v in sorted(jvp_counts.items(), key=lambda x: -x[1]):
    jvp_extra = v - fwd_counts.get(k, 0)
    print(f"  {k:30s} {v:4d}   (+{jvp_extra} vs fwd)")
