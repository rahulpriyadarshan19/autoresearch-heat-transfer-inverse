"""ID case2_ar_posterior: bank-averaged Laplace posterior of saved session designs, on the held-out bank.

Smoke test: none; checks.txt compares the recomputed Φ_val of each final design with results.csv.
Average over the bank: log-Euclidean mean C̄ = expm(mean_s logm C_s), C_s = A_s⁻¹, because then
log det C̄ = mean_s log det C_s = Φ_val exactly. The arithmetic mean of C_s and the inverse of the mean A_s
are recorded only for the Jensen check log det (mean A_s)⁻¹ ≤ Φ_val ≤ log det mean C_s.
Outputs: outputs/case2_ar_posterior_<stamp>/{checks.txt, posterior.csv, posterior.npz}; plot with plot_posterior.py.

Run:  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 \
      venv_ar/bin/python scripts/posterior_session.py --session <session_id> --designs <trial>:start|final ...
"""

from __future__ import annotations

import argparse
import csv
import time

import jax
import jax.numpy as jnp
import numpy as np

from autoresearch import evaluator, harness
from autoresearch import problem as P
from heat_inverse import config

LEVEL = 0.95
CHI2 = -2.0 * np.log(1.0 - LEVEL)  # LEVEL quantile of chi-square with 2 degrees of freedom
N_OUTLINE = 361
TOL_PHI = 1e-10
STAGE_KEY = {"start": "X_start", "final": "X"}
COLUMNS = ("label", "trial", "config", "stage", "phi_val", "phi_logged", "logdet_le", "logdet_c_mean",
           "logdet_a_mean_inv", "C11", "C12", "C22", "sd1", "sd2", "rho", "area", "feasible", "wall_s")


def spd_fn(f):
    """Matrix function of symmetric positive definite (..., 2, 2) arrays via eigh."""
    def apply(M):
        w, V = jnp.linalg.eigh(M)
        return jnp.einsum("...ij,...j,...kj->...ik", V, f(w), V)
    return apply


@jax.jit
def averages(X, B):
    """(log-Euclidean mean of C_s, arithmetic mean of C_s, (mean A_s)⁻¹, log det C_s) of design X on bank B."""
    A = P.information(X, B)
    C = jnp.linalg.inv(A)
    C_le = spd_fn(jnp.exp)(jnp.mean(spd_fn(jnp.log)(C), axis=0))
    return C_le, jnp.mean(C, axis=0), jnp.linalg.inv(jnp.mean(A, axis=0)), jnp.linalg.slogdet(C)[1]


@jax.jit
def outline(C):
    """(N_OUTLINE, 2) points of the LEVEL contour of N(0, C)."""
    t = jnp.linspace(0.0, 2.0 * jnp.pi, N_OUTLINE)
    return jnp.sqrt(CHI2) * jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1) @ jnp.linalg.cholesky(C).T


def shape(C):
    sd1, sd2 = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    return dict(C11=C[0, 0], C12=C[0, 1], C22=C[1, 1], sd1=sd1, sd2=sd2, rho=C[0, 1] / (sd1 * sd2),
                area=np.pi * CHI2 * np.sqrt(np.linalg.det(C)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--designs", nargs="+", required=True, help="<trial id>:start|final")
    args = ap.parse_args()
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_posterior"))
    t_all = time.perf_counter()
    _, B_val = P.load_banks()
    with open(config.PROJECT_ROOT / "sessions" / args.session / "results.csv", newline="") as fh:
        logged = {r["id"]: r for r in csv.DictReader(fh)}

    lines, ok = [f"INFO  session {args.session}; bank val S = {B_val.shape[0]}; level {LEVEL}"], True
    C0 = np.diag(P.PRIOR_SD**2)
    rows = [dict(label="prior", trial="", config="", stage="prior", phi_val=float(np.linalg.slogdet(C0)[1]),
                 phi_logged="", logdet_le="", logdet_c_mean="", logdet_a_mean_inv="", feasible="", wall_s="",
                 **shape(C0))]
    arrays = {"theta0": P.LOG_PRIOR_MU, "outline_prior": np.asarray(outline(jnp.asarray(C0)))}

    t0 = time.perf_counter()
    z = np.load(config.OUTPUTS / args.session / args.designs[0].split(":")[0] / "trial.npz")
    jax.block_until_ready(averages(jnp.asarray(z["X"]), B_val))
    evaluator.evaluate(z["X"], z["faces"], B_val)
    lines.append(f"INFO  JIT warm-up {time.perf_counter() - t0:.1f} s")

    for spec in args.designs:
        tid, stage = spec.split(":")
        z = np.load(config.OUTPUTS / args.session / tid / "trial.npz")
        X, faces = z[STAGE_KEY[stage]], z["faces"]
        t0 = time.perf_counter()
        C_le, C_mean, A_mean_inv, logdet = (np.asarray(v) for v in averages(jnp.asarray(X), B_val))
        ev = evaluator.evaluate(X, faces, B_val)
        wall = time.perf_counter() - t0
        phi_bank = float(np.mean(logdet))
        ld_le, ld_c, ld_a = (float(np.linalg.slogdet(M)[1]) for M in (C_le, C_mean, A_mean_inv))
        label = f"{tid}:{stage}"
        logged_phi = float(logged[tid]["score"]) if stage == "final" else None
        for name, passed, detail in (
                ("log det C_le = mean log det C_s", abs(ld_le - phi_bank) <= TOL_PHI, f"{abs(ld_le - phi_bank):.2e}"),
                ("evaluator Φ_val = mean log det C_s", abs(ev["score"] - phi_bank) <= TOL_PHI,
                 f"{abs(ev['score'] - phi_bank):.2e}"),
                ("Jensen log det (mean A)⁻¹ ≤ Φ_val ≤ log det mean C", ld_a <= ev["score"] <= ld_c,
                 f"{ld_a:.6f} ≤ {ev['score']:.6f} ≤ {ld_c:.6f}")):
            ok &= bool(passed)
            lines.append(f"{'PASS' if passed else 'FAIL'}  {label}: {name}: {detail}")
        if logged_phi is not None:
            d = abs(ev["score"] - logged_phi)
            ok &= d <= TOL_PHI
            lines.append(f"{'PASS' if d <= TOL_PHI else 'FAIL'}  {label}: Φ_val = results.csv: {d:.2e}")
        lines.append(f"INFO  {label}: config {logged[tid]['config']}, Φ_val {ev['score']:.6f}, "
                     f"feasible {ev['feasible']}, wall {1e3 * wall:.1f} ms")
        rows.append(dict(label=label, trial=tid, config=logged[tid]["config"], stage=stage, phi_val=ev["score"],
                         phi_logged="" if logged_phi is None else logged_phi, logdet_le=ld_le, logdet_c_mean=ld_c,
                         logdet_a_mean_inv=ld_a, feasible=ev["feasible"], wall_s=wall, **shape(C_le)))
        arrays[f"outline_{label}"] = np.asarray(outline(jnp.asarray(C_le)))
        arrays[f"X_{label}"], arrays[f"faces_{label}"] = X, faces

    with open(out / "posterior.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "posterior.npz", **arrays)
    lines.append(f"INFO  total wall {time.perf_counter() - t_all:.1f} s (includes JIT)")
    lines.append(f"RESULT {'PASS' if ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
