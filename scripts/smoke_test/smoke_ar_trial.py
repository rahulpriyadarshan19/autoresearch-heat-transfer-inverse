"""ID case2_ar_smoke_trial: checks of the ported problem, trial and evaluator code, before any session runs.

Context: src/heat_inverse is a verbatim copy and src/autoresearch/{problem,trial}.py port the D-opt functions of
heat_transfer_inverse/scripts/run_dopt_case2.py (commit a4ca30c). This test reproduces the saved §6.1 design
dopt_x1z1_5 (data/reference/), checks the gradient, the Halton start rule and the spacing check, and runs one
N = 30 trial. Pass thresholds:
    phi of the saved design      |Δ| <= 5e-6 against laplace.csv (printed to 6 significant digits)
    rerun of its optimisation    nit and phi_val against laplace.csv (975, same tolerance), reported
    g_PIELM vs exact series      max |Δg| < 0.1 K at θ* (σ = 1 K)
    JAX gradient vs central FD   relative error <= 1e-6
Outputs: outputs/case2_ar_smoke_trial_<stamp>/checks.txt, rerun.npz, trial.npz; plot with plot_smoke_ar_trial.py.

Run:  venv_ar/bin/python scripts/smoke_test/smoke_ar_trial.py
"""

from __future__ import annotations

import csv
import time

import jax
import jax.numpy as jnp
import numpy as np

from autoresearch import evaluator, harness, layout, trial
from autoresearch import problem as P
from heat_inverse import config
from heat_inverse.cases import case2

REF = config.PROJECT_ROOT / "data" / "reference" / "case2_dopt_x1z1_20261101T164737_30f5"
CFG = "dopt_x1z1_5"
TOL_PHI = 5e-6


class Checks:
    def __init__(self):
        self.lines, self.ok = [], True

    def add(self, name, passed, detail):
        self.ok &= bool(passed)
        self.lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    def info(self, text):
        self.lines.append(f"INFO  {text}")


def main() -> int:
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_smoke_trial"))
    c, t_all = Checks(), time.perf_counter()
    B_train, B_val = P.load_banks()
    c.info(f"bank sha256 {P.file_sha256(P.BANK_FILE)}")

    z = np.load(REF / "layouts.npz")
    ref = next(r for r in csv.DictReader(open(REF / "laplace.csv")) if r["config"] == CFG)
    X_opt, faces = z[f"X_{CFG}"], z[f"faces_{CFG}"]

    # 1. phi of the saved optimised design on both banks
    for bank, B in (("val", B_val), ("train", B_train)):
        v = float(P.phi(jnp.asarray(X_opt), B))
        d = abs(v - float(ref[f"phi_{bank}"]))
        c.add(f"phi_{bank} of saved {CFG}", d <= TOL_PHI, f"{v:.8f} vs {ref[f'phi_{bank}']} (|Δ| {d:.2e})")

    # 2. rerun of the saved optimisation from its start design
    res = trial.optimise(z[f"X_start_{CFG}"], faces, B_train, maxiter=1000)
    ev = evaluator.evaluate(res.X, faces, B_val)
    d = abs(ev["score"] - float(ref["phi_val"]))
    c.add(f"rerun of {CFG}: nit", res.nit == int(ref["nit"]), f"{res.nit} vs {ref['nit']}")
    c.add(f"rerun of {CFG}: phi_val", d <= TOL_PHI, f"{ev['score']:.8f} vs {ref['phi_val']} (|Δ| {d:.2e})")
    c.info(f"rerun: {res.wall_time:.2f} s for {res.nit} iterations incl. compile; "
           f"{1e3 * np.median(np.diff(res.t_iter)):.2f} ms median per iteration; message: {res.message}")
    np.savez(out / "rerun.npz", X_start=res.X_start, X=res.X, faces=res.faces, loss=res.loss, t_iter=res.t_iter)

    # 3. PIELM sensitivities against the exact series at θ* (smoke test only; the evaluator never sees θ*)
    log_k = np.log(np.asarray(case2.K_TRUE[:2]))[None]
    t0 = time.perf_counter()
    B_star = jnp.asarray(P.B_of(P.beta_batch_fn(), log_k))
    g_pielm = np.asarray(P.g_bank(jnp.asarray(X_opt), B_star))[0]
    g_series = np.zeros_like(g_pielm)
    for j in range(2):
        e = P.STEP * np.eye(2)[j]
        T = [case2.SeriesReference(tuple(case2.expand_k(*np.exp(log_k[0] + s * e)).tolist())).T(X_opt)
             for s in (1, -1)]
        g_series[:, j] = (np.asarray(T[0]) - np.asarray(T[1])) / (2 * P.STEP)
    dg = float(np.abs(g_pielm - g_series).max())
    c.add("g_PIELM vs series at θ*", dg < 0.1, f"max |Δg| = {dg:.4f} K ({time.perf_counter() - t0:.1f} s)")

    # 4. JAX gradient of the training loss against central differences along a random direction
    X0, f0 = layout.start_design((6, 6, 6, 6, 6))
    ip, lo, hi, u0 = trial.bounds_and_start(X0, f0)
    v = np.random.default_rng(0).standard_normal(u0.shape)
    _, grad = trial.loss_and_grad(jnp.asarray(u0), B_train, ip, X0)
    eps = 1e-6
    lp = float(trial.loss_and_grad(jnp.asarray(u0 + eps * v), B_train, ip, X0)[0])
    lm = float(trial.loss_and_grad(jnp.asarray(u0 - eps * v), B_train, ip, X0)[0])
    fd, ad = (lp - lm) / (2 * eps), float(np.dot(np.asarray(grad), v))
    rel = abs(fd - ad) / abs(ad)
    c.add("JAX gradient vs central differences", rel <= 1e-6, f"relative error {rel:.2e}")

    # 5. Halton start rule
    for counts in ((6, 6, 6, 6, 6), (20, 5, 0, 5, 0), (30, 0, 0, 0, 0)):
        Xa, fa = layout.start_design(counts)
        Xb, _ = layout.start_design(counts)
        c.add(f"start {layout.fmt(counts)} on faces, inside margins", evaluator.on_faces(Xa, fa),
              f"{len(Xa)} sensors")
        c.add(f"start {layout.fmt(counts)} deterministic", np.array_equal(Xa, Xb), "two calls equal")
    Xs, fs = layout.start_design((6, 6, 6, 6, 6))
    same = np.array_equal(Xs[fs == "x0"][:, 1:], Xs[fs == "x1"][:, 1:]) and \
        np.array_equal(Xs[fs == "z0"][:, :2], Xs[fs == "z1"][:, :2])
    c.add("start: equal face-local points on mirror faces", same, "x0 = x1 in (y, z), z0 = z1 in (x, y)")

    # 6. spacing check of the evaluator
    for d_mm, expect in ((5.0, False), (9.94, False), (9.96, True), (10.0, True)):
        X = np.array([[0.05, 0.0, 0.05], [0.05 + 1e-3 * d_mm, 0.0, 0.05]])
        feas = evaluator.evaluate(X, np.array(["y0", "y0"]), B_val)["feasible"]
        c.add(f"spacing check at {d_mm} mm", feas == expect, f"feasible = {feas}")

    # 7. one N = 30 trial, short
    X0, f0 = layout.start_design((6, 6, 6, 6, 6))
    r = trial.optimise(X0, f0, B_train, maxiter=50, snapshot_iter=20)
    ev = evaluator.evaluate(r.X, f0, B_val)
    c.add("N = 30 trial (maxiter 50): sensors on faces", ev["on_faces"], f"nit {r.nit}, {r.wall_time:.2f} s")
    c.add("N = 30 trial: finite metrics", all(np.isfinite([ev["score"], ev["s1"], ev["s2"], ev["rho"]])),
          f"score {ev['score']:.5f}, s1 {ev['s1']:.4f}, s2 {ev['s2']:.4f}, rho {ev['rho']:+.3f}, "
          f"min spacing {ev['min_spacing_mm']:.3f} mm")
    c.add("N = 30 trial: snapshot at iteration 20", r.X_snapshot is not None and len(r.loss) >= 20,
          f"{len(r.loss)} losses stored")
    np.savez(out / "trial.npz", X_start=r.X_start, X=r.X, faces=r.faces, loss=r.loss, t_iter=r.t_iter,
             **{k: v for k, v in ev.items()})

    c.info(f"total wall time {time.perf_counter() - t_all:.1f} s")
    c.lines.append(f"RESULT {'PASS' if c.ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(c.lines) + "\n")
    print("\n".join(c.lines), f"\n-> {out}", sep="")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
