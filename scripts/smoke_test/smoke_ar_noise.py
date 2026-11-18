"""ID case2_ar_smoke_noise: calibration of the start noise of the best-of-R score; sets R, τ, ε_eff and B_max for the
learnings run.

Context: §3 (C1–C14), J8 and J10 of project_docs/case2_dopt_autoresearch_learnings_design.md. The optimiser is the
SciPy L-BFGS-B trial path (trial.optimise), one start after another: case2_ar_smoke_multistart showed that the batched
GPU L-BFGS has the same best-of-R statistics but is 2.4x slower on this GPU. Starts: the members of the constraint
x0 = x1 = z0 = z1 (25-0-0-0-0, 21-1-1-1-1, …, 1-6-6-6-6) and the mirror pair 9-5-3-4-4 / 9-3-5-4-4, each with
M = 32 scrambled Halton seeds 0…31; N = 25, maxiter 5000, kill time 60 s per start (v1 I12). A start is valid if it
passes the hard spacing check. Results:
    σ_R     for R in {1, 2, 4, 8}: standard deviation of the best-of-R Φ_val over 1000 bootstrap draws, maximum over
            the sensor_layouts (C5, C6)
    τ(R)    2√2 σ_R; R = the smallest R with τ(R) <= ε = 2 ln 1.10, else 8 (C7, C8); ε_eff = max(τ(R), ε)
    B_max   min(32, ⌊0.05 × 10 h / (R t̄_start)⌋), t̄_start = mean wall time of one start (C12)
Checks: p̂^R <= 1e-3, with p̂ the largest fraction of starts of one sensor_layout that fail the hard spacing check
(the chance that a trial has no valid start); no start reaches the kill time; the mirror pair differs by <= τ(R) in
mean best-of-R Φ_val (J10).
Outputs: outputs/case2_ar_smoke_noise_<stamp>/checks.txt, starts.csv, tau_R.csv, calibration.json, designs.npz;
plots with plot_smoke_ar_noise.py.

Run:  scripts/smoke_test/run_smoke_gpu.sh 0 0,1,4,5 scripts/smoke_test/smoke_ar_noise.py
"""

from __future__ import annotations

import csv
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from autoresearch import evaluator, harness, layout, noise, trial
from autoresearch import problem as P
from heat_inverse import config

LAYOUTS = tuple((25 - 4 * m, m, m, m, m) for m in range(7)) + ((9, 5, 3, 4, 4), (9, 3, 5, 4, 4))
MIRROR = ("9-5-3-4-4", "9-3-5-4-4")
SEEDS = tuple(range(32))
MAXITER = 5000
KILL_TIME = 60.0
T_SESSION = 10 * 3600.0
F_HYP = 0.05
B_CAP = 32
P_NO_VALID = 1e-3


class Checks:
    def __init__(self):
        self.lines, self.ok = [], True

    def add(self, name, passed, detail):
        self.ok &= bool(passed)
        self.lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
        print(self.lines[-1], flush=True)

    def info(self, text):
        self.lines.append(f"INFO  {text}")
        print(self.lines[-1], flush=True)


def main() -> int:
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_smoke_noise"))
    c, t_all = Checks(), time.perf_counter()
    B_train, B_val = P.load_banks()
    keys = [(layout.fmt(L), s) for L in LAYOUTS for s in SEEDS]
    c.info(f"{len(keys)} starts, maxiter {MAXITER}, kill time {KILL_TIME:.0f} s per start")

    runs = []
    for i, (L, s) in enumerate((L, s) for L in LAYOUTS for s in SEEDS):
        runs.append(trial.optimise(*layout.start_design(L, seed=s), B_train, maxiter=MAXITER, kill_time=KILL_TIME))
        if s == SEEDS[-1]:
            done = runs[-len(SEEDS):]
            c.info(f"{layout.fmt(L)}: {sum(r.wall_time for r in done):.1f} s, nit median "
                   f"{np.median([r.nit for r in done]):.0f}, converged {sum(r.converged for r in done)} of {len(done)}")
    X = np.stack([r.X for r in runs])
    faces = np.stack([r.faces for r in runs])
    wall = np.array([r.wall_time for r in runs])
    pt = np.asarray(jax.jit(jax.vmap(P.phi, in_axes=(0, None)))(jnp.asarray(X), B_train))
    ev = [evaluator.evaluate(x, f, B_val) for x, f in zip(X, faces)]
    val, valid = np.array([e["score"] for e in ev]), np.array([e["feasible"] for e in ev])
    cfg = np.array([k[0] for k in keys])
    with open(out / "starts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("config", "seed", "phi_train", "phi_val", "s1", "s2", "rho", "nit", "converged", "killed",
                    "wall_time", "min_spacing_mm", "feasible"))
        for k, p, e, r in zip(keys, pt, ev, runs):
            w.writerow((*k, f"{p:.10f}", f"{e['score']:.10f}", f"{e['s1']:.8f}", f"{e['s2']:.8f}", f"{e['rho']:.8f}",
                        r.nit, r.converged, r.killed, f"{r.wall_time:.3f}", f"{e['min_spacing_mm']:.4f}",
                        e["feasible"]))
    np.savez(out / "designs.npz", X_start=np.stack([r.X_start for r in runs]), X=X, faces=faces, config=cfg,
             seed=[k[1] for k in keys])

    sigma, mean = {}, {}
    with open(out / "tau_R.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("config", "R", "n_valid", "mean_best", "sigma_R", "tau_R"))
        for name in dict.fromkeys(cfg):
            sel = cfg == name
            for R in noise.R_GRID:
                b = noise.best_of_r(pt[sel], val[sel], valid[sel], R)
                sigma[name, R], mean[name, R] = float(np.nanstd(b, ddof=1)), float(np.nanmean(b))
                w.writerow((name, R, int(valid[sel].sum()), f"{mean[name, R]:.10f}", f"{sigma[name, R]:.6e}",
                            f"{float(noise.tau(sigma[name, R])):.6e}"))
    sigma_max = {R: max(v for (_, r), v in sigma.items() if r == R) for R in noise.R_GRID}
    tau = {R: float(noise.tau(sigma_max[R])) for R in noise.R_GRID}
    R = noise.choose_R(tau)
    eps_eff = max(tau[R], noise.EPS_GAIN)
    for r in noise.R_GRID:
        arg = max((v, n) for (n, rr), v in sigma.items() if rr == r)[1]
        c.info(f"R = {r}: σ_R = {sigma_max[r]:.4f} (largest at {arg}), τ(R) = {tau[r]:.4f}")
    c.info(f"ε = {noise.EPS_GAIN:.4f}; chosen R = {R}; ε_eff = {eps_eff:.4f}")

    n_valid = {name: int(valid[cfg == name].sum()) for name in dict.fromkeys(cfg)}
    p_invalid = {name: 1 - n / len(SEEDS) for name, n in n_valid.items()}
    worst = max(p_invalid, key=p_invalid.get)
    c.info(f"invalid starts (hard spacing check): {len(keys) - sum(n_valid.values())} of {len(keys)}; smallest "
           f"spacing {min(e['min_spacing_mm'] for e in ev):.4f} mm")
    c.add(f"a trial has a valid start: p̂^R <= {P_NO_VALID:g}", p_invalid[worst] ** R <= P_NO_VALID,
          f"p̂ = {p_invalid[worst]:.4f} ({worst}), p̂^{R} = {p_invalid[worst] ** R:.2e}")
    killed = int(sum(r.killed for r in runs))
    c.add(f"no start reaches the kill time ({KILL_TIME:.0f} s)", killed == 0,
          f"{killed} killed; longest start {wall.max():.1f} s")
    d_mirror = abs(mean[MIRROR[0], R] - mean[MIRROR[1], R])
    c.add(f"mirror pair {MIRROR[0]} / {MIRROR[1]}: |Δ mean best of R| <= τ(R)", d_mirror <= tau[R],
          f"{d_mirror:.4f} vs τ({R}) = {tau[R]:.4f}")

    t_start = float(wall.mean())
    B_max = int(min(B_CAP, max(1, F_HYP * T_SESSION // (R * t_start))))
    c.info(f"t̄_start {t_start:.2f} s (max {wall.max():.1f} s); one batch of B_max = {B_max} sensor_layouts × R = {R} "
           f"starts ≈ {B_max * R * t_start:.0f} s")

    cal = dict(eps=noise.EPS_GAIN, R_grid=list(noise.R_GRID), sigma_R=sigma_max, tau_R=tau, R=R, tau=tau[R],
               eps_eff=eps_eff, n_valid=n_valid, p_invalid=p_invalid, mirror_diff=d_mirror, t_start_mean=t_start,
               t_start_max=float(wall.max()), kill_time=KILL_TIME, B_max=B_max, configs=list(dict.fromkeys(cfg)),
               seeds=list(SEEDS), maxiter=MAXITER, wall_time_starts=float(wall.sum()))
    (out / "calibration.json").write_text(json.dumps(cal, indent=2, default=float) + "\n")

    c.info(f"total wall time {time.perf_counter() - t_all:.1f} s")
    c.lines.append(f"RESULT {'PASS' if c.ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(c.lines) + "\n")
    print(c.lines[-1], f"-> {out}", sep="\n")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
