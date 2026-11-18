"""ID case2_ar_smoke_multistart: the batched GPU L-BFGS (src/autoresearch/multistart.py) against the SciPy L-BFGS-B
trial path, before the calibration case2_ar_smoke_noise uses it.

Context: the learnings run (project_docs/case2_dopt_autoresearch_learnings_design.md, J9 and J12) runs all R starts
of a batch in one vectorised GPU loop, with its own line search and active-set rule, so a start can end in a
different local optimum than SciPy. Starts: 13-3-3-3-3, 5-5-5-5-5 and 1-6-6-6-6 (members of the calibration
constraint x0 = x1 = z0 = z1), scrambled Halton seeds 0…31, N = 25, maxiter 5000. Checks:
    (1) physics: batched loss and gradient at the start designs against trial.loss_and_grad,
        relative difference <= 1e-10
    (2) statistics: for each sensor_layout and R in {1, 2, 4, 8}, |E[best of R] GPU − E[best of R] SciPy| of Φ_val
        <= max(τ(R), 1e-6), τ(R) = 2√2 σ_R of the SciPy starts (bootstrap of noise.py, paired draws)
    (3) time: s per loop iteration against batch size; wall time of the full batch against the SciPy runs, speed-up > 1
    batch invariance (info): the first 8 starts alone against the same starts in the full batch
Outputs: outputs/case2_ar_smoke_multistart_<stamp>/checks.txt, starts.csv, bootstrap.csv, physics.csv, timing.csv,
hist.npz; plots with plot_smoke_ar_multistart.py.

Run:  scripts/smoke_test/run_smoke_gpu.sh 0 0,1,4,5 scripts/smoke_test/smoke_ar_multistart.py
"""

from __future__ import annotations

import csv
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from autoresearch import evaluator, harness, layout, multistart, noise, trial
from autoresearch import problem as P
from heat_inverse import config

LAYOUTS = ((13, 3, 3, 3, 3), (5, 5, 5, 5, 5), (1, 6, 6, 6, 6))
SEEDS = tuple(range(32))
MAXITER = 5000
TOL_PHYS = 1e-10
TAU_FLOOR = 1e-6  # scale of the stop tests in Φ: τ(R) is 0 when all starts reach one optimum
SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
N_TIME = 20
N_INV = 8
FIELDS = ("path", "config", "seed", "phi_train", "phi_val", "s1", "s2", "rho", "nit", "status", "converged",
          "min_spacing_mm", "feasible")


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


def evaluate_all(X, faces, B_train, B_val):
    """Φ_train (no penalty) and the evaluator dict of each final design."""
    phi_train = np.asarray(multistart.phi_batch(jnp.asarray(X), B_train))
    return phi_train, [evaluator.evaluate(x, f, B_val) for x, f in zip(X, faces)]


def main() -> int:
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_smoke_multistart"))
    c, t_all = Checks(), time.perf_counter()
    B_train, B_val = P.load_banks()
    keys = [(layout.fmt(L), s) for L in LAYOUTS for s in SEEDS]
    starts = [layout.start_design(L, seed=s) for L in LAYOUTS for s in SEEDS]
    c.info(f"N = {len(starts[0][0])}, {len(starts)} starts, maxiter {MAXITER}")

    # (1) physics at the start designs
    u0, _, _, ip, X0, _ = multistart.stack_starts(starts, len(starts))
    fb, gb = (np.asarray(a) for a in multistart.loss_and_grad_batch(u0, B_train, ip, X0))
    phys = []
    for i in range(len(starts)):
        f1, g1 = trial.loss_and_grad(u0[i], B_train, ip[i], X0[i])
        f1, g1 = float(f1), np.asarray(g1)
        phys.append((abs(fb[i] - f1) / abs(f1), np.abs(gb[i] - g1).max() / np.abs(g1).max()))
    phys = np.array(phys)
    with open(out / "physics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("config", "seed", "rel_loss", "rel_grad"))
        w.writerows((*k, f"{a:.3e}", f"{b:.3e}") for k, (a, b) in zip(keys, phys))
    c.add("(1) batched loss vs single start", phys[:, 0].max() <= TOL_PHYS,
          f"max relative difference {phys[:, 0].max():.1e}")
    c.add("(1) batched gradient vs single start", phys[:, 1].max() <= TOL_PHYS,
          f"max relative difference {phys[:, 1].max():.1e} (of max |g|)")

    # both paths from the same starts
    res = multistart.optimise(starts, B_train, MAXITER)
    c.info(f"GPU: {len(starts)} starts in {res.wall_time:.1f} s incl. compile; nit median {np.median(res.nit):.0f}, "
           f"max {res.nit.max()}; status {dict(Counter(res.status))}")
    t0 = time.perf_counter()
    sc = [trial.optimise(X0_, f_, B_train, maxiter=MAXITER) for X0_, f_ in starts]
    t_scipy = time.perf_counter() - t0
    nit_s = np.array([r.nit for r in sc])
    c.info(f"SciPy: {len(starts)} starts in {t_scipy:.1f} s; nit median {np.median(nit_s):.0f}, max {nit_s.max()}; "
           f"converged {sum(r.converged for r in sc)} of {len(sc)}")
    paths = {"gpu": (res.X, res.nit, res.status, res.converged),
             "scipy": (np.stack([r.X for r in sc]), nit_s, np.array([r.message for r in sc]),
                       np.array([r.converged for r in sc]))}
    data = {}
    with open(out / "starts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for name, (X, nit, status, conv) in paths.items():
            pt, ev = evaluate_all(X, res.faces, B_train, B_val)
            data[name] = (pt, np.array([e["score"] for e in ev]), np.array([e["feasible"] for e in ev]))
            for k, p, e, n, st, cv in zip(keys, pt, ev, nit, status, conv):
                w.writerow((name, *k, f"{p:.10f}", f"{e['score']:.10f}", f"{e['s1']:.8f}", f"{e['s2']:.8f}",
                            f"{e['rho']:.8f}", n, st, cv, f"{e['min_spacing_mm']:.4f}", e["feasible"]))
    hist_s = np.full((len(sc), MAXITER), np.nan)
    for i, r in enumerate(sc):
        hist_s[i, :len(r.loss)] = r.loss
    np.savez(out / "hist.npz", hist_gpu=res.hist, hist_scipy=hist_s, config=[k[0] for k in keys],
             seed=[k[1] for k in keys])
    same = np.abs(data["gpu"][0] - data["scipy"][0]) <= 1e-6
    c.info(f"{same.sum()} of {len(same)} starts end at the same Φ_train (|Δ| <= 1e-6) on both paths; feasible: "
           f"GPU {data['gpu'][2].sum()}, SciPy {data['scipy'][2].sum()}")

    # (2) best-of-R statistics
    worst, rows = 0.0, []
    for L in LAYOUTS:
        sel = np.array([k[0] == layout.fmt(L) for k in keys])
        for R in noise.R_GRID:
            bs, bg = (noise.best_of_r(*(a[sel] for a in data[p]), R) for p in ("scipy", "gpu"))
            sigma = float(np.nanstd(bs, ddof=1))
            tau = max(float(noise.tau(sigma)), TAU_FLOOR)
            d = abs(float(np.nanmean(bg) - np.nanmean(bs)))
            worst = max(worst, d / tau)
            rows.append((layout.fmt(L), R, f"{np.nanmean(bs):.10f}", f"{np.nanmean(bg):.10f}", f"{sigma:.6e}",
                         f"{float(np.nanstd(bg, ddof=1)):.6e}", f"{tau:.6e}", f"{d:.6e}"))
    with open(out / "bootstrap.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("config", "R", "mean_best_scipy", "mean_best_gpu", "sigma_R_scipy", "sigma_R_gpu", "tau", "abs_diff"))
        w.writerows(rows)
    c.add("(2) E[best of R] GPU vs SciPy within τ(R)", worst <= 1.0, f"largest |Δ| / τ(R) = {worst:.2f}")

    # (3) time
    t_it_s = float(np.median([r.wall_time / max(r.nit, 1) for r in sc]))
    with open(out / "timing.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("batch", "s_per_iter", "s_per_start_iter", "scipy_s_per_iter"))
        for b in SIZES:
            t = multistart.iteration_time([starts[i % len(starts)] for i in range(b)], B_train, N_TIME, MAXITER)
            w.writerow((b, f"{t:.6e}", f"{t / b:.6e}", f"{t_it_s:.6e}"))
            c.info(f"batch {b:4d}: {1e3 * t:8.2f} ms per iteration, {1e3 * t / b:7.3f} ms per start-iteration "
                   f"(SciPy {1e3 * t_it_s:.2f} ms)")
    c.add("(3) speed-up of the full batch against SciPy", t_scipy / res.wall_time > 1,
          f"{t_scipy:.1f} s / {res.wall_time:.1f} s = {t_scipy / res.wall_time:.1f}x")

    # batch invariance
    sub = multistart.optimise(starts[:N_INV], B_train, MAXITER)
    d_sub = np.abs(np.asarray(multistart.phi_batch(jnp.asarray(sub.X), B_train)) - data["gpu"][0][:N_INV])
    c.info(f"batch invariance: first {N_INV} starts alone vs in the batch of {len(starts)}: max |ΔΦ_train| "
           f"{d_sub.max():.2e}, nit equal for {(sub.nit == res.nit[:N_INV]).sum()} of {N_INV}")

    c.info(f"total wall time {time.perf_counter() - t_all:.1f} s")
    c.lines.append(f"RESULT {'PASS' if c.ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(c.lines) + "\n")
    print(c.lines[-1], f"-> {out}", sep="\n")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
