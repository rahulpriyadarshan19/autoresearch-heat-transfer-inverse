"""ID case2_ar_smoke_mirror: mirror symmetry of the model, the start rule and the trial at N = 30.

Context: with mirror_reduce on, the cache of scripts/run_session.py gives a mirror image the score of its
canonical counterpart and does not run it. This is exact only if, for each mirror m (x -> L_x - x, z -> L_z - z,
or both):
    (1) the model is symmetric: Φ(m X) = Φ(X), and the sensitivities G of m X equal those of X;
    (2) the start rule is equivariant: start_design(m L) = m start_design(L);
    (3) the trial is equivariant: the final design of m L is m of the final design of L.
In exact arithmetic (1) and (2) give (3), because L-BFGS-B with symmetric bounds is equivariant under u -> 1 - u.
The harness start rule (layout.start_design, the old rule here) puts the Halton start on each sensor_layout alone,
which breaks (2); the new rule (mirrored_start) maps the canonical start instead.
Checks:
    M1  start rule: each mirror image starts from the mirror of the canonical start, on its faces
    M2  model: Φ_train, Φ_val, s1, s2, ρ and G of m X against X, for 20 seeded random designs and the 2 canonical
        final designs of M3; tolerance 1e-9 on Φ (float64 rounding is about 1e-13)
    M3  trials at maxiter 5000: the 4 mirror variants of 20-5-0-5-0 and of 0-8-7-8-7, with the start rule (new)
        and with the Halton start of each variant alone (old); |Δscore|, nit and max |X - m X_c| against the
        canonical trial; tolerance 1e-6 on |Δscore| for the new rule (rounding can grow along the path)
Outputs: outputs/case2_ar_smoke_mirror_<stamp>/{checks.txt, model.csv, trials.csv, trials/<rule>_<config>.npz};
plots with plot_smoke_ar_mirror.py.

Run:  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 \
      venv_ar/bin/python scripts/smoke_test/smoke_ar_mirror.py
"""

from __future__ import annotations

import csv
import time

import jax
import numpy as np

from autoresearch import evaluator, harness, layout, trial
from autoresearch import problem as P
from heat_inverse import config

N, MAXITER = 30, 5000
CANONICAL = ((20, 5, 0, 5, 0), (0, 8, 7, 8, 7))
MIRRORS = {"x": (True, False), "z": (False, True), "xz": (True, True)}
N_RANDOM, SEED = 20, 0
TOL_PHI, TOL_TRIAL = 1e-9, 1e-6
MODEL_COLUMNS = ("design", "mirror", "d_phi_train", "d_phi_val", "d_s1", "d_s2", "d_rho", "rel_dG")
TRIAL_COLUMNS = ("rule", "config", "canonical", "mirror", "nit", "nit_canonical", "converged", "wall", "score",
                 "d_score", "max_dX_m", "s1", "s2", "rho", "feasible")


class Checks:
    def __init__(self):
        self.lines, self.ok = [], True

    def add(self, name, passed, detail=""):
        self.ok &= bool(passed)
        self.lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    def info(self, text):
        self.lines.append(f"INFO  {text}")


def variant(counts, flags):
    """Counts of the mirror image of counts under flags."""
    c = list(counts)
    for flip, (a, b) in zip(flags, layout.MIRROR_PAIRS):
        if flip:
            c[a], c[b] = c[b], c[a]
    return tuple(c)


def mirror_flags(counts):
    """For each pair in MIRROR_PAIRS: True if the mirror of that pair turns canonical(counts) into counts."""
    return tuple(bool(counts[a] < counts[b]) for a, b in layout.MIRROR_PAIRS)


def mirror_design(X, faces, flags):
    """Image of a design under the mirrors selected by flags (x -> L_x - x, z -> L_z - z); sensor order kept."""
    X, faces = np.array(X, dtype=float), np.array(faces)
    for flip, (a, b) in zip(flags, layout.MIRROR_PAIRS):
        if flip:
            fa, fb = P.FACES[a], P.FACES[b]
            axis = P.FACE[fa].axis
            X[:, axis] = P.L3[axis] - X[:, axis]
            faces = np.where(faces == fa, fb, np.where(faces == fb, fa, faces))
    return X, faces


def mirrored_start(counts):
    """New rule: layout.start_design of canonical(counts), mapped by the mirrors that turn it into counts."""
    return mirror_design(*layout.start_design(layout.canonical(counts)), mirror_flags(counts))


def random_design(rng):
    """N sensors on random allowed faces, uniform inside the margins."""
    faces = rng.choice(P.FACES, size=N)
    X = np.zeros((N, 3))
    for i, name in enumerate(faces):
        f = P.FACE[name]
        ip = P.IN_PLANE[f.axis]
        X[i, f.axis] = P.L3[f.axis] * f.side
        X[i, ip] = P.M_LO[ip] + rng.random(2) * (P.L3[ip] - P.M_LO[ip] - P.M_HI[ip])
    return X, faces


def main() -> int:
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_smoke_mirror"))
    (out / "trials").mkdir()
    c, t_all = Checks(), time.perf_counter()
    B_train, B_val = P.load_banks()
    phi = jax.jit(P.phi)
    g_bank = jax.jit(P.g_bank)

    ok_start, ok_face, changed = True, True, 0
    for cc in CANONICAL:
        X_c, f_c = layout.start_design(cc)
        ok_start &= np.array_equal(mirrored_start(cc)[0], X_c)
        for name, flags in MIRRORS.items():
            v = variant(cc, flags)
            X_v, f_v = mirrored_start(v)
            X_m, f_m = mirror_design(X_c, f_c, flags)
            ok_start &= np.array_equal(X_v, X_m) and np.array_equal(f_v, f_m)
            ok_start &= tuple(int((f_v == fn).sum()) for fn in P.FACES) == v
            ok_face &= evaluator.on_faces(X_v, f_v)
            changed += not np.array_equal(np.sort(X_v, axis=0), np.sort(layout.start_design(v)[0], axis=0))
    c.add("M1: start of each mirror image = mirror of the canonical start, counts kept", ok_start)
    c.add("M1: mirrored starts on their faces, inside the margins", ok_face)
    c.info(f"M1: {changed} of {len(CANONICAL) * len(MIRRORS)} mirrored starts differ from the Halton start alone")

    t0 = time.perf_counter()
    Xw, fw = layout.start_design(CANONICAL[0])
    evaluator.evaluate(trial.optimise(Xw, fw, B_train, 2).X, fw, B_val)
    c.info(f"JIT warm-up: {time.perf_counter() - t0:.1f} s")

    rows, finals = [], {}
    with open(out / "trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIAL_COLUMNS)
        w.writeheader()
        for cc in CANONICAL:
            jobs = [("new", cc, "none", (False, False))]
            for name, flags in MIRRORS.items():
                jobs += [("new", variant(cc, flags), name, flags), ("old", variant(cc, flags), name, flags)]
            for rule, v, name, flags in jobs:
                X0, f0 = mirrored_start(v) if rule == "new" else layout.start_design(v)
                t0 = time.perf_counter()
                r = trial.optimise(X0, f0, B_train, MAXITER)
                ev = evaluator.evaluate(r.X, r.faces, B_val)
                wall = time.perf_counter() - t0
                cfg = layout.fmt(v)
                np.savez(out / "trials" / f"{rule}_{cfg}.npz", X_start=r.X_start, X=r.X, faces=r.faces, loss=r.loss)
                if name == "none":
                    finals[cc] = (r.X, r.faces, ev["score"], r.nit)
                X_c, f_c, s_c, nit_c = finals[cc]
                dX = (float(np.abs(r.X - mirror_design(X_c, f_c, flags)[0]).max()) if rule == "new"
                      else float("nan"))
                row = dict(rule=rule, config=cfg, canonical=layout.fmt(cc), mirror=name, nit=r.nit, nit_canonical=nit_c,
                           converged=r.converged, wall=wall, score=ev["score"], d_score=abs(ev["score"] - s_c),
                           max_dX_m=dX, s1=ev["s1"], s2=ev["s2"], rho=ev["rho"], feasible=ev["feasible"])
                w.writerow(row)
                fh.flush()
                rows.append(row)
                print(f"{rule} {cfg}: nit {r.nit}, {wall:.1f} s, score {ev['score']:.9f}, |Δscore| {row['d_score']:.3e}",
                      flush=True)

    rng = np.random.default_rng(SEED)
    designs = [(f"random_{k:02d}", *random_design(rng)) for k in range(N_RANDOM)]
    designs += [(f"final_{layout.fmt(cc)}", finals[cc][0], finals[cc][1]) for cc in CANONICAL]
    model_rows = []
    for label, X, f in designs:
        ev = evaluator.evaluate(X, f, B_val)
        G = np.asarray(g_bank(X, B_val))
        p_train = float(phi(X, B_train))
        for name, flags in MIRRORS.items():
            Xm, fm = mirror_design(X, f, flags)
            em = evaluator.evaluate(Xm, fm, B_val)
            Gm = np.asarray(g_bank(Xm, B_val))
            model_rows.append(dict(design=label, mirror=name, d_phi_train=abs(float(phi(Xm, B_train)) - p_train),
                                   d_phi_val=abs(em["score"] - ev["score"]), d_s1=abs(em["s1"] - ev["s1"]),
                                   d_s2=abs(em["s2"] - ev["s2"]), d_rho=abs(em["rho"] - ev["rho"]),
                                   rel_dG=float(np.abs(Gm - G).max() / np.abs(G).max())))
    with open(out / "model.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MODEL_COLUMNS)
        w.writeheader()
        w.writerows(model_rows)
    for name in MIRRORS:
        sel = [r for r in model_rows if r["mirror"] == name]
        d_phi = max(max(r["d_phi_train"], r["d_phi_val"]) for r in sel)
        c.add(f"M2: model symmetric under mirror {name}", d_phi <= TOL_PHI,
              f"max |ΔΦ| {d_phi:.3e}, max |Δs1| {max(r['d_s1'] for r in sel):.3e}, max |Δs2| "
              f"{max(r['d_s2'] for r in sel):.3e}, max |Δρ| {max(r['d_rho'] for r in sel):.3e}, max rel |ΔG| "
              f"{max(r['rel_dG'] for r in sel):.3e} over {len(sel)} designs")

    for r in rows:
        if r["mirror"] == "none":
            continue
        detail = (f"{r['config']} vs {r['canonical']}: |Δscore| {r['d_score']:.3e}, nit {r['nit']} vs "
                  f"{r['nit_canonical']}" + (f", max |ΔX| {r['max_dX_m']:.3e} m" if r["rule"] == "new" else ""))
        if r["rule"] == "new":
            c.add(f"M3: trial equivariant, new rule, mirror {r['mirror']}", r["d_score"] <= TOL_TRIAL, detail)
        else:
            c.info(f"M3: old rule, mirror {r['mirror']}: {detail}")
    walls = [r["wall"] for r in rows]
    c.info(f"M3: {len(rows)} trials, wall {sum(walls):.1f} s, mean {np.mean(walls):.1f} s, max {max(walls):.1f} s")
    c.info(f"total wall time {time.perf_counter() - t_all:.1f} s")
    c.lines.append(f"RESULT {'PASS' if c.ok else 'FAIL'}")
    (out / "checks.txt").write_text("\n".join(c.lines) + "\n")
    print("\n".join(c.lines), f"\n-> {out}", sep="")
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
