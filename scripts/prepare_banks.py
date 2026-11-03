"""ID case2_ar_banks: put the prior banks in outputs/case2_ar_cache/ and check them against a recompute.

The banks are checked again by smoke_test/smoke_ar_trial.py, which reproduces a saved §6.1 design score.

Run:  venv_ar/bin/python scripts/prepare_banks.py [--source <prior_banks.npz>]
"""

from __future__ import annotations

import argparse
import shutil
import time

import jax
import numpy as np

from autoresearch import harness
from autoresearch import problem as P
from heat_inverse import config

TOL_REL = 1e-8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="existing prior_banks.npz to copy if the cache file is absent")
    args = ap.parse_args()
    assert jax.devices()[0].platform == "gpu", "JAX does not run on the GPU"
    out = config.run_dir(harness.run_id("case2_ar_banks"))
    lines, t_all = [], time.perf_counter()

    if P.BANK_FILE.exists():
        lines.append(f"cache present: {P.BANK_FILE}")
    elif args.source:
        P.BANK_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, P.BANK_FILE)
        lines.append(f"copied {args.source} -> {P.BANK_FILE}")
    else:
        _, _, sec = P.compute_banks()
        lines.append(f"computed {4 * (P.S_TRAIN + P.S_VAL)} solves in {sec:.1f} s -> {P.BANK_FILE}")

    z = np.load(P.BANK_FILE)
    key_ok = str(z["key"]) == P.BANK_KEY
    lines += [f"sha256 {P.file_sha256(P.BANK_FILE)}", f"key equals BANK_KEY: {key_ok}",
              f"B_train {z['B_train'].shape}, B_val {z['B_val'].shape}, finite: "
              f"{bool(np.isfinite(z['B_train']).all() and np.isfinite(z['B_val']).all())}"]

    # the first CHUNK solves of each bank: the same batch shape as the cached computation
    n_check = P.CHUNK // 4
    t0 = time.perf_counter()
    beta_batch = P.beta_batch_fn()
    t_setup = time.perf_counter() - t0
    ok = key_ok
    for name, S, seed in (("B_train", P.S_TRAIN, P.SEED_TRAIN), ("B_val", P.S_VAL, P.SEED_VAL)):
        t0 = time.perf_counter()
        B_new = P.B_of(beta_batch, P.prior_bank(S, seed)[:n_check])
        sec = time.perf_counter() - t0
        B_old = z[name][:n_check]
        rel = float(np.abs(B_new - B_old).max() / np.abs(B_old).max())
        ok &= rel <= TOL_REL
        lines.append(f"{name}: recomputed first {n_check} samples ({4 * n_check} solves, {sec:.1f} s incl. compile): "
                     f"max |dB| / max |B| = {rel:.3e}  {'PASS' if rel <= TOL_REL else 'FAIL'} (<= {TOL_REL:g})")
    lines += [f"problem setup {t_setup:.1f} s", f"total wall time {time.perf_counter() - t_all:.1f} s",
              f"RESULT {'PASS' if ok else 'FAIL'}"]
    (out / "check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), f"\n-> {out / 'check.txt'}", sep="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
