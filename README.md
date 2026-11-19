# autoresearch_heat_transfer_inverse

## Directories

| Path | Content |
|---|---|
| `program.md` | fixed planner instructions (`--mode llm`) |
| `program_learn.md` | fixed planner instructions (`--mode learn`) |
| `src/heat_inverse/` | NP-PIELM forward model, verbatim copy of `heat_transfer_inverse` commit `a4ca30c` |
| `src/autoresearch/` | `problem` (frozen constants, banks, Φ), `layout`, `trial`, `multistart` (batched GPU L-BFGS, unused), `noise` (best-of-R bootstrap, c(n), B_max, calibration), `evaluator` (frozen), `hypothesis` (constraint, S(h), τ_h, verdict), `learnings` (groups, promotion, flags), `state`, `planner`, `log`, `harness` (`Session`, `LearnSession`), `plotting` |
| `proxy/` | API-key proxy (user `llmkey`): `llmproxy.py`, `llmproxy.service`, `install_proxy.sh`, `uninstall_proxy.sh` |
| `scripts/` | `prepare_banks.py` (case2_ar_banks), `run_session.py` + `run_session.sh` (case2_ar_session), `plot_session.py` (W1–W6), `plot_learn_session.py` (W7–W12) |
| `scripts/smoke_test/` | `smoke_ar_trial.py`, `smoke_ar_harness.py`, `smoke_ar_proxy.py`, `smoke_ar_pilot.py` + `run_pilot.sh`, `smoke_ar_multistart.py`, `smoke_ar_noise.py`, `smoke_ar_learn.py`, `run_smoke_gpu.sh`, plot scripts |
| `data/reference/` | saved §6.1 design `dopt_x1z1_5`, used by `smoke_ar_trial.py` |
| `sessions/<session_id>/` | `results.csv`, `calls.csv`, `session.json`; `--mode learn` and `--r` also: `starts.csv`, `hypotheses.csv`, `learnings.csv`, `confirm.csv`, `confirm_starts.csv` (tracked) |
| `outputs/` | bank cache, trial files, call files, smoke and pilot outputs, plots (not tracked) |
| `project_docs/` | papers, session summaries, `case2_dopt_autoresearch_design.md`, `case2_dopt_autoresearch_learnings_design.md` |

## Setup

```bash
/home/user/miniconda3/bin/python3.12 -m venv venv_ar
venv_ar/bin/pip install -r requirements.txt
venv_ar/bin/pip install -e . --no-deps
venv_ar/bin/python scripts/prepare_banks.py --source <existing prior_banks.npz>   # without --source: 2560 solves
sudo proxy/install_proxy.sh [port] [model]       # read proxy/install_proxy.sh and proxy/llmproxy.py first
```

## Run order

```bash
venv_ar/bin/python scripts/smoke_test/smoke_ar_trial.py
venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_trial.py outputs/case2_ar_smoke_trial_<stamp>
venv_ar/bin/python scripts/smoke_test/smoke_ar_harness.py
venv_ar/bin/python scripts/smoke_test/smoke_ar_proxy.py
scripts/smoke_test/run_pilot.sh
```

Learnings run (`--mode learn`), smoke tests in order:

```bash
scripts/smoke_test/run_smoke_gpu.sh 0 0,1,4,5 scripts/smoke_test/smoke_ar_multistart.py
venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_multistart.py outputs/case2_ar_smoke_multistart_<stamp>
scripts/smoke_test/run_smoke_gpu.sh 0 0,1,4,5 scripts/smoke_test/smoke_ar_noise.py
venv_ar/bin/python scripts/smoke_test/plot_smoke_ar_noise.py outputs/case2_ar_smoke_noise_<stamp>
scripts/smoke_test/run_smoke_gpu.sh 0 0,1,4,5 scripts/smoke_test/smoke_ar_learn.py
```

Learnings session (branch `ar/n25`, `src/` committed); R, σ_R, τ, maxiter, kill time and B_max come from the calibration:

```bash
git checkout -b ar/n25
scripts/run_session.sh 0 0,1,4,5 --mode learn --n 25 --calibration outputs/case2_ar_smoke_noise_<stamp> [--k-stop 5] [--n-promote 2] [--n-confirm 5] [--effort high] [--cache-ttl 5m|1h]
venv_ar/bin/python scripts/plot_learn_session.py <session_id>
venv_ar/bin/python scripts/plot_session.py --llm <session_id>
```

Exhaustive run with R starts per sensor_layout (seeds 0…R−1): add `--r <R>` to `--mode exh`.

Sessions run on a branch per N, with `src/` committed:

```bash
git checkout -b ar/n30
scripts/run_session.sh 0 0,1,4,5 --mode llm --n 30 --b <B> --m <M> --maxiter <maxiter> --kill-time <s> [--effort high] [--cache-ttl 5m|1h] [--no-mirror-reduce]
scripts/run_session.sh 1 2,3,6,7 --mode exh --n 30 --b 64 --maxiter <maxiter> --kill-time <s> [--no-mirror-reduce]
scripts/run_session.sh 1 2,3,6,7 --mode exh --n 30 --b 64 ... --resume --session-id <id>    # continue an exhaustive run
venv_ar/bin/python scripts/plot_session.py --llm <session_id> --exh <session_id>
```

Remove the proxy: `sudo proxy/uninstall_proxy.sh`.
