# Reproducing Wainwright (2009): Sharp Thresholds for Lasso Support Recovery

This project reproduces the simulation study (Figs. 1–3) of
M. J. Wainwright (2009), *Sharp Thresholds for High-Dimensional and Noisy Sparsity
Recovery Using ℓ1-Constrained Quadratic Programming (Lasso)*, IEEE Trans. Inf. Theory 55(5),
2183–2202, [doi:10.1109/TIT.2009.2016018](https://doi.org/10.1109/TIT.2009.2016018).
The paper shows that the Lasso recovers the exact signed support of a sparse vector
with probability → 1 when the rescaled sample size θ = n / (2k log(p − k)) exceeds a
threshold (θ* = 1 for i.i.d. Gaussian designs), and fails with probability → 1 below it.

The full simulation design (aims, data-generating mechanisms, estimand, methods and
performance measures) is in [ADEMP.md](ADEMP.md). The paper itself is not included in
this repository (licensed copy); see the DOI above.

## Key findings

The qualitative behaviour is reproduced: rescaling by θ = n/(2k log(p − k)) lines the curves
up across p, the identity-design transition sits near θ = 1, and the Toeplitz transition lies
between the eq. (29) bounds. However, our success rates are noticeably lower than the paper's
in the transition region, and the paper's figures are not fully consistent with its stated
rescaling formula.

## Setup

Requires Python 3.10 or newer (tested with 3.14) and `make`. From the project root:

```bash
make setup
```

This creates a virtual environment in `venv/` (if it does not exist yet) and installs
`requirements.txt` into it. All other `make` targets use `venv/bin/python` automatically,
so there is no need to activate the environment.

## Running the analysis

```bash
make test    # run the test suite (41 tests, a few seconds)
make all     # simulation -> summary tables -> figures
```

| Target | What it does |
|---|---|
| `make setup` | create `venv/` and install the dependencies |
| `make simulate` | run all 432 conditions × 200 replicates and save the raw results to `results/raw/` |
| `make analyze` | compute success rates, CIs, thresholds, curve collapse, the comparison with the paper and the theoretical bounds |
| `make figures` | draw all figures (PDF and PNG) into `results/figures/` |
| `make all` | `simulate`, then `analyze`, then `figures` (each step only reruns when its inputs changed) |
| `make test` | run the test suite |
| `make clean` | remove all generated files, so that `make all` rebuilds everything from scratch |

**Estimated runtime** (laptop, single core):
- `make setup`: about 1 minute (downloads packages)
- `make test`: about 5 seconds
- `make all` from scratch: about 4 minutes. The simulation step dominates (215 s for 86,400 replicates); summaries and figures take a few seconds.

The simulation can also be run directly, for example
`venv/bin/python -m src.simulation --n-reps 10` for a quick test run, or
`--check pdw_closed_form` to use the closed-form primal-dual witness instead of the default.

## Outputs

| File | Contents |
|---|---|
| `results/raw/simulation_raw.csv` | one row per replicate (not committed; rebuilt by `make simulate`) |
| `results/raw/simulation_config.json` | exact configuration, seed rule, package versions and git commit of the run |
| `results/summary.csv` | one row per condition: success rate, Monte Carlo SE, 95% Wilson CI, diagnostics |
| `results/thresholds.csv` | per curve: θ where the success rate first reaches 0.1, 0.5 and 0.9 |
| `results/collapse.csv` | per θ: largest gap between the curves for different p (two-proportion z) |
| `results/paper_comparison.csv` | our success rates next to approximate values read off the paper's Fig. 1(b) |
| `results/theory_bounds.csv` | eq. (29) thresholds θ_ℓ, θ_u and ρ_ℓ, ρ_u, averaged over random supports |
| `results/figures/fig1_success_vs_n_and_theta` | Fig. 1: success probability vs. n and vs. θ |
| `results/figures/fig1_pdw_diagnostics` | diagnostic figure: which step of the primal-dual witness fails, and how badly |
| `results/figures/fig2_fig3_success_grid` | Figs. 2–3: success probability by design and sparsity regime, with the theoretical thresholds |

## Repository structure

```
├── src/
│   ├── dgps.py          # data-generating functions (identity and Toeplitz designs, beta*, noise)
│   ├── methods.py       # support-recovery checks: primal-dual witness (two versions), full Lasso
│   ├── metrics.py       # performance measures and theoretical thresholds
│   ├── simulation.py    # configuration (CONFIG), seeding and the simulation loop
│   └── figures.py       # all figures
├── tests/               # DGP verification, method correctness, output validation, reproducibility
├── docs/paper_values.csv  # approximate values read off the paper's Fig. 1(b)
├── results/             # summary tables and figures (raw results are git-ignored)
├── data/simulated/      # placeholder for cached replicates (unused: replicates are regenerated from seeds)
├── ADEMP.md             # simulation design document
├── Makefile
└── requirements.txt
```
