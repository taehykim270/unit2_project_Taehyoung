# Reproducing Wainwright (2009): Sharp Thresholds for Lasso Support Recovery

This project reproduces the simulation study (Figs. 1–3) of
M. J. Wainwright (2009), *Sharp Thresholds for High-Dimensional and Noisy Sparsity
Recovery Using ℓ1-Constrained Quadratic Programming (Lasso)*, IEEE Trans. Inf. Theory 55(5),
2183–2202, [doi:10.1109/TIT.2009.2016018](https://doi.org/10.1109/TIT.2009.2016018).
The paper studies whether the Lasso can select exactly the nonzero variables and recover
their signs (**signed-support recovery**). As the problem size grows, success probability
approaches 1 above a threshold in θ = n / (2k log(p − k)) and 0 below it. Here, n is the
sample size, p is the number of variables, and k is the number of nonzero coefficients.
For independent Gaussian predictors (the identity design), the threshold is θ* = 1.

The full simulation plan is in [ADEMP.md](ADEMP.md): the aims, how data are generated,
the target probability, the methods, and how performance is measured. The paper itself
is not included in this repository; see the DOI above.

## Key findings

The main patterns are reproduced: plotting success against θ brings the curves for different
p close together, with the rise near θ = 1 for the identity design and between the eq. (29)
bounds for the Toeplitz design. Our success rates are lower than the paper's in the transition
region, and the captions of the paper's Figs. 2 and 3 give a different rescaling formula from
its main text.

## Reproducing everything

Requires Python 3.10 or newer (tested with 3.14) and `make`. From the project root:

```bash
make reproduce
```

This command creates a virtual environment in `venv/`, installs the packages listed in
`requirements.txt`, removes generated files, runs the tests, and rebuilds the simulations,
summary tables and figures. It takes about 9 minutes. All `make` targets use
`venv/bin/python` automatically, so there is no need to activate the environment.

## Running individual steps

| Target | What it does |
|---|---|
| `make reproduce` | run `setup`, `clean`, `test`, then `all` to rebuild everything |
| `make setup` | create `venv/` and install the dependencies |
| `make simulate` | run all 864 conditions × 200 replicates and save the raw results to `results/raw/` |
| `make analyze` | compute success rates, confidence intervals, thresholds, overlap of curves across p, comparisons with the paper, and theoretical bounds |
| `make figures` | draw all figures (PDF and PNG) into `results/figures/` |
| `make all` | `simulate`, then `analyze`, then `figures` (each step only reruns when its inputs changed) |
| `make test` | run the test suite |
| `make clean` | remove all generated files, so that `make all` rebuilds everything from scratch |

**Estimated runtime** (laptop, single core):

- `make setup`: about 1 minute (downloads packages)
- `make test`: about 5 seconds
- `make reproduce`: about 9 minutes in total
- `make all` from scratch: about 8 minutes. Most of this is simulation time (451 s for 172,800 replicates); summaries and figures take a few seconds.

The simulation can also be run directly, for example
`venv/bin/python -m src.simulation --n-reps 10` for a quick test run, or
`--check pdw_closed_form` to check recovery using the closed-form primal-dual witness
(PDW), which uses a linear solve. See [ADEMP.md](ADEMP.md) for the method details.

## Outputs

| File | Contents |
|---|---|
| `results/raw/simulation_raw.csv` | one row per replicate (not committed; rebuilt by `make simulate`) |
| `results/raw/simulation_config.json` | settings, random-seed rule, package versions and git commit used for the run |
| `results/summary.csv` | one row per condition: success rate, Monte Carlo standard error (SE), 95% Wilson confidence interval (CI), and failure diagnostics |
| `results/thresholds.csv` | per curve: θ where the success rate first reaches 0.1, 0.5 and 0.9 |
| `results/collapse.csv` | per θ: largest standardized difference between curves for different p (two-proportion z statistic) |
| `results/paper_comparison.csv` | our success rates alongside approximate values read from the paper's Figs. 1(b), 2 and 3 |
| `results/theory_bounds.csv` | eq. (29) thresholds θ_ℓ, θ_u and ρ_ℓ, ρ_u, averaged over random supports |
| `results/figures/fig1_success_vs_n_and_theta` | Fig. 1: success probability vs. n and vs. θ |
| `results/figures/fig1_pdw_diagnostics` | diagnostic figure: which PDW check fails and how many wrong variables are selected |
| `results/figures/fig2_identity_by_regime` | Fig. 2: identity design, success probability by sparsity regime |
| `results/figures/fig3_toeplitz_by_regime` | Fig. 3: Toeplitz design, with the theoretical thresholds |
| `results/figures/fig4_toeplitz_strong_by_regime` | extension (not in the paper): Toeplitz design with μ = 0.5 |
| `results/figures/fig5_sparse_by_regime` | extension (not in the paper): sparse random design, q = 0.1 |

## Repository structure

```
├── src/
│   ├── dgps.py          # generate design matrices, beta* and noise
│   ├── methods.py       # support-recovery checks: primal-dual witness (two versions), full Lasso
│   ├── metrics.py       # performance measures and theoretical thresholds
│   ├── simulation.py    # configuration (CONFIG), seeding and the simulation loop
│   └── figures.py       # all figures
├── tests/               # DGP verification, method correctness, output validation, reproducibility
├── results/             # summary tables and figures (raw results are git-ignored)
├── data/
│   ├── paper_values.csv # approximate values read off the paper's Figs. 1(b), 2 and 3
│   └── simulated/       # placeholder for cached replicates (unused: replicates are regenerated from seeds)
├── ADEMP.md             # simulation design document
├── ANALYSIS.md          # critical analysis
├── Makefile
└── requirements.txt
```
