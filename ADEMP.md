# ADEMP: Reproducing the Simulations in Wainwright (2009)

**Paper.** M. J. Wainwright (2009). *Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using ℓ1-Constrained Quadratic Programming (Lasso).* IEEE Trans. Inf. Theory 55(5), 2183–2202.

**Notation.** $`n`$ is the sample size, $`p`$ is the number of variables, and $`k`$ is the number of nonzero coefficients. The **support** $`S`$ is the set of variables with nonzero coefficients. A **replicate** is one simulated data set and its recovery result. SE means standard error; CI means confidence interval.

## ADEMP summary

| | |
|---|---|
| **Aims** | Reproduce Figs. 1–3. <br> (1) Does rescaling by $`\theta = n/(2k\log(p-k))`$ line up the success curves for different $`p`$? <br> (2) For $`\Sigma = I`$, is there a sharp transition at $`\theta^{\ast} = 1`$ in every sparsity regime? <br> (3) For a Toeplitz $`\Sigma`$, does the transition lie between the bounds $`\theta_\ell(\Sigma)`$ and $`\theta_u(\Sigma)`$ of eq. (29)? |
| **Data-generating mechanisms** | $`y = X\beta^{\ast} + w`$ with $`w \sim N(0, \sigma^2 I)`$, $`\sigma = 0.5`$. <br> Gaussian designs have independent rows from $`N(0, \Sigma)`$: identity $`\Sigma = I`$, or Toeplitz $`\Sigma_{ij} = \mu^{\lvert i-j\rvert}`$ with $`\mu = 0.10`$ (paper) or $`\mu = 0.50`$ (extension). <br> A fourth design is sparse random with $`\Sigma = I`$: entries are $`\pm 1/\sqrt{q}`$ with probability $`q/2`$ each, otherwise 0 ($`q = 0.10`$; extension). <br> $`\beta^{\ast}`$ has a uniformly random support of size $`k`$ with entries $`\pm 0.5`$. <br> All combinations of design (4), sparsity regime $`k(p)`$ (linear, sublinear, fractional power; 3), $`p = 128, 256, 512`$ (3), and $`\theta = 0.1, 0.2, \dots, 2.4`$ (24), with $`n = \lceil 2\theta k\log(p-k)\rceil`$: 864 conditions. |
| **Estimand (target probability)** | $`P_{\text{succ}}`$: the probability that the Lasso has a unique solution selecting exactly the true variables with the correct signs, $`\mathrm{sign}(\hat\beta) = \mathrm{sign}(\beta^{\ast})`$, for each condition. |
| **Methods** | The Lasso with the paper's fixed $`\lambda_n = \sqrt{2\sigma^2\log(k)\log(p-k)/n}`$. <br> The primal-dual witness (PDW) checks whether the Lasso recovers the true variables and signs (Lemmas 2–3). It solves the Lasso on $`S`$, then checks dual feasibility and sign consistency; see Section 4. |
| **Performance measures** | $`\hat P_{\text{succ}}`$ with Monte Carlo SE and 95% Wilson CI; estimated thresholds $`\hat\theta_{10}, \hat\theta_{50}, \hat\theta_{90}`$; overlap of curves across $`p`$; failure diagnostics; comparisons with the paper and the eq. (29) bounds. |
| **Replicates** | $`R = 200`$ per condition, as in the paper (172,800 in total); worst-case Monte Carlo SE 0.035. One master seed with an independent random stream per replicate. |

**Scope.** This study reproduces all three simulation figures in the paper and adds two extensions:

| Figure | Design covariance Σ | Sparsity regime(s) | x-axis |
|---|---|---|---|
| Fig. 1(a) | Identity | Fractional power | raw sample size $`n`$ |
| Fig. 1(b) | Identity | Fractional power | rescaled sample size $`\theta`$ |
| Fig. 2(a–c) | Identity | Linear, sublinear, fractional power | $`\theta`$ |
| Fig. 3(a–c) | Toeplitz, $`\mu = 0.10`$ | Linear, sublinear, fractional power | $`\theta`$ |
| Fig. 4 (extension, not in the paper) | Toeplitz, $`\mu = 0.50`$ | Linear, sublinear, fractional power | $`\theta`$ |
| Fig. 5 (extension, not in the paper) | Sparse random, $`q = 0.10`$ | Linear, sublinear, fractional power | $`\theta`$ |

The paper's main result (Theorems 3 and 4): when the rescaled sample size

```math
\theta(n,p,k) = \frac{n}{2k\log(p-k)}
```

is above an upper threshold $`\theta_u(\Sigma)`$, the probability of **signed-support recovery** approaches 1 as the problem size grows. This means selecting exactly the true variables with the correct signs. Below a lower threshold $`\theta_\ell(\Sigma)`$, the probability approaches 0. For $`\Sigma = I`$, the thresholds coincide: $`\theta_\ell = \theta_u = 1`$.

---

## 1. Aims

**Fig. 1: Is $`\theta`$ the right scaling?**

- (a): Plotted against the raw sample size $`n`$, the success-probability curves for $`p = 128, 256, 512`$ should be clearly separated: larger problems need more samples.
- (b): Plotted against $`\theta = n / (2k\log(p-k))`$, the same curves should overlap (called *curve collapse*).
- *Hypothesis:* the three curves are separated in $`n`$ and overlap within Monte Carlo error in $`\theta`$.

**Fig. 2: Is there a sharp threshold at $`\theta^{\ast} = 1`$ for the standard Gaussian design?**

- Check that for $`\Sigma = I`$ success probability rises from near 0 to near 1 around $`\theta = 1`$ in all three sparsity regimes: (a) linear, (b) sublinear, (c) fractional power.
- *Hypotheses:* the transition is centered near $`\theta = 1`$ in every regime and gets steeper as $`p`$ grows.

**Fig. 3: Do the bounds hold for a correlated design?**

- Check that for a Toeplitz covariance ($`\Sigma_{ij} = \mu^{|i-j|}`$, $`\mu = 0.10`$) the transition happens between the theoretical bounds $`\theta_\ell(\Sigma)`$ and $`\theta_u(\Sigma)`$.
- *Hypothesis:* success probability is near 0 for $`\theta < \theta_\ell(\Sigma)`$, near 1 for $`\theta > \theta_u(\Sigma)`$, and the curves for different $`p`$ collapse as in Fig. 2.

**Quick check:** for $`\theta`$ near 0 (and whenever $`n \le k`$) the success probability must be exactly 0. For $`\theta`$ near 2.4 it should be near 1.

---

## 2. Data-generating mechanisms

### 2.1 Observation model

For each replicate:

```math
y = X\beta^{\ast} + w, \qquad X \in \mathbb{R}^{n\times p}, \quad w \sim N(0, \sigma^2 I_n), \quad \sigma = 0.5.
```

### 2.2 Design matrix $`X`$ (four designs)

A new $`X`$ is generated for every replicate. For the Gaussian designs, rows $`x_i`$ are independent draws from $`N(0, \Sigma)`$. The sparse random design has independent entries as described below.

| Design | Σ or entry distribution | Used in |
|---|---|---|
| Standard (uniform) Gaussian | $`\Sigma = I_p`$ | Figs. 1, 2 |
| Toeplitz | $`\Sigma_{ij} = \mu^{\lvert i-j\rvert}`$, $`\mu = 0.10`$ | Fig. 3 |
| Toeplitz, strong correlation (extension) | $`\Sigma_{ij} = \mu^{\lvert i-j\rvert}`$, $`\mu = 0.50`$ | Fig. 4 |
| Sparse random (extension) | independent entries $`\pm 1/\sqrt{q}`$ with probability $`q/2`$ each, otherwise 0 ($`q = 0.10`$); $`\Sigma = I`$ | Fig. 5 |

For each Toeplitz design, generate $`X = Z L^\top`$, where $`L`$ is the Cholesky factor of $`\Sigma`$ and $`Z`$ is an $`n\times p`$ matrix of independent standard Gaussian entries. Compute $`L`$ once for each combination of $`p`$ and $`\mu`$.

> [Note] Eq. (46) calls the Toeplitz parameter $`\mu`$, but the Fig. 3 panel titles say "ρ = 0.10". They are the same parameter.

### 2.3 True coefficient vector $`\beta^{\ast}`$

1. Choose $`k`$ distinct indices from $`1, \dots, p`$ uniformly at random to form the support $`S`$, with $`|S| = k`$.
2. For each $`i \in S`$, set $`\beta^{\ast}_i = +\beta_{\min}`$ or $`-\beta_{\min}`$ with probability 1/2 each.
3. Set $`\beta^{\ast}_j = 0`$ for $`j \notin S`$.

The paper uses $`\beta_{\min} = 0.5`$. A new $`S`$ and new signs are drawn in every replicate.

### 2.4 Sparsity regimes ($`k`$ as a function of $`p`$), with $`\alpha = 0.40`$, $`\delta = 0.75`$

| Regime | Formula | $`p=128`$ | $`p=256`$ | $`p=512`$ |
|---|---|---|---|---|
| Linear | $`k = \lceil \alpha p \rceil`$ | 52 | 103 | 205 |
| Sublinear | $`k = \lceil \alpha p / \log(\alpha p) \rceil`$ | 14 | 23 | 39 |
| Fractional power | $`k = \lceil \alpha p^{\delta} \rceil`$ | 16 | 26 | 44 |

(All logs are natural logs.)

### 2.5 Sample size

Set the sample size using $`\theta`$:

```math
n = \lceil 2 \theta  k \log(p-k) \rceil .
```

- The paper varies $`\theta`$ over $`(0, 2.4)`$.
- **[Decision]** The paper does not give the grid. We use $`\theta = 0.1, 0.2, \dots, 2.4`$ (24 values) to show the transition in more detail.
- **[Decision]** If $`n \le k`$, the replicate counts as a failure. Justification:
  - For $`n < k`$, $`X_S^\top X_S`$ is singular and a unique Lasso solution has at most $`n`$ nonzeros, so a size-$`k`$ support cannot be recovered. For $`n = k`$, evaluating the PDW gave 0 successes in 2000 trials, so counting it as a failure changes nothing. The paper also plots the first point of every curve as 0.
  - On our grid this only happens at $`\theta = 0.1`$ for $`p = 128`$ (linear, sublinear and fractional regimes).
- **[Decision]** The rescaling follows Section VII: $`n = \lceil 2\theta k\log(p-k) \rceil`$. The paper is inconsistent here: Section VII and the Fig. 1 caption use $`n/[2k\log(p-k)]`$, while the Fig. 2 and Fig. 3 captions use $`n/[k\log(p-k)]`$, without the 2.

### 2.6 Simulation conditions

| Factor | Levels | # |
|---|---|---|
| Design | Identity, Toeplitz ($`\mu=0.10`$), Toeplitz ($`\mu=0.50`$, extension), sparse random ($`q=0.10`$, extension) | 4 |
| Sparsity regime | Linear, sublinear, fractional power | 3 |
| Problem size $`p`$ | 128, 256, 512 | 3 |
| Rescaled sample size $`\theta`$ | 0.1, 0.2, …, 2.4 | 24 |
| **Total conditions** | all combinations of the factors above | **864** |
| Replicates per condition | $`R = 200`$ (as in paper) | |
| **Total replicates** | | **172,800** |

Fixed across all conditions: $`\sigma = 0.5`$, $`\beta_{\min} = 0.5`$, $`\alpha = 0.40`$, $`\delta = 0.75`$, $`\lambda_n`$ as in Section 4.

Fig. 1 uses the identity design with fractional-power sparsity: the same data as Fig. 2(c), plotted once against $`n`$ and once against $`\theta`$.

### 2.7 Random seeds

The master seed is `master_seed = 607`. Each replicate gets its own random-number generator using `numpy.random.SeedSequence(master_seed, spawn_key=(design, regime, p, round(1000·θ), rep))`, where design and regime are fixed integer codes.

The seed depends only on the condition and replicate index. Any replicate can therefore be rerun on its own, and adding or reordering conditions leaves earlier results unchanged. Tests check both properties.

- Generate data in this order: $`X`$, $`\beta^{\ast}`$ (support, then signs), then $`w`$.
- Store a `seed_key` in each raw row (e.g. `0-2-128-1000-17`) so it can be regenerated.
- Record the run details in `simulation_config.json`: settings, generator (PCG64), seed rule, Python and package versions, timestamp, and git commit. A "dirty" flag marks uncommitted changes; the commit is `"unavailable"` until the repository has one.

---

## 3. Estimands / targets

The target is the **probability of exact signed-support recovery**:

```math
P_{\text{succ}}(\theta, p, \text{regime}, \Sigma) = \mathbb{P}[  \mathbb{S}_\pm(\hat\beta) = \mathbb{S}_\pm(\beta^{\ast})  ],
```

Here, $`\mathbb{S}_\pm(\beta_i) = \mathrm{sign}(\beta_i)`$ records whether each coefficient is negative, zero or positive: $`-1`$, $`0`$ or $`+1`$ (eq. 6).

- A replicate succeeds only if the Lasso solution is unique, selects exactly the support of $`\beta^{\ast}`$, and gives every selected coefficient the correct sign.
- This probability accounts for randomness in $`X`$, $`w`$, $`S`$ and the coefficient signs.

**Theoretical benchmarks:**

- Identity Σ: as $`p \to \infty`$, $`P_{\text{succ}}`$ tends to a step function at $`\theta^{\ast} = 1`$.
- Toeplitz Σ: the step lies between $`\theta_\ell(\Sigma)`$ and $`\theta_u(\Sigma)`$ (eq. 29):

```math
\theta_\ell(\Sigma) = \frac{\rho_\ell(\Sigma_{S^c|S})}{C_{\max}(2-\gamma(\Sigma))^2}, \qquad \theta_u(\Sigma) = \frac{\rho_u(\Sigma_{S^c|S})}{C_{\min} \gamma^2(\Sigma)},
```

with $`\gamma`$ the incoherence parameter (26a), $`C_{\min}, C_{\max}`$ the eigenvalue bounds on $`\Sigma_{SS}`$ (26b–c), and $`\rho_\ell, \rho_u`$ from (27).

> [Decision] The paper does not explain how it computed the vertical lines in Fig. 3. They are the same in all three panels, near 0.9 and 1. With random supports, eq. (29) gives $`\theta_\ell \approx 0.52`$–$`0.59`$ and $`\theta_u \approx 1.63`$–$`1.87`$. Only the numerators, $`\rho_\ell(\Sigma_{S^c|S}) \approx 0.89`$ and $`\rho_u(\Sigma_{S^c|S}) = 1`$, match the paper's lines.
>
> Our Fig. 3 shows both the paper-style lines ($`\rho_\ell`$, $`\rho_u`$) and the full eq. (29) bounds.
>
> For each (design, regime, $`p`$), compute the constants in eqs. (26)–(29) for 50 uniformly drawn supports $`S`$. The random stream is keyed by (design, regime, $`p`$). Save their means and standard deviations in `results/theory_bounds.csv`. Each panel shows the average over the three values of $`p`$. For the identity design, all four lines equal 1.

---

## 4. Methods

**Method under evaluation:** the Lasso (eq. 3)

```math
\hat\beta = \arg\min_{\beta\in\mathbb{R}^p} \Big[ \frac{1}{2n}\Vert y - X\beta\Vert_2^2 + \lambda_n \Vert\beta\Vert_1 \Big],
```

with the regularization parameter fixed as in the paper (not tuned by cross-validation):

```math
\lambda_n = \sqrt{\frac{2\sigma^2 \log(k) \log(p-k)}{n}}.
```

**Checking success: the primal-dual witness (PDW), Section II-B.**
By Lemmas 2–3, the PDW succeeds exactly when the Lasso has a unique solution with the correct signed support. The closed-form version checks this without an iterative solver; the default version solves a smaller Lasso problem on $`S`$. The formulas below use $`z_S = \mathrm{sign}(\beta^{\ast}_S)`$:

1. **Restricted solution:** compute the error on the true support (eq. 11):

   $`\displaystyle \Delta = \Big(\frac{1}{n}X_S^\top X_S\Big)^{-1}\Big[\frac{1}{n}X_S^\top w - \lambda_n z_S\Big].`$

2. **Subgradient on $`S`$:** set it to $`z_S = \mathrm{sign}(\beta^{\ast}_S)`$.
3. **Strict dual feasibility:** for $`j \in S^c`$, compute (eq. 10)

   $`\displaystyle Z_j = X_j^\top\Big[X_S(X_S^\top X_S)^{-1}z_S + \Pi_{X_S^\perp}\Big(\frac{w}{\lambda_n n}\Big)\Big],`$

   with $`\Pi_{X_S^\perp} = I - X_S(X_S^\top X_S)^{-1}X_S^\top`$. Check that $`\max_{j\in S^c}|Z_j| < 1`$.

4. **Sign consistency:** check $`\mathrm{sign}(\beta^{\ast}_i + \Delta_i) = \mathrm{sign}(\beta^{\ast}_i)`$ for all $`i \in S`$.

The replicate succeeds if and only if Steps 3 and 4 both pass.

**Implementation.** There are three recovery checks in `src/methods.py`. All use `check(data, λ)`, and the simulation settings select which one runs.

- **`pdw_restricted_lasso` (default, the paper's order):**
  - Step 1 solves the restricted Lasso (eq. 9) on the columns in $`S`$ numerically (scikit-learn, tolerance $`10^{-10}`$).
  - Step 2 takes the subgradient implied by the restricted problem's optimality condition, $`z_S = X_S^\top(y - X_S\check\beta_S)/(n\lambda_n)`$. This equals $`\mathrm{sign}(\check\beta_i)`$ wherever $`\check\beta_i \neq 0`$.
  - Step 4 therefore checks $`\mathrm{sign}(\check\beta_S) = \mathrm{sign}(\beta^{\ast}_S)`$, counting $`|\check\beta_i| < 10^{-8}`$ as 0.
- **`pdw_closed_form` (suggested alternative):**
  - Step 2 comes first: guess $`z_S = \mathrm{sign}(\beta^{\ast}_S)`$. With the signs fixed, Step 1 becomes linear: $`\check\beta_S = (X_S^\top X_S)^{-1}(X_S^\top y - n\lambda_n z_S)`$, which equals $`\beta^{\ast}_S + \Delta`$.
  - If the guessed signs are wrong, Step 4 fails. By Lemma 3(b), the Lasso also fails to recover the signed support. No iterative solver or zero cutoff is needed.
  - It agreed with the default on all 14,400 Fig. 1 replicates.
- **`full_lasso` (cross-check):** solve eq. (3) on all $`p`$ columns and compare signs; see below.

For both PDW versions, Step 3 uses $`z_{S^c} = X_{S^c}^\top (y - X_S\check\beta_S)/(n\lambda_n)`$. This equals eq. (10), which uses an $`n \times n`$ projection matrix; a test checks the equality. We evaluate Steps 3 and 4 on every replicate to record which checks fail. The paper's procedure would stop at the first failure.

The PDW uses the true support $`S`$ to check what the full Lasso would do. The `full_lasso` check fits the model without using $`S`$, then compares its result with the truth.

**Checking the implementation (tests only).**
The PDW avoids solving the full $`p`$-column Lasso in each simulation replicate. Even a sign error in its code could produce a plausible curve, so the tests compare both PDW versions with `full_lasso` at several $`\theta`$ values. This uses `sklearn.linear_model.Lasso(alpha=λ_n, fit_intercept=False)` to minimize the same objective (eq. 3), treating $`|\hat\beta_j| < 10^{-8}`$ as 0. The tests also check that $`\check\beta_S`$ from both PDW versions matches the full Lasso solution on $`S`$ whenever the PDW succeeds.

**Saved for each replicate** (before computing summaries):

- condition settings ($`\Sigma`$, regime, $`p`$, $`k`$, $`\theta`$, $`n`$, actual $`\theta`$ after rounding $`n`$ up, $`\lambda_n`$) and replicate index (see §2.7 for seeding)
- success indicator (0/1)
- dual-feasibility pass/fail and sign-consistency pass/fail, recorded separately
- $`\max_{j \in S^c}|Z_j|`$ (dual-feasibility margin) and $`\min_{i\in S}\mathrm{sign}(\beta^{\ast}_i)(\beta^{\ast}_i + \Delta_i)`$ (sign margin)
- failure counts:
  - the number of $`j \in S^c`$ with $`|z_j| \ge 1`$, i.e. the wrong variables that enter;
  - the number of $`i \in S`$ where $`\check\beta_i`$ is zero or has the wrong sign.

  Full $`z`$ and $`\check\beta`$ vectors can be regenerated from the seed key, so they are not stored.
- `seed_key` and `solver_converged`. Solver `ConvergenceWarning`s are recorded as convergence failures. The closed-form check uses no iterative solver and records `True`.

---

## 5. Performance measures

| Measure | Definition | Purpose |
|---|---|---|
| **Success probability** | $`\hat P = \frac{1}{R}\sum_{r=1}^R \mathbb{1}(\text{success}_r)`$ | Main quantity in all figures |
| **Monte Carlo SE** | $`\sqrt{\sum_r(\phi_r - \bar\phi)^2 / (R(R-1))}`$; for 0/1 outcomes this is $`\sqrt{\hat P(1-\hat P)/(R-1)}`$ (≤ 0.035 for $`R=200`$) | Measures simulation uncertainty; equals 0 when $`\hat P = 0`$ or $`1`$, so figure bands use Wilson intervals |
| **Solver convergence** | Number of replicates per condition where the numerical solver did not converge | Shows that no result relies on an unconverged fit |
| **95% interval** | Wilson score interval for $`\hat P`$ | More reliable near 0 and 1 than the usual normal (Wald) interval |
| **Estimated threshold** $`\hat\theta_{50}`$ | The actual $`\theta`$ where $`\hat P`$ first reaches 0.5, estimated by linear interpolation between grid points (same for $`\hat\theta_{10}, \hat\theta_{90}`$) | Compare with $`\theta^{\ast}=1`$ (Fig. 2) and $`[\theta_\ell, \theta_u]`$ (Fig. 3) |
| **Transition width** | $`\hat\theta_{90} - \hat\theta_{10}`$ | Tests whether the transition gets steeper as $`p`$ grows |
| **Failure type** | Share of failures involving dual feasibility (Step 3) or sign consistency (Step 4) | Shows which checks fail below the threshold |
| **Failure severity** | Mean number of wrong variables entering, among replicates that fail Step 3 (and mean number of sign errors, among those that fail Step 4), with a normal-approximation 95% CI | Diagnostic: shows *how badly* the Lasso fails, e.g. by one extra variable or by many |
| **Curve collapse (Fig. 1)** | At each $`\theta`$: the largest two-proportion $`z`$ statistic $`\lvert \hat P_{p_1} - \hat P_{p_2}\rvert  / \text{SE}_{\text{pooled}}`$ over all pairs of $`p`$ (`results/collapse.csv`); $`z < 2`$ means overlap within Monte Carlo error | Measures how well the rescaling works |

**Comparison with the paper.** `results/paper_comparison.csv` compares approximate success rates read from the paper's Figs. 1(b), 2 and 3 (`data/paper_values.csv`) with our rates and Wilson CIs at the same $`\theta`$. We use linear interpolation between our grid points.

**Figures:**

All figures are saved in `results/figures/` as PDF (vector) and PNG.

- **Fig. 1:** `fig1_success_vs_n_and_theta`, two panels as in the paper. (a) $`\hat P`$ vs. $`n`$; (b) $`\hat P`$ vs. actual $`\theta`$. Colours and marker shapes distinguish $`p`$. Includes 95% Wilson confidence bands and a reference line at $`\theta = 1`$.
- **Fig. 1, diagnostic:** `fig1_pdw_diagnostics`, 1×2. (a) pass rates of PDW Step 3 (dual feasibility, solid) and Step 4 (sign consistency, dashed) vs. $`\theta`$: which step fails; (b) mean number of wrong variables entering, given Step 3 fails (log scale, points with ≥ 10 failing replicates): how badly it fails.
- **Fig. 2:** `fig2_identity_by_regime`, three panels with shared axes, one per sparsity regime (linear, sublinear, fractional power). Shows $`\hat P`$ vs. actual $`\theta`$, with colours and markers for $`p`$, 95% Wilson confidence bands, and a reference line at $`\theta^{\ast} = 1`$.
- **Fig. 3:** `fig3_toeplitz_by_regime`, the same layout for the Toeplitz design. The vertical lines described in §3 show the eq. (29) bounds $`\theta_\ell, \theta_u`$ as dashed lines and $`\rho_\ell, \rho_u`$ of $`\Sigma_{S^c|S}`$ as dotted lines.
- **Fig. 4, extension (not in the paper):** `fig4_toeplitz_strong_by_regime`, the same layout as Fig. 3 for the Toeplitz design with $`\mu = 0.50`$, a condition less favourable to the Lasso.
- **Fig. 5, extension (not in the paper):** `fig5_sparse_by_regime`, the same layout for the sparse random design ($`q = 0.10`$), with the reference line at $`\theta^{\ast} = 1`$ since $`\Sigma = I`$.

---

## 6. Key results

Full run: 864 conditions × 200 replicates. See [ANALYSIS.md](ANALYSIS.md) for discussion.

1. **Overall.** The main patterns are reproduced, but our success rates are lower than the paper's in the transition region across both original designs and all sparsity regimes (`results/paper_comparison.csv`).
2. **Identity design (Figs. 1–2).** Rescaling by $`\theta`$ lines up the curves across $`p`$. The empirical threshold is $`\hat\theta_{50} = 0.99`$–$`1.23`$ across regimes and $`p`$, close to the theoretical $`\theta^{\ast} = 1`$ (`results/thresholds.csv`).
3. **Toeplitz design (Fig. 3).** The transition lies between the eq. (29) bounds, $`\theta_\ell \approx 0.52`$–$`0.59`$ and $`\theta_u \approx 1.63`$–$`1.87`$, with $`\hat\theta_{50} = 1.09`$–$`1.29`$. The paper's vertical lines match only $`\rho_\ell \approx 0.89`$ and $`\rho_u = 1`$, not the full bounds (`results/theory_bounds.csv`).
4. **Rescaling formula.** Section VII and the Fig. 1 caption use $`n/[2k\log(p-k)]`$, while the Fig. 2 and Fig. 3 captions use $`n/[k\log(p-k)]`$. This study follows Section VII.
5. **Diagnostics.** Below the threshold the Lasso fails mostly through dual feasibility (PDW Step 3), not sign consistency. Near the threshold a failure usually means only one or two wrong variables enter (`fig1_pdw_diagnostics`).
6. **Method checks.** Both PDW versions agreed on all 14,400 Fig. 1 replicates and match the full Lasso in the tests. No solver convergence failures were recorded across the 172,800 replicates.
7. **Extension: stronger correlation (Toeplitz, $`\mu = 0.50`$; Fig. 4).** Success is much lower in this setting. Within $`\theta \le 2.4`$, it reaches only about 0.35–0.45 for sublinear and fractional-power sparsity. It stays near 0 for linear sparsity, where larger $`p`$ performs worse. Incoherence still holds ($`\gamma \approx 0.20`$–$`0.24`$), but the eq. (29) bounds are too far apart to locate the transition precisely ($`\theta_\ell \approx 0.05`$–$`0.09`$, $`\theta_u \approx 40`$–$`71`$).
8. **Extension: sparse random design ($`q = 0.10`$, $`\Sigma = I`$; Fig. 5).** A non-Gaussian design not covered by the theorems. The transition stays close to the identity design: $`\hat\theta_{50} = 1.05`$–$`1.35`$, at most 0.17 to the right of the identity design, and the gap shrinks as $`p`$ grows, consistent with the Gaussian threshold carrying over for large $`p`$.
