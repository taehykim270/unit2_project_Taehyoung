# ADEMP: Reproducing the Simulations in Wainwright (2009)

**Paper.** M. J. Wainwright (2009). *Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using ℓ1-Constrained Quadratic Programming (Lasso).* IEEE Trans. Inf. Theory 55(5), 2183–2202.

## ADEMP glimpse

| | |
|---|---|
| **Aims** | Reproduce Figs. 1–3. <br> (1) Does rescaling by $`\theta = n/(2k\log(p-k))`$ line up the success curves for different $`p`$? <br> (2) For $`\Sigma = I`$, is there a sharp transition at $`\theta^{\ast} = 1`$ in every sparsity regime? <br> (3) For a Toeplitz $`\Sigma`$, does the transition lie between the bounds $`\theta_\ell(\Sigma)`$ and $`\theta_u(\Sigma)`$ of eq. (29)? |
| **Data-generating mechanisms** | $`y = X\beta^{\ast} + w`$ with $`w \sim N(0, \sigma^2 I)`$, $`\sigma = 0.5`$. <br> Rows of $`X`$ i.i.d. $`N(0, \Sigma)`$, with $`\Sigma = I`$ or Toeplitz $`\Sigma_{ij} = \mu^{\lvert i-j\rvert}`$, $`\mu = 0.10`$. <br> $`\beta^{\ast}`$ has a uniformly random support of size $`k`$ with entries $`\pm 0.5`$. <br> Fully crossed factors: design (2) × sparsity regime $`k(p)`$ (linear, sublinear, fractional power; 3) × $`p = 128, 256, 512`$ (3) × $`\theta = 0.1, 0.2, \dots, 2.4`$ (24), with $`n = \lceil 2\theta k\log(p-k)\rceil`$: 432 conditions. |
| **Estimand** | $`P_{\text{succ}}`$: the probability that the Lasso has a unique solution with exactly the correct signed support, $`\mathrm{sign}(\hat\beta) = \mathrm{sign}(\beta^{\ast})`$, for each condition. |
| **Methods** | The Lasso with the paper's fixed $`\lambda_n = \sqrt{2\sigma^2\log(k)\log(p-k)/n}`$. <br> Success is decided exactly by the primal-dual witness (restricted Lasso on $`S`$, then the dual-feasibility and sign-consistency checks), which is equivalent to solving the Lasso (Lemmas 2–3). |
| **Performance measures** | $`\hat P_{\text{succ}}`$ with Monte Carlo SE and 95% Wilson CI; empirical thresholds $`\hat\theta_{10}, \hat\theta_{50}, \hat\theta_{90}`$; curve collapse across $`p`$; diagnostics (which witness step fails, and by how many wrong variables); comparison with the paper and with the eq. (29) bounds. |
| **Replicates** | $`R = 200`$ per condition, as in the paper (86,400 in total); worst-case Monte Carlo SE 0.035. One master seed with an independent random stream per replicate. |

Details for each part follow below.

**Scope.** This study reproduces all three simulation figures in the paper:

| Figure | Design covariance Σ | Sparsity regime(s) | x-axis |
|---|---|---|---|
| Fig. 1(a) | Identity | Fractional power | raw sample size $`n`$ |
| Fig. 1(b) | Identity | Fractional power | rescaled sample size $`\theta`$ |
| Fig. 2(a–c) | Identity | Linear, sublinear, fractional power | $`\theta`$ |
| Fig. 3(a–c) | Toeplitz, $`\mu = 0.10`$ | Linear, sublinear, fractional power | $`\theta`$ |

The paper's main result (Theorems 3 and 4): when the rescaled sample size

```math
\theta(n,p,k) = \frac{n}{2k\log(p-k)}
```

is above an upper threshold $`\theta_u(\Sigma)`$, the Lasso recovers the **signed support** of $`\beta^{\ast}`$ with probability tending to 1. When it is below a lower threshold $`\theta_\ell(\Sigma)`$, the probability tends to 0. For $`\Sigma = I`$, the threshold is sharp since $`\theta_\ell = \theta_u = 1`$.

---

## 1. Aims

**Fig. 1: Is $`\theta`$ the right scaling?**
- (a): Plotted against the raw sample size $`n`$, the success-probability curves for $`p = 128, 256, 512`$ should be clearly separated: larger problems need more samples.
- (b): Plotted against $`\theta = n / (2k\log(p-k))`$, the same curves should collapse onto one curve.
- *Hypothesis:* the three curves are separated in $`n`$ and overlap within Monte Carlo error in $`\theta`$.

**Fig. 2: Is there a sharp threshold at $`\theta^{\ast} = 1`$ for the standard Gaussian design?**
- Check that for $`\Sigma = I`$ the success probability goes from ≈0 to ≈1 around $`\theta = 1`$ in all three sparsity regimes ((a):linear, (b):sublinear, (c):fractional power).
- *Hypotheses:* (i) the transition is centered near $`\theta = 1`$ whatever the sparsity regime; (ii) the transition gets steeper as $`p`$ grows.

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

### 2.2 Design matrix $`X`$ (two random ensembles)
The rows $`x_i`$ are i.i.d. $`N(0, \Sigma)`$, and a new $`X`$ is drawn for every replicate.

| Ensemble | Σ | Used in |
|---|---|---|
| Standard (uniform) Gaussian | $`\Sigma = I_p`$ | Figs. 1, 2 |
| Toeplitz | $`\Sigma_{ij} = \mu^{\lvert i-j\rvert}`$, $`\mu = 0.10`$ | Fig. 3 |

For the Toeplitz case, generate $`X = Z L^\top`$, where $`L`$ is the Cholesky factor of $`\Sigma`$ (computed once per $`p`$) and $`Z`$ is $`n\times p`$ standard Gaussian, drawn exactly as in the identity case.

> [Note] Eq. (46) calls the Toeplitz parameter $`\mu`$, but the Fig. 3 panel titles say "ρ = 0.10". They are the same parameter.

### 2.3 True coefficient vector $`\beta^{\ast}`$
1. Choose the support $`S`$ of the indices $`1, \dots, p`$ with $`|S| = k`$ uniformly at random.
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
The sample size is set by the control parameter $`\theta`$:

```math
n = \lceil 2 \theta  k \log(p-k) \rceil .
```

- The paper varies $`\theta`$ over $`(0, 2.4)`$.
- **[Decision]** The paper does not give the grid. This study uses the grid $`\theta = 0.1, 0.2, \dots, 2.4`$ (24 values) for all figures, for high resolution in the transition region.
- **[Decision]** If $`n \le k`$, the replicate counts as a failure. Justification:
  - For $`n < k`$, $`X_S^\top X_S`$ is singular and a unique Lasso solution has at most $`n`$ nonzeros, so a size-$`k`$ support cannot be recovered. For $`n = k`$, evaluating the PDW gave 0 successes in 2000 trials, so counting it as a failure changes nothing. The paper also plots the first point of every curve as 0.
  - On our grid this only happens at $`\theta = 0.1`$ for $`p = 128`$ (linear, sublinear and fractional regimes).
- **[Decision]** The rescaling follows Section VII: $`n = \lceil 2\theta k\log(p-k) \rceil`$. The paper is inconsistent here: Section VII and the Fig. 1 caption use $`n/[2k\log(p-k)]`$, while the Fig. 2 and Fig. 3 captions use $`n/[k\log(p-k)]`$, without the 2.

### 2.6 Simulation design matrix

| Factor | Levels | # |
|---|---|---|
| Covariance Σ | Identity, Toeplitz ($`\mu=0.10`$) | 2 |
| Sparsity regime | Linear, sublinear, fractional power | 3 |
| Problem size $`p`$ | 128, 256, 512 | 3 |
| Rescaled sample size $`\theta`$ | 0.1, 0.2, …, 2.4 | 24 |
| **Total conditions** | fully crossed | **432** |
| Replicates per condition | $`R = 200`$ (as in paper) | |
| **Total replicates** | | **86,400** |

Fixed across all conditions: $`\sigma = 0.5`$, $`\beta_{\min} = 0.5`$, $`\alpha = 0.40`$, $`\delta = 0.75`$, $`\lambda_n`$ as in Section 4.

Fig. 1 is the (Identity, fractional power) slice. It uses the same simulated data as Fig. 2(c), plotted once against $`n`$ and once against $`\theta`$.

### 2.7 Random seeds
- One global master seed is set in the config (`master_seed = 607`).
- Each **replicate** gets its own independent generator: `numpy.random.SeedSequence(master_seed, spawn_key=(design, regime, p, round(1000·θ), rep))`, where design and regime are fixed integer codes.
- The seed depends only on the condition and the replicate index, not on the order or number of conditions. So:
  - any condition or replicate can be rerun on its own and give identical results;
  - adding conditions later (Figs. 2–3) never changes results already computed (checked by a test).
- Within a replicate, draws are made in a fixed order: $`X`$, then $`\beta^{\ast}`$ (support, then signs), then $`w`$.
- Each raw row stores its `seed_key` (e.g. `0-2-128-1000-17`), so any row can be regenerated on its own (checked by a test).
- Each run's `simulation_config.json` records the provenance:
  - the full config;
  - the generator (PCG64) and the seed rule;
  - the Python and package versions;
  - a timestamp;
  - the git commit with a "dirty" flag (`"unavailable"` until the repository has a commit).

---

## 3. Estimands / targets

The target is the **probability of exact signed-support recovery**:

```math
P_{\text{succ}}(\theta, p, \text{regime}, \Sigma) = \mathbb{P}[  \mathbb{S}_\pm(\hat\beta) = \mathbb{S}_\pm(\beta^{\ast})  ],
```

where $`\mathbb{S}_\pm(\beta_i) = \mathrm{sign}(\beta_i)`$, which takes the values $`-1`$, $`0`$ or $`+1`$, is the extended sign vector (eq. 6).

- A replicate succeeds only if the Lasso solution is unique, has exactly the same support as $`\beta^{\ast}`$, and has the correct sign on every coordinate of that support.
- The probability is taken over the random $`X`$, $`w`$, $`S`$ and signs.

**Theoretical benchmarks:**
- Identity Σ: as $`p \to \infty`$, $`P_{\text{succ}}`$ tends to a step function at $`\theta^{\ast} = 1`$.
- Toeplitz Σ: the step lies between $`\theta_\ell(\Sigma)`$ and $`\theta_u(\Sigma)`$ (eq. 29):

```math
\theta_\ell(\Sigma) = \frac{\rho_\ell(\Sigma_{S^c|S})}{C_{\max}(2-\gamma(\Sigma))^2}, \qquad \theta_u(\Sigma) = \frac{\rho_u(\Sigma_{S^c|S})}{C_{\min} \gamma^2(\Sigma)},
```

with $`\gamma`$ the incoherence parameter (26a), $`C_{\min}, C_{\max}`$ the eigenvalue bounds on $`\Sigma_{SS}`$ (26b–c), and $`\rho_\ell, \rho_u`$ from (27).

> [Decision] The paper does not say how it computed the vertical lines in Fig. 3; they are identical in all three panels and lie close to 0.9 and 1. Eq. (29) with random supports gives $`\theta_\ell \approx 0.52`$–$`0.59`$ and $`\theta_u \approx 1.63`$–$`1.87`$, depending on the panel; only its numerators, $`\rho_\ell(\Sigma_{S^c|S}) \approx 0.89`$ and $`\rho_u(\Sigma_{S^c|S}) = 1`$, match the paper's lines.
>
> Our Fig. 3 draws **both**:
> - the paper-style lines ($`\rho_\ell`$, $`\rho_u`$);
> - the full eq. (29) bounds, averaged over random draws of $`S`$ for each panel.
>
> How they are computed:
> - For each (design, regime, $`p`$), all constants of eqs. (26)–(29) are computed for 50 random supports $`S`$, drawn uniformly as in the DGP from a stream keyed by (design, regime, $`p`$).
> - Their mean and SD are saved in `results/theory_bounds.csv`.
> - Each panel draws the average over the three values of $`p`$, which barely differ. For the identity design all four lines equal 1.

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

**How success is decided: the primal-dual witness (PDW) construction (Section II-B).**
By Lemmas 2–3, the PDW succeeds exactly when the Lasso has a unique solution with the correct signed support. That means the success indicator can be computed exactly in closed form, without running an iterative solver. Writing $`z_S = \mathrm{sign}(\beta^{\ast}_S)`$:

1. **Restricted solution:** compute the error on the true support (eq. 11):

   $`\displaystyle \Delta = \Big(\frac{1}{n}X_S^\top X_S\Big)^{-1}\Big[\frac{1}{n}X_S^\top w - \lambda_n z_S\Big].`$

2. **Subgradient on $`S`$:** set it to $`z_S = \mathrm{sign}(\beta^{\ast}_S)`$.
3. **Strict dual feasibility:** for $`j \in S^c`$, compute (eq. 10)

   $`\displaystyle Z_j = X_j^\top\Big[X_S(X_S^\top X_S)^{-1}z_S + \Pi_{X_S^\perp}\Big(\frac{w}{\lambda_n n}\Big)\Big],`$

   with $`\Pi_{X_S^\perp} = I - X_S(X_S^\top X_S)^{-1}X_S^\top`$. Check that $`\max_{j\in S^c}|Z_j| < 1`$.

4. **Sign consistency:** check $`\mathrm{sign}(\beta^{\ast}_i + \Delta_i) = \mathrm{sign}(\beta^{\ast}_i)`$ for all $`i \in S`$.

The replicate succeeds if and only if Steps 3 and 4 both pass.

**Implementation.** The support-recovery check is a class with one interface (`check(data, λ)`), with three interchangeable implementations in `src/methods.py`. The simulation uses the one named in the config.
- **`pdw_restricted_lasso` (default, the paper's order):**
  - Step 1 solves the restricted Lasso (eq. 9) on the columns in $`S`$ numerically (scikit-learn, tolerance $`10^{-10}`$).
  - Step 2 takes the subgradient implied by the restricted problem's optimality condition, $`z_S = X_S^\top(y - X_S\check\beta_S)/(n\lambda_n)`$. This equals $`\mathrm{sign}(\check\beta_i)`$ wherever $`\check\beta_i \neq 0`$.
  - Step 4 therefore checks $`\mathrm{sign}(\check\beta_S) = \mathrm{sign}(\beta^{\ast}_S)`$, counting $`|\check\beta_i| < 10^{-8}`$ as 0.
- **`pdw_closed_form` (suggested alternative):**
  - Step 2 comes first: guess $`z_S = \mathrm{sign}(\beta^{\ast}_S)`$. With the signs fixed, Step 1 becomes linear: $`\check\beta_S = (X_S^\top X_S)^{-1}(X_S^\top y - n\lambda_n z_S)`$, which equals $`\beta^{\ast}_S + \Delta`$.
  - If the guess is wrong, Step 4 fails, and by Lemma 3(b) the Lasso fails too, so the verdict is still exact. No solver or zero cutoff is needed.
  - It gave identical verdicts to the default on all 14,400 Fig. 1 replicates.
- **`full_lasso` (cross-check):** solve eq. (3) on all $`p`$ columns and compare signs; see below.

For both PDW versions, Step 3 is $`z_{S^c} = X_{S^c}^\top (y - X_S\check\beta_S)/(n\lambda_n)`$. This is algebraically identical to eq. (10), which is written with an $`n \times n`$ projection matrix; a test checks the equality. Both Step 3 and Step 4 are evaluated on every replicate (the paper's procedure would stop at the first failure), so each failure can be attributed to a step.

The PDW uses the true support $`S`$. That is legitimate: it certifies what the full Lasso would do and is not an estimator. Only `full_lasso` is a method that never sees the truth.

**Cross-check (method correctness, test suite only).**
- By Lemmas 2–3 the PDW result is exact, so the simulation does not solve the full $`p`$-column Lasso.
- The lemmas justify the mathematics, not our transcription of it into code. A one-character sign error in Step 1 still produces a smooth, plausible-looking success curve.
- The test suite therefore checks that both PDW versions agree with `full_lasso`, which uses `sklearn.linear_model.Lasso(alpha=λ_n, fit_intercept=False)` and minimizes the same objective (eq. 3), at several $`\theta`$ ($`|\hat\beta_j| < 10^{-8}`$ is treated as 0).
- The tests also check that both PDW versions' $`\check\beta_S`$ equal the full Lasso solution on $`S`$ whenever the PDW succeeds.

**Quantities recorded per replicate** (saved raw, before summarizing):
- condition identifiers ($`\Sigma`$, regime, $`p`$, $`k`$, $`\theta`$, $`n`$, realized $`\theta`$ after rounding $`n`$ up, $`\lambda_n`$) and replicate index (the seed is determined by these; see §2.7)
- success indicator (0/1)
- dual-feasibility pass/fail and sign-consistency pass/fail, recorded separately
- $`\max_{j \in S^c}|Z_j|`$ (dual-feasibility margin) and $`\min_{i\in S}\mathrm{sign}(\beta^{\ast}_i)(\beta^{\ast}_i + \Delta_i)`$ (sign margin)
- failure counts:
  - the number of $`j \in S^c`$ with $`|z_j| \ge 1`$, i.e. the wrong variables that enter;
  - the number of $`i \in S`$ where $`\check\beta_i`$ is zero or has the wrong sign.

  Full $`z`$ / $`\check\beta`$ vectors are not stored, since every replicate can be regenerated from its seed key.
- `seed_key` and `solver_converged`. A solver `ConvergenceWarning` is caught and recorded, not silently ignored; the closed form uses no solver and always records `True`.

---

## 5. Performance measures

| Measure | Definition | Purpose |
|---|---|---|
| **Success probability** | $`\hat P = \frac{1}{R}\sum_{r=1}^R \mathbb{1}(\text{success}_r)`$ | Main quantity in all figures |
| **Monte Carlo SE** | $`\sqrt{\sum_r(\phi_r - \bar\phi)^2 / (R(R-1))}`$; for 0/1 outcomes this is $`\sqrt{\hat P(1-\hat P)/(R-1)}`$ (≤ 0.035 for $`R=200`$) | Uncertainty of every estimated mean; 0 when $`\hat P = 0`$ or $`1`$, so the figure bands use Wilson instead |
| **Solver convergence** | Number of replicates per condition where the numerical solver did not converge | Shows that no result relies on an unconverged fit |
| **95% interval** | Wilson score interval for $`\hat P`$ | Better-behaved near 0 and 1 than the Wald interval |
| **Empirical threshold** $`\hat\theta_{50}`$ | The realized $`\theta`$ where $`\hat P`$ first reaches 0.5 (linear interpolation between grid points; same for $`\hat\theta_{10}, \hat\theta_{90}`$) | Compare against $`\theta^{\ast}=1`$ (Fig. 2) and against $`[\theta_\ell, \theta_u]`$ (Fig. 3) |
| **Transition width** | $`\hat\theta_{90} - \hat\theta_{10}`$ | Tests whether the transition gets steeper as $`p`$ grows |
| **Failure-mode breakdown** | Share of failures due to dual feasibility (Step 3) vs. sign consistency (Step 4) | Diagnostic: shows *why* the Lasso fails below the threshold |
| **Failure severity** | Mean number of wrong variables entering, among replicates that fail Step 3 (and mean number of sign errors, among those that fail Step 4), with a normal-approximation 95% CI | Diagnostic: shows *how badly* the Lasso fails, e.g. by one extra variable or by many |
| **Curve collapse (Fig. 1)** | At each $`\theta`$: the largest two-proportion $`z`$ statistic $`\lvert \hat P_{p_1} - \hat P_{p_2}\rvert  / \text{SE}_{\text{pooled}}`$ over all pairs of $`p`$ (`results/collapse.csv`); $`z < 2`$ means overlap within Monte Carlo error | Measures how well the rescaling works |

**Comparison with the paper** (`results/paper_comparison.csv`): approximate success rates read off the paper's Figs. 1(b), 2 and 3 (`data/paper_values.csv`) are compared with our success rate and Wilson CI, interpolated along our curve at the same $`\theta`$.

**Figures:**
All figures are saved in `results/figures/` as PDF (vector) and PNG.
- **Fig. 1, publication-quality:** `fig1_success_vs_n_and_theta`, 1×2 like the paper. (a) $`\hat P`$ vs. $`n`$; (b) $`\hat P`$ vs. realized $`\theta`$. Colored and marker-coded by $`p`$, 95% Wilson CI bands, reference line at $`\theta = 1`$.
- **Fig. 1, diagnostic:** `fig1_pdw_diagnostics`, 1×2. (a) pass rates of PDW Step 3 (dual feasibility, solid) and Step 4 (sign consistency, dashed) vs. $`\theta`$: which step fails; (b) mean number of wrong variables entering, given Step 3 fails (log scale, points with ≥ 10 failing replicates): how badly it fails.
- **Fig. 2, publication-quality:** `fig2_identity_by_regime`, 1×3 like the paper, one panel per sparsity regime (linear, sublinear, fractional power), shared axes. $`\hat P`$ vs. realized $`\theta`$, coloured and marker-coded by $`p`$, 95% Wilson CI bands, reference line at $`\theta^{\ast} = 1`$.
- **Fig. 3, publication-quality:** `fig3_toeplitz_by_regime`, the same layout for the Toeplitz design, with the vertical lines of §3: the eq. (29) bounds $`\theta_\ell, \theta_u`$ dashed, and $`\rho_\ell, \rho_u`$ of $`\Sigma_{S^c|S}`$ dotted.

---

## 6. Key results

Full run: 432 conditions × 200 replicates. The discussion of these results is in ANALYSIS.md.

1. **Overall.** The qualitative behaviour is reproduced, but our success rates are noticeably lower than the paper's in the transition region, in every design and sparsity regime (`results/paper_comparison.csv`).
2. **Identity design (Figs. 1–2).** Rescaling by $`\theta`$ lines up the curves across $`p`$. The empirical threshold is $`\hat\theta_{50} = 0.99`$–$`1.23`$ across regimes and $`p`$, close to the theoretical $`\theta^{\ast} = 1`$ (`results/thresholds.csv`).
3. **Toeplitz design (Fig. 3).** The transition lies between the eq. (29) bounds, $`\theta_\ell \approx 0.52`$–$`0.59`$ and $`\theta_u \approx 1.63`$–$`1.87`$, with $`\hat\theta_{50} = 1.09`$–$`1.29`$. The paper's vertical lines match only $`\rho_\ell \approx 0.89`$ and $`\rho_u = 1`$, not the full bounds (`results/theory_bounds.csv`).
4. **Rescaling formula.** The paper is inconsistent about it: Section VII and the Fig. 1 caption use $`n/[2k\log(p-k)]`$, the Fig. 2 and Fig. 3 captions use $`n/[k\log(p-k)]`$. This study follows Section VII.
5. **Diagnostics.** Below the threshold the Lasso fails mostly through dual feasibility (PDW Step 3), not sign consistency. Near the threshold a failure usually means only one or two wrong variables enter (`fig1_pdw_diagnostics`).
6. **Method checks.** Both PDW versions gave identical verdicts on all 14,400 Fig. 1 replicates and agree with solving the full Lasso in the tests; the numerical solver converged on all 86,400 replicates.
