# ADEMP: Reproducing the Simulations in Wainwright (2009)

**Paper.** M. J. Wainwright (2009). *Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using ℓ1-Constrained Quadratic Programming (Lasso).* IEEE Trans. Inf. Theory 55(5), 2183–2202.

**Scope.** This study reproduces all three simulation figures in the paper:

| Figure | Design covariance Σ | Sparsity regime(s) | x-axis |
|---|---|---|---|
| Fig. 1(a) | Identity | Fractional power | raw sample size $n$ |
| Fig. 1(b) | Identity | Fractional power | rescaled sample size $\theta$ |
| Fig. 2(a–c) | Identity | Linear, sublinear, fractional power | $\theta$ |
| Fig. 3(a–c) | Toeplitz, $\mu = 0.10$ | Linear, sublinear, fractional power | $\theta$ |

The paper's main result (Theorems 3 and 4): when the rescaled sample size

$$\theta(n,p,k) = \frac{n}{2k\log(p-k)}$$

is above an upper threshold $\theta_u(\Sigma)$, the Lasso recovers the **signed support** of $\beta^{\ast}$ with probability tending to 1. When it is below a lower threshold $\theta_\ell(\Sigma)$, the probability tends to 0. For $\Sigma = I$, the threshold is sharp since $\theta_\ell = \theta_u = 1$.

---

## 1. Aims

**Fig. 1: Is $\theta$ the right scaling?**
- (a): Plotted against the raw sample size $n$, the success-probability curves for $p \in \lbrace 128, 256, 512\rbrace $ should be clearly separated: larger problems need more samples.
- (b): Plotted against $\theta = n / (2k\log(p-k))$, the same curves should collapse onto one curve.
- *Hypothesis:* the three curves are separated in $n$ and overlap within Monte Carlo error in $\theta$.

**Fig. 2: Is there a sharp threshold at $\theta^{\ast} = 1$ for the standard Gaussian design?**
- Check that for $\Sigma = I$ the success probability goes from ≈0 to ≈1 around $\theta = 1$ in all three sparsity regimes ((a):linear, (b):sublinear, (c):fractional power).
- *Hypotheses:* (i) the transition is centered near $\theta = 1$ whatever the sparsity regime; (ii) the transition gets steeper as $p$ grows.

**Fig. 3: Do the bounds hold for a correlated design?**
- Check that for a Toeplitz covariance ($\Sigma_{ij} = \mu^{|i-j|}$, $\mu = 0.10$) the transition happens between the theoretical bounds $\theta_\ell(\Sigma)$ and $\theta_u(\Sigma)$.
- *Hypothesis:* success probability is near 0 for $\theta < \theta_\ell(\Sigma)$, near 1 for $\theta > \theta_u(\Sigma)$, and the curves for different $p$ collapse as in Fig. 2.

**Quick check:** for $\theta$ near 0 (and whenever $n \le k$) the success probability must be exactly 0. For $\theta$ near 2.4 it should be near 1.

---

## 2. Data-generating mechanisms

### 2.1 Observation model
For each replicate:

$$y = X\beta^{\ast} + w, \qquad X \in \mathbb{R}^{n\times p}, \quad w \sim N(0, \sigma^2 I_n), \quad \sigma = 0.5.$$

### 2.2 Design matrix $X$ (two random ensembles)
The rows $x_i$ are i.i.d. $N(0, \Sigma)$, and a new $X$ is drawn for every replicate.

| Ensemble | Σ | Used in |
|---|---|---|
| Standard (uniform) Gaussian | $\Sigma = I_p$ | Figs. 1, 2 |
| Toeplitz | $\Sigma_{ij} = \mu^{\lvert i-j\rvert}$, $\mu = 0.10$ | Fig. 3 |

For the Toeplitz case, generate $X = Z L^\top$, where $L$ is the Cholesky factor of $\Sigma$ (computed once per $p$) and $Z$ is $n\times p$ standard Gaussian, drawn exactly as in the identity case.

> [Note] Eq. (46) calls the Toeplitz parameter $\mu$, but the Fig. 3 panel titles say "ρ = 0.10". They are the same parameter.

### 2.3 True coefficient vector $\beta^{\ast}$
1. Choose the support $S \subset \lbrace 1,\dots,p\rbrace $ with $|S| = k$ uniformly at random.
2. For each $i \in S$, set $\beta^{\ast}_i = +\beta_{\min}$ or $-\beta_{\min}$ with probability 1/2 each.
3. Set $\beta^{\ast}_j = 0$ for $j \notin S$.

The paper uses $\beta_{\min} = 0.5$. A new $S$ and new signs are drawn in every replicate.

### 2.4 Sparsity regimes ($k$ as a function of $p$), with $\alpha = 0.40$, $\delta = 0.75$

| Regime | Formula | $p=128$ | $p=256$ | $p=512$ |
|---|---|---|---|---|
| Linear | $k = \lceil \alpha p \rceil$ | 52 | 103 | 205 |
| Sublinear | $k = \lceil \alpha p / \log(\alpha p) \rceil$ | 14 | 23 | 39 |
| Fractional power | $k = \lceil \alpha p^{\delta} \rceil$ | 16 | 26 | 44 |

(All logs are natural logs.)

### 2.5 Sample size
The sample size is set by the control parameter $\theta$:

$$n = \lceil 2 \theta  k \log(p-k) \rceil .$$

- The paper varies $\theta$ over $(0, 2.4)$.
- **[Decision]** The paper does not give the grid. Its figures show 13 points (Figs. 1(b), 2; step ≈0.19) and 17 points (Fig. 3; step ≈0.14). This study uses the finer grid $\theta \in \lbrace 0.1, 0.2, \dots, 2.4\rbrace $ (24 values) for all figures, to get higher resolution in the transition region.
- **[Decision]** If $n \le k$, the replicate counts as a failure. Justification:
  - The paper plots the first point of every curve in Figs. 1–3 as 0.
  - $n < k$: $X_S^\top X_S$ is singular, so the PDW is undefined. Also, a unique Lasso solution has at most $n$ nonzeros, so it cannot recover a size-$k$ support.
  - $n = k$: $X_S$ is square and invertible, so the PDW can be evaluated. Doing so gave 0 successes in 2000 trials ($p=128$, $k=16$; $\max_{j\notin S}|z_j| \ge 2.4$ in every trial), so counting it as a failure changes nothing.
  - On the 0.1 grid this only happens at $\theta = 0.1$ for $p = 128$: linear ($n=46 < k=52$), sublinear ($n = k = 14$), fractional ($n = k = 16$).
- **[Decision]** The rescaling follows Section VII: $n = \lceil 2\theta k\log(p-k) \rceil$. The paper is inconsistent here:
  - the Fig. 1 caption also uses $n/[2k\log(p-k)]$;
  - the Fig. 2 and Fig. 3 captions use $n/[k\log(p-k)]$, without the 2;
  - the $n$-spacing of the markers in Fig. 1(a) matches the version without the 2.

Example values of $n$ at $\theta = 1$: 451 / 1037 / 2349 (linear), 133 / 251 / 481 (sublinear), 151 / 283 / 542 (fractional) for $p = 128 / 256 / 512$.

### 2.6 Simulation design matrix

| Factor | Levels | # |
|---|---|---|
| Covariance Σ | Identity, Toeplitz ($\mu=0.10$) | 2 |
| Sparsity regime | Linear, sublinear, fractional power | 3 |
| Problem size $p$ | 128, 256, 512 | 3 |
| Rescaled sample size $\theta$ | 0.1, 0.2, …, 2.4 | 24 |
| **Total conditions** | fully crossed | **432** |
| Replicates per condition | $R = 200$ (as in paper) | |
| **Total replicates** | | **86,400** |

Fixed across all conditions: $\sigma = 0.5$, $\beta_{\min} = 0.5$, $\alpha = 0.40$, $\delta = 0.75$, $\lambda_n$ as in Section 4.

Fig. 1 is the (Identity, fractional power) slice. It uses the same simulated data as Fig. 2(c), plotted once against $n$ and once against $\theta$.

### 2.7 Random seeds
- One global master seed is set in the config (`master_seed = 607`).
- Each **replicate** gets its own independent generator: `numpy.random.SeedSequence(master_seed, spawn_key=(design, regime, p, round(1000·θ), rep))`, where design and regime are fixed integer codes.
- The seed depends only on the condition and the replicate index, not on the order or number of conditions. So:
  - any condition or replicate can be rerun on its own and give identical results;
  - adding conditions later (Figs. 2–3) never changes results already computed (checked by a test).
- Within a replicate, draws are made in a fixed order: $X$, then $\beta^{\ast}$ (support, then signs), then $w$.
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

$$P_{\text{succ}}(\theta, p, \text{regime}, \Sigma) = \mathbb{P}[  \mathbb{S}_\pm(\hat\beta) = \mathbb{S}_\pm(\beta^{\ast})  ],$$

where $\mathbb{S}_\pm(\beta_i) = \mathrm{sign}(\beta_i) \in \lbrace -1, 0, +1\rbrace $ is the extended sign vector (eq. 6).

- A replicate succeeds only if the Lasso solution is unique, has exactly the same support as $\beta^{\ast}$, and has the correct sign on every coordinate of that support.
- The probability is taken over the random $X$, $w$, $S$ and signs.

**Theoretical benchmarks:**
- Identity Σ: as $p \to \infty$, $P_{\text{succ}}$ tends to a step function at $\theta^{\ast} = 1$.
- Toeplitz Σ: the step lies between $\theta_\ell(\Sigma)$ and $\theta_u(\Sigma)$ (eq. 29):

$$\theta_\ell(\Sigma) = \frac{\rho_\ell(\Sigma_{S^c|S})}{C_{\max}(2-\gamma(\Sigma))^2}, \qquad \theta_u(\Sigma) = \frac{\rho_u(\Sigma_{S^c|S})}{C_{\min} \gamma^2(\Sigma)},$$

with $\gamma$ the incoherence parameter (26a), $C_{\min}, C_{\max}$ the eigenvalue bounds on $\Sigma_{SS}$ (26b–c), and $\rho_\ell, \rho_u$ from (27).

> [Decision] The paper does not say how it computed the vertical lines in Fig. 3. They sit at ≈0.91 and ≈1.01 and are identical in all three panels.
>
> Eq. (29) depends on $S$:
> - with random $S$ it gives $\theta_\ell \approx 0.52$–$0.60$ and $\theta_u \approx 1.62$–$1.87$, varying by panel;
> - the worst case gives $\theta_u \approx 1.90$.
>
> Only the numerators match the paper: $\rho_\ell(\Sigma_{S^c|S}) \approx 0.89$ and $\rho_u(\Sigma_{S^c|S}) = 1$, nearly independent of $S$.
>
> Our Fig. 3 draws **both**:
> - the paper-style lines ($\rho_\ell$, $\rho_u$);
> - the full eq. (29) bounds, averaged over random draws of $S$ for each panel.
>
> How they are computed:
> - For each (design, regime, $p$), all constants of eqs. (26)–(29) are computed for 50 random supports $S$, drawn uniformly as in the DGP from a stream keyed by (design, regime, $p$).
> - Their mean and SD are saved in `results/theory_bounds.csv`.
> - Each panel draws the average over the three values of $p$, which barely differ. For the identity design all four lines equal 1.

---

## 4. Methods

**Method under evaluation:** the Lasso (eq. 3)

$$\hat\beta = \arg\min_{\beta\in\mathbb{R}^p} \lbrace  \frac{1}{2n}\Vert y - X\beta\Vert _2^2 + \lambda_n \Vert \beta\Vert _1 \rbrace ,$$

with the regularization parameter fixed as in the paper (not tuned by cross-validation):

$$\lambda_n = \sqrt{\frac{2\sigma^2 \log(k) \log(p-k)}{n}}.$$

**How success is decided: the primal-dual witness (PDW) construction (Section II-B).**
By Lemmas 2–3, the PDW succeeds exactly when the Lasso has a unique solution with the correct signed support. That means the success indicator can be computed exactly in closed form, without running an iterative solver. Writing $z_S = \mathrm{sign}(\beta^{\ast}_S)$:

1. **Restricted solution:** compute the error on the true support (eq. 11):
   $$\Delta = (\tfrac{1}{n}X_S^\top X_S)^{-1}[\tfrac{1}{n}X_S^\top w - \lambda_n z_S].$$
2. **Subgradient on $S$:** set it to $z_S = \mathrm{sign}(\beta^{\ast}_S)$.
3. **Strict dual feasibility:** for $j \in S^c$, compute (eq. 10)
   $$Z_j = X_j^\top[X_S(X_S^\top X_S)^{-1}z_S + \Pi_{X_S^\perp}(\tfrac{w}{\lambda_n n})],$$
   with $\Pi_{X_S^\perp} = I - X_S(X_S^\top X_S)^{-1}X_S^\top$. Check that $\max_{j\in S^c}|Z_j| < 1$.
4. **Sign consistency:** check $\mathrm{sign}(\beta^{\ast}_i + \Delta_i) = \mathrm{sign}(\beta^{\ast}_i)$ for all $i \in S$.

The replicate succeeds if and only if Steps 3 and 4 both pass.

**Implementation.** The support-recovery check is a class with one interface (`check(data, λ)`), with three interchangeable implementations in `src/methods.py`. The simulation uses the one named in the config.
- **`pdw_restricted_lasso` (default, the paper's order):**
  - Step 1 solves the restricted Lasso (eq. 9) on the columns in $S$ numerically (scikit-learn, tolerance $10^{-10}$).
  - Step 2 takes the subgradient implied by the restricted problem's optimality condition, $z_S = X_S^\top(y - X_S\check\beta_S)/(n\lambda_n)$. This equals $\mathrm{sign}(\check\beta_i)$ wherever $\check\beta_i \neq 0$.
  - Step 4 therefore checks $\mathrm{sign}(\check\beta_S) = \mathrm{sign}(\beta^{\ast}_S)$, counting $|\check\beta_i| < 10^{-8}$ as 0.
- **`pdw_closed_form` (suggested alternative):**
  - Step 2 comes first: guess $z_S = \mathrm{sign}(\beta^{\ast}_S)$. With the signs fixed, Step 1 becomes linear: $\check\beta_S = (X_S^\top X_S)^{-1}(X_S^\top y - n\lambda_n z_S)$, which equals $\beta^{\ast}_S + \Delta$.
  - If the guess is wrong, Step 4 fails, and by Lemma 3(b) the Lasso fails too, so the verdict is still exact. No solver or zero cutoff is needed.
  - It gave identical verdicts to the default on all 14,400 Fig. 1 replicates.
- **`full_lasso` (cross-check):** solve eq. (3) on all $p$ columns and compare signs; see below.

For both PDW versions, Step 3 is $z_{S^c} = X_{S^c}^\top (y - X_S\check\beta_S)/(n\lambda_n)$. This is algebraically identical to eq. (10), which is written with an $n \times n$ projection matrix; a test checks the equality. Both Step 3 and Step 4 are evaluated on every replicate (the paper's procedure would stop at the first failure), so each failure can be attributed to a step.

The PDW uses the true support $S$. That is legitimate: it certifies what the full Lasso would do and is not an estimator. Only `full_lasso` is a method that never sees the truth.

**Cross-check (method correctness, test suite only).**
- By Lemmas 2–3 the PDW result is exact, so the simulation does not solve the full $p$-column Lasso.
- The lemmas justify the mathematics, not our transcription of it into code. A one-character sign error in Step 1 still produces a smooth, plausible-looking success curve.
- The test suite therefore checks that both PDW versions agree with `full_lasso`, which uses `sklearn.linear_model.Lasso(alpha=λ_n, fit_intercept=False)` and minimizes the same objective (eq. 3), at several $\theta$ ($|\hat\beta_j| < 10^{-8}$ is treated as 0).
- The tests also check that both PDW versions' $\check\beta_S$ equal the full Lasso solution on $S$ whenever the PDW succeeds.

**Quantities recorded per replicate** (saved raw, before summarizing):
- condition identifiers ($\Sigma$, regime, $p$, $k$, $\theta$, $n$, realized $\theta$ after rounding $n$ up, $\lambda_n$) and replicate index (the seed is determined by these; see §2.7)
- success indicator (0/1)
- dual-feasibility pass/fail and sign-consistency pass/fail, recorded separately
- $\max_{j \in S^c}|Z_j|$ (dual-feasibility margin) and $\min_{i\in S}\mathrm{sign}(\beta^{\ast}_i)(\beta^{\ast}_i + \Delta_i)$ (sign margin)
- failure counts:
  - the number of $j \in S^c$ with $|z_j| \ge 1$, i.e. the wrong variables that enter;
  - the number of $i \in S$ where $\check\beta_i$ is zero or has the wrong sign.

  Full $z$ / $\check\beta$ vectors are not stored, since every replicate can be regenerated from its seed key.
- `seed_key` and `solver_converged`. A solver `ConvergenceWarning` is caught and recorded, not silently ignored; the closed form uses no solver and always records `True`.

---

## 5. Performance measures

| Measure | Definition | Purpose |
|---|---|---|
| **Success probability** | $\hat P = \frac{1}{R}\sum_{r=1}^R \mathbb{1}\lbrace \text{success}_r\rbrace $ | Main quantity in all figures |
| **Monte Carlo SE** | $\sqrt{\sum_r(\phi_r - \bar\phi)^2 / (R(R-1))}$; for 0/1 outcomes this is $\sqrt{\hat P(1-\hat P)/(R-1)}$ (≤ 0.035 for $R=200$) | Uncertainty of every estimated mean; 0 when $\hat P \in \lbrace 0,1\rbrace $, so the figure bands use Wilson instead |
| **Solver convergence** | Number of replicates per condition where the numerical solver did not converge | Shows that no result relies on an unconverged fit |
| **95% interval** | Wilson score interval for $\hat P$ | Better-behaved near 0 and 1 than the Wald interval |
| **Empirical threshold** $\hat\theta_{50}$ | The realized $\theta$ where $\hat P$ first reaches 0.5 (linear interpolation between grid points; same for $\hat\theta_{10}, \hat\theta_{90}$) | Compare against $\theta^{\ast}=1$ (Fig. 2) and against $[\theta_\ell, \theta_u]$ (Fig. 3) |
| **Transition width** | $\hat\theta_{90} - \hat\theta_{10}$ | Tests whether the transition gets steeper as $p$ grows |
| **Failure-mode breakdown** | Share of failures due to dual feasibility (Step 3) vs. sign consistency (Step 4) | Diagnostic: shows *why* the Lasso fails below the threshold |
| **Failure severity** | Mean number of wrong variables entering, among replicates that fail Step 3 (and mean number of sign errors, among those that fail Step 4), with a normal-approximation 95% CI | Diagnostic: shows *how badly* the Lasso fails, e.g. by one extra variable or by many |
| **Curve collapse (Fig. 1)** | At each $\theta$: the largest two-proportion $z$ statistic $\lvert \hat P_{p_1} - \hat P_{p_2}\rvert  / \text{SE}_{\text{pooled}}$ over all pairs of $p$ (`results/collapse.csv`); $z < 2$ means overlap within Monte Carlo error | Measures how well the rescaling works |

**Comparison with the paper** (`results/paper_comparison.csv`): approximate success rates read off the paper's Fig. 1(b) (`docs/paper_values.csv`) are compared with our success rate and Wilson CI, interpolated along our curve at the same $\theta$.

**Figures:**
All figures are saved in `results/figures/` as PDF (vector) and PNG.
- **Fig. 1, publication-quality:** `fig1_success_vs_n_and_theta`, 1×2 like the paper. (a) $\hat P$ vs. $n$; (b) $\hat P$ vs. realized $\theta$. Colored and marker-coded by $p$, 95% Wilson CI bands, reference line at $\theta = 1$.
- **Fig. 1, diagnostic:** `fig1_pdw_diagnostics`. (a) median and 10–90% band of $\max_{j\in S^c}|z_j|$ vs. $\theta$, with the bound $|z| = 1$; (b) pass rate of Step 3 (dual feasibility); (c) pass rate of Step 4 (sign consistency); (d) mean number of wrong variables entering, given Step 3 fails (points with ≥ 10 failing replicates).
- **Figs. 2–3, publication-quality:** `fig2_fig3_success_grid`, one 2 × 3 grid of small multiples. Rows are the design (identity = Fig. 2, Toeplitz = Fig. 3), columns the sparsity regime (linear, sublinear, fractional power); axes are shared.
  - Each panel shows $\hat P$ vs. realized $\theta$, coloured and marker-coded by $p$, with 95% Wilson CI bands.
  - Vertical lines (see §3): the eq. (29) bounds $\theta_\ell, \theta_u$ dashed, and $\rho_\ell, \rho_u$ of $\Sigma_{S^c|S}$ dotted.
