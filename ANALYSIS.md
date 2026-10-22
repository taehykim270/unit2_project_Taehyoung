# Critical Analysis: Reproducing Wainwright (2009)

**Use of AI.** I used Claude (Anthropic) to write code, tests and documentation drafts. I planned the work, made design decisions, checked the code and outputs, and requested corrections. My decisions included the primal-dual-witness step order, the θ grid, and which checks, results and figures to keep. The development history is recorded in the repository's git commits. Codex helped simplify the wording of this analysis.

## 1. How well were the results reproduced?

**The main patterns were reproduced.** Rescaling by θ = n/(2k log(p − k)) brings the curves for different p close together. For the identity design, success reaches 50% near θ = 1 in every sparsity regime (θ̂₅₀ = 0.99–1.23). For the Toeplitz design, the transition lies between the bounds in eq. (29): θ_ℓ ≈ 0.52–0.59 and θ_u ≈ 1.63–1.87.

**Our success rates are lower than the paper's** in the transition region across both designs and all sparsity regimes. The average gap is about 30 percentage points at θ = 1 and 15 at θ = 1.5.

A possible reason is the success criterion. We use the primal-dual witness (PDW) to check whether the Lasso selects exactly the true variables with the correct signs. If the paper treated small estimated coefficients as zero, it could have counted some near misses as successes. In an exploratory run, treating coefficients below 0.01 in magnitude as zero raised success at θ = 1 from 0.33 to 0.57, closer to the paper's ≈ 0.65. The paper does not describe its solver, so this explanation remains unconfirmed.

The paper is also inconsistent: the Fig. 2–3 captions give a different θ formula from Section VII, and the Fig. 3 lines match only part of eq. (29). We follow Section VII and plot the full bounds.

## 2. Neutrality of the design

The study evaluates only the Lasso, so it cannot establish how well it performs relative to other methods. It includes no comparison with methods such as orthogonal matching pursuit or thresholded least squares, or with the information-theoretic limit cited in the paper.

The settings also favour the Lasso: λ uses the true noise level σ, the signal β_min = 0.5 is large, and both designs satisfy the incoherence condition. The simulations use Gaussian designs and noise, random supports, and only weak correlation (Toeplitz μ = 0.1). These idealized conditions limit how broadly the findings apply. I added one less favourable condition and one non-Gaussian design to test this (§3).

## 3. What I would change

I added two extensions. With stronger Toeplitz correlation (μ = 0.5; Fig. 4), success stays below about 0.45 for θ ≤ 2.4 (near 0 for linear sparsity), and the eq. (29) bounds become too wide to be informative (θ_u ≈ 40–71), although incoherence still holds. A sparse random design not covered by the theorems (entries ±1/√q with probability q/2 each, otherwise 0; q = 0.1; Fig. 5) behaves much like the identity design (θ̂₅₀ = 1.05–1.35, at most 0.17 later, with the gap shrinking as p grows).

With more time, I would try μ = 0.9, designs that violate incoherence, heavy-tailed noise, and other α and δ values (to test whether θ still aligns the curves). I would also compare the fixed λ with a tuned λ and with a competing method.

## 4. Visualizations

I recreated Figs. 1–3 with 95% Wilson confidence bands and labelled theoretical reference lines. I also added a diagnostic figure showing that failures mainly involve extra variables entering the model (dual-feasibility failures). The signs of the true coefficients are usually recovered at smaller θ. Among failing replicates, the diagnostic flags about 1.8 extra variables on average at θ = 1 and about 1 above θ = 1.5.

The curves do not overlap perfectly after rescaling. For larger p, success rises more steeply above θ ≈ 1, with significant gaps at 9–17 of the 24 θ values per panel. These differences are consistent with finite-p effects that are less visible in the paper's figures.

## 5. What was surprising

I expected smoother, steadily increasing curves. On our finer θ grid, 4 of the 9 identity-design curves dip by more than 0.03 (e.g., p = 128 at θ = 1.2). These dips are within Monte Carlo error (SE ≤ 0.035), but noise at one grid point can shift the estimated threshold. The consistently lower success rates were also unexpected, as was larger p doing worse for linear sparsity with μ = 0.5 (success at θ = 2.4 fell from 0.075 to 0 as p grew from 128 to 512).

## 6. Most challenging aspects

- **Implementing the PDW correctly.** Even a sign error can produce a plausible curve. I cross-checked both PDW versions against the full Lasso solver.
- **Interpreting the paper.** This included following its step order, handling n ≤ k, and resolving the unclear θ formula and threshold lines.
- **Reproducible seeding.** Adding conditions had to leave earlier results unchanged.
- **Version control.** Changes often affected code, tests, documentation and outputs together, making it difficult to keep commits focused and results consistent with the code.

## 7. Confidence in the results

**I am confident in the computation.** Each replicate has its own seed key, so individual rows and the full simulation can be reproduced. Every success rate has a 95% Wilson interval (Monte Carlo SE ≤ 0.035). Both PDW versions agree with the full Lasso solver in the tests, and no solver convergence failures were recorded across the 172,800 replicates.

**The main uncertainty is how closely our choices match the paper's.**

- **Rescaling and success criterion:** the ambiguities discussed in §1 could explain some differences.
- **Fixed settings:** I used the true σ in λ, β_min = 0.5, natural logs, and rounded k and n up. Different interpretations could shift the curves.
- **Grid, edge cases and theoretical lines:** θ increases by 0.1, n ≤ k counts as failure (only at θ = 0.1, p = 128), and the eq. (29) bounds are averaged over 50 random supports and then over p.
- **Replication:** R = 200 matches the paper and gives useful curves, but limits precise comparisons of θ̂₅₀ across p.
