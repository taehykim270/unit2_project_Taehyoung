"""
Statistical method being evaluated: the Lasso with the paper's fixed lambda_n,

    beta_hat = argmin_beta  (1 / 2n) ||y - X beta||_2^2 + lambda_n ||beta||_1     (eq. 3)

Success = exact signed-support recovery. Three interchangeable checks decide it; all
follow the same contract (`SupportRecoveryCheck.check(data, lam) -> dict`):

    PDWRestrictedLasso  primal-dual witness (Section II-B) in the paper's order:
                        numerical restricted Lasso on S, then Steps 2-4      [default]
    PDWClosedForm       same PDW, with Step 2's subgradient guessed as sign(beta*_S),
                        which turns Step 1 into a linear solve (no solver needed)
    FullLasso           solve eq. (3) on all p columns and compare signs (cross-check)

By Lemmas 2-3 the PDW succeeds iff the Lasso has a unique solution with the correct
signed support. The PDW checks use the true support S: they certify what the Lasso
would do and are not estimators themselves. FullLasso never sees the truth until
its output is compared with beta*.
"""

import math
import warnings
from abc import ABC, abstractmethod

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


def regularization_lambda(n, p, k, sigma):
    """lambda_n = sqrt(2 sigma^2 log(k) log(p - k) / n)   (Section VII)."""
    return math.sqrt(2.0 * sigma**2 * math.log(k) * math.log(p - k) / n)


def lasso_fit_status(X, y, lam, tol=1e-10, max_iter=100_000):
    """Coordinate-descent Lasso; returns (coef, converged).

    scikit-learn's objective (1 / 2n) ||y - X b||^2 + alpha ||b||_1 is exactly eq. (3)
    with alpha = lambda_n. A ConvergenceWarning is caught and reported as converged=False
    instead of being printed, so non-converged fits are recorded rather than silently used.
    """
    model = Lasso(alpha=lam, fit_intercept=False, tol=tol, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model.coef_, converged


def lasso_fit(X, y, lam, tol=1e-10, max_iter=100_000):
    """Coordinate-descent Lasso coefficients (see lasso_fit_status)."""
    return lasso_fit_status(X, y, lam, tol=tol, max_iter=max_iter)[0]


def signed_support_recovered(beta_hat, beta, zero_tol=1e-8):
    """True if sign(beta_hat) == sign(beta*) coordinate-wise, treating |beta_hat_j| < zero_tol as 0."""
    beta_hat = np.where(np.abs(beta_hat) < zero_tol, 0.0, beta_hat)
    return bool(np.array_equal(np.sign(beta_hat), np.sign(beta)))


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------

class SupportRecoveryCheck(ABC):
    """Decides whether the Lasso recovers the signed support of beta* on one data set."""

    name = "base"

    @abstractmethod
    def check(self, data, lam):
        """Return a dict with
            success         : bool, the Lasso recovers the signed support
            dual_feasible   : bool, PDW Step 3 passes  (max_{j in S^c} |z_j| < 1)
            sign_consistent : bool, PDW Step 4 passes  (sign(beta_check_S) = sign(beta*_S))
            max_abs_z       : max_{j in S^c} |z_j|
            sign_margin     : min_{i in S} sign(beta*_i) * beta_check_i
            n_violations    : #{j in S^c : |z_j| >= 1}, wrong variables that enter the Lasso
            n_sign_errors   : #{i in S : beta_check_i zero or of the wrong sign}
            solver_converged: bool, the numerical solver converged (True if no solver is used)
        `data` is the dict returned by dgps.generate_replicate.
        """


# ---------------------------------------------------------------------------
# Primal-dual witness (PDW)
# ---------------------------------------------------------------------------

class PrimalDualWitness(SupportRecoveryCheck):
    """Shared PDW logic; subclasses differ only in how Steps 1-2 produce beta_check_S.

    sign_tol: in Step 4, sign(beta*_i) * beta_check_i must exceed this. 0 for an exact
    computation; a small positive cutoff when beta_check_S comes from an iterative solver.
    """

    def __init__(self, sign_tol=0.0):
        self.sign_tol = sign_tol

    @abstractmethod
    def restricted_solution(self, X_S, y, beta_S, lam):
        """Steps 1-2: return (beta_check_S, converged), the restricted Lasso solution on the support."""

    def check(self, data, lam):
        X, y, beta, support = data["X"], data["y"], data["beta"], data["support"]
        n, p = X.shape
        k = len(support)
        off_support = np.setdiff1d(np.arange(p), support)

        # X_S^T X_S is singular when n < k; n <= k counts as failure (see ADEMP 2.5)
        if n <= k:
            return {"success": False, "dual_feasible": False, "sign_consistent": False,
                    "max_abs_z": np.nan, "sign_margin": np.nan,
                    "n_violations": np.nan, "n_sign_errors": np.nan, "solver_converged": np.nan}

        X_S = X[:, support]
        beta_check_S, converged = self.restricted_solution(X_S, y, beta[support], lam)

        # Step 3: solve the zero-subgradient condition (8) for z on S^c (eq. 10 / 52):
        #   z_{S^c} = X_{S^c}^T (y - X_S beta_check_S) / (n lam)
        residual = y - X_S @ beta_check_S
        z_Sc = X[:, off_support].T @ residual / (n * lam)
        max_abs_z = float(np.max(np.abs(z_Sc))) if len(off_support) else 0.0
        dual_feasible = max_abs_z < 1.0  # strict dual feasibility -> uniqueness

        # Step 4: sign consistency, sign(beta_check_S) == sign(beta*_S) (eq. 13)
        signed_check = np.sign(beta[support]) * beta_check_S
        sign_margin = float(np.min(signed_check))
        sign_consistent = sign_margin > self.sign_tol

        return {"success": bool(dual_feasible and sign_consistent),
                "dual_feasible": bool(dual_feasible),
                "sign_consistent": bool(sign_consistent),
                "max_abs_z": max_abs_z,
                "sign_margin": sign_margin,
                "n_violations": int(np.sum(np.abs(z_Sc) >= 1.0)),
                "n_sign_errors": int(np.sum(signed_check <= self.sign_tol)),
                "solver_converged": bool(converged)}


class PDWRestrictedLasso(PrimalDualWitness):
    """PDW in the paper's order.

    Step 1: solve the restricted Lasso (eq. 9) on the columns in S numerically.
    Step 2: the subgradient is the one implied by the restricted problem's optimality
            condition, z_S = X_S^T (y - X_S beta_check_S) / (n lam). It equals
            sign(beta_check_i) where beta_check_i != 0. Step 4 therefore reduces to
            sign(beta_check_S) == sign(beta*_S), with |beta_check_i| < zero_tol counted as 0.
    """

    name = "pdw_restricted_lasso"

    def __init__(self, zero_tol=1e-8, solver_tol=1e-10, max_iter=100_000):
        super().__init__(sign_tol=zero_tol)
        self.solver_tol = solver_tol
        self.max_iter = max_iter

    def restricted_solution(self, X_S, y, beta_S, lam):
        return lasso_fit_status(X_S, y, lam, tol=self.solver_tol, max_iter=self.max_iter)


class PDWClosedForm(PrimalDualWitness):
    """PDW with Step 2 done first: guess z_S = sign(beta*_S).

    With the signs fixed, Step 1's optimality condition is linear:
        (1/n) X_S^T (X_S b - y) + lam z_S = 0  =>  b = (X_S^T X_S)^{-1} (X_S^T y - n lam z_S),
    which equals beta*_S + Delta from eq. (11). If the guess is wrong, Step 4 fails,
    and by Lemma 3(b) the Lasso fails too, so the verdict is still exact.
    The linear solve is accurate to rounding error, so sign_tol defaults to 0.
    """

    name = "pdw_closed_form"

    def restricted_solution(self, X_S, y, beta_S, lam):
        n = X_S.shape[0]
        z_S = np.sign(beta_S)
        return np.linalg.solve(X_S.T @ X_S, X_S.T @ y - n * lam * z_S), True   # no iterative solver


# ---------------------------------------------------------------------------
# Cross-check: solve the full Lasso numerically
# ---------------------------------------------------------------------------

class FullLasso(SupportRecoveryCheck):
    """Solve eq. (3) on all p columns and compare sign(beta_hat) with sign(beta*).

    Independent of the PDW algebra, so it verifies the PDW implementations.
    PDW-specific diagnostics are not defined here and are returned as NaN.
    """

    name = "full_lasso"

    def __init__(self, zero_tol=1e-8, solver_tol=1e-10, max_iter=100_000):
        self.zero_tol = zero_tol
        self.solver_tol = solver_tol
        self.max_iter = max_iter

    def check(self, data, lam):
        beta_hat, converged = lasso_fit_status(data["X"], data["y"], lam,
                                               tol=self.solver_tol, max_iter=self.max_iter)
        return {"success": signed_support_recovered(beta_hat, data["beta"], self.zero_tol),
                "dual_feasible": np.nan, "sign_consistent": np.nan,
                "max_abs_z": np.nan, "sign_margin": np.nan,
                "n_violations": np.nan, "n_sign_errors": np.nan,
                "solver_converged": bool(converged)}


# Registry used by simulation.py to pick a check by name from CONFIG
CHECKS = {cls.name: cls for cls in (PDWRestrictedLasso, PDWClosedForm, FullLasso)}
