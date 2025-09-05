""" Basic Bayesian Optimization utilities (for PyCall.jl)

- Expected Improvement (maximization)
- 5/2-Matern kernel suggested in Fit.jl
- Robust candidate sampling + optional local refinement (L-BFGS-B)
- Restart from CSV log

Notes
-----
* This module assumes we MAXIMIZE the objective (y).
  If you want to minimize f, pass y = -f in your driver.
"""
import os
import warnings
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm

# Optional plotting is handled from Julia side.

# sklearn imports are kept here so Fit.jl can use them from the same Python context
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

try:
    import pandas as pd
except Exception as e:
    pd = None  # Fit.jl should import pandas; this is a fallback


# ------------------------------
# Utilities
# ------------------------------
def _as_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[None, :]
    return X

def _sample_candidates(bounds, n_cand=4096, param_order=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if param_order is None:
        param_order = list(bounds.keys())
    lo = np.array([bounds[p][0] for p in param_order], dtype=float)
    hi = np.array([bounds[p][1] for p in param_order], dtype=float)
    U = rng.random((n_cand, len(param_order)))
    XR = lo + U*(hi - lo)
    return XR, lo, hi

def _dedup(x, X, atol=1e-10, rtol=1e-12):
    """Return True if x already appears in rows of X (within tolerance)."""
    if X is None or len(X) == 0:
        return False
    diffs = np.abs(X - x)
    eq = np.all(diffs <= (atol + rtol*np.abs(x)), axis=1)
    return bool(np.any(eq))


# ------------------------------
# GP helpers
# ------------------------------
def surrogate(model, X):
    """Return GP posterior mean and std at X.
    Parameters
    ----------
    model : GaussianProcessRegressor
    X : array-like (n, d) or (d,)
    Returns
    -------
    (mu, std) : 1D arrays of length n
    """
    warnings.filterwarnings("ignore")
    X = _as_2d(X)
    mu, std = model.predict(X, return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    return mu, std


def Expected_Improvement(X_obs, XS, model, explore, y_obs=None):
    """Compute EI at candidate set XS.

    EI(x) = (μ - y_best - ξ) Φ(Z) + σ φ(Z),   Z = (μ - y_best - ξ)/σ
    We *maximize* y, so y_best = max observed y.
    """
    X_obs = _as_2d(X_obs)
    XS = _as_2d(XS)
    if y_obs is None:
        raise ValueError("Expected_Improvement: please pass observed y as y_obs")
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    y_best = np.max(y_obs) if y_obs.size else -np.inf

    mu, std = surrogate(model, XS)
    imp = mu - y_best - float(explore)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.zeros_like(std)
        nz = std > 0
        Z[nz] = imp[nz] / std[nz]
        ei = np.zeros_like(std)
        ei[nz] = imp[nz]*norm.cdf(Z[nz]) + std[nz]*norm.pdf(Z[nz])
    # Numerical guard
    ei[~np.isfinite(ei)] = 0.0
    return ei.reshape(-1)


def Opt_Acquisition(X, y_obs, model, bounds, explore=0.0, n_cand=4096, k_refine=8, param_order=None, rng=None):
    """Return a single next point (1, d) that maximizes EI.

    - Sample n_cand random points in box(bounds)
    - Evaluate EI; pick top-k_refine as starts
    - Locally refine each start with L-BFGS-B (box constraints)
    - Return the best refined point (fall back to random if duplicates)
    """
    X = _as_2d(X)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    XR, lo, hi = _sample_candidates(bounds, n_cand=n_cand, param_order=param_order, rng=rng)

    # Score candidates
    EI = Expected_Improvement(X, XR, model, explore=explore, y_obs=y_obs)
    order = np.argsort(-EI)
    starts = XR[order[:max(1, int(k_refine))], :]

    # Prepare acq function for local opt
    def neg_ei(x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        ei = Expected_Improvement(X, x, model, explore=explore, y_obs=y_obs)[0]
        return -float(ei)

    bounds_list = list(zip(lo, hi))
    best_x = starts[0].copy()
    best_val = -neg_ei(best_x)

    for s in starts:
        res = minimize(neg_ei, s, method="L-BFGS-B", bounds=bounds_list, options={"maxiter": 150})
        if not res.success:
            # fall back to start if needed
            x_cand = s
            val = -neg_ei(x_cand)
        else:
            x_cand = res.x
            val = -res.fun
        if val > best_val + 1e-14:
            best_val = val
            best_x = x_cand

    # Deduplicate w.r.t. existing X; if duplicate, walk down sorted XR list
    if _dedup(best_x, X):
        for idx in order:
            x_alt = XR[idx]
            if not _dedup(x_alt, X):
                best_x = x_alt
                break
    return best_x.reshape(1, -1)


def Restart(bounds, file_name, param_order=None):
    """Load previous log if present; else return empty X, y and idx_list=[[0]].

    CSV format expected:
        columns = ["ID"] + param_order + ["Obj"]
        separator = '\t' (tab)
    """
    if param_order is None:
        param_order = list(bounds.keys())
    csv_path = f"{file_name}.csv"
    if not os.path.exists(csv_path) or pd is None:
        X = np.empty((0, len(param_order)), dtype=float)
        y = np.array([], dtype=float)
        idx_list = np.array([[0]], dtype=int)
        return X, y, idx_list

    try:
        df = pd.read_csv(csv_path, sep="\t")
        # Ensure required columns exist
        cols_needed = ["ID"] + list(param_order) + ["Obj"]
        for c in cols_needed:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in log")
        X = df[param_order].to_numpy(dtype=float)
        y = df["Obj"].to_numpy(dtype=float).reshape(-1)
        idx = df["ID"].to_numpy(dtype=int).reshape(-1, 1)
        if idx.size == 0:
            idx = np.array([[0]], dtype=int)
        return X, y, idx
    except Exception as e:
        # Fallback to empty if parsing fails
        X = np.empty((0, len(param_order)), dtype=float)
        y = np.array([], dtype=float)
        idx_list = np.array([[0]], dtype=int)
        return X, y, idx_list
