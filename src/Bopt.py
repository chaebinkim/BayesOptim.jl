"""
Bopt.py — Bayesian Optimization helpers (unit-space wrapper, trust-region aware)

Contents
--------
- UnitSpaceGP: wraps sklearn GPR to work in [0,1]^d (input normalization)
- surrogate(): GP predict mean/std
- Expected_Improvement(): EI for maximization (uses observed y_best)
- Opt_Acquisition(): sample→rank→local-refine (L-BFGS-B), trust-region aware
- optimal_std_via_sampling(): posterior function sampling to estimate std of optimum
- Restart(): load previous CSV log

Notes
-----
* We MAXIMIZE y. For chi2 minimization, we transform inside Fit.jl via y = -log(chi2 + eps).
* Bounds are given in original parameter scales; UnitSpaceGP handles normalization.
"""
import os
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.stats import norm

try:
    import pandas as pd
except Exception:
    pd = None

# ------------------------------
# Bounds helpers & normalization
# ------------------------------
def _bounds_arrays(bounds, param_order):
    lo = np.array([bounds[p][0] for p in param_order], dtype=float)
    hi = np.array([bounds[p][1] for p in param_order], dtype=float)
    return lo, hi

def to_unit_batch(bounds, param_order, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    lo, hi = _bounds_arrays(bounds, param_order)
    return (X - lo) / (hi - lo)

def from_unit_batch(bounds, param_order, U):
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    lo, hi = _bounds_arrays(bounds, param_order)
    return lo + U * (hi - lo)

# ------------------------------
# Unit-space GP wrapper
# ------------------------------
class UnitSpaceGP:
    """Wrap sklearn GPR so that fit/predict/sample_y are done in unit space [0,1]^d."""
    def __init__(self, model, bounds, param_order):
        self.model = model
        self.bounds = bounds
        self.param_order = list(param_order)

    def _to_unit(self, X):
        return to_unit_batch(self.bounds, self.param_order, X)

    def fit(self, X, y):
        X_u = self._to_unit(X)
        return self.model.fit(X_u, y)

    def predict(self, X, return_std=True):
        X_u = self._to_unit(X)
        return self.model.predict(X_u, return_std=return_std)

    def sample_y(self, X, n_samples=1, random_state=None):
        X_u = self._to_unit(X)
        return self.model.sample_y(X_u, n_samples=n_samples, random_state=random_state)

# ------------------------------
# Utils
# ------------------------------
def _as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[None, :] if X.ndim == 1 else X

def _dedup_unit(x_u, X_u, tol=1e-6):
    if X_u is None or len(X_u) == 0:
        return False
    diffs = np.abs(X_u - x_u)
    eq = np.all(diffs <= tol, axis=1)
    return bool(np.any(eq))

def _sample_candidates(bounds, param_order, n_cand=4096, trust_region=None, rng=None):
    """
    Sample candidates in ORIGINAL space, possibly inside a trust-region in unit space.
    trust_region: dict or None. If dict:
        keys:
          - 'L' : side length in unit space (0 < L <= 1)
          - 'center_u' : 1D array in [0,1]^d specifying box center in unit space
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(param_order)
    if trust_region is None:
        U = rng.random((n_cand, d))  # global uniform in unit space
    else:
        L = float(trust_region.get('L', 1.0))
        L = max(1e-6, min(1.0, L))
        center_u = np.asarray(trust_region['center_u'], dtype=float).reshape(-1)
        if center_u.size != d:
            raise ValueError("center_u size mismatch.")
        U = center_u + (rng.random((n_cand, d)) - 0.5) * L
        U = np.clip(U, 0.0, 1.0)
    XR = from_unit_batch(bounds, param_order, U)  # map to original
    return XR, U

# ------------------------------
# GP helpers
# ------------------------------
def surrogate(model, X):
    warnings.filterwarnings("ignore")
    X = _as_2d(X)
    mu, std = model.predict(X, return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    return mu, std

def Expected_Improvement(X_obs, XS, model, explore, y_obs):
    """EI(x) = (μ - y_best - ξ) Φ(Z) + σ φ(Z), with y_best from observed y."""
    X_obs = _as_2d(X_obs)
    XS = _as_2d(XS)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    y_best = -np.inf if y_obs.size == 0 else float(np.max(y_obs))
    mu, std = surrogate(model, XS)
    imp = mu - y_best - float(explore)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.zeros_like(std)
        nz = std > 0
        Z[nz] = imp[nz] / std[nz]
        ei = np.zeros_like(std)
        ei[nz] = imp[nz] * norm.cdf(Z[nz]) + std[nz] * norm.pdf(Z[nz])
    ei[~np.isfinite(ei)] = 0.0
    return ei.reshape(-1)

# ------------------------------
# Acquisition (single point)
# ------------------------------
def Opt_Acquisition(X, y_obs, model, bounds, explore=0.01, n_cand=4096, k_refine=8, param_order=None, trust_region=None, rng=None):
    """Return one next point (1,d). Trust region {'L', 'auto_center'| 'center_u'} supported."""
    X = _as_2d(X)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    if param_order is None:
        param_order = list(bounds.keys())
    if rng is None:
        rng = np.random.default_rng()

    # Resolve trust-region center
    tr = None
    if trust_region is not None:
        if trust_region.get('auto_center', False) and y_obs.size > 0:
            best_idx = int(np.argmax(y_obs))
            x_best = X[best_idx:best_idx+1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}

    # Sample and score
    XR, U = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=tr, rng=rng)
    EI = Expected_Improvement(X, XR, model, explore=explore, y_obs=y_obs)
    order = np.argsort(-EI)
    starts = XR[order[:max(1, int(k_refine))], :]

    lo, hi = _bounds_arrays(bounds, param_order)
    bounds_list = list(zip(lo, hi))

    def neg_ei(x):
        ei = Expected_Improvement(X, x, model, explore=explore, y_obs=y_obs)[0]
        return -float(ei)

    best_x = starts[0].copy()
    best_val = -neg_ei(best_x)

    for s in starts:
        res = minimize(neg_ei, s, method="L-BFGS-B", bounds=bounds_list, options={"maxiter": 150})
        if not res.success:
            x_cand = s
            val = -neg_ei(x_cand)
        else:
            x_cand = res.x
            val = -res.fun
        if val > best_val + 1e-14:
            best_val = val
            best_x = x_cand

    # Dedup in unit space
    X_u = to_unit_batch(bounds, param_order, X)
    x_u = to_unit_batch(bounds, param_order, best_x)
    if _dedup_unit(x_u.reshape(-1), X_u, tol=1e-6):
        for idx in order:
            x_alt = XR[idx]
            x_alt_u = to_unit_batch(bounds, param_order, x_alt).reshape(-1)
            if not _dedup_unit(x_alt_u, X_u):
                best_x = x_alt
                break

    return best_x.reshape(1, -1)

# ------------------------------
# Posterior sampling-based uncertainty of optimum
# ------------------------------
def optimal_std_via_sampling(model, bounds, param_order, X=None, y=None, n_funcs=200, n_cand=2000,
                             trust_region=None, rng=None, eps=1e-12):
    """Estimate std of optimal y and chi2 via posterior function sampling on a candidate set."""
    if rng is None:
        rng = np.random.default_rng()

    # Resolve trust-region
    tr = None
    if trust_region is not None:
        if trust_region.get('auto_center', False) and (X is not None) and (y is not None) and (len(y) > 0):
            X = _as_2d(X)
            y = np.asarray(y, dtype=float).reshape(-1)
            best_idx = int(np.argmax(y))
            x_best = X[best_idx:best_idx+1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}

    # Candidate set
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=tr, rng=rng)

    # Posterior function samples: (n_cand, n_funcs)
    YS = model.sample_y(XR, n_samples=int(n_funcs), random_state=None)
    if YS.ndim == 1:
        YS = YS[:, None]
    if YS.shape[0] != XR.shape[0]:
        YS = YS.T

    # Optimum per sample
    idx_max = np.argmax(YS, axis=0)               # (n_funcs,)
    y_star = YS[idx_max, np.arange(YS.shape[1])]  # (n_funcs,)
    X_star = XR[idx_max, :]                       # (n_funcs, d)

    # Convert back to chi2 (clip to >= 0 for numerical safety)
    chi2_star = np.maximum(np.exp(-y_star) - float(eps), 0.0)

    out = {
        "y_star_mean": float(np.mean(y_star)),
        "y_star_std":  float(np.std(y_star, ddof=1)),
        "chi2_star_mean": float(np.mean(chi2_star)),
        "chi2_star_std":  float(np.std(chi2_star, ddof=1)),
        "X_star_mean":    np.mean(X_star, axis=0),
        "X_star_std":     np.std(X_star, axis=0, ddof=1),
    }
    return out

# ------------------------------
# Restart helper
# ------------------------------
def Restart(bounds, file_name, param_order=None):
    """Load previous CSV if present. Columns: [ID] + param_order + [Chi2, Y] (or legacy [Obj])."""
    if param_order is None:
        param_order = list(bounds.keys())
    csv_path = f"{file_name}.csv"
    if (pd is None) or (not os.path.exists(csv_path)):
        X = np.empty((0, len(param_order)), dtype=float)
        y = np.array([], dtype=float)
        idx_list = np.array([[0]], dtype=int)
        return X, y, idx_list

    try:
        df = pd.read_csv(csv_path, sep="\t")
        if "Y" in df.columns:
            X = df[param_order].to_numpy(dtype=float)
            y = df["Y"].to_numpy(dtype=float).reshape(-1)
        elif "Obj" in df.columns:  # legacy
            X = df[param_order].to_numpy(dtype=float)
            y = df["Obj"].to_numpy(dtype=float).reshape(-1)
        else:
            raise ValueError("No usable objective column found (need 'Y' or 'Obj').")
        idx = df["ID"].to_numpy(dtype=int).reshape(-1, 1)
        if idx.size == 0:
            idx = np.array([[0]], dtype=int)
        return X, y, idx
    except Exception:
        X = np.empty((0, len(param_order)), dtype=float)
        y = np.array([], dtype=float)
        idx_list = np.array([[0]], dtype=int)
        return X, y, idx_list
