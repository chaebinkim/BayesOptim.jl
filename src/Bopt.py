"""
Bopt.py — Minimal-yet-robust helpers for GP-based Bayesian Optimization (for PyCall.jl)

This file is designed to be imported from Julia via PyCall (@pyinclude).
It provides:
- Input scaling to unit space [0,1]^d
- A UnitSpaceGP wrapper supporting return_std and return_cov
- Expected Improvement acquisition
- Candidate sampling (global or inside a unit-space trust box)
- An acquisition optimizer with multi-start L-BFGS-B
- A simple Restart() that loads/saves CSV logs compatible with Fit.jl
- Posterior sampling of the optimum with options for noise-free/global sampling
"""
import os
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist

try:
    import pandas as pd
except Exception:
    pd = None

# ---------------------------------------------------------------------
# Bounds helpers & normalization
# ---------------------------------------------------------------------
def _bounds_arrays(bounds, param_order):
    lo = np.array([bounds[p][0] for p in param_order], dtype=float)
    hi = np.array([bounds[p][1] for p in param_order], dtype=float)
    return lo, hi

def to_unit_batch(bounds, param_order, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    lo, hi = _bounds_arrays(bounds, param_order)
    return (X - lo) / (hi - lo + 1e-15)

def from_unit_batch(bounds, param_order, U):
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    lo, hi = _bounds_arrays(bounds, param_order)
    return lo + U * (hi - lo)

def _as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[None, :] if X.ndim == 1 else X

# ---------------------------------------------------------------------
# Unit-space GP wrapper
# ---------------------------------------------------------------------
class UnitSpaceGP:
    """Wrap sklearn GPR so that fit/predict/sample_y operate in unit space [0,1]^d.
       Supports return_std and return_cov for predict().
    """
    def __init__(self, model, bounds, param_order):
        self.model = model
        self.bounds = bounds
        self.param_order = list(param_order)

    def _to_unit(self, X):
        return to_unit_batch(self.bounds, self.param_order, X)

    def fit(self, X, y):
        X_u = self._to_unit(X)
        return self.model.fit(X_u, y)

    def predict(self, X, return_std=True, return_cov=False):
        """Return mu,(std|cov) or mu depending on flags.
        Only one of (return_std, return_cov) may be True.
        The covariance returned is the noise-free predictive covariance.
        """
        if return_std and return_cov:
            raise ValueError("Only one of return_std or return_cov can be True.")
        X_u = self._to_unit(X)
        if return_cov:
            mu, cov = self.model.predict(X_u, return_cov=True)
            return np.asarray(mu), np.asarray(cov)
        elif return_std:
            mu, std = self.model.predict(X_u, return_std=True)
            return np.asarray(mu), np.asarray(std)
        else:
            mu = self.model.predict(X_u, return_std=False)
            return np.asarray(mu)

    def sample_y(self, X, n_samples=1, random_state=None):
        X_u = self._to_unit(X)
        return self.model.sample_y(X_u, n_samples=n_samples, random_state=random_state)

# ---------------------------------------------------------------------
# Surrogate helpers
# ---------------------------------------------------------------------
def surrogate(model, X):
    warnings.filterwarnings("ignore")
    X = _as_2d(X)
    mu, std = model.predict(X, return_std=True)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    return mu, std

def Expected_Improvement(X_obs, XS, model, explore, y_obs):
    """EI(x) = (μ - y_best - ξ) Φ(Z) + σ φ(Z), maximize y = -log(chi2+eps)."""
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

# ---------------------------------------------------------------------
# Candidate sampling & acquisition
# ---------------------------------------------------------------------
def _sample_candidates(bounds, param_order, n_cand=4096, trust_region=None, rng=None):
    """Sample candidate points in ORIGINAL space. If trust_region is not None,
       sample a unit-space box with side length L around center_u (both in [0,1]^d).
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(param_order)
    if trust_region is None:
        U = rng.random((n_cand, d))  # global
    else:
        L = float(trust_region.get('L', 1.0))
        L = max(1e-6, min(1.0, L))
        center_u = np.asarray(trust_region['center_u'], dtype=float).reshape(-1)
        U = center_u + (rng.random((n_cand, d)) - 0.5) * L
        U = np.clip(U, 0.0, 1.0)
    XR = from_unit_batch(bounds, param_order, U)
    return XR, U

def Opt_Acquisition(X, y_obs, model, bounds, explore=0.01, n_cand=4096, k_refine=8,
                    param_order=None, trust_region=None, rng=None):
    """Return one next point (1,d). Multi-start EI + L-BFGS-B within (optional) trust region."""
    X = _as_2d(X)
    y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
    if param_order is None:
        param_order = list(bounds.keys())
    if rng is None:
        rng = np.random.default_rng()

    # Sample and score
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=trust_region, rng=rng)
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

    return best_x.reshape(1, -1)

# ---------------------------------------------------------------------
# Posterior sampling of optimum (for uncertainty of optimum)
# ---------------------------------------------------------------------
def optimal_std_via_sampling(model, bounds, param_order, X=None, y=None,
                             n_funcs=200, n_cand=2000, trust_region=None,
                             rng=None, eps=1e-12, posterior_seed=12345,
                             noise_free=True, global_scope=True):
    """Monte Carlo posterior estimate of optimum's mean/std (in y and chi2 domains).
       - noise_free=True : use noise-free predictive covariance for sampling
       - global_scope=True: ignore trust_region and sample globally
    """
    if rng is None:
        rng = np.random.default_rng()

    tr = None
    if (not global_scope) and (trust_region is not None):
        # center TR at current best (in unit space)
        if trust_region.get('auto_center', False) and (X is not None) and (y is not None) and (len(y) > 0):
            X = _as_2d(X); y = np.asarray(y, dtype=float).reshape(-1)
            best_idx = int(np.argmax(y))
            x_best = X[best_idx:best_idx+1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}

    # Candidate set
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=tr, rng=rng)

    # Draw functions from GP posterior
    if noise_free:
        mu, cov = model.predict(XR, return_cov=True)  # noise-free cov
        cov = np.asarray(cov, dtype=float)
        # numerical jitter
        cov.flat[::cov.shape[0]+1] += 1e-10
        rng2 = np.random.default_rng(posterior_seed)
        YS = rng2.multivariate_normal(mean=mu, cov=cov, size=int(n_funcs)).T  # (n_cand, n_funcs)
    else:
        YS = model.sample_y(XR, n_samples=int(n_funcs), random_state=posterior_seed)
        if YS.ndim == 1: YS = YS[:, None]
        if YS.shape[0] != XR.shape[0]: YS = YS.T

    # For each sampled function, take argmax in y (equiv. min chi2)
    idx_max = np.argmax(YS, axis=0)
    y_star = YS[idx_max, np.arange(YS.shape[1])]
    X_star = XR[idx_max, :]
    chi2_star = np.maximum(np.exp(-y_star) - float(eps), 0.0)

    return {
        "y_star_mean": float(np.mean(y_star)),
        "y_star_std":  float(np.std(y_star, ddof=1)),
        "chi2_star_mean": float(np.mean(chi2_star)),
        "chi2_star_std":  float(np.std(chi2_star, ddof=1)),
        "X_star_mean":    np.mean(X_star, axis=0),
        "X_star_std":     np.std(X_star, axis=0, ddof=1),
    }

# ---------------------------------------------------------------------
# Restart helper (load CSV if exists)
# ---------------------------------------------------------------------
def Restart(bounds, file_name, param_order=None):
    if param_order is None:
        param_order = list(bounds.keys())
    csv_path = f"{file_name}.csv"
    if (pd is None) or (not os.path.exists(csv_path)):
        X = np.empty((0, len(param_order)), dtype=float)
        y = np.array([], dtype=float)
        idx_list = np.array([[0]], dtype=int)
        return X, y, idx_list
    try:
        df = pd.read_csv(csv_path, sep="\t", engine="c")  # enforce C engine
        if "Y" in df.columns:
            X = df[param_order].to_numpy(dtype=float)
            y = df["Y"].to_numpy(dtype=float).reshape(-1)
        elif "Obj" in df.columns:
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
