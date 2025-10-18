"""
Bopt.py - Bayesian Optimization helpers
(unit-space wrapper, trust-region aware acquisition policies, posterior analysis utilities)
"""
import os
import json
import warnings
import itertools
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm, qmc

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

# ------------------------------
# Bounds helpers & normalization
# ------------------------------
def _bounds_arrays(bounds, param_order):
    """Return low/high arrays respecting param_order."""
    lo = np.array([bounds[p][0] for p in param_order], dtype=float)
    hi = np.array([bounds[p][1] for p in param_order], dtype=float)
    return lo, hi


def to_unit_batch(bounds, param_order, X):
    """Map points from original scale to [0,1]^d."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    lo, hi = _bounds_arrays(bounds, param_order)
    return (X - lo) / (hi - lo + 1e-15)


def from_unit_batch(bounds, param_order, U):
    """Map unit cube points back to original scale."""
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

    def predict(self, X, return_std=True, return_cov=False):
        X_u = self._to_unit(X)
        if return_cov:
            return self.model.predict(X_u, return_cov=True)
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


def _sample_candidates(bounds, param_order, n_cand=4096, trust_region=None, rng=None):
    """
    Sample candidates in ORIGINAL space, possibly inside a trust-region in unit space.

    trust_region: dict or None. If dict it can specify:
        - 'L'        : side length in unit space (0< L <=1)
        - 'center_u' : 1D array in [0,1]^d specifying TR centre in unit space
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(param_order)
    if trust_region is None:
        U = rng.random((int(n_cand), d))
    else:
        L = float(trust_region.get("L", 1.0))
        L = max(1e-6, min(1.0, L))
        center_u = np.asarray(trust_region["center_u"], dtype=float).reshape(-1)
        if center_u.size != d:
            raise ValueError("center_u size mismatch.")
        U = center_u + (rng.random((int(n_cand), d)) - 0.5) * L
        U = np.clip(U, 0.0, 1.0)
    XR = from_unit_batch(bounds, param_order, U)
    return XR, U


def _far_mask(bounds, param_order, XR, X, min_dist=0.15):
    """Boolean mask for candidates that are farther than min_dist (unit space)."""
    Uc = to_unit_batch(bounds, param_order, XR)
    if X is None or len(X) == 0:
        return np.ones(len(XR), dtype=bool)
    Ux = to_unit_batch(bounds, param_order, X)
    dmin = cdist(Uc, Ux).min(axis=1)
    return dmin > float(min_dist)


def _candidate_pool(bounds, param_order, n_cand=4096, *, trust_region=None,
                    rng=None, X=None, min_dist=None):
    """Sample candidates and optionally apply min-distance filter."""
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand,
                               trust_region=trust_region, rng=rng)
    mask = None
    if min_dist is not None:
        mask = _far_mask(bounds, param_order, XR, X, min_dist=min_dist)
    return XR, mask


def _select_candidate(scores, XR, mask):
    """Return candidate (1,d) with max score, preferring those passing mask."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if mask is not None and np.any(mask):
        idx_pool = np.where(mask)[0]
        best_idx = idx_pool[int(np.argmax(scores[idx_pool]))]
        return XR[best_idx:best_idx + 1, :]
    best_idx = int(np.argmax(scores))
    return XR[best_idx:best_idx + 1, :]


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
    """Compute Expected Improvement at XS given observed data."""
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
# Exploration helpers
# ------------------------------
def Propose_Thompson(model, bounds, param_order, X=None, n_cand=4096, min_dist=0.15, rng=None):
    """Force exploration via Thompson sampling draws."""
    XR, mask = _candidate_pool(bounds, param_order, n_cand=n_cand,
                               trust_region=None, rng=rng, X=X, min_dist=min_dist)
    rs = None
    if rng is not None:
        rs = int(np.uint32(rng.integers(0, 2 ** 32 - 1)))
    YS = model.sample_y(XR, n_samples=1, random_state=rs)
    if YS.ndim > 1:
        YS = YS[:, 0]
    return _select_candidate(YS, XR, mask)


def Propose_MaxStd(model, bounds, param_order, X=None, n_cand=4096, min_dist=0.15, rng=None):
    """Pick candidate with largest posterior standard deviation."""
    XR, mask = _candidate_pool(bounds, param_order, n_cand=n_cand,
                               trust_region=None, rng=rng, X=X, min_dist=min_dist)
    _, std = surrogate(model, XR)
    return _select_candidate(std, XR, mask)


def Global_PI(model, bounds, param_order, X, y, delta=1e-3, n_cand=4000, min_dist=0.15, rng=None):
    """Return global PI score focusing on far candidates."""
    XR, mask = _candidate_pool(bounds, param_order, n_cand=n_cand,
                               trust_region=None, rng=rng, X=X, min_dist=min_dist)
    mu, std = surrogate(model, XR)
    y_best = float(np.max(y)) if len(y) else -np.inf
    z = (mu - (y_best + float(delta))) / (std + 1e-12)
    PI = norm.cdf(z)
    if mask is not None and np.any(mask):
        return float(np.max(PI[mask]))
    return float(np.max(PI))


# ------------------------------
# Acquisition with min-distance preference
# ------------------------------
def Opt_Acquisition(X, y_obs, model, bounds, explore=0.01, n_cand=4096, k_refine=8,
                    param_order=None, trust_region=None, rng=None, min_dist=0.10):
    """Return one next point (1,d). Prefers candidates far from observed points."""
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
            x_best = X[best_idx:best_idx + 1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}

    # Sample candidates & evaluate EI
    XR, mask = _candidate_pool(bounds, param_order, n_cand=n_cand,
                               trust_region=tr, rng=rng, X=X, min_dist=min_dist)
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
        if res.success:
            x_cand = res.x
            val = -res.fun
        else:
            x_cand = s
            val = -neg_ei(x_cand)
        if val > best_val + 1e-14:
            best_val = val
            best_x = x_cand

    if mask is not None and np.any(mask):
        far_idx = set(np.where(mask)[0])
        for idx0 in order:
            if idx0 in far_idx:
                best_x = XR[idx0]
                break

    return best_x.reshape(1, -1)


# ------------------------------
# Posterior sampling-based uncertainty of optimum
# ------------------------------
def _posterior_samples(model, XR, n_funcs=200, noise_free=False, posterior_seed=None):
    """
    Return array of sampled latent values y = f(XR) with shape (n_cand, n_funcs).
    """
    n_funcs = int(n_funcs)
    if noise_free:
        mu, cov = model.predict(XR, return_cov=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        rng = np.random.default_rng(posterior_seed)
        YS = rng.multivariate_normal(mean=mu, cov=cov, size=n_funcs).T
    else:
        rs = None if posterior_seed is None else int(np.uint32(posterior_seed))
        YS = model.sample_y(XR, n_samples=n_funcs, random_state=rs)
        if YS.ndim == 1:
            YS = YS[:, None]
        if YS.shape[0] != XR.shape[0]:
            YS = YS.T
    return YS


def optimal_std_via_sampling(model, bounds, param_order, X=None, y=None,
                             n_funcs=200, n_cand=2000, trust_region=None, rng=None, eps=1e-12,
                             posterior_seed=None, noise_free=False, global_scope=False):
    """
    Estimate uncertainty of optimum via posterior sampling.
    """
    if rng is None:
        seed = posterior_seed if posterior_seed is not None else 0
        rng = np.random.default_rng(seed)

    tr = None
    if (not global_scope) and (trust_region is not None):
        if trust_region.get('auto_center', False) and (X is not None) and (y is not None) and len(y) > 0:
            X = _as_2d(X)
            y = np.asarray(y, dtype=float).reshape(-1)
            best_idx = int(np.argmax(y))
            x_best = X[best_idx:best_idx + 1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}

    XR, _ = _sample_candidates(bounds, param_order, n_cand=int(n_cand), trust_region=tr, rng=rng)
    YS = _posterior_samples(model, XR, n_funcs=n_funcs, noise_free=bool(noise_free),
                            posterior_seed=posterior_seed)

    idx_max = np.argmax(YS, axis=0)
    y_star = YS[idx_max, np.arange(YS.shape[1])]
    X_star = XR[idx_max, :]
    chi2_star = np.maximum(np.exp(-y_star) - float(eps), 0.0)

    out = {
        "y_star_mean": float(np.mean(y_star)),
        "y_star_std": float(np.std(y_star, ddof=1)),
        "chi2_star_mean": float(np.mean(chi2_star)),
        "chi2_star_std": float(np.std(chi2_star, ddof=1)),
        "X_star_mean": np.mean(X_star, axis=0),
        "X_star_std": np.std(X_star, axis=0, ddof=1),
    }
    return out


# ------------------------------
# Pairwise heatmaps (E[chi2] / PI)
# ------------------------------
def _grid_for_pair(bounds, param_order, pair, x_fixed, grid_n=80):
    """Build grid for a given parameter pair while fixing others."""
    p, q = pair
    i = param_order.index(p)
    j = param_order.index(q)
    lo, hi = _bounds_arrays(bounds, param_order)
    xs = np.linspace(lo[i], hi[i], grid_n)
    ys = np.linspace(lo[j], hi[j], grid_n)
    base = np.array([x_fixed[k] for k in param_order], dtype=float)
    grid = []
    for yy in ys:
        for xx in xs:
            v = base.copy()
            v[i] = xx
            v[j] = yy
            grid.append(v)
    XR = np.array(grid, dtype=float)
    return XR, xs, ys, i, j


def pairwise_heatmap_plot(model, bounds, param_order, x_fixed,
                          pairs=None, grid_n=80, mode="Echi2",
                          eta=0.0, X_hist=None, chi2_hist=None,
                          out_prefix="pair", eps=1e-12,
                          save_data=False, data_format="npz", save_hist=False):
    """
    Draw pairwise heatmaps and (optionally) save the underlying arrays.
    """
    if pairs is None:
        pairs = list(itertools.combinations(param_order, 2))

    chi2_best = None
    if mode.upper() == "PI":
        if chi2_hist is not None and len(chi2_hist):
            chi2_best = float(np.nanmin(chi2_hist))
        else:
            mu0, _ = model.predict(np.array([[x_fixed[k] for k in param_order]]), return_std=True)
            chi2_best = max(np.exp(-float(mu0)) - eps, 0.0)

    for (p, q) in pairs:
        XR, xs, ys, ii, jj = _grid_for_pair(bounds, param_order, (p, q), x_fixed, grid_n=grid_n)
        mu, std = model.predict(XR, return_std=True)
        mu = mu.reshape(-1)
        std = std.reshape(-1)

        if mode.upper() == "ECHI2":
            Z = np.maximum(np.exp(-mu + 0.5 * std ** 2) - eps, 0.0)
            label = "E[$\\chi^2$]"
        else:
            y_thr = -np.log((1.0 + float(eta)) * chi2_best + eps)
            z = (mu - y_thr) / (std + 1e-12)
            Z = norm.cdf(z)
            label = "PI"

        Z2 = Z.reshape(len(ys), len(xs))
        MU2 = mu.reshape(len(ys), len(xs))
        SD2 = std.reshape(len(ys), len(xs))

        stem = f"{out_prefix}_{p}_vs_{q}_{mode}"
        if save_data:
            dfmt = str(data_format).lower()
            if dfmt == "npz":
                np.savez(stem + ".npz", xs=np.asarray(xs), ys=np.asarray(ys),
                         Z=Z2, mu=MU2, std=SD2)
                if save_hist and (X_hist is not None) and len(X_hist):
                    hist = np.asarray(X_hist)[:, [ii, jj]]
                    np.save(stem + "_hist.npy", hist)
            elif dfmt == "csv":
                XX, YY = np.meshgrid(xs, ys)
                arr = np.column_stack([XX.ravel(), YY.ravel(),
                                       Z2.ravel(), MU2.ravel(), SD2.ravel()])
                header = f"{p},{q},Z,mu,std"
                np.savetxt(stem + ".csv", arr, delimiter=",", header=header, comments="")
                if save_hist and (X_hist is not None) and len(X_hist):
                    hist = np.asarray(X_hist)[:, [ii, jj]]
                    np.savetxt(stem + "_hist.csv", hist, delimiter=",",
                               header=f"{p},{q}", comments="")
            else:
                raise ValueError("data_format must be 'npz' or 'csv'")

        fig, ax = plt.subplots(figsize=(4.2, 3.8), layout='constrained')
        im = ax.imshow(Z2, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       origin='lower', aspect='auto')
        ax.set_xlabel(p)
        ax.set_ylabel(q)
        cb = fig.colorbar(im, ax=ax, shrink=0.84)
        cb.set_label(label)

        if X_hist is not None and len(X_hist):
            pts = np.asarray(X_hist)[:, [ii, jj]]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c='k', alpha=0.35, linewidths=0)
        ax.scatter([x_fixed[p]], [x_fixed[q]], s=160, marker='*',
                   facecolors='none', edgecolors='w', linewidths=1.8)

        fig.savefig(stem + ".png", dpi=200)
        plt.close(fig)


# ------------------------------
# levelset_region_sampling
# ------------------------------
def levelset_region_sampling(model, bounds, param_order,
                             chi2_min=None, delta=1.0,
                             n_samples=500,
                             X_hist=None, chi2_hist=None,
                             eps=1e-12, seed=None, pad=0.02,
                             outfile="posterior_levelset_samples.npz",
                             posterior=False,
                             n_funcs=400,
                             posterior_seed=None,
                             save_all_draws=False,
                             q=None, batch=5000,
                             include_topK=5,
                             min_pts_factor=5,
                             quantile_box=None):
    """
    Sample level-set region where chi2 <= chi2_min + delta.
    """
    if X_hist is None or chi2_hist is None:
        raise ValueError("X_hist and chi2_hist are required.")

    rng = np.random.default_rng(seed)
    d = len(param_order)
    lo = np.array([bounds[p][0] for p in param_order], dtype=float)
    hi = np.array([bounds[p][1] for p in param_order], dtype=float)
    width = hi - lo

    X_hist = np.asarray(X_hist, dtype=float).reshape(-1, d)
    chi2_hist = np.asarray(chi2_hist, dtype=float).reshape(-1)
    valid = np.isfinite(chi2_hist)
    if not np.any(valid):
        raise ValueError("chi2_hist contains no finite values.")
    Xv = X_hist[valid]
    cv = chi2_hist[valid]
    N = Xv.shape[0]

    cmin_data = np.min(cv) if chi2_min is None else float(chi2_min)
    thr = float(cmin_data) + float(delta)

    mask = (cv <= thr)
    pts = Xv[mask]

    min_need = max(2 * d, min_pts_factor * d)
    if pts.shape[0] < min_need:
        K_raw = max(min_need, min(2000, N))
        K = int(min(K_raw, N))
        kth = max(1, min(K - 1, N - 1))
        idx = np.argpartition(cv, kth)[:K]
        pts = Xv[idx]

    if quantile_box is not None:
        q_lo, q_hi = float(quantile_box[0]), float(quantile_box[1])
        box_lo0 = np.quantile(pts, q_lo, axis=0)
        box_hi0 = np.quantile(pts, q_hi, axis=0)
    else:
        box_lo0 = np.min(pts, axis=0)
        box_hi0 = np.max(pts, axis=0)

    box_lo = np.maximum(lo, box_lo0 - width * float(pad))
    box_hi = np.minimum(hi, box_hi0 + width * float(pad))

    epsw = 1e-9 + 0.01 * width
    tight = (box_hi - box_lo) < 1e-12
    box_lo[tight] = np.maximum(lo[tight], box_lo[tight] - 0.5 * epsw[tight])
    box_hi[tight] = np.minimum(hi[tight], box_hi[tight] + 0.5 * epsw[tight])
    box = np.vstack([box_lo, box_hi]).T

    S = rng.uniform(low=box_lo, high=box_hi, size=(int(n_samples), d))

    if include_topK and include_topK > 0:
        k = int(min(include_topK, N))
        top_idx = np.argsort(cv)[:k]
        S = np.vstack([S, Xv[top_idx]])
    S = S.astype(np.float64, copy=False)

    out = {
        "samples": S,
        "box": box.astype(np.float64),
        "chi2_min": float(cmin_data),
        "delta": float(delta),
        "threshold": float(thr),
        "n_samples": int(S.shape[0]),
        "posterior": int(bool(posterior)),
        "n_funcs": int(n_funcs if posterior else 0),
        "seed": int(seed) if seed is not None else -1,
        "posterior_seed": int(posterior_seed) if posterior_seed is not None else -1,
        "save_all_draws": int(bool(save_all_draws)),
        "include_topK": int(include_topK),
    }

    if not posterior:
        mu_s, std_s = model.predict(S, return_std=True)
        mu_s = np.asarray(mu_s).reshape(-1)
        std_s = np.asarray(std_s).reshape(-1)
        chi2_plugin = np.exp(-mu_s) - float(eps)
        chi2_mean = np.exp(-mu_s + 0.5 * std_s * std_s) - float(eps)
        out["chi2_plugin"] = chi2_plugin.astype(np.float64)
        out["chi2_mean"] = chi2_mean.astype(np.float64)
    else:
        cov_supported = True
        try:
            mu, cov = model.predict(S, return_cov=True)
            mu = np.asarray(mu, dtype=float).reshape(-1)
            cov = np.asarray(cov, dtype=float)
        except TypeError:
            cov_supported = False
            mu, std = model.predict(S, return_std=True)
            mu = np.asarray(mu, dtype=float).reshape(-1)
            std = np.asarray(std, dtype=float).reshape(-1)

        rng_ps = np.random.default_rng(posterior_seed)
        if cov_supported:
            diag = np.clip(np.diag(cov), 0.0, None)
            base = float(np.mean(diag) + 1e-16)
            ok = False
            for ktry in range(7):
                try:
                    YS = rng_ps.multivariate_normal(mean=mu, cov=cov, size=int(n_funcs))
                    ok = True
                    break
                except np.linalg.LinAlgError:
                    cov = cov + (10.0 ** ktry) * (1e-12 + 1e-6 * base) * np.eye(cov.shape[0], dtype=float)
            if not ok:
                std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                YS = mu[None, :] + std[None, :] * rng_ps.standard_normal(size=(int(n_funcs), mu.size))
        else:
            YS = mu[None, :] + std[None, :] * rng_ps.standard_normal(size=(int(n_funcs), mu.size))

        CH = np.maximum(np.exp(-YS) - float(eps), 0.0)
        out["chi2_ps_mean"] = np.mean(CH, axis=0).astype(np.float64)
        out["chi2_ps_median"] = np.median(CH, axis=0).astype(np.float64)
        out["chi2_ps_q05"] = np.quantile(CH, 0.05, axis=0).astype(np.float64)
        out["chi2_ps_q95"] = np.quantile(CH, 0.95, axis=0).astype(np.float64)
        out["chi2_ps_min"] = np.min(CH, axis=0).astype(np.float64)
        if save_all_draws:
            out["chi2_draws"] = CH.T.astype(np.float64)

    np.savez(outfile, **out)
    return {"outfile": outfile, "box": box, "n_kept_for_box": int(pts.shape[0]),
            "threshold": thr}


# ------------------------------
# Logging & post-processing helpers
# ------------------------------
def log_progress(file_name, fig_name, param_order, idx_list, X, chi2s, y, sep="\t"):
    """
    Persist optimization trace to CSV and generate diagnostic plots.
    """
    data = np.hstack([
        idx_list,
        X,
        np.asarray(chi2s, dtype=float).reshape(-1, 1),
        np.asarray(y, dtype=float).reshape(-1, 1),
    ])
    header = ["ID"] + list(param_order) + ["Chi2", "Y"]

    if pd is None:
        raise RuntimeError("pandas is required for logging progress.")

    df = pd.DataFrame(data, columns=header)
    df["ID"] = df["ID"].astype(int)
    df.to_csv(file_name + ".csv", sep=sep, index=False)

    if fig_name is None:
        return

    fig, ax = plt.subplots(layout='constrained')
    iters = np.arange(1, len(chi2s) + 1)
    ax.scatter(iters, chi2s, s=70)
    if len(chi2s):
        imin = int(np.argmin(chi2s)) + 1
        ax.scatter(imin, float(np.min(chi2s)), marker='*', s=200)
        ax.set_title(f"Minimum is Idx = {imin}", fontsize=20)
    else:
        ax.set_title("No evaluations yet", fontsize=20)
    ax.set_xlabel('Idx', fontsize=15)
    ax.set_ylabel(r'$\chi^2$', fontsize=15)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.savefig(fig_name + "_vs_Idx.png")
    plt.close(fig)

    if X.size == 0:
        return
    n_dim = X.shape[1]
    fig, axs = plt.subplots(1, n_dim, figsize=(3 * n_dim, 3), layout='constrained')
    if n_dim == 1:
        axs = [axs]
    if len(chi2s):
        ibest = int(np.argmin(chi2s))
        chi2_best = float(chi2s[ibest])
    else:
        ibest = 0
        chi2_best = float("nan")
    for i in range(n_dim):
        axs[i].scatter(X[:, i], chi2s, s=40)
        if len(chi2s):
            pbest = float(X[ibest, i])
            axs[i].scatter(pbest, chi2_best, marker='*', s=200)
            axs[i].set_title(f"Min at {param_order[i]} = {pbest:.5f}", fontsize=10)
        axs[i].set_xlabel(param_order[i], fontsize=10)
        axs[i].set_ylabel(r'$\chi^2$', fontsize=10)
        axs[i].grid(True)
        axs[i].set_axisbelow(True)
    fig.align_labels()
    fig.savefig(fig_name + "_vs_params.png")
    plt.close(fig)


def run_postprocessing(GP, bounds, param_order, X, y, chi2s, fig_name, *,
                       rng=None, levelset_kwargs=None,
                       uncertainty_kwargs=None, heatmap_kwargs=None):
    """
    Execute final uncertainty / visualization routines in a single call.
    """
    if len(X) == 0 or len(chi2s) == 0:
        return

    if rng is None:
        rng = np.random.default_rng()

    try:
        ukw = {} if uncertainty_kwargs is None else dict(uncertainty_kwargs)
        post_seed = int(rng.integers(1, 2 ** 31 - 1))
        ukw.setdefault("posterior_seed", post_seed)
        summary = optimal_std_via_sampling(
            GP, bounds, param_order,
            X=X, y=y,
            trust_region=None,
            eps=1e-12,
            noise_free=True,
            global_scope=True,
            **ukw,
        )
        print("[Uncertainty@final] y* std=%.4g  chi2* std=%.4g" %
              (summary["y_star_std"], summary["chi2_star_std"]))
        with open(fig_name + "_uncert.json", "w") as f:
            json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v)
                       for k, v in summary.items()}, f)
    except Exception as exc:  # pragma: no cover - best effort
        print("[Uncertainty@final] sampling failed:", exc)

    try:
        hkw = {"grid_n": 80, "mode": "Echi2",
               "save_data": True, "data_format": "npz", "save_hist": False}
        if heatmap_kwargs:
            hkw.update(heatmap_kwargs)
        ib = int(np.argmin(chi2s))
        x_best = {p: float(X[ib, k]) for k, p in enumerate(param_order)}
        pairwise_heatmap_plot(
            GP, bounds, param_order, x_best,
            X_hist=X, chi2_hist=chi2s,
            out_prefix=fig_name + "_pair",
            **hkw,
        )
        print("[Pairwise] heatmaps saved with prefix:", fig_name + "_pair")
    except Exception as exc:  # pragma: no cover
        print("[Pairwise] plotting failed:", exc)

    try:
        lkw = {"delta": 1.0, "n_samples": 10000,
               "posterior": True, "n_funcs": 400, "save_all_draws": False}
        if levelset_kwargs:
            lkw.update(levelset_kwargs)
        chi2_min = float(np.min(chi2s))
        out = levelset_region_sampling(
            GP, bounds, param_order,
            chi2_min=chi2_min,
            X_hist=X, chi2_hist=chi2s,
            outfile=fig_name + "_levelset_samples_ps.npz",
            **lkw,
        )
        print("[LevelSet] saved:", out["outfile"])
        print("[LevelSet] box (lo,hi) per dim:\n", out["box"])
    except Exception as exc:  # pragma: no cover
        print("[LevelSet] sampling failed:", exc)


# ------------------------------
# Restart helper
# ------------------------------
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
        df = pd.read_csv(csv_path, sep="\t", engine="c")
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


# ------------------------------
# Latin Hypercube Sampling for better initial design
# ------------------------------
def latin_hypercube_sampling(bounds, param_order, n_samples, seed=None):
    """Generate Latin Hypercube samples for better space-filling initial design."""
    d = len(param_order)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    U = sampler.random(n=int(n_samples))
    return from_unit_batch(bounds, param_order, U)


# ------------------------------
# Improved trust-region with more aggressive expansion
# ------------------------------
def update_trust_region_aggressive(L, succ, fail, improved, succ_th=2, fail_th=2):
    """
    More aggressive trust-region updates for expensive objectives.
    Expand faster on success, contract more conservatively on failure.
    """
    L_min = 0.05

    if improved:
        succ += 1
        fail = 0
        L = min(1.0, L * 2.0)
        if succ >= succ_th:
            L = min(1.0, L * 1.5)
            succ = 0
    else:
        fail += 1
        succ = 0
        if fail >= fail_th:
            L *= 0.7
            fail = 0

    if L < L_min:
        L = 0.8
        succ, fail = 0, 0

    return L, succ, fail
