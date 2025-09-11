"""
Bopt.py — Bayesian Optimization helpers
(unit-space wrapper, trust-region aware, exploration helpers, pairwise heatmaps)

Additions in this patch:
- UnitSpaceGP: GP wrapper to operate in unit space [0,1]^d
- Propose_Thompson / Propose_MaxStd: forced exploration proposals
- Global_PI: global Probability-of-Improvement score (far from existing points)
- Min-distance preference inside Opt_Acquisition to reduce local clustering
- Pairwise heatmap utilities (E[chi2] and PI maps)
- Other utilities retained (EI, TuRBO-lite sampling, restart, posterior sampling)
"""
import os
import numpy as np
import warnings
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist

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
    return (X - lo) / (hi - lo + 1e-15)

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
# Exploration helpers
# ------------------------------
def _far_mask(bounds, param_order, XR, X, min_dist=0.15):
    Uc = to_unit_batch(bounds, param_order, XR)
    if X is None or len(X) == 0:
        return np.ones(len(XR), dtype=bool)
    Ux = to_unit_batch(bounds, param_order, X)
    dmin = cdist(Uc, Ux).min(axis=1)
    return dmin > float(min_dist)

def Propose_Thompson(model, bounds, param_order, X=None, n_cand=4096, min_dist=0.15, rng=None):
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=None, rng=rng)
    YS = model.sample_y(XR, n_samples=1)  # (n_cand,1) or (n_cand,)
    if YS.ndim > 1:
        YS = YS[:, 0]
    mask = _far_mask(bounds, param_order, XR, X, min_dist=min_dist)
    if np.any(mask):
        idx = int(np.argmax(YS[mask]))
        return XR[mask][idx:idx+1, :]
    return XR[np.argmax(YS):np.argmax(YS)+1, :]

def Propose_MaxStd(model, bounds, param_order, X=None, n_cand=4096, min_dist=0.15, rng=None):
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=None, rng=rng)
    _, std = surrogate(model, XR)
    mask = _far_mask(bounds, param_order, XR, X, min_dist=min_dist)
    if np.any(mask):
        idx = int(np.argmax(std[mask]))
        return XR[mask][idx:idx+1, :]
    return XR[np.argmax(std):np.argmax(std)+1, :]

def Global_PI(model, bounds, param_order, X, y, delta=1e-3, n_cand=4000, min_dist=0.15, rng=None):
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=None, rng=rng)
    mu, std = surrogate(model, XR)
    y_best = float(np.max(y)) if len(y) else -np.inf
    z = (mu - (y_best + delta)) / (std + 1e-12)
    PI = norm.cdf(z)
    mask = _far_mask(bounds, param_order, XR, X, min_dist=min_dist)
    return float(np.max(PI[mask])) if np.any(mask) else float(np.max(PI))

# ------------------------------
# Acquisition with min-distance preference
# ------------------------------
def Opt_Acquisition(X, y_obs, model, bounds, explore=0.01, n_cand=4096, k_refine=8,
                    param_order=None, trust_region=None, rng=None, min_dist=0.10):
    """Return one next point (1,d). Prefers candidates far from observed points (min_dist in unit space)."""
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
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=tr, rng=rng)
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

    # Prefer far-enough candidate among EI-ranked list
    X_u = to_unit_batch(bounds, param_order, X)
    for idx0 in order:
        x_try = XR[idx0]
        x_u = to_unit_batch(bounds, param_order, x_try).reshape(1, -1)
        if X_u is None or len(X_u) == 0:
            best_x = x_try; break
        dmin = np.min(np.linalg.norm(X_u - x_u, axis=1))
        if dmin > float(min_dist):
            best_x = x_try; break
    # else keep best_x from local refine

    return best_x.reshape(1, -1)

# ------------------------------
# Posterior sampling-based uncertainty of optimum
# ------------------------------
def optimal_std_via_sampling(model, bounds, param_order, X=None, y=None,
                             n_funcs=200, n_cand=2000, trust_region=None,
                             rng=None, eps=1e-12, posterior_seed=12345,
                             noise_free=True, global_scope=False):
    if rng is None:
        rng = np.random.default_rng()
    # (1) 신뢰영역 사용 여부
    tr = None
    if (not global_scope) and (trust_region is not None):
        if trust_region.get('auto_center', False) and (X is not None) and (y is not None) and (len(y) > 0):
            X = np.atleast_2d(X); y = np.asarray(y).reshape(-1)
            best_idx = int(np.argmax(y)); x_best = X[best_idx:best_idx+1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}
    # (2) 후보 생성
    XR, _ = _sample_candidates(bounds, param_order, n_cand=n_cand, trust_region=tr, rng=rng)
    # (3) 노이즈 없는 포스터리어에서 샘플
    if noise_free:
        mu, cov = model.predict(XR, return_cov=True)  # noise-free predictive cov
        cov = np.asarray(cov, dtype=float)
        # 수치 안정화
        cov.flat[::cov.shape[0]+1] += 1e-10
        rng2 = np.random.default_rng(posterior_seed)
        YS = rng2.multivariate_normal(mean=mu, cov=cov, size=int(n_funcs)).T  # (n_cand, n_funcs)
    else:
        YS = model.sample_y(XR, n_samples=int(n_funcs), random_state=posterior_seed)
        if YS.ndim == 1: YS = YS[:, None]
        if YS.shape[0] != XR.shape[0]: YS = YS.T
    # (4) 각 샘플 함수에서의 최적
    idx_max = np.argmax(YS, axis=0)
    y_star = YS[idx_max, np.arange(YS.shape[1])]
    X_star = XR[idx_max, :]
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
# Pairwise heatmaps (E[chi2] / PI)
# ------------------------------
def _grid_for_pair(bounds, param_order, pair, x_fixed, grid_n=80):
    p, q = pair
    i = param_order.index(p); j = param_order.index(q)
    lo, hi = _bounds_arrays(bounds, param_order)
    xs = np.linspace(lo[i], hi[i], grid_n)
    ys = np.linspace(lo[j], hi[j], grid_n)
    # build full-dim grid with others fixed to x_fixed
    base = np.array([x_fixed[k] for k in param_order], dtype=float)
    grid = []
    for b in ys:
        for a in xs:
            v = base.copy()
            v[i] = a; v[j] = b
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

    mode:
      - 'Echi2' : E[chi2] = exp(-mu + 0.5*std^2) - eps   (y ~ N(mu,std^2), chi2 = exp(-y)-eps)
      - 'PI'    : P(chi2 <= (1+eta)*chi2_best) == P(y >= y_thr)

    If save_data:
      data_format='npz'  -> <prefix>_<p>_vs_<q>_<mode>.npz  (xs, ys, Z, mu, std, meta…)
      data_format='csv'  -> <prefix>_<p>_vs_<q>_<mode>.csv  (long format: p, q, Z, mu, std)
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
        mu = mu.reshape(-1); std = std.reshape(-1)

        if mode.upper() == "ECHI2":
            Z = np.maximum(np.exp(-mu + 0.5*std**2) - eps, 0.0)   # expected chi^2
            label = "E[$\\chi^2$]"
        else:  # PI
            y_thr = -np.log((1.0 + float(eta)) * chi2_best + eps)
            z = (mu - y_thr) / (std + 1e-12)
            Z = norm.cdf(z)
            label = "PI"

        # reshape to 2D for saving/plotting
        Z2  = Z.reshape(len(ys), len(xs))
        MU2 = mu.reshape(len(ys), len(xs))
        SD2 = std.reshape(len(ys), len(xs))

        # ---------- (A) 데이터 저장 ----------
        stem = f"{out_prefix}_{p}_vs_{q}_{mode}"
        if save_data:
            dfmt = str(data_format).lower()
            if dfmt == "npz":
                np.savez(stem + ".npz",
                         xs=np.asarray(xs), ys=np.asarray(ys),
                         Z=Z2, mu=MU2, std=SD2)
                if save_hist and (X_hist is not None) and (len(X_hist) > 0):
                    hist = np.asarray(X_hist)[:, [ii, jj]]
                    np.save(stem + "_hist.npy", hist)
            elif dfmt == "csv":
                # long format: columns -> p, q, Z, mu, std
                XX, YY = np.meshgrid(xs, ys)  # shape (ny, nx)
                arr = np.column_stack([XX.ravel(), YY.ravel(), Z2.ravel(), MU2.ravel(), SD2.ravel()])
                header = f"{p},{q},Z,mu,std"
                np.savetxt(stem + ".csv", arr, delimiter=",", header=header, comments="")
                if save_hist and (X_hist is not None) and (len(X_hist) > 0):
                    hist = np.asarray(X_hist)[:, [ii, jj]]
                    np.savetxt(stem + "_hist.csv", hist, delimiter=",", header=f"{p},{q}", comments="")
            else:
                raise ValueError("data_format must be 'npz' or 'csv'")

        # ---------- (B) 그림 저장 ----------
        fig, ax = plt.subplots(figsize=(4.2, 3.8), layout='constrained')
        im = ax.imshow(Z2, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       origin='lower', aspect='auto')
        ax.set_xlabel(p); ax.set_ylabel(q)
        cb = fig.colorbar(im, ax=ax, shrink=0.84)
        cb.set_label(label)

        # overlay: past evals and current best (x_fixed)
        if X_hist is not None and len(X_hist):
            pts = np.asarray(X_hist)[:, [ii, jj]]
            ax.scatter(pts[:,0], pts[:,1], s=14, c='k', alpha=0.35, linewidths=0)
        ax.scatter([x_fixed[p]], [x_fixed[q]], s=160, marker='*',
                   facecolors='none', edgecolors='w', linewidths=1.8)

        fig.savefig(stem + ".png", dpi=200)
        plt.close(fig)
        
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
        # use C engine explicitly to avoid regex-sep fallback warning
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
