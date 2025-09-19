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
from math import ceil
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
    rs = None
    if rng is not None:
        rs = int(np.uint32(rng.integers(0, 2**32 - 1)))
    YS = model.sample_y(XR, n_samples=1, random_state=rs)
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
def _posterior_samples(model, XR, n_funcs=200, noise_free=False, posterior_seed=None):
    """
    Return shape: (n_cand, n_funcs) of sampled latent values y=f(XR)
    noise_free=True면 predict(return_cov=True)로 얻은 f의 공분산에서 직접 MVN 샘플.
    """
    n_funcs = int(n_funcs)
    if noise_free:
        mu, cov = model.predict(XR, return_cov=True)  # UnitSpaceGP가 return_cov=True 지원
        mu = np.asarray(mu, dtype=float).reshape(-1)
        rng = np.random.default_rng(posterior_seed)
        # size=n_funcs => (n_funcs, n_cand) -> T
        YS = rng.multivariate_normal(mean=mu, cov=cov, size=n_funcs).T
    else:
        rs = None if posterior_seed is None else int(np.uint32(posterior_seed))
        YS = model.sample_y(XR, n_samples=n_funcs, random_state=rs)
        if YS.ndim == 1: YS = YS[:, None]
        if YS.shape[0] != XR.shape[0]: YS = YS.T
    return YS


def optimal_std_via_sampling(model, bounds, param_order, X=None, y=None,
                             n_funcs=200, n_cand=2000, trust_region=None, rng=None, eps=1e-12,
                             posterior_seed=None, noise_free=False, global_scope=False):
    """
    posterior_seed: 최종 샘플링 고정 시드
    noise_free:     WhiteKernel 등 관측노이즈 제거한 f posterior에서 샘플
    global_scope:   True면 trust_region 무시하고 전역 범위에서 후보 생성
    """
    if rng is None:
        rng = np.random.default_rng(posterior_seed if posterior_seed is not None else 0)

    # trust-region 처리
    tr = None
    if (not global_scope) and (trust_region is not None):
        if trust_region.get('auto_center', False) and (X is not None) and (y is not None) and (len(y) > 0):
            X = _as_2d(X); y = np.asarray(y, dtype=float).reshape(-1)
            best_idx = int(np.argmax(y))
            x_best = X[best_idx:best_idx+1, :]
            center_u = to_unit_batch(bounds, param_order, x_best).reshape(-1)
            tr = {'L': float(trust_region.get('L', 1.0)), 'center_u': center_u}
        elif 'center_u' in trust_region:
            tr = {'L': float(trust_region.get('L', 1.0)),
                  'center_u': np.asarray(trust_region['center_u'], dtype=float).reshape(-1)}

    # 후보 생성 (전역/지역)
    XR, _ = _sample_candidates(bounds, param_order, n_cand=int(n_cand), trust_region=tr, rng=rng)

    # 포스터리어 함수 샘플링
    YS = _posterior_samples(model, XR, n_funcs=n_funcs, noise_free=bool(noise_free), posterior_seed=posterior_seed)

    idx_max = np.argmax(YS, axis=0)
    y_star = YS[idx_max, np.arange(YS.shape[1])]
    X_star = XR[idx_max, :]
    chi2_star = np.maximum(np.exp(-y_star) - float(eps), 0.0)

    out = {
        "y_star_mean":   float(np.mean(y_star)),
        "y_star_std":    float(np.std(y_star, ddof=1)),
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
# levelset_region_sampling
# ------------------------------
def levelset_region_sampling(model, bounds, param_order,
                             chi2_min=None, delta=1.0,
                             n_samples=500,                 # 저장할 샘플 수
                             X_hist=None, chi2_hist=None,   # 관측 이력 (필수)
                             eps=1e-12, seed=None, pad=0.02,
                             outfile="posterior_levelset_samples.npz",
                             # --- posterior sampling 옵션 ---
                             posterior=False,               # True면 포스터리어 샘플링
                             n_funcs=400,                   # 함수 실현 개수
                             posterior_seed=None,           # 샘플링 시드
                             save_all_draws=False,          # 모든 실현 저장(큰 파일!)
                             q=None, batch=5000,
                             # --- 안정성/재현성/커버리지 옵션 ---
                             include_topK=5,                # 관측 상위 K(작은 χ²) 점을 S에 포함 (0이면 비활성)
                             min_pts_factor=5,              # 폴백 시 최소 필요 표본 크기의 계수
                             quantile_box=None):            # (q_lo,q_hi) 지정시 분위수 박스 사용
    """
    레벨셋 박스는 관측 이력으로 정의(χ² ≤ chi2_min+δ). 박스 안에서 균일 샘플 S를 뽑아:
      - posterior=False: plug-in/mean 기반 χ² 추정
      - posterior=True : 노이즈-프리 포스터리어(잠재함수)에서 MVN 샘플링 → χ² 요약 저장

    저장(.npz)에는 숫자 배열만 사용해 Unicode dtype 문제를 회피.
    """
    import numpy as np

    if X_hist is None or chi2_hist is None:
        raise ValueError("X_hist와 chi2_hist를 제공해야 합니다.")

    rng = np.random.default_rng(seed)
    d = len(param_order)
    lo = np.array([bounds[p][0] for p in param_order], dtype=float)
    hi = np.array([bounds[p][1] for p in param_order], dtype=float)
    width = (hi - lo)

    # --- 이력 & 유효값 필터 ---
    X_hist = np.asarray(X_hist, dtype=float).reshape(-1, d)
    chi2_hist = np.asarray(chi2_hist, dtype=float).reshape(-1)
    valid = np.isfinite(chi2_hist)
    if not np.any(valid):
        raise ValueError("chi2_hist에 유효한 값이 없습니다 (모두 NaN/Inf).")
    Xv = X_hist[valid]
    cv = chi2_hist[valid]
    N = Xv.shape[0]

    # --- 임계값 ---
    cmin_data = np.min(cv) if (chi2_min is None) else float(chi2_min)
    thr = float(cmin_data) + float(delta)

    # --- 임계 이내 관측점 ---
    mask = (cv <= thr)
    pts = Xv[mask]

    # --- 폴백: 임계 이내가 너무 적으면 상위 K개로 대체(안전한 K/kth 계산) ---
    min_need = max(2*d, min_pts_factor*d)  # 차원 대비 충분한 표본
    if pts.shape[0] < min_need:
        K_raw = max(min_need, min(2000, N))
        K = int(min(K_raw, N))                   # K ≤ N
        kth = max(1, min(K-1, N-1))              # 1..N-1
        idx = np.argpartition(cv, kth)[:K]
        pts = Xv[idx]

    # --- 박스 산정: 분위수 박스 or min/max 박스 ---
    if quantile_box is not None:
        q_lo, q_hi = float(quantile_box[0]), float(quantile_box[1])
        box_lo0 = np.quantile(pts, q_lo, axis=0)
        box_hi0 = np.quantile(pts, q_hi, axis=0)
    else:
        box_lo0 = np.min(pts, axis=0)
        box_hi0 = np.max(pts, axis=0)

    box_lo = np.maximum(lo, box_lo0 - width*float(pad))
    box_hi = np.minimum(hi, box_hi0 + width*float(pad))

    # zero-width 보정
    epsw = 1e-9 + 0.01 * width
    tight = (box_hi - box_lo) < 1e-12
    box_lo[tight] = np.maximum(lo[tight], box_lo[tight] - 0.5*epsw[tight])
    box_hi[tight] = np.minimum(hi[tight], box_hi[tight] + 0.5*epsw[tight])
    box = np.vstack([box_lo, box_hi]).T  # (d,2)

    # --- 박스 내부 균일 샘플 ---
    S = rng.uniform(low=box_lo, high=box_hi, size=(int(n_samples), d))

    # (선택) 관측 상위 K개를 포함시켜 최적 근방 커버리지 보장
    if include_topK and include_topK > 0:
        k = int(min(include_topK, N))
        top_idx = np.argsort(cv)[:k]
        S = np.vstack([S, Xv[top_idx]])
    S = S.astype(np.float64, copy=False)

    # --- χ² 계산 / 저장용 dict ---
    out = {
        "samples": S,
        "box": box.astype(np.float64),
        "chi2_min": float(cmin_data),
        "delta": float(delta),
        "threshold": float(thr),
        "n_samples": int(S.shape[0]),
        "posterior": int(bool(posterior)),
        "n_funcs": int(n_funcs if posterior else 0),
        "seed": int(seed) if (seed is not None) else -1,
        "posterior_seed": int(posterior_seed) if (posterior_seed is not None) else -1,
        "save_all_draws": int(bool(save_all_draws)),
        "include_topK": int(include_topK),
    }

    if not posterior:
        # ------- plug-in/mean 기반 -------
        try:
            mu_s, std_s = model.predict(S, return_std=True)
            mu_s = np.asarray(mu_s).reshape(-1)
            std_s = np.asarray(std_s).reshape(-1)
        except TypeError:
            # return_std 미지원 시 폴백(평균만)
            mu_s = np.asarray(model.predict(S)).reshape(-1)
            std_s = np.zeros_like(mu_s)

        chi2_plugin = np.exp(-mu_s) - float(eps)                      # median/plug-in
        chi2_mean   = np.exp(-mu_s + 0.5*std_s*std_s) - float(eps)    # E[χ²]
        out["chi2_plugin"] = chi2_plugin.astype(np.float64)
        out["chi2_mean"]   = chi2_mean.astype(np.float64)

    else:
        # ------- 포스터리어 샘플링 (노이즈-프리) -------
        # return_cov 미지원 시 안전 폴백: 독립 가우시안 근사
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
            # 수치안정화: 대각선 평균을 스케일로 지터 증가
            diag = np.clip(np.diag(cov), 0.0, None)
            base = float(np.mean(diag) + 1e-16)
            ok = False
            for ktry in range(7):
                try:
                    YS = rng_ps.multivariate_normal(mean=mu, cov=cov, size=int(n_funcs))
                    ok = True
                    break
                except np.linalg.LinAlgError:
                    cov = cov + (10.0**ktry) * (1e-12 + 1e-6*base) * np.eye(cov.shape[0], dtype=float)
            if not ok:
                std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                YS = mu[None, :] + std[None, :] * rng_ps.standard_normal(size=(int(n_funcs), mu.size))
        else:
            # 독립 근사
            YS = mu[None, :] + std[None, :] * rng_ps.standard_normal(size=(int(n_funcs), mu.size))

        CH = np.maximum(np.exp(-YS) - float(eps), 0.0)   # (n_funcs, N_tot)
        chi2_ps_mean   = np.mean(CH, axis=0)
        chi2_ps_median = np.median(CH, axis=0)
        chi2_ps_q05    = np.quantile(CH, 0.05, axis=0)
        chi2_ps_q95    = np.quantile(CH, 0.95, axis=0)
        chi2_ps_min    = np.min(CH, axis=0)

        out["chi2_ps_mean"]   = chi2_ps_mean.astype(np.float64)
        out["chi2_ps_median"] = chi2_ps_median.astype(np.float64)
        out["chi2_ps_q05"]    = chi2_ps_q05.astype(np.float64)
        out["chi2_ps_q95"]    = chi2_ps_q95.astype(np.float64)
        out["chi2_ps_min"]    = chi2_ps_min.astype(np.float64)

        if save_all_draws:
            out["chi2_draws"] = CH.T.astype(np.float64)  # (N_tot, n_funcs)

    np.savez(outfile, **out)
    return {"outfile": outfile, "box": box, "n_kept_for_box": int(pts.shape[0]),
            "threshold": thr}

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
