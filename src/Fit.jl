function Fit(Objective, interval, max_iter; file_name = "Bopt_Log", fig_name = "chi2", ref_point = nothing, delta = 1.0)
    DIR = @__DIR__
    @pyinclude(DIR*"/Bopt.py")
    py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

SEP = "\t"  # use a real tab and force C engine on read_csv

# ---- Helpers for safe log-transform ----
def sanitize_chi2(chi2, chi2_bad=1e30):
    z = float(chi2)
    if not np.isfinite(z) or z <= 0:
        z = chi2_bad
    return z

def chi2_to_y(chi2, eps=1e-12, chi2_bad=1e30):
    z = sanitize_chi2(chi2, chi2_bad=chi2_bad)
    return -np.log(z + eps)

# ---- Config from Julia ----
bounds = $interval
param_order = list($((collect(keys(interval)))))
max_iter = int($max_iter)
file_name = $file_name
fig_name = $fig_name
ref_point = $ref_point

# ---- Seeds for reproducibility ----
seed_base = 12345                    # 전체 고정 시드 베이스
rng = np.random.default_rng(seed_base)
gp_seed = seed_base                  # GPR 하이퍼파라미터 최적화용 시드

# ---- GP model (ARD + normalize_y) wrapped in UnitSpaceGP ----
d = len(param_order)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d),
                                      length_scale_bounds=(1e-5, 1e5),
                                      nu=2.5) \
         + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
GP_model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=10,
    random_state=gp_seed          # << 추가: GPR 내부 최적화 재현성
)
GP = UnitSpaceGP(GP_model, bounds, param_order)  # unit-space wrapper

# ----------------- Random RNG per run -----------------
ss = np.random.SeedSequence()            # entropy from OS
_seed = int(ss.entropy % (2**32))        # reduce to 32-bit for logging/compat
rng = np.random.default_rng(ss)
print(f"[RNG] run seed = {_seed}")
with open(file_name + "_seed.txt", "w") as f:
    f.write(str(_seed) + "\n")

# ---- Restart or init ----
X, y, idx_list = Restart(bounds, file_name, param_order=param_order)
start = int(idx_list[-1, 0]) + 1
chi2s = []
try:
    # Force C engine to avoid regex-sep fallback warning
    df_prev = pd.read_csv(file_name + ".csv", sep=SEP, engine="c")
    if "Chi2" in df_prev.columns: chi2s = df_prev["Chi2"].astype(float).tolist()
    elif "Obj" in df_prev.columns: chi2s = [np.nan]*len(df_prev["Obj"])
except Exception:
    pass

# ---- Initial design ----
if start == 1:
    print("Initial Run")
    n_init = max(1, min(10, 2*d))

    # Build reference vector (provided or midpoints)
    ref_vec = np.zeros((1, d), dtype=float)
    for j, p in enumerate(param_order):
        lo, hi = bounds[p]
        if isinstance(ref_point, dict) and (p in ref_point):
            v = float(ref_point[p])
            # clip into bounds
            if v < lo: v = lo
            if v > hi: v = hi
        else:
            v = 0.5*(lo + hi)  # midpoint if not provided
        ref_vec[0, j] = v

    # Fill remaining random initial points (if any)
    m = max(0, n_init - 1)
    if m > 0:
        Xrand = np.array([[rng.uniform(bounds[p][0], bounds[p][1]) for p in param_order] for _ in range(m)], dtype=float)
        X0 = np.vstack([ref_vec, Xrand])
    else:
        X0 = ref_vec

    # Evaluate initial design
    y0_list, chi20_list = [], []
    for i in range(n_init):
        params = {"ID": i+1}
        for j, p in enumerate(param_order):
            params[p] = float(X0[i, j])
        chi2_i = $Objective(params)
        y_i = chi2_to_y(chi2_i)
        y0_list.append(y_i); chi20_list.append(sanitize_chi2(chi2_i))
    X = X0
    y = np.array(y0_list, dtype=float)
    chi2s = chi20_list
    idx_list = np.arange(1, n_init+1).reshape(-1, 1)
    start = n_init + 1

# ---- Trust-region state (TuRBO-lite) ----
L = 0.8      # side length in unit space
L_min = 0.05 # below this, reset to global
succ, fail = 0, 0
succ_th = 3
fail_th = 3

# ---- Adaptive EI exploration rate ----
xi0, xi_min, decay = 0.05, 0.005, 0.5
xi = xi0
plateau, W = 0, 10   # plateau counter over a window of W

# ---- Main loop ----
for idx in range(start, max_iter + 1):
    print(f"Bayesian Opt Step :: {idx}")

    # Fit GP
    GP.fit(X, y)

    # Trust-region dict
    tr = None if (len(y) < 2) else {'L': L, 'auto_center': True}

    # --- Decide proposal (EI vs forced exploration) ---
    force_explore = False
    if (fail >= fail_th and L <= 0.2) or (plateau >= 2):
        try:
            pi_far = Global_PI(GP, bounds, param_order, X, y,
                               delta=1e-3, n_cand=3000, min_dist=0.15, rng=rng)
            force_explore = (pi_far >= 0.10)
        except Exception:
            force_explore = True

    if force_explore and len(y) >= 5:
        if (idx % 2) == 0:
            x_next = Propose_Thompson(GP, bounds, param_order, X=X,
                                      n_cand=4000, min_dist=0.15, rng=rng)
        else:
            x_next = Propose_MaxStd(GP, bounds, param_order, X=X,
                                    n_cand=4000, min_dist=0.15, rng=rng)
    else:
        x_next = Opt_Acquisition(
            X, y, GP, bounds=bounds,
            explore=xi, n_cand=4096, k_refine=8,
            param_order=param_order, trust_region=tr,
            rng=rng, min_dist=0.10
        )

    # ---- Evaluate objective
    params = {"ID": idx}
    for i, p in enumerate(param_order):
        params[p] = float(x_next[0, i])
    chi2_next = $Objective(params)
    y_next = chi2_to_y(chi2_next)

    # ---- Update datasets
    X = np.vstack([X, x_next])
    y = np.append(y, y_next)
    idx_list = np.vstack([idx_list, [idx]])
    chi2s.append(float(sanitize_chi2(chi2_next)))

    # ---- Success / fail logic
    i_best = int(np.argmax(y[:-1])) if len(y) > 1 else 0
    improved = (y_next > y[i_best] + 1e-6)
    if improved:
        succ += 1; fail = 0
        L = min(1.0, L * 1.5)  # expand region on success
        if succ >= succ_th:
            L = min(1.0, L * 1.2); succ = 0
    else:
        fail += 1; succ = 0
        if fail >= fail_th:
            L *= 0.5; fail = 0
    if L < L_min:
        L = 0.8; succ, fail = 0, 0  # reset to global

    # ---- Adaptive xi update (plateau detection)
    if len(chi2s) >= W + 1:
        best_prev = np.min(chi2s[:-W]); best_now = np.min(chi2s)
        rel = (best_prev - best_now) / (best_prev + 1e-12)
        if rel < 1e-3:    # <0.1% improvement over window
            plateau += 1
        else:
            plateau = max(0, plateau - 1)
    xi = max(xi_min, xi0 * (decay ** plateau))

    # ---- Logging (CSV)
    data = np.hstack([idx_list, X, np.array(chi2s).reshape(-1,1), y.reshape(-1,1)])
    header = ["ID"] + param_order + ["Chi2", "Y"]
    df = pd.DataFrame(data, columns=header)
    df["ID"] = df["ID"].astype(int)
    df.to_csv(file_name + ".csv", sep=SEP, index=False)

    # ---- Plots
    # 1) Chi2 vs iteration
    fig, ax = plt.subplots(layout='constrained')
    iters = np.arange(1, len(chi2s) + 1)
    ax.scatter(iters, chi2s, s=70)
    imin = int(np.argmin(chi2s)) + 1
    ax.scatter(imin, np.min(chi2s), marker='*', s=200)
    ax.set_xlabel('Idx', fontsize=15)
    ax.set_ylabel(r'$\chi^2$', fontsize=15)
    ax.set_title(f"Minimum is Idx = {imin}", fontsize=20)
    ax.grid(True); ax.set_axisbelow(True)
    fig.savefig(fig_name + "_vs_Idx.png"); plt.close(fig)

    # 2) Chi2 vs parameters
    fig, axs = plt.subplots(1, X.shape[1], figsize=(3*X.shape[1], 3), layout='constrained')
    if X.shape[1] == 1:
        axs = [axs]
    for i in range(X.shape[1]):
        axs[i].scatter(X[:, i], chi2s, s=40)
        pbest = X[np.argmin(chi2s), i]
        ybest = np.min(chi2s)
        axs[i].scatter(pbest, ybest, marker='*', s=200)
        axs[i].set_xlabel(param_order[i], fontsize=10)
        axs[i].set_ylabel(r'$\chi^2$', fontsize=10)
        axs[i].set_title(f"Min at {param_order[i]} = {pbest:.5f}", fontsize=10)
        axs[i].grid(True); axs[i].set_axisbelow(True)
    fig.align_labels()
    fig.savefig(fig_name + "_vs_params.png"); plt.close(fig)
    
if len(y) >= 2 and not hasattr(GP.model, "X_train_"):
    GP.fit(X, y)
    
# ---- Posterior sampling uncertainty (ONLY at the very end)
try:
    post_seed = int(rng.integers(1, 2**31-1))   # per-run random seed
    tr_final = {'L': L, 'auto_center': True} if len(y) > 2 else None
    summary = optimal_std_via_sampling(
        GP, bounds, param_order,
        X=X, y=y,
        n_funcs=500,
        n_cand=5000,
        trust_region=None,
        eps=1e-12,
        posterior_seed=post_seed,   # << 고정 시드
        noise_free=True,           # << 노이즈-프리 공분산
        global_scope=True,         # << 전역 스코프 샘플링
    )

    print("[Uncertainty@final] y* std=%.4g  chi2* std=%.4g" % (summary["y_star_std"], summary["chi2_star_std"]))
    with open(file_name + "_uncert.json", "w") as f:
        json.dump({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in summary.items()}, f)
except Exception as e:
    print("[Uncertainty@final] sampling failed:", e)

# ---- Pairwise heatmaps (E[chi2] and PI) at the very end
try:
    ib = int(np.argmin(chi2s))
    x_best = {p: float(X[ib, k]) for k, p in enumerate(param_order)}

    # Expected chi^2 maps (smooth landscape)
    pairwise_heatmap_plot(
        GP, bounds, param_order, x_best,
        pairs=None,        # or e.g. [("J3","J4"), ("J3","Jnnn")]
        grid_n=80,
        mode="Echi2",
        X_hist=X, chi2_hist=chi2s,
        out_prefix=fig_name + "_pair",
        save_data = True, data_format="npz", save_hist = False,
    )

    print("[Pairwise] heatmaps saved with prefix:", fig_name + "_pair")
except Exception as e:
    print("[Pairwise] plotting failed:", e)

# ---- 레벨셋 샘플링 & 저장 ----
chi2_min = float(np.min(chi2s))
out = levelset_region_sampling(
    GP, bounds, param_order,
    chi2_min=chi2_min,
    delta=delta,            # 옵션: 1 대신 다른 값
    n_scan=120_000,       # 전역 스캔 점 수
    n_samples=500,        # 저장할 샘플 수
    q=0.95,               # 보수적: 95% 분위수 기준. 기대값 쓰려면 None
    seed=20250911,        # 재현성
    pad=0.02,
    batch=6000,
    outfile=fig_name + "_levelset_samples.npz",
)
print("[LevelSet] saved:", out["outfile"])
print("[LevelSet] box (lo,hi) per dim:\n", out["box"])
"""
end

