function Fit(Objective, interval, max_iter; file_name = "Bopt_Log", fig_name = "chi2", ref_point = nothing, delta = 1.0, plateau_rel = 1e-4, pi_threshold = 0.1)
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
delta = $delta
plateau_rel = float($plateau_rel)
pi_threshold = float($pi_threshold)

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

# ---- Initial design with Latin Hypercube Sampling ----
if start == 1:
    print("Initial Run with Latin Hypercube Sampling")
    n_init = max(2, min(15, 3*d))  # Increased initial samples for better coverage

    # Use LHS for better space-filling design
    try:
        X_lhs = latin_hypercube_sampling(bounds, param_order, n_samples=n_init-1, seed=_seed)
    except Exception as e:
        print(f"LHS failed: {e}, using random sampling")
        X_lhs = np.array([[rng.uniform(bounds[p][0], bounds[p][1]) for p in param_order] 
                         for _ in range(n_init-1)], dtype=float)

    # Build reference vector
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

    X0 = np.vstack([ref_vec, X_lhs])

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

# ---- Trust-region state (more aggressive) ----
L = 0.9      # Start larger for expensive objectives
L_min = 0.05
succ, fail = 0, 0
succ_th = 2  # Faster expansion trigger
fail_th = 2  # Slower contraction trigger

# ---- Adaptive EI exploration rate (more conservative initially) ----
xi0, xi_min, decay = 0.02, 0.001, 0.6  # Lower initial xi for expensive objectives
xi = xi0
plateau, W = 0, 12   # Longer window for plateau detection

# ---- Main loop ----
for idx in range(start, max_iter + 1):
    print(f"Bayesian Opt Step :: {idx}")

    # Fit GP
    GP.fit(X, y)

    # Trust-region dict
    tr = None if (len(y) < 2) else {'L': L, 'auto_center': True}

    # --- Decide proposal strategy ---
    force_explore = False
    if (fail >= fail_th and L <= 0.2) or (plateau >= 3):
        try:
            pi_far = Global_PI(GP, bounds, param_order, X, y,
                               delta=1e-3, n_cand=3000, min_dist=0.15, rng=rng)
            force_explore = (pi_far >= pi_threshold)
        except Exception:
            force_explore = True

    # Propose next point
    if force_explore and len(y) >= 5:
        if (idx % 2) == 0:
            x_next = Propose_Thompson(GP, bounds, param_order, X=X,
                                      n_cand=5000, min_dist=0.15, rng=rng)
        else:
            x_next = Propose_MaxStd(GP, bounds, param_order, X=X,
                                    n_cand=5000, min_dist=0.15, rng=rng)
        x_next = x_next.reshape(1, -1)
    else:
        x_next = Opt_Acquisition(
            X, y, GP, bounds=bounds,
            explore=xi, n_cand=5120, k_refine=10,  # More candidates & refinements
            param_order=param_order, trust_region=tr,
            rng=rng, min_dist=0.12
        )
        x_next = x_next.reshape(1, -1)

    # ---- Evaluate objective ----
    params = {"ID": idx}
    for j, p in enumerate(param_order):
        params[p] = float(x_next[0, j])
    chi2_val = sanitize_chi2($Objective(params))
    y_val = chi2_to_y(chi2_val)

    # ---- Update datasets
    X = np.vstack([X, x_next])
    y = np.append(y, y_val)
    idx_list = np.vstack([idx_list, np.array([[idx]], dtype=int)])
    chi2s.append(float(chi2_val))

    # ---- Success / fail logic (more aggressive)
    i_best = int(np.argmax(y[:-1])) if len(y) > 1 else 0
    improved = (y_val > y[i_best] + 1e-6)
    
    L, succ, fail = update_trust_region_aggressive(L, succ, fail, improved, 
                                                   succ_th=succ_th, fail_th=fail_th)

    # ---- Adaptive xi update
    if len(chi2s) >= W + 1:
        best_prev = np.min(chi2s[:-W]); best_now = np.min(chi2s)
        rel = (best_prev - best_now) / (best_prev + 1e-12)
        if rel < plateau_rel:
            plateau += 1
        else:
            plateau = max(0, plateau - 1)
    xi = max(xi_min, xi0 * (decay ** plateau))

    # ---- Logging & diagnostics
    log_progress(file_name, fig_name, param_order, idx_list, X, chi2s, y, sep=SEP)
    
if len(y) >= 2 and not hasattr(GP.model, "X_train_"):
    GP.fit(X, y)
    
run_postprocessing(
    GP, bounds, param_order, X, y, chi2s, fig_name,
    rng=rng,
    uncertainty_kwargs={"n_funcs": 500, "n_cand": 5000},
    heatmap_kwargs={"pairs": None},
    levelset_kwargs={
        "delta": delta,
        "n_samples": 10000,
        "q": None,
        "seed": _seed,
        "posterior": True,
        "n_funcs": 400,
        "posterior_seed": _seed,
        "save_all_draws": False,
    },
)

"""
end


