function Fit(Objective, interval, max_iter; file_name = "Bopt_Log", fig_name = "chi2", ref_point = nothing)
    DIR = @__DIR__
    @pyinclude(DIR*"/Bopt.py")
    py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

# ----------------- Config from Julia -----------------
bounds = $interval
param_order = list($((collect(keys(interval)))))
max_iter = int($max_iter)
file_name = $file_name
fig_name = $fig_name
ref_point = $ref_point   # Dict or None

# ----------------- CSV I/O settings -----------------
SEP = "\t"  # real tab (one char)
# When reading, force C engine to avoid regex fallback warnings.

# ----------------- Helpers for safe log-transform -----------------
def sanitize_chi2(chi2, chi2_bad=1e30):
    z = float(chi2)
    if not np.isfinite(z) or z <= 0:
        z = chi2_bad
    return z

def chi2_to_y(chi2, eps=1e-12, chi2_bad=1e30):
    z = sanitize_chi2(chi2, chi2_bad=chi2_bad)
    return -np.log(z + eps)

# ----------------- Random RNG per run -----------------
ss = np.random.SeedSequence()            # entropy from OS
_seed = int(ss.entropy % (2**32))        # reduce to 32-bit for logging/compat
rng = np.random.default_rng(ss)
print(f"[RNG] run seed = {_seed}")
with open(file_name + "_seed.txt", "w") as f:
    f.write(str(_seed) + "\n")

# ----------------- Define GP model (ARD + normalize_y) -----------------
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
    random_state=0,   # hyperparameter optimization reproducibility
)
GP = UnitSpaceGP(GP_model, bounds, param_order)

# ----------------- Restart or initialize -----------------
X, y, idx_list = Restart(bounds, file_name, param_order=param_order)
start = int(idx_list[-1, 0]) + 1
chi2s = []
try:
    df_prev = pd.read_csv(file_name + ".csv", sep=SEP, engine="c")
    if "Chi2" in df_prev.columns:
        chi2s = df_prev["Chi2"].astype(float).tolist()
    elif "Obj" in df_prev.columns:
        chi2s = [np.nan]*len(df_prev["Obj"])
except Exception:
    pass

# ----------------- Initial design with optional reference -----------------
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

# ----------------- Trust-region (optional, simple) -----------------
L = 0.8
def _tr_state():
    if len(y) < 2:
        return None
    ib = int(np.argmax(y))
    from Bopt import to_unit_batch
    center_u = to_unit_batch(bounds, param_order, X[ib:ib+1, :]).reshape(-1)
    return {'L': L, 'center_u': center_u}

# ----------------- BO main loop -----------------
xi = 0.01
for idx in range(start, max_iter + 1):
    print(f"Bayesian Opt Step :: {idx}")
    GP.fit(X, y)

    tr = _tr_state()
    x_next = Opt_Acquisition(
        X, y, GP, bounds=bounds,
        explore=xi, n_cand=4096, k_refine=8,
        param_order=param_order, trust_region=tr, rng=rng
    )

    # Evaluate objective
    params = {"ID": idx}
    for i, p in enumerate(param_order):
        params[p] = float(x_next[0, i])
    chi2_next = $Objective(params)
    y_next = chi2_to_y(chi2_next)

    # Update datasets
    X = np.vstack([X, x_next])
    y = np.append(y, y_next)
    idx_list = np.vstack([idx_list, [idx]])
    chi2s.append(float(sanitize_chi2(chi2_next)))

    # Logging (CSV)
    data = np.hstack([idx_list, X, np.array(chi2s).reshape(-1,1), y.reshape(-1,1)])
    header = ["ID"] + param_order + ["Chi2", "Y"]
    df = pd.DataFrame(data, columns=header)
    df["ID"] = df["ID"].astype(int)
    df.to_csv(file_name + ".csv", sep=SEP, index=False)

    # Plots
    # 1) Chi2 vs iteration
    fig, ax = plt.subplots(layout='constrained')
    iters = np.arange(1, len(chi2s) + 1)
    ax.scatter(iters, chi2s, s=60)
    imin = int(np.argmin(chi2s)) + 1
    ax.scatter(imin, np.min(chi2s), marker='*', s=160)
    ax.set_xlabel('Idx', fontsize=13)
    ax.set_ylabel(r'$\\chi^2$', fontsize=13)
    ax.set_title(f"Minimum is Idx = {imin}", fontsize=16)
    ax.grid(True); ax.set_axisbelow(True)
    fig.savefig(fig_name + "_vs_Idx.png"); plt.close(fig)

    # 2) Chi2 vs parameters
    fig, axs = plt.subplots(1, X.shape[1], figsize=(3*X.shape[1], 3), layout='constrained')
    if X.shape[1] == 1:
        axs = [axs]
    for i in range(X.shape[1]):
        axs[i].scatter(X[:, i], chi2s, s=36)
        pbest = X[np.argmin(chi2s), i]
        ybest = np.min(chi2s)
        axs[i].scatter(pbest, ybest, marker='*', s=160)
        axs[i].set_xlabel(param_order[i], fontsize=10)
        axs[i].set_ylabel(r'$\\chi^2$', fontsize=10)
        axs[i].set_title(f"Min at {param_order[i]} = {pbest:.5f}", fontsize=10)
        axs[i].grid(True); axs[i].set_axisbelow(True)
    fig.align_labels()
    fig.savefig(fig_name + "_vs_params.png"); plt.close(fig)

# ----------------- Final analysis (posterior sampling, one-time) -----------------
# Ensure GP is fit even if no new iterations were performed on resume
if len(y) >= 2 and not hasattr(GP.model, "X_train_"):
    GP.fit(X, y)

try:
    post_seed = int(rng.integers(1, 2**31-1))   # per-run random seed
    summary = optimal_std_via_sampling(
        GP, bounds, param_order,
        X=X, y=y,
        n_funcs=200, n_cand=2000,
        trust_region=None,    # global
        eps=1e-12,
        posterior_seed=post_seed,
        noise_free=True,
        global_scope=True,
    )
    print("[Uncertainty@final] y* std=%.4g  chi2* std=%.4g" % (summary["y_star_std"], summary["chi2_star_std"]))
    with open(file_name + "_uncert.json", "w") as f:
        json.dump({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in summary.items()}, f, indent=2)
except Exception as e:
    print("[Uncertainty@final] sampling failed:", e)
"""
end
