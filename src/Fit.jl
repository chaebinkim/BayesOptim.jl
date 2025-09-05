function Fit(Objective, interval, max_iter; file_name = "Bopt_Log", fig_name = "chi2")
    DIR = @__DIR__
    @pyinclude(DIR*"/Bopt.py")
    py"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    # Julia -> Python bridges
    bounds = $interval
    param_order = list($((collect(keys(interval)))))
    max_iter = int($max_iter)

    $SafetyChecks(bounds)
    
    # --- Gaussian Process model (ARD + normalize_y for stability)
    d = len(param_order)
    kernel = Matern(length_scale=np.ones(d), length_scale_bounds=(1e-3, 1e3), nu=2.5)
    GP_model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=10
    )

    # --- Restart from CSV if present
    X, y, idx_list = Restart(bounds, $file_name, param_order=param_order)
    start = int(idx_list[-1, 0]) + 1

    rng = np.random.default_rng(12345)  # set a fixed seed for reproducibility

    # --- Initial point if first run
    if start == 1:
        print("Initial Run")
        x0 = np.array([rng.uniform(bounds[p][0], bounds[p][1]) for p in param_order])[None, :]
        params = {"ID": 1}
        for i, p in enumerate(param_order):
            params[p] = float(x0[0, i])
        y0 = $Objective(params)
        X = x0
        y = np.array([y0], dtype=float)
        idx_list = np.array([[1]], dtype=int)
        start += 1

    # --- Main BO Loop
    # Use a small constant exploration (xi) for EI; you may tune this schedule.
    xi = 0.01

    for idx in range(start, max_iter + 1):
        print(f"Bayesian Opt Step :: {idx}")
        # Fit GP on current data
        GP_model.fit(X, y)

        # Propose next point
        x_next = Opt_Acquisition(
            X, y, GP_model,
            bounds=bounds,
            explore=xi,
            n_cand=4096,
            k_refine=8,
            param_order=param_order,
            rng=rng
        )

        # Evaluate objective at the proposed point
        params = {"ID": idx}
        for i, p in enumerate(param_order):
            params[p] = float(x_next[0, i])
        y_next = $Objective(params)

        # Update datasets
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        idx_list = np.vstack([idx_list, [idx]])

        best_idx = int(np.argmax(y))
        print("Best Loss", y[best_idx], "\n", "Params = ", X[best_idx])

        # --- Logging to CSV
        data = np.hstack([idx_list, X, y.reshape(-1, 1)])
        header = ["ID"] + param_order + ["Obj"]
        df = pd.DataFrame(data, columns=header)
        df["ID"] = df["ID"].astype(int)
        df.to_csv($file_name + ".csv", sep='\t', index=False)

        # --- Plots
        # 1) Objective vs iteration index
        fig, ax = plt.subplots(layout='constrained')
        iters = np.arange(1, len(y) + 1)
        ax.scatter(iters, -y, s=70)
        imin = int(np.argmin(-y)) + 1  # 1-based index for display
        ax.scatter(imin, np.min(-y), marker='*', s=200)
        ax.set_xlabel('Idx', fontsize=15)
        ax.set_ylabel(r'$\chi^2$', fontsize=15)
        ax.set_title(f"Minimum is Idx = {imin}", fontsize=20)
        ax.grid(True)
        ax.set_axisbelow(True)
        fig.savefig($fig_name + "_vs_Idx.png")
        plt.close(fig)

        # 2) Objective vs parameters
        fig, axs = plt.subplots(1, X.shape[1], figsize=(3*X.shape[1], 3), layout='constrained')
        if X.shape[1] == 1:
            axs = [axs]
        data_np = data.astype(float)
        for i in range(X.shape[1]):
            axs[i].scatter(data_np[:, 1 + i], -y, s=40)
            pbest = data_np[np.argmin(-y), 1 + i]
            ybest = np.min(-y)
            axs[i].scatter(pbest, ybest, marker='*', s=200)
            axs[i].set_xlabel(header[1 + i], fontsize=10)
            axs[i].set_ylabel(r'$\chi^2$', fontsize=10)
            axs[i].set_title(f"Min at {header[1 + i]} = {pbest:.5f}", fontsize=10)
            axs[i].grid(True)
            axs[i].set_axisbelow(True)
        fig.align_labels()
        fig.savefig($fig_name + "_vs_params.png")
        plt.close(fig)
    """
end
