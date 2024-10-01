

function Fit(Objective, interval, max_iter; file_name = "Bopt_Log", fig_name = "chi2")
    DIR = @__DIR__
    @pyinclude(DIR*"/Bopt.py")
    py"""
    bounds = $interval
    max_iter = $max_iter

    $SafetyChecks(bounds)

    # Define GP process. 
    print("Setting up Kernel")
    kernel = Matern(
        length_scale=1.0, 
        length_scale_bounds=(1e-03, 1e3), 
        nu=2.5
    )
    GP_model = GaussianProcessRegressor(
        kernel=kernel, 
        normalize_y=False, 
        optimizer='fmin_l_bfgs_b',  
        n_restarts_optimizer=30  
    )
    
    X, y, idx_list = Restart(bounds, $file_name)
    start = int(idx_list[-1][0]) + 1

    if start == 1:
        print("Initial Run")
        idx_list = np.array([start])
        X = np.vstack([np.random.uniform(low=bounds[p][0], high=bounds[p][1], size=50-len(bounds)) for p in bounds]).T
        X = X.round(decimals = 5, out = None)
        params ={"ID":1}
        for p,i in zip(bounds, range(len(bounds))):
            params[p] = X[0,i]
        y = np.array([$Objective(params)])
        start = start + 1

    for idx in range(start, max_iter):
        print("Bayesian Opt Step :: %i"%idx)
        # Fit GP
        GP_model.fit(X, y)
    
        # Enforce alteration of high exploration and exploitation
        if idx % 4 == 0 : 
            exploreRate = 0.25 
        else : 
            exploreRate = 0.0
        
        # Evaluate Objective
        x_next = Opt_Acquisition(X, GP_model, bounds=bounds, explore=exploreRate)
        x_next = x_next.round(decimals=5, out=None)
        params ={"ID":idx}
        for p,i in zip(bounds, range(len(bounds))):
            params[p] = x_next[0,i]
        X = np.vstack([X, x_next])
        y = np.vstack([y, $Objective(params) ])
        idx_list = np.vstack([idx_list, idx])
        
        best_so_far = np.argmax(y)
        print("Best Loss ", np.max(y), '\n', "Params = ", X[best_so_far])
        
        # Update Log
        data = np.hstack((idx_list, X, y))
        header = [p for p in bounds]
        header.append("Obj")
        header.insert(0, "ID")
        df = pd.DataFrame(data, columns=header,)
        df["ID"] = df["ID"].astype(int)
        df.to_csv($file_name+".csv", sep='\t')
    """ 
    df = CSV.read(file_name*".csv", DataFrame)
    
    idx = df[!, 2];
    Params = df[!, 3:end-1];
    chi2 = -1 * df[!, end];
    
    f = Figure();
    ax = Axis(f[1, 1], ylabel = "χ² (a.u)", xlabel = "Idx", xlabelsize = 18, ylabelsize = 18)
    scatter!(ax, idx, chi2)
    scatter!(ax, findmin(chi2)[2], findmin(chi2)[1], marker = :star5, markersize = 20)
    ax.title = "Minimum ID = $(findmin(chi2)[2])"
    xlims!(0, maximum(idx))

    save(fig_name*"_idx.png", f)
    f = Figure(size=(1400, 400));
    for i in 1:size(Params)[2]
        ax = Axis(f[1, i], title = "Best value = $(Params[findmin(chi2)[2], i])")
        scatter!(ax, Params[:, i], chi2)
        scatter!(ax, Params[findmin(chi2)[2], i], findmin(chi2)[1], marker = :star5, markersize = 20)
        ylims!(minimum(chi2) * 0.9, minimum(chi2) * 2.0)
        xlims!(minimum(Params[:,i])*0.9, maximum(Params[:,i])*3.0)
    end
    save(fig_name*"_params.png", f)
    
end
