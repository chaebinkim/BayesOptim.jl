

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
        X = np.vstack([np.random.uniform(low=bounds[p][0], high=bounds[p][1]) for p in bounds]).T
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

        fig, ax = plt.subplots(layout = 'constrained')
        ax.scatter(data[:,0], -data[:,-1], s = 70)
        ax.scatter(np.argmin(-data[:,-1])+1, np.min(-data[:,-1]), marker = '*', s = 200)
        ax.set_xlabel('Idx', fontsize = 15)
        ax.set_ylabel(r'$\chi^2$', fontsize = 15)
        ax.set_title('Minimum is Idx = {}'.format(np.argmin(-data[:,-1])+1), fontsize= 20)
        ax.grid(True)
        ax.set_axisbelow(True)
        fig.savefig($fig_name+"_vs_Idx.png")
        
        
        fig, axs = plt.subplots(1, X.shape[1], figsize = (3*X.shape[1], 3), layout = 'constrained')
        for i in range(0, X.shape[1]):
            axs[i].scatter(data[:,i+1], -data[:,-1])
            axs[i].scatter(data[np.argmin(-data[:,-1]), i+1], np.min(-data[:,-1]), marker = '*', s = 200)
            axs[i].set_xlabel(header[i+1], fontsize = 10)
            axs[i].set_ylabel(r'$\chi^2$', fontsize = 10)
            axs[i].set_title('Min at {} = {}'.format(header[i+1], data[np.argmin(-data[:,-1]), i].round(decimals = 5, out = None)), fontsize= 10)
            axs[i].grid(True)
            axs[i].set_axisbelow(True)
        
        fig.align_labels()

        fig.savefig($fig_name+"_vs_params.png")

    # best_so_far = np.argmax(y)
    # Params_best = X[best_so_far]
    # md, std = surrogate(GP_model, X)
    # print(md)
    # print(std)
    
    """ 
    
end
