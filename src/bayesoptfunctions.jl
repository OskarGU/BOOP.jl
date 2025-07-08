# Propose next point using multiple random restarts
function propose_next(gp, ybest; bounds=(-1.0, 1.0), d=1, n_restarts=20, acq=expected_improvement, tuningPar = 0.10)
    best_val = Inf
    best_x = zeros(d)
    function objective_function(x)
        if acq == expected_improvement
            return acq(gp, x, ybest; ξ=tuningPar)
        elseif acq == upper_confidence_bound
            return acq(gp, x; κ=tuningPar)
        else
            error("Unknown acquisition function: $acq")
        end
    end

    for _ in 1:n_restarts
        x0 = rand(Uniform(bounds[1], bounds[2]), d)
        res = optimize(objective_function,
                       bounds[1] * ones(d), bounds[2] * ones(d), x0,
                       Fminbox(LBFGS()); autodiff = :forward)
        if res.minimum < best_val
            best_val = res.minimum
            best_x = res.minimizer
        end
    end

    return best_x
end

# Function to find the minimum of the posterior mean even in unvisited points.
function posterior_min(gp, bounds, d)
    acq_mean(x) = predict_f(gp,x)[1][1]
    x0 = rand(d)
    res = optimize(acq_mean,
                   bounds[1] * ones(d), bounds[2] * ones(d),
                   x0, Fminbox(LBFGS()); autodiff = :forward)
    return res.minimum
end

# Function to find the minimum of the posterior mean only in observed points.
function posteriorMinObs(gp, X)
    μ, _ = predict_f(gp, X')
    return minimum(μ)
end


# Rescaling functions used for GP to ensure inputs are in a suitable range. (Easier to set lengthscales and mor robust optimization).
rescale(x, lo, hi) = 2 * (x .- lo) ./ (hi - lo) .- 1
inv_rescale(x, lo, hi) = 0.5 * (x .+ 1) * (hi - lo) .+ lo


####################
# Main BO function:
# Bayesian Optimization loop
function BO(f, modelSettings, optimizationSettings, warmStart)
    X, y = warmStart
    Xscaled= rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2])
    for i in 1:optimizationSettings.nIter

        gp = GP(Xscaled', y[:], modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=100)) 
    
    
        # Get current best y
        ybest = minimum(y)
        #ybest = posterior_min(gp, bounds, d)
        #ybest = posteriorMinObs(gp, Xscaled)
        # Propose next point
        x_next = propose_next(gp, ybest, d=modelSettings.xdim, n_restarts=20, acq=optimizationSettings.acq, tuningPar=optimizationSettings.tuningPar)
                 #propose_next(gp, ybest; bounds=(-1.0, 1.0), d=1, n_restarts=20, acq=expected_improvement, ξ = 0.10)
        xNextRescaled = inv_rescale(x_next, modelSettings.xBounds[1], modelSettings.xBounds[2])  # Rescale back to original bounds
        if(modelSettings.xdim ==1) y_next = f(xNextRescaled[1])
          else y_next = f(xNextRescaled)
        end
    
        # Add to dataset
        X = vcat(X, xNextRescaled')
        Xscaled = vcat(Xscaled, x_next')
        y = vcat(y, y_next)
    
        println("Iter $i: x = $(round.(xNextRescaled, digits=3)), y = $(round(y_next, digits=3))")
    end
    gp = GP(Xscaled', y[:], modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=500)) 
    return gp, X, y
end
