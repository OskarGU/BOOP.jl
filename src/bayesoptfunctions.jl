# Propose next point using multiple random restarts
function propose_next(gp, f_max; n_restarts=20, acq=expected_improvement, tuningPar = 0.10)
    d = gp.dim
    best_acq_val = -Inf # Start at -Inf because we're looking for a maximum
    best_x = zeros(d)

    function objective_to_minimize(x)
        val = 0.0
        if acq == expected_improvement
            val = acq(gp, x, f_max; ξ=tuningPar)
        elseif acq == upper_confidence_bound
            val = acq(gp, x; κ=tuningPar)
        else
            error("Unknown acquisition function: $acq")
        end
        return -val
    end

    for _ in 1:n_restarts
        x0 = rand(Uniform(-1., 1.), d)
        res = optimize(objective_to_minimize,
                       -1. * ones(d), 1. * ones(d), x0,
                       Fminbox(LBFGS()); autodiff = :forward)

        # The acquisition score is the negative of the optimizer's minimum
        current_acq_val = -res.minimum
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = res.minimizer
        end
    end

    return best_x
end


# Function to find the maximum of the posterior mean even in unvisited points.
function posterior_max(gp; n_starts=20)
    d = gp.dim
    acq_mean(x) = predict_f(gp, reshape(x, d, 1))[1][1]
    best_val, best_x = multi_start_maximize(acq_mean, -1. * ones(d), 1. * ones(d), n_starts=n_starts)
    return (fX_max = best_val, X_max = best_x)
end

# Function to find the maximum of the posterior mean over observed points.
function posteriorMaxObs(gp, X_scaled)
    μ, _ = predict_f(gp, X_scaled)
    return maximum(μ)
end


# Rescaling functions used for GP to ensure inputs are in a suitable range. (Easier to set lengthscales and mor robust optimization).
#rescale(x, lo, hi) = 2 * (x .- lo) ./ (hi - lo) .- 1¨
function rescale(X, lo, hi)
    # X is n×d matrix, lo and hi are length-d vectors
    return 2 .* (X .- lo') ./ (hi' .- lo') .- 1
end

inv_rescale(X_scaled, lo, hi) = ((X_scaled .+ 1) ./ 2) .* (hi' .- lo') .+ lo'
#inv_rescale(x, lo, hi) = 0.5 * (x .+ 1) .* (hi - lo) .+ lo




# Main BO function
function BO(f, modelSettings, optimizationSettings, warmStart)
    X, y = warmStart
    Xscaled = rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2])

    for i in 1:optimizationSettings.nIter
        # The GP models the original y values directly for a maximization problem
        gp = GP(Xscaled', y[:], modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=100))
    
        # Get current best y by finding the MAXIMUM of observed values.
        f_max = maximum(y)

        # Propose next point by MAXIMIZING the acquisition function
        x_next_scaled = propose_next(gp, f_max,
                                     n_restarts=optimizationSettings.n_restarts,
                                     acq=optimizationSettings.acq,
                                     tuningPar=optimizationSettings.tuningPar) 
        
        # Rescale back to original bounds to evaluate the true function
        x_next_original = inv_rescale(x_next_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
        
        # Handle 1D vs multi-D function calls
        y_next = 0.0
        if modelSettings.xdim == 1
            y_next = f(x_next_original[1])
        else
            y_next = f(x_next_original)
        end

        # Add new data to dataset
        X = vcat(X, x_next_original')
        Xscaled = vcat(Xscaled, x_next_scaled')
        y = vcat(y, y_next)
    
        println("Iter $i: x = $(round.(x_next_original, digits=3)), y = $(round(y_next, digits=3))")
    end

    # Final GP model on all data
    gp = GP(Xscaled', y[:], modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=500))

    # --- Final Results, now finding MAXIMA ---

    # (1) Global posterior mean maximum (can be unobserved)
    final_posterior_max_result = posterior_max(gp; n_starts=40)
    objectMaximizer_scaled = final_posterior_max_result.X_max
    objectMaximizer = inv_rescale(objectMaximizer_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]

    # (2) Maximum over points with the highest posterior mean among observed points
    μ, _ = predict_f(gp, Xscaled')
    maxIdx = argmax(μ)
    postMaxObserved_scaled = Xscaled[maxIdx, :]
    postMaxObserved = inv_rescale(postMaxObserved_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
    postMaxObservedY = μ[maxIdx]
    
    return gp, X, y, objectMaximizer, postMaxObserved, postMaxObservedY
end

####################
# Main BO function:
# Bayesian Optimization loop
#function BO(f, modelSettings, optimizationSettings, warmStart)
#    X, y = warmStart
#    Xscaled= rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2])
#    for i in 1:optimizationSettings.nIter
#
#        gp = GP(Xscaled', y[:], modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
#        optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=100)) 
#    
#    
#        # Get current best y
#        #ybest = minimum(y)
#        #ybest = posterior_min(gp, bounds, d).fX_min
#        ybest = posteriorMinObs(gp, Xscaled)
#        # Propose next point
#        x_next = propose_next(gp, ybest, n_restarts=20, acq=optimizationSettings.acq, tuningPar=optimizationSettings.tuningPar)
#                 #propose_next(gp, ybest; n_restarts=20, acq=expected_improvement, ξ = 0.10)
#        xNextRescaled = inv_rescale(x_next[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]  # Rescale back to original bounds
#        if(modelSettings.xdim ==1) 
#        y_next = f(xNextRescaled[1])
#          else y_next = f(xNextRescaled)
#        end
#    
#        # Add to dataset
#        X = vcat(X, xNextRescaled')
#        Xscaled = vcat(Xscaled, x_next')
#        y = vcat(y, y_next)
#    
#        println("Iter $i: x = $(round.(xNextRescaled, digits=3)), y = $(round(y_next, digits=3))")
#    end
#    gp = GP(Xscaled', y[:], modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
#    optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=500)) 
#
#    # (1) Global posterior mean minimum (can be unobserved)
#    objectMinimizer_scaled = posterior_min(gp).X_min
#    objectMinimizer = inv_rescale(objectMinimizer_scaled, modelSettings.xBounds[1], modelSettings.xBounds[2])
#    
#    # (2) Minimum over observed X points
#    μ, _ = predict_f(gp, Xscaled')
#    minIdx = argmin(μ)
#    postMinObserved_scaled = Xscaled[minIdx, :]
#    postMinObserved = inv_rescale(postMinObserved_scaled, modelSettings.xBounds[1], modelSettings.xBounds[2])
#    postMinObservedY = μ[minIdx]
#    return gp, X, y, objectMinimizer, postMinObserved, postMinObservedY 
#end
#