# Propose next point using multiple random restarts
function propose_next(gp, f_max; n_restarts=20, acq=expected_improvement, tuningPar = 0.10, nq=20, dmp)
    d = gp.dim
    best_acq_val = -Inf # Start at -Inf because we're looking for a maximum
    best_x = zeros(d)

    function objective_to_minimize(x)
        val = 0.0
        if acq == expected_improvement
            val = acq(gp, x, f_max; ξ=tuningPar)
        elseif acq == upper_confidence_bound
            val = acq(gp, x; κ=tuningPar)
        elseif acq == knowledgeGradientHybrid
            val = acq(gp, x; n_z = nq)
        elseif acq == knowledgeGradientDiscrete
            val = acq(gp, x, dmp)
        elseif acq == posterior_variance            
            val = acq(gp, x)
        else
            error("Unknown acquisition function: $acq")
        end
        return -val
    end

    for _ in 1:n_restarts
        x0 = rand(Uniform(-1., 1.), d)


        if acq == knowledgeGradientHybrid || acq == knowledgeGradientDiscrete
            # Use a robust, derivative-free optimizer for KG functions
            res = optimize(objective_to_minimize,
                           -1, 1, x0,
                           Fminbox(NelderMead()))
        else
            res = optimize(objective_to_minimize,
                       -1. * ones(d), 1. * ones(d), x0,
                       Fminbox(LBFGS()); autodiff = :forward)
        end
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
function posteriorMax(gp; n_starts=20)
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
        # --- NEW: Standardize y-values at the start of the loop ---
        μ_y = mean(y)
        # Use max to set a minimum standard deviation (jitter) in one line
        σ_y = max(std(y), 1e-6)
        y_scaled = (y .- μ_y) ./ σ_y
        
        # Train the GP on the STANDARDIZED y-values
        gp = GP(Xscaled', y_scaled, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=100))
    
        # Get the current best STANDARDIZED y-value
        if optimizationSettings.acq == knowledgeGradientHybrid || optimizationSettings.acq == knowledgeGradientDiscrete
            # For KG, we need the maximum posterior mean over the whole search area, also nonvisited points.
            f_max_scaled = posteriorMax(gp, n_starts=10)
        else
            # For EI and UCB, we can use the posterior maximum at visited x-values
            f_max_scaled = posteriorMaxObs(gp, Xscaled')
        end
        #f_max_scaled = maximum(y_scaled) # This option would be for noisefree functions.

        # Propose the next point using the GP trained on scaled data
        x_next_scaled = propose_next(gp, f_max_scaled,
                                     n_restarts=optimizationSettings.n_restarts,
                                     acq=optimizationSettings.acq,
                                     tuningPar=optimizationSettings.tuningPar,
                                     nq=optimizationSettings.nq,
                                     dmp=optimizationSettings.dmp)
        
        # Rescale back to original bounds to evaluate the true function
        x_next_original = inv_rescale(x_next_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
        
        # Handle 1D vs multi-D function calls
        y_next = 0.0
        if modelSettings.xdim == 1
            y_next = f(x_next_original[1])
        else
            y_next = f(x_next_original)
        end

        # Add the new ORIGINAL y-value to the dataset
        X = vcat(X, x_next_original')
        Xscaled = vcat(Xscaled, x_next_scaled')
        y = vcat(y, y_next)
    
        println("Iter $i: x = $(round.(x_next_original, digits=3)), y = $(round(y_next, digits=3))")
    end

    # --- Final GP model on all data ---
    # Standardize final y-data before fitting the final model
    μ_y_final = mean(y)
    # Use max to set a minimum standard deviation (jitter) in one line
    σ_y_final = max(std(y), 1e-6)
    y_scaled_final = (y .- μ_y_final) ./ σ_y_final
    
    gp = GP(Xscaled', y_scaled_final, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=500))

    # (1) Global posterior mean maximum (can be unobserved)
    final_posterior_max_result = posteriorMax(gp; n_starts=40)
    objectMaximizer_scaled = final_posterior_max_result.X_max
    objectMaximizer = inv_rescale(objectMaximizer_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]

    # (2) Maximum over points with the highest posterior mean among observed points
    μ_scaled, _ = predict_f(gp, Xscaled')
    maxIdx = argmax(μ_scaled)
    postMaxObserved_scaled = Xscaled[maxIdx, :]
    postMaxObserved = inv_rescale(postMaxObserved_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
    
    # --- NEW: Rescale the final predicted mean back to the original y-scale ---
    postMaxObservedY_scaled = μ_scaled[maxIdx]
    postMaxObservedY = postMaxObservedY_scaled * σ_y_final + μ_y_final
    
    return gp, X, y, objectMaximizer, postMaxObserved, postMaxObservedY
end
