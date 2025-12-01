
# Helper function that returns the objective function to be minimized. Gives nice dispatch in next function instead of ugly if.
_get_objective(gp, f_max, config::EIConfig) = x -> -expected_improvement(gp, x, f_max; ξ=config.ξ)
_get_objective(gp, f_max, config::UCBConfig) = x -> -upper_confidence_bound(gp, x; κ=config.κ)
_get_objective(gp, f_max, config::KGHConfig) = x -> -knowledgeGradientHybrid(gp, x; n_z=config.n_z)
_get_objective(gp, f_max, config::KGDConfig) = x -> -knowledgeGradientDiscrete(gp, x, config.domain_points)
_get_objective(gp, f_max, config::PosteriorVarianceConfig) = x -> -posterior_variance(gp, x)
# Experiment med kvadratur.
_get_objective(gp, f_max, config::KGQConfig) = x -> -knowledgeGradientQuadrature(gp, x; n_z=config.n_z, alpha=config.alpha, n_starts=config.n_starts)


"""
    propose_next(gp::GPE, f_max; n_restarts::Int, acq_config::AcquisitionConfig)

Optimizes the acquisition function to find the best next point to sample.
The optimization is performed in the scaled `[-1, 1]^d` space.

# Arguments
- `gp::GPE`: The current trained Gaussian Process model.
- `f_max`: The current best observed value (or posterior mean max), used by some acquisition functions like EI.
- `n_restarts::Int`: The number of restarts for the multi-start optimization of the acquisition function.
- `acq_config::AcquisitionConfig`: A struct holding the configuration for the chosen acquisition function (e.g., `EIConfig`, `UCBConfig`, `KGQConfig`).

# Returns
- `Vector{Float64}`: The coordinates of the next best point to sample, in the **scaled `[-1, 1]^d` space**.
"""
function propose_next(gp, f_max; n_restarts::Int, acq_config::AcquisitionConfig)
    d = gp.dim
    best_acq_val = -Inf
    best_x = zeros(d)

    # Dispatch to the correct helper to get the objective function
    objective_to_minimize = _get_objective(gp, f_max, acq_config)
    #starts =  [-ones(d) .+ (ones(d) .- -ones(d)) .* ((i .+ 0.5) ./ n_restarts) for i in 0:(n_restarts - 1)]
    starts = [2 .* rand(d) .- 1 for _ in 1:n_restarts]
    for i in 1:n_restarts
        x0 =  starts[i]#rand(Uniform(-1., 1.), d)

        # Use the type to select the optimizer
        res = if acq_config isa KnowledgeGradientConfig # Use the abstract type
            optimize(objective_to_minimize, -1.0, 1.0, x0, Fminbox(NelderMead()))
        else
            optimize(objective_to_minimize, 
         -1.0 * ones(d), 
         1.0 * ones(d), 
         x0, 
         Fminbox(LBFGS()), 
         # Add options here as the last positional argument
         Optim.Options(time_limit = 0.5, show_trace = false); 
         autodiff = :forward)
        end

        current_acq_val = -Optim.minimum(res)
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = Optim.minimizer(res)
        end
    end
    return best_x
end


# Utan Autodiff (NelderMead) för att undvika Dual-number krascher.
function propose_next(gp::GPE{X,Y,M,K,P}, f_max; n_restarts::Int, acq_config::BOOP.AcquisitionConfig) where {X, Y, M, P, K<:GarridoMerchanKernel}
    # 1. Hämta info
    disc_dims = gp.kernel.integer_dims
    disc_ranges = gp.kernel.integer_ranges
    d = gp.dim
    cont_dims = setdiff(1:d, disc_dims)

    # 2. Skapa kombinationer
    discrete_combinations = vec(collect(Iterators.product(disc_ranges...)))
    full_objective = _get_objective(gp, f_max, acq_config)
    
    best_acq_val = -Inf
    best_x_full = zeros(d)

    # Slumpa startpunkter (med marginal från kanten för säkerhets skull)
    n_cont = length(cont_dims)
    random_starts = [2 .* rand(n_cont) .- 1 for _ in 1:n_restarts]

    # 3. Loop
    for d_vals in discrete_combinations

        # Wrapper
        function sub_objective(x_cont)
            # --- FIX 1: CLAMP INPUT ---
            # Tvinga in värdet i [-1, 1] innan vi skickar det till GP:n.
            # Detta skyddar mot att NelderMead testar 1.0000001
            x_safe = clamp.(x_cont, -1.0, 1.0)
            T = eltype(x_safe)
            
            x_full = zeros(T,d) 
            if !isempty(cont_dims)
                x_full[cont_dims] = x_safe
            end
            x_full[disc_dims] .= d_vals
            return full_objective(x_full)
        end

        if !isempty(cont_dims)
            for x0 in random_starts
                # Optimera med NelderMead (Robust, inga gradienter)
                #res = optimize(sub_objective, -1.0 * ones(n_cont), 1.0 * ones(n_cont), x0, Fminbox(NelderMead()))
                res = optimize(sub_objective, -1.0 * ones(n_cont), 1.0 * ones(n_cont), x0, Fminbox(LBFGS()), 
               Optim.Options(time_limit=0.5); 
               autodiff = :forward)

                curr_val = -Optim.minimum(res)
                
                if curr_val > best_acq_val
                    best_acq_val = curr_val
                    
                    # --- FIX 2: CLAMP OUTPUT ---
                    # Ibland returnerar optimeraren 1.00000002. Kläm åt det.
                    raw_minimizer = Optim.minimizer(res)
                    best_x_full[cont_dims] = clamp.(raw_minimizer, -1.0, 1.0)
                    
                    best_x_full[disc_dims] .= d_vals
                end
            end
        else
            # Endast diskreta
            val = -sub_objective(Float64[])
            if val > best_acq_val
                best_acq_val = val
                best_x_full[disc_dims] .= d_vals
            end
        end
    end

    return best_x_full
end


"""
    posteriorMax(gp; n_starts=20)

Finds the global maximum of the GP's posterior mean over the entire continuous,
scaled domain `[-1, 1]^d` using multi-start optimization. This maximum can
occur at a previously unobserved point.

# Arguments
- `gp`: The trained Gaussian Process model.
- `n_starts::Int`: The number of restarts for the optimization.

# Returns
- `NamedTuple{(:fX_max, :X_max)}`: A named tuple containing the maximum value
  of the posterior mean (`fX_max`) and its location (`X_max`) in the scaled space.
"""
function posteriorMax(gp; n_starts=30)
    d = gp.dim
    acq_mean(x) = predict_f(gp, reshape(x, d, 1))[1][1]
    best_val, best_x = multi_start_maximize(acq_mean, -1. * ones(d), 1. * ones(d), n_starts=n_starts)
    return (fX_max = best_val, X_max = best_x)
end



#### For dispatcing to match G-M:
function posteriorMax(gp::GPE{X,Y,M,K,P}; n_starts=30) where {X,Y,M,P,K<:BOOP.GarridoMerchanKernel}
    # Reuse logic from propose_next but optimize post mean directly
    disc_dims = gp.kernel.integer_dims
    disc_ranges = gp.kernel.integer_ranges
    d = gp.dim
    cont_dims = setdiff(1:d, disc_dims)
    
    # Hämta posterior mean funktionen
    # Notera: predict_f returnerar (mean, variance), vi vill ha mean[1]
    acq_mean(x) = predict_f(gp, reshape(x, d, 1))[1][1]

    best_val = -Inf
    best_x_full = zeros(d)
    
    discrete_combinations = vec(collect(Iterators.product(disc_ranges...)))

    for d_vals in discrete_combinations
        
        # Sub-funktion för kontinuerlig optimering
        function sub_mean(x_cont)
             T = eltype(x_cont)
             x_full = zeros(T, d)
             if !isempty(cont_dims); x_full[cont_dims] = x_cont; end
             x_full[disc_dims] .= d_vals
             return -acq_mean(x_full) # Minimera negativt mean = Maximera mean
        end

        if !isempty(cont_dims)
            n_cont = length(cont_dims)
            starts = [-ones(n_cont) .+ (ones(n_cont) .- -ones(n_cont)) .* ((i .+ 0.5) ./ n_starts) for i in 0:(n_starts - 1)]
            
            for i in 1:n_starts
                x0 = starts[i]
                res = optimize(sub_mean, -1.0 * ones(n_cont), 1.0 * ones(n_cont), x0, Fminbox(LBFGS()); autodiff = :forward)
                
                curr_val = -Optim.minimum(res) # Invertera tillbaka
                if curr_val > best_val
                    best_val = curr_val
                    best_x_full[cont_dims] = Optim.minimizer(res)
                    best_x_full[disc_dims] .= d_vals
                end
            end
        else
            # Bara diskreta
            val = acq_mean(float([d_vals...])) # predict_f vill ha float-vektor även om det är heltal
            if val > best_val
                best_val = val
                best_x_full[disc_dims] .= d_vals
            end
        end
    end
    
    return (fX_max = best_val, X_max = best_x_full)
end






"""
    posteriorMaxObs(gp, X_scaled)

Finds the maximum of the GP's posterior mean evaluated only at the points
that have already been observed.

# Arguments
- `gp`: The trained Gaussian Process model.
- `X_scaled`: A `d x n` matrix of the `n` locations already observed,
  in the scaled space.

# Returns
- `Float64`: The maximum value of the posterior mean among the observed points.
"""
function posteriorMaxObs(gp, X_scaled)
    μ, _ = predict_f(gp, X_scaled)
    return maximum(μ)
end


"""
    rescale(X, lo, hi; integ=0)

Scales an `n x d` matrix `X` from an original domain to the hypercube `[-1, 1]^d`.

# Arguments
- `X`: An `n x d` matrix where each row is a point in the original domain.
- `lo`: A `d`-element vector of lower bounds for the original domain.
- `hi`: A `d`-element vector of upper bounds for the original domain.
- `Integ`: An integer deciding whether the last column should be scaled or not.


# Returns
- The scaled `n x d` matrix where all values are in `[-1, 1]`.

# Examples
```julia-repl
julia> lo = [0.0, 10.0];
julia> hi = [1.0, 20.0];
julia> X = [0.5 15.0; 0.0 10.0];
julia> rescale(X, lo, hi)
2×2 Matrix{Float64}:
 0.0  0.0
-1.0 -1.0
```
"""
function rescale(X, lo, hi; integ=0)
    # X is n×d matrix, lo and hi are length-d vectors
    if integ == 0
        return 2 .* (X .- lo') ./ (hi' .- lo') .- 1
    else
        return hcat(2 .* (X[:,1:end-integ] .- lo') ./ (hi' .- lo') .- 1, [round.(X[:,end]);])
    end
end

"""
    inv_rescale(X_scaled, lo, hi; integ = 0)

Performs the inverse of `rescale`, scaling an `n x d` matrix `X_scaled` from
the hypercube `[-1, 1]^d` back to the original domain.

# Arguments
- `X_scaled`: An `n x d` matrix where each row is a point in `[-1, 1]^d`.
- `lo`: A `d`-element vector of lower bounds for the original domain.
- `hi`: A `d`-element vector of upper bounds for the original domain.
- `Integ`: An integer deciding whether the last column should be scaled or not.

# Returns
- The un-scaled `n x d` matrix in the original domain.

# Examples
```julia-repl
julia> lo = [0.0, 10.0];
julia> hi = [1.0, 20.0];
julia> X_scaled = [0.0 0.0; -1.0 -1.0];
julia> inv_rescale(X_scaled, lo, hi)
2×2 Matrix{Float64}:
 0.5  15.0
 0.0  10.0
```
""" 
function inv_rescale(XScaled, lo, hi; integ = 0) 
    if integ == 0
        ((XScaled .+ 1) ./ 2) .* (hi' .- lo') .+ lo'
    else
        hcat(((XScaled[:,1:end-integ] .+ 1) ./ 2) .* (hi' .- lo') .+ lo', [round.(XScaled[:,end]);])
    end
end





# Main BO function
"""
    BO(f, modelSettings, optimizationSettings, warmStart)

Performs a full Bayesian Optimization run.
... (resten av docstringen) ...

# Returns
- A `Tuple` containing:
    - `gp`: The final, trained GP model.
    - `X`: The full `n x d` matrix of all evaluated points.
    - `y`: The full `n`-element vector of all observations.
    - `objectMaximizer`: The location `x` of the global maximum of the final posterior mean.
    - `objectMaximizerY`: The predicted posterior mean value at `objectMaximizer`. # <-- NEW
    - `postMaxObserved`: The location `x` of the observed point with the highest posterior mean.
    - `postMaxObservedY`: The predicted posterior mean value at `postMaxObserved`.
...
"""
function BO(f, modelSettings, optimizationSettings, warmStart; DiscreteKern=0)
    X, y = warmStart
    # Work in the scaled space [-1, 1]^d.
    Xscaled = rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2], integ = DiscreteKern)

    for i in 1:optimizationSettings.nIter
        # Standardize y-values at the start of the loop
        μ_y = mean(y)
        # Use max to set a robust standard deviation (jitter).
        σ_y = max(std(y), 1e-6)
        y_scaled = (y .- μ_y) ./ σ_y
        
        # Train the GP on the Standardized y-values
        gp = GP(Xscaled', y_scaled, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, 
                      options=Optim.Options(iterations=100))
    
        # Select what to optimize based on the acquisition strategy.
        if optimizationSettings.acq_config isa KnowledgeGradientConfig
            # This is for any knowledge gradient method!
            f_max_scaled = posteriorMax(gp; n_starts=10)
        else
            # For all other methods (EI, UCB, etc.)
            f_max_scaled = posteriorMaxObs(gp, Xscaled')
        end

        # Select next evaluation point.
        x_next_scaled = propose_next(gp, f_max_scaled,
                                     n_restarts=optimizationSettings.n_restarts,
                                     acq_config=optimizationSettings.acq_config
        )
        
        # Rescale back to original scale to evaluate the true function
        x_next_original = inv_rescale(x_next_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2], integ=DiscreteKern)[:]
        
        # Handle 1D vs multi-D function calls
        y_next = 0.0
        if modelSettings.xdim == 1
            y_next = f(x_next_original[1])
        else
            y_next = f(x_next_original)
        end

        # Add the new original-scale y-value to the dataset
        X = vcat(X, x_next_original')
        Xscaled = vcat(Xscaled, x_next_scaled')
        y = vcat(y, y_next)
    
        println("Iter $i: x = $(round.(x_next_original, digits=3)), y = $(round(y_next, digits=3))")
    end

    # --- Final GP model on all data ---
    # Standardize final y-data before fitting the final model
    μ_y_final = mean(y)

    # Use max to set a minimum standard deviation (jitter).
    σ_y_final = max(std(y), 1e-6)
    y_scaled_final = (y .- μ_y_final) ./ σ_y_final
    
    gp = GP(Xscaled', y_scaled_final, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
    optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=500))

    # (1) Global posterior mean maximum.
    final_posterior_max_result = posteriorMax(gp; n_starts=40)
    objectMaximizer_scaled = final_posterior_max_result.X_max
    objectMaximizer = inv_rescale(objectMaximizer_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2], integ=DiscreteKern)[:]
    
    # --- NEW CODE BLOCK: Rescale the global max value ---
    objectMaximizerY_scaled = final_posterior_max_result.fX_max
    objectMaximizerY = objectMaximizerY_scaled * σ_y_final + μ_y_final
    # ----------------------------------------------------

    # (2) Maximum over points with the highest posterior mean among observed points
    μ_scaled, _ = predict_f(gp, Xscaled')
    maxIdx = argmax(μ_scaled)
    postMaxObserved_scaled = Xscaled[maxIdx, :]
    postMaxObserved = inv_rescale(postMaxObserved_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2], integ=DiscreteKern)[:]
    
    # Rescale the final predicted mean back to the original y-scale.
    postMaxObservedY_scaled = μ_scaled[maxIdx]
    postMaxObservedY = postMaxObservedY_scaled * σ_y_final + μ_y_final
    
    return gp, X, y, objectMaximizer, objectMaximizerY, postMaxObserved, postMaxObservedY
end

# New version that takes a GP as an input. Then the user can easily specify a prior on the lengthscales and so on.
# Main BO function
"""
    BO(f, gp_template, modelSettings, optimizationSettings, warmStart)

Performs a full Bayesian Optimization run using a GP template.

# Arguments
- `gp_template`: A `GPE` object acting as a blueprint (containing kernel, mean, priors).
- ... other arguments as before ...
"""

# With debuggtimer.
function BO(f, gpTemplate::GPE, modelSettings, optimizationSettings, warmStart; DiscreteKern=0)
    println("--- Startar Bayesiansk Optimering ---")
    
    X, y = warmStart
    # Work in the scaled space [-1, 1]^d.
    Xscaled = rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2], integ = DiscreteKern)

    currentKernel = deepcopy(gpTemplate.kernel)
    currentLogNoise = gpTemplate.logNoise.value
    currentMean = gpTemplate.mean
    gp = nothing 
    # -------------------------------------------------------------

    for i in 1:optimizationSettings.nIter
        println("\n=== Iteration $i ===")
        
        # 1. Standardize Data
        μ_y = mean(y)
        σ_y = max(std(y), 1e-6)
        y_scaled = (y .- μ_y) ./ σ_y
        
        # t_gp = time() # debugg-timer
        # We need to clamp parameters so they ar within the specified bounds.
        safeParams = clamp.(get_params(currentKernel), 
                    modelSettings.kernelBounds[1] .+ 1e-4, 
                    modelSettings.kernelBounds[2] .- 1e-4
        )
        set_params!(currentKernel, safeParams)

    
        if !isnothing(modelSettings.noiseBounds)
            nlb, nub = modelSettings.noiseBounds
            safeNoise = clamp(currentLogNoise, nlb + 1e-4, nub - 1e-4)
        else
            safeNoise = Float64(currentLogNoise)
        end

        gp = GP(Xscaled', y_scaled, currentMean, deepcopy(currentKernel), safeNoise)
        
        # Stänga av optiomering av noise ger bättre speed.
        #optimize!(gp; kernbounds = modelSettings.kernelBounds, noise=false, 
        #          options=Optim.Options(iterations=100, time_limit = 5.0))
        optimize!(gp; 
            kernbounds = modelSettings.kernelBounds, 
            noisebounds = modelSettings.noiseBounds,
            method = NelderMead(),     
            options = Optim.Options(iterations=500, time_limit = 4.0)
        )
        
        #t_gp_end = time() - t_gp # debugg-timer
        #println("  [TIMER] GP Training:    $(round(t_gp_end, digits=4)) s")
        # -----------------------------

        currentKernel = gp.kernel
        currentMean = gp.mean
        currentLogNoise = max(gp.logNoise.value, -2.) # max() to avoid numerical issues.     

        #t_acq = time() # debugg-timer

        # Select what to optimize based on the acquisition strategy.
        if optimizationSettings.acq_config isa KnowledgeGradientConfig
            fMaxScaled = posteriorMax(gp; n_starts=10)
        else
            fMaxScaled = posteriorMaxObs(gp, Xscaled')
        end
        
        # Select next evaluation point.
        xNextScaled = propose_next(gp, fMaxScaled,
                                     n_restarts=optimizationSettings.n_restarts,
                                     acq_config=optimizationSettings.acq_config
        )
        
       # t_acq_end = time() - t_acq # debugg-timer
       # println("  [TIMER] Propose Next:   $(round(t_acq_end, digits=4)) s")
        # --------------------------------

        # Rescale back to original scale to evaluate the true function
        xNextOriginal = inv_rescale(xNextScaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2], integ=DiscreteKern)[:]
        
        #t_func = time() # debugg-timer
        
        y_next = 0.0
        if gp.dim == 1
            y_next = f(xNextOriginal[1])
        else
            y_next = f(xNextOriginal)
        end
        
        #t_func_end = time() - t_func # debug-timer
        #println("  [TIMER] Function Eval:  $(round(t_func_end, digits=4)) s") # debug-timer
        # ----------------------------------

        # Add the new original-scale y-value to the dataset
        X = vcat(X, xNextOriginal')
        Xscaled = vcat(Xscaled, xNextScaled')
        y = vcat(y, y_next)
        
        println("  >> Resultat: y = $(round(y_next, digits=3))")
    end

    println("\n=== Finalizing Model ===")

    # --- Final GP model on all data ---
    μ_y_final = mean(y)
    σ_y_final = max(std(y), 1e-6)
    yScaledFinal = (y .- μ_y_final) ./ σ_y_final
    
    #t_final_gp = time() # debugg-timer
    safeParams = clamp.(get_params(currentKernel), 
                modelSettings.kernelBounds[1] .+ 1e-4, 
                modelSettings.kernelBounds[2] .- 1e-4
    )
    set_params!(currentKernel, safeParams)
          
    if !isnothing(modelSettings.noiseBounds)
        nlb, nub = modelSettings.noiseBounds
        safeNoise = clamp(currentLogNoise, nlb + 1e-4, nub - 1e-4)
    else
        safeNoise = Float64(currentLogNoise)
    end

    gpOut = GP(Xscaled', yScaledFinal, currentMean, deepcopy(currentKernel), safeNoise)
    optimize!(gpOut; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, method = NelderMead(),
    options=Optim.Options(iterations=500, time_limit = 5.0))
    
    #println("  [TIMER] Final GP Opt:   $(round(time() - t_final_gp, digits=4)) s") # debugg-timer
    # -----------------------------

    #t_final_search = time() # debugg-timer

    # (1) Global posterior mean maximum.
    finalPosteriorMaxResult = posteriorMax(gpOut; n_starts=40)
    objectMaximizer_scaled = finalPosteriorMaxResult.X_max
    objectMaximizer = inv_rescale(objectMaximizer_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2], integ=DiscreteKern)[:]
    
    objectMaximizerY_scaled = finalPosteriorMaxResult.fX_max
    objectMaximizerY = objectMaximizerY_scaled * σ_y_final + μ_y_final
    
    # (2) Maximum over points with the highest posterior mean among observed points
    μ_scaled, _ = predict_f(gpOut, Xscaled')
    maxIdx = argmax(μ_scaled)
    postMaxObserved_scaled = Xscaled[maxIdx, :]
    postMaxObserved = inv_rescale(postMaxObserved_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2], integ=DiscreteKern)[:]
    
    postMaxObservedY_scaled = μ_scaled[maxIdx]
    postMaxObservedY = postMaxObservedY_scaled * σ_y_final + μ_y_final
    
    #println("  [TIMER] Final Max Search: $(round(time() - t_final_search, digits=4)) s") # debugg-timer

    return gpOut, X, y, objectMaximizer, objectMaximizerY, postMaxObserved, postMaxObservedY
end