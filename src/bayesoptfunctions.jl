
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

    for _ in 1:n_restarts
        x0 = rand(Uniform(-1., 1.), d)

        # Use the type to select the optimizer
        res = if acq_config isa KnowledgeGradientConfig # Use the abstract type
            optimize(objective_to_minimize, -1.0, 1.0, x0, Fminbox(NelderMead()))
        else
            optimize(objective_to_minimize, -1.0 * ones(d), 1.0 * ones(d), x0, Fminbox(LBFGS()); autodiff = :forward)
        end

        current_acq_val = -Optim.minimum(res)
        if current_acq_val > best_acq_val
            best_acq_val = current_acq_val
            best_x = Optim.minimizer(res)
        end
    end
    return best_x
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
function posteriorMax(gp; n_starts=20)
    d = gp.dim
    acq_mean(x) = predict_f(gp, reshape(x, d, 1))[1][1]
    best_val, best_x = multi_start_maximize(acq_mean, -1. * ones(d), 1. * ones(d), n_starts=n_starts)
    return (fX_max = best_val, X_max = best_x)
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
    rescale(X, lo, hi)

Scales an `n x d` matrix `X` from an original domain to the hypercube `[-1, 1]^d`.

# Arguments
- `X`: An `n x d` matrix where each row is a point in the original domain.
- `lo`: A `d`-element vector of lower bounds for the original domain.
- `hi`: A `d`-element vector of upper bounds for the original domain.

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
"""
function rescale(X, lo, hi)
    # X is n×d matrix, lo and hi are length-d vectors
    return 2 .* (X .- lo') ./ (hi' .- lo') .- 1
end

"""
    inv_rescale(X_scaled, lo, hi)

Performs the inverse of `rescale`, scaling an `n x d` matrix `X_scaled` from
the hypercube `[-1, 1]^d` back to the original domain.

# Arguments
- `X_scaled`: An `n x d` matrix where each row is a point in `[-1, 1]^d`.
- `lo`: A `d`-element vector of lower bounds for the original domain.
- `hi`: A `d`-element vector of upper bounds for the original domain.

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
inv_rescale(X_scaled, lo, hi) = ((X_scaled .+ 1) ./ 2) .* (hi' .- lo') .+ lo'




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
function BO(f, modelSettings, optimizationSettings, warmStart)
    X, y = warmStart
    # Work in the scaled space [-1, 1]^d.
    Xscaled = rescale(X, modelSettings.xBounds[1], modelSettings.xBounds[2])

    for i in 1:optimizationSettings.nIter
        # Standardize y-values at the start of the loop
        μ_y = mean(y)
        # Use max to set a robust standard deviation (jitter).
        σ_y = max(std(y), 1e-6)
        y_scaled = (y .- μ_y) ./ σ_y
        
        # Train the GP on the Standardized y-values
        gp = GP(Xscaled', y_scaled, modelSettings.mean, modelSettings.kernel, modelSettings.logNoise)
        optimize!(gp; kernbounds = modelSettings.kernelBounds, noisebounds = modelSettings.noiseBounds, options=Optim.Options(iterations=100))
    
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
        x_next_original = inv_rescale(x_next_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
        
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
    objectMaximizer = inv_rescale(objectMaximizer_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
    
    # --- NEW CODE BLOCK: Rescale the global max value ---
    objectMaximizerY_scaled = final_posterior_max_result.fX_max
    objectMaximizerY = objectMaximizerY_scaled * σ_y_final + μ_y_final
    # ----------------------------------------------------

    # (2) Maximum over points with the highest posterior mean among observed points
    μ_scaled, _ = predict_f(gp, Xscaled')
    maxIdx = argmax(μ_scaled)
    postMaxObserved_scaled = Xscaled[maxIdx, :]
    postMaxObserved = inv_rescale(postMaxObserved_scaled[:]', modelSettings.xBounds[1], modelSettings.xBounds[2])[:]
    
    # Rescale the final predicted mean back to the original y-scale.
    postMaxObservedY_scaled = μ_scaled[maxIdx]
    postMaxObservedY = postMaxObservedY_scaled * σ_y_final + μ_y_final
    
    # --- UPDATED RETURN STATEMENT ---
    return gp, X, y, objectMaximizer, objectMaximizerY, postMaxObserved, postMaxObservedY
end