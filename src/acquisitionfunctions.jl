# This file contain acquisition functions that can be used in the Bayesian optimization.
# Note that all functions are defined in order to minimize a function!

""" 
```
  expected_improvement(gp, xnew, ybest; ξ = 0.01)
```
Computes the expected improvement given a Gaussian process (`gp`) object, and the earliest best evaluation (`ybest`) at the new evaluation point `xnew`). ξ is a tuning parameter that controls the exploration-exploitation trade-off. A large value of ξ encourages exploration and vice versa.

Returns the expected improvement at `xnew`.
# Examples
```julia-repl
julia> Set up GP model.
julia> X_train = [1.0, 2.5, 4.0]; julia> y_train = [sin(x) for x in X_train];
julia> gp_model = GP(X_train', y_train, MeanZero(), SE(0.0, 0.0));
julia> optimize!(gp_model);

julia> # 2. Define the best observed value and a candidate point
julia> y_best = minimum(y_train);
julia> x_candidate = 3.0;

julia> # 3. Compute Expected Improvement
julia> ei = expected_improvement(gp_model, x_candidate, y_best; ξ=0.01)
≈ 4.89.
```
""" 
function expected_improvement(gp, xnew, ybest; ξ = 0.10)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 0.1^6)  # Ensure positive std deviation
    Δ = ybest .- μ .- ξ
    Z = Δ ./ σ
    Φ = cdf.(Normal(), Z)
    ϕ = pdf.(Normal(), Z)
    ei =  Δ .* Φ .+ σ .* ϕ
    return -ei[1]
end



""" 
```
  upper_confidence_bound(gp, xnew; κ = 2.0)
```
Computes the upper confidence bound criteria at the new point `xnew` given a Gaussian process (`gp`) object, and the exploitation/exploitation parameter `κ`. High values of κ encourage exploration.

Returns the upper confidence bound criteria at `xnew`.
# Examples
```julia-repl
julia> Set up GP model.
julia> X_train = [1.0, 2.5, 4.0]; julia> y_train = [sin(x) for x in X_train];
julia> gp_model = GP(X_train', y_train, MeanZero(), SE(0.0, 0.0));
julia> optimize!(gp_model);
julia> x_candidate = 3.0;

julia> ucb = upper_confidence_bound(gp_model, x_candidate, κ = 2.0)
-0.22727575567547253
```
""" 
function upper_confidence_bound(gp, xnew; κ = 2.0)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 1e-6)  # Avoid zero std
    ucb = μ[1] - κ * σ
    return ucb
end



# Expected Improvement (EI) acquisition function with boundary penalty
function expected_improvement_boundary(gp, xnew, ybest; ξ = 0.10, bounds=nothing)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 0.1^6)  # Ensure positive std deviation
    Δ = ybest .- μ .- ξ
    Z = Δ ./ σ
    Φ = cdf.(Normal(), Z)
    ϕ = pdf.(Normal(), Z)
    ei = Δ .* Φ .+ σ .* ϕ
    ei_val = ei[1]

    # Apply boundary penalty if bounds are provided
    if bounds !== nothing
        a, b = bounds
        # Normalize x to [-1, 1] per dimension
        xnorm = 2 .* ((xvec[:,1] .- a) ./ (b .- a)) .- 1
        # Compute weight: product of (1 - normalized^2)
        w = prod(1 .- xnorm.^2)
        ei_val *= w
    end

    return ei_val
end



#########################
"""
    knowledge_gradient(gp, xnew, domain_points; n_samples=20)

Calculates the Knowledge Gradient (KG) acquisition function for a candidate point `xnew`.

The Knowledge Gradient quantifies the expected improvement in the best estimated
value after sampling at `xnew`. It is particularly effective for noisy objective
functions. This implementation uses Monte Carlo simulation to approximate the expectation.

# Arguments
- `gp::GPE`: The trained Gaussian Process model from GaussianProcesses.jl.
- `xnew::Vector{Float64}`: The candidate point at which to evaluate the KG.
- `domain_points::Matrix{Float64}`: A matrix of points discretizing the search domain.
  This is used to find the minimum of the posterior mean efficiently. Each column is a point.
- `n_samples::Int=20`: The number of Monte Carlo samples to draw to estimate the expectation.
  Higher values give a more accurate estimate but increase computation time.

# Returns
- `Float64`: The negative Knowledge Gradient value. It is returned as negative because
  standard optimizers perform minimization, and we want to maximize the acquisition function.
"""

function knowledge_gradient(gp::GPE, xnew, lower, upper; n_samples::Int=20)
    xvec = xnew isa Number ? [xnew] : xnew
    xnew = reshape(xvec, :, 1)
    # 1. Find the minimum of the *current* posterior mean. This is our baseline.
    # We find it by evaluating the posterior mean over our discretized domain.
    #μ_current, _ = predict_f(gp, domain_points)
    #min_μ_current = minimum(μ_current)
    μ_current(x) = predict_f(gp, reshape(x, :, 1))[1][1]
    min_μ_current = multi_start_minimize(μ_current, lb, ub; n_starts=40)


    # 2. Get the predictive distribution at the candidate point `xnew`.
    # This tells us what we expect to observe if we sample there.
    μ_new_point, σ²_new_point = predict_y(gp, reshape(xnew, :, 1))
   
    # Add model noise to the predictive variance for simulating a noisy observation
    predictive_dist = Normal(μ_new_point[1], sqrt(max(σ²_new_point[1], 1e-6)))

    # 3. Run Monte Carlo simulation to estimate the expected future minimum.
    future_minimums = zeros(n_samples)
    for i in 1:n_samples
        # a. Draw one potential future observation `y_sample` at `xnew`.
        y_sample = rand(predictive_dist)

        # b. Create a temporary, updated GP model by "fantasizing" that we
        #    observed (xnew, y_sample).
        x_updated = hcat(gp.x, xnew)
        y_updated = vcat(gp.y, y_sample)

        # Re-fit the GP with this new fantasized data point.
        # Note: In a real application, you might want to avoid re-optimizing
        # hyperparameters here for speed, but for correctness, we re-fit.
        gp_fantasy = GPE(x_updated, y_updated, gp.mean, gp.kernel, gp.logNoise)

        # c. Find the minimum of the posterior mean of this *new* fantasy GP.
        #μ_fantasy, _ = predict_f(gp_fantasy, domain_points)
        #future_minimums[i] = minimum(μ_fantasy)
        μ_fantasy(x) = predict_f(gp_fantasy, reshape(x, :, 1))[1][1]
        future_minimums[i] = multi_start_minimize(μ_fantasy, lb, ub; n_starts=10)
    end

    # 4. Calculate the expected value of the future minimum by averaging the simulations.
    expected_min_μ_future = mean(future_minimums)

    # 5. The Knowledge Gradient is the difference between our current best
    #    estimate and the expected future best estimate.
    kg = min_μ_current - expected_min_μ_future

    # 6. Return the negative value for maximization with a minimizer.
    return -kg
end

"""
    multi_start_minimize(f, lower, upper; n_starts=20)

Performs multi-start constrained optimization using `Fminbox` with `BFGS()` to minimize the objective function `f` over the box-constrained domain defined by `lower` and `upper`. Supports both 1D and multi-dimensional inputs.

The algorithm initializes `n_starts` evenly spread starting points within the domain and selects the best minimum found across all runs.

# Arguments
- `f::Function`: The function to minimize.
- `lower::Union{Number, AbstractVector}`: Lower bound(s) of the search domain.
- `upper::Union{Number, AbstractVector}`: Upper bound(s) of the search domain.
- `n_starts::Int`: Number of starting points. Defaults to 20.

# Returns
- `Float64`: The minimum value of `f` found across all runs.

# Examples
```julia-repl
julia> f(x) = sum((x .- 1.0).^2)
julia> minimum = multi_start_minimize(f, [-5.0, -5.0], [5.0, 5.0], n_starts=10)
0.0

julia> g(x) = (x[1] - 2)^2 + sin(5x[1])
julia> minimum = multi_start_minimize(g, -3.0, 3.0, n_starts=15)
-0.3784012476539641
function multi_start_minimize(f, lower, upper; n_starts=20)
      if !isa(lower, AbstractVector)
        lower = [lower]
        upper = [upper]
    end
    dim = length(lower)
    # Generate evenly spaced starting points in each dimension
    starts = [lower .+ (upper .- lower) .* ((i .+ 0.5) ./ n_starts) for i in 0:(n_starts - 1)]

    mins = Float64[]
    for x0 in starts
        res = optimize(f, lower, upper, x0, Fminbox(BFGS()))
        push!(mins, Optim.minimum(res))
    end
    return minimum(mins)
end
"""

function multi_start_minimize(f, lower, upper; n_starts=20)
      if !isa(lower, AbstractVector)
        lower = [lower]
        upper = [upper]
    end
    dim = length(lower)
    # Generate evenly spaced starting points in each dimension
    starts = [lower .+ (upper .- lower) .* ((i .+ 0.5) ./ n_starts) for i in 0:(n_starts - 1)]

    mins = Float64[]
    for x0 in starts
        res = optimize(f, lower, upper, x0, Fminbox(BFGS()))
        push!(mins, Optim.minimum(res))
    end
    return minimum(mins)
end