# This file contain acquisition functions that can be used in the Bayesian optimization.

""" 
```
  expected_improvement(gp, xnew, fMax; ξ = 0.01)
```
Computes the expected improvement given a Gaussian process (`gp`) object, and the earliest best evaluation (`fMax`) at the new evaluation point `xnew`). ξ is a tuning parameter that controls the exploration-exploitation trade-off. A large value of ξ encourages exploration and vice versa.

Returns the expected improvement at `xnew`.
# Examples
```julia-repl
julia> Set up GP model.
julia> X_train = [1.0, 2.5, 4.0]; julia> y_train = [sin(x) for x in X_train];
julia> gp_model = GP(X_train', y_train, MeanZero(), SE(0.0, 0.0));
julia> optimize!(gp_model);

julia> # 2. Define the best observed value and a candidate point
julia> fMax = maximum(y_train);
julia> x_candidate = 3.0;

julia> # 3. Compute Expected Improvement
julia> ei = expected_improvement(gp_model, x_candidate, fMax; ξ=0.01)
≈ 4.89.
```
""" 
function expected_improvement(gp, xnew, fMax; ξ = 0.10)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 0.1^6)  # Ensure positive std deviation
    Δ = μ .- fMax .- ξ
    Z = Δ ./ σ
    Φ = cdf.(Normal(), Z)
    ϕ = pdf.(Normal(), Z)
    ei =  Δ .* Φ .+ σ .* ϕ
    return ei[1]
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
-0.22727575567547253   # Updete this example, it is from when i used minimum instead of maximum.
```
""" 
function upper_confidence_bound(gp, xnew; κ = 2.0)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 1e-6)  # Avoid zero std
    ucb = μ[1] + κ * σ
    return ucb
end



# Expected Improvement (EI) acquisition function with boundary penalty
function expected_improvement_boundary(gp, xnew, ybest; ξ = 0.10, bounds=nothing)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 0.1^6)  # Ensure positive std deviation
    Δ = μ .- ybest .- ξ
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






"""
    knowledge_gradient(gp, xnew, lower, upper; n_samples=20)

Calculates the Knowledge Gradient (KG) acquisition function for a candidate point `xnew`.
This version is designed for MAXIMIZATION problems.

The Knowledge Gradient quantifies the expected increase in the maximum estimated
value after sampling at `xnew`. This implementation uses Monte Carlo simulation.

# Arguments
- `gp::GPE`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the KG.
- `lower`, `upper`: The bounds of the search domain for finding the posterior maximum.
- `n_samples::Int`: The number of Monte Carlo samples to use.

# Returns
- `Float64`: The positive Knowledge Gradient value. This function returns the natural KG
  score, which should be maximized by the optimization loop.
"""
function knowledge_gradient(gp::GPE, xnew, lower, upper; n_samples::Int=20)
    xvec = xnew isa Number ? [xnew] : xnew
    xnew = reshape(xvec, :, 1)

    # 1. Find the MAXIMUM of the *current* posterior mean. This is our baseline.
    μ_current(x) = predict_f(gp, reshape(x, :, 1))[1][1]
    max_μ_current = multi_start_maximize(μ_current, lower, upper; n_starts=40)

    # 2. Get the predictive distribution at the candidate point `xnew`. (No change needed)
    μ_new_point, σ²_new_point = predict_y(gp, xnew)
    predictive_dist = Normal(μ_new_point[1], sqrt(max(σ²_new_point[1], 1e-6)))

    # 3. Run Monte Carlo simulation to estimate the expected future MAXIMUM.
    future_maximums = zeros(n_samples)
    for i in 1:n_samples
        # a. Draw one potential future observation `y_sample` at `xnew`.
        y_sample = rand(predictive_dist)

        # b. Create a temporary, "fantasy" GP model.
        x_updated = hcat(gp.x, xnew)
        y_updated = vcat(gp.y, y_sample)
        gp_fantasy = GPE(x_updated, y_updated, gp.mean, gp.kernel, gp.logNoise)

        # c. Find the MAXIMUM of the posterior mean of this new fantasy GP.
        μ_fantasy(x) = predict_f(gp_fantasy, reshape(x, :, 1))[1][1]
        future_maximums[i] = multi_start_maximize(μ_fantasy, lower, upper; n_starts=10)
    end

    # 4. Calculate the expected value of the future maximum.
    expected_max_μ_future = mean(future_maximums)

    # 5. The KG is the expected INCREASE in the MAXIMUM.
    kg = expected_max_μ_future - max_μ_current

    # 6. Return the natural, positive KG value.
    #    The optimization loop is responsible for negating this if it uses a minimizer.
    return kg
end





function multi_start_minimize(f, lower, upper; n_starts=20)
     if !isa(lower, AbstractVector)
         lower = [lower]
         upper = [upper]
     end
     dim = length(lower)
     starts = [lower .+ (upper .- lower) .* ((i .+ 0.5) ./ n_starts) for i in 0:(n_starts - 1)]

     mins = Float64[]
     for x0 in starts
         res = optimize(f, lower, upper, x0, Fminbox(BFGS()))
         push!(mins, Optim.minimum(res))
     end
     return minimum(mins)
end


"""
    multi_start_maximize(f, lower, upper; n_starts=20)

Performs multi-start constrained optimization to MAXIMIZE the objective function `f`.
It works by minimizing the negative of the function `f`.
"""
function multi_start_maximize(f, lower, upper; n_starts=20)
    # To maximize f(x), we can simply minimize -f(x).
    objective_to_minimize = x -> -f(x)
    
    # Use your existing multi_start_minimize function to find the minimum of -f(x).
    min_val = multi_start_minimize(objective_to_minimize, lower, upper; n_starts=n_starts)
    
    # The maximum of f(x) is the negative of the minimum of -f(x).
    return -min_val
end


##########################
"""
Calculates E[max(μ + σZ)] where Z ~ N(0,1).
This is the stable, core algorithm.
"""
function knowledge_gradient_discrete(μ::Vector{Float64}, σ::Vector{Float64})
    if length(μ) != length(σ)
        error("Input vectors μ and σ must have the same length.")
    end
    d = length(μ)
    if d == 0
        return 0.0
    elseif d == 1
        return 0.0 # KG is E[max] - max, for one line this is E[μ+σZ] - μ = 0
    end
    
    O = sortperm(σ)
    μ_sorted, σ_sorted = μ[O], σ[O]
    
    I = [1, 2]
    Z_tilde = [-Inf, (μ_sorted[1] - μ_sorted[2]) / (σ_sorted[2] - σ_sorted[1])]

    for i in 3:d
        while true
            j = last(I)
            z = (μ_sorted[j] - μ_sorted[i]) / (σ_sorted[i] - σ_sorted[j])
            if z >= last(Z_tilde)
                push!(I, i)
                push!(Z_tilde, z)
                break
            else
                pop!(I)
                pop!(Z_tilde)
                if isempty(I)
                    push!(I, i); push!(Z_tilde, -Inf); break
                end
            end
        end
    end
    
    push!(Z_tilde, Inf)
    norm_dist = Normal(0, 1)
    
    z_upper, z_lower = Z_tilde[2:end], Z_tilde[1:end-1]
    
    A_vec = pdf.(norm_dist, z_lower) - pdf.(norm_dist, z_upper)
    
    B_vec = cdf.(norm_dist, z_upper) - cdf.(norm_dist, z_lower)
    
    μ_I, σ_I = μ_sorted[I], σ_sorted[I]
    
    expected_max = (B_vec' * μ_I) + (A_vec' * σ_I)
    max_μ_current = maximum(μ)
    
    return expected_max - max_μ_current
end

"""
Computes the Knowledge Gradient acquisition function for a multi-dimensional GP.

This version correctly handles multi-dimensional inputs by expecting `domain_points`
to be a D x d matrix.
"""
function kg_acquisition_function_multid(gp::GPE, xnew::Vector{Float64}, domain_points::Matrix{Float64})
    # --- Step A: Get GP-specific values ---
    # Get the current posterior mean over the discrete domain points.
    μ_current, _ = predict_f(gp, domain_points)
    
    # Get the predictive distribution of a noisy observation y at xnew.
    xnew_mat = reshape(xnew, :, 1)
    _, σ²_y_new = predict_y(gp, xnew_mat)
    σ_y_new = sqrt(max(σ²_y_new[1], 1e-12))

    # Calculate the covariance vector between the domain points and xnew.
    K_domain_xnew = cov(gp.kernel, domain_points, xnew_mat)

    # This vector describes the change in the posterior mean.
    update_vector = K_domain_xnew ./ σ²_y_new[1]

    # --- Step B: Construct μ and σ for the core algorithm ---
    μ = vec(μ_current)
    σ = vec(update_vector) .* σ_y_new
    
    # --- Step C: Call the robust, standalone algorithm ---
    kg_value = knowledge_gradient_discrete(μ, σ)
    
    return kg_value
end
