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



# Expected Improvement (EI) acquisition function with boundary penalty. EXPERIMENTAL!
"""
    expected_improvement_boundary(gp, xnew, ybest; ξ = 0.10, bounds=nothing)

Computes the Expected Improvement (EI) with an optional penalty for points
near the boundary of the search space. VERY EXPERIMENTAL.

The penalty is a multiplicative weight `w = prod(1 - x_norm^2)` that smoothly
pushes the search away from the edges of the scaled `[-1, 1]` domain. This
can be useful to prevent the optimizer from selecting points at the very edge
of the feasible region. The idea comes from practical experience that EI otherwise spent too much time exploring the boundaries.

# Arguments
- `gp`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the EI.
- `ybest`: The current best observed value.
- `ξ`: The exploration-exploitation trade-off parameter.
- `bounds`: A tuple `(lower, upper)` defining the original search space, required for the penalty.

# Returns
- `Float64`: The (potentially penalized) Expected Improvement value.
"""
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
        xnorm = 2 .* ((xvec[:,1] .- a) ./ (b .- a)) .- 1
        w = prod(1 .- xnorm.^2)
        ei_val *= w
    end

    return ei_val
end



"""
    knowledgeGradientMonteCarlo(gp, xnew, lower, upper; n_samples=20)

Calculates the Knowledge Gradient (KG) acquisition function for a candidate point `xnew` using Monte Carlo.
This version gives a noisy surface to the optimization landscape due to MC variation. Try the "quadrature methods below".

The Knowledge Gradient quantifies the expected increase in the maximum estimated
value after sampling at `xnew`.

# Arguments
- `gp::GPE`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the KG.
- `n_samples::Int`: The number of Monte Carlo samples to use.

# Returns
- `Float64`: The positive Knowledge Gradient value. This function returns the natural KG
  score, which should be maximized by the optimization loop.
"""
function knowledgeGradientMonteCarlo(gp, xnew; n_samples::Int=200)
    xvec = xnew isa Number ? [xnew] : xnew
    xnew = reshape(xvec, :, 1)

    # 1. Find the MAXIMUM of the *current* posterior mean. This is our baseline.
    μ_current(x) = predict_f(gp, reshape(x, :, 1))[1][1]
    d = gp.dim
    lower= fill(-1.0, d)
    upper= fill(1.0, d)
    max_μ_current, _ = multi_start_maximize(μ_current, lower, upper; n_starts=40)

    # 2. Get the predictive distribution at the candidate point `xnew`.
    μ_new_point, σ²_new_point = predict_y(gp, xnew)
    predictive_dist = Normal(μ_new_point[1], sqrt(max(σ²_new_point[1], 1e-6)))

    # 3. Run Monte Carlo simulation to estimate the expected future maximum.
    future_maximums = zeros(n_samples)
    for i in 1:n_samples
        # a. Draw one potential future observation `y_sample` at `xnew`.
        y_sample = [rand(predictive_dist)]

        # b. Create a temporary, "fantasy" GP model.
        x_updated = hcat(gp.x, xnew)
        y_updated = vcat(gp.y, y_sample)
        gp_fantasy = GP(x_updated, y_updated, gp.mean, gp.kernel, gp.logNoise)

        # c. Find the maximum of the posterior mean of this new fantasy GP.
        μ_fantasy(x) = predict_f(gp_fantasy, reshape(x, :, 1))[1][1]
        future_maximums[i], _ = multi_start_maximize(μ_fantasy, lower, upper; n_starts=10)
    end

    # 4. Calculate the expected value of the future maximum.
    expected_max_μ_future = mean(future_maximums)

    # 5. The KG is the expected increase in the maximum.
    kg = expected_max_μ_future - max_μ_current

    return kg
end




"""
    multi_start_minimize(f, lower, upper; n_starts=20)

Finds the global minimum of a function `f` within a given box `[lower, upper]`
by using multiple starting points.

This function uses the L-BFGS optimizer from `Optim.jl` together with autodiff, starting from `n_starts`
evenly spaced points within the domain to increase the probability of finding a
global, rather than local, minimum.

# Arguments
- `f`: The function to minimize.
- `lower`: A vector of lower bounds.
- `upper`: A vector of upper bounds.
- `n_starts::Int`: The number of distinct starting points to use.

# Returns
- `Tuple{Float64, Vector{Float64}}`: A tuple `(best_min, best_argmin)` containing
  the minimum value found and the location where it was found.
"""
function multi_start_minimize(f, lower, upper; n_starts=20)
     if !isa(lower, AbstractVector)
         lower = [lower]
         upper = [upper]
     end
     dim = length(lower)
     starts = [lower .+ (upper .- lower) .* ((i .+ 0.5) ./ n_starts) for i in 0:(n_starts - 1)]

     best_min = Inf
     best_argmin = similar(lower)

     for x0 in starts
         #res = optimize(f, lower, upper, x0, Fminbox(BFGS()))
         #res = optimize(f, lower, upper, x0, Fminbox(NelderMead()))
         res = optimize(f, lower, upper, x0, Fminbox(LBFGS()),
                       Optim.Options(iterations=100); # Add options if needed
                       autodiff = :forward)

         if Optim.minimum(res) < best_min
            best_min = Optim.minimum(res)
            best_argmin = Optim.minimizer(res)
         end
     end
     return (best_min, best_argmin)
end


"""
    multi_start_maximize(f, lower, upper; n_starts=20)

Finds the global maximum of a function `f` within a given box `[lower, upper]`
by using multiple starting points. This is a wrapper around `multi_start_minimize`
that simply minimizes the negative of the function.

# Arguments
- `f`: The function to maximize.
- `lower`: A vector of lower bounds.
- `upper`: A vector of upper bounds.
- `n_starts::Int`: The number of distinct starting points to use.

# Returns
- `Tuple{Float64, Vector{Float64}}`: A tuple `(best_max, best_argmax)` containing
  the maximum value found and the location where it was found.
"""
function multi_start_maximize(f, lower, upper; n_starts=20)
    objective_to_minimize = x -> -f(x)
    min_val, argmin_x = multi_start_minimize(objective_to_minimize, lower, upper; n_starts=n_starts)
    return (-min_val, argmin_x)
end



"""
    ExpectedMaxGaussian(μ::Vector{Float64}, σ::Vector{Float64})

Analytically computes the expectation `E[max(μ + σZ)]` where `Z ~ N(0,1)`.

This is the core for `knowledgeGradientDiscrete`. It calculates
the expected maximum of a set of correlated Gaussian random variables that share
a single source of randomness, `Z`. The function is robust to cases where
slopes (`σ`) are equal or nearly equal.

# Arguments
- `μ::Vector{Float64}`: A vector of the current means of the random variables.
- `σ::Vector{Float64}`: A vector of the sensitivities ("slopes") of each random
  variable with respect to the common random factor `Z`.

# Returns
- `Float64`: The analytically computed expected maximum value.
"""
function ExpectedMaxGaussian(μ::Vector{Float64}, σ::Vector{Float64})
    if length(μ) != length(σ)
        error("Input vectors μ and σ must have the same length.")
    end
    d = length(μ)
    if d == 0
        return 0.0
    elseif d == 1
        return μ[1] # E[μ+σZ] = μ
    end
    
    # Sort by slope in ascending order
    O = sortperm(σ)
    μ_sorted, σ_sorted = μ[O], σ[O]
    
    I = [1]
    Z_tilde = [-Inf]

    for i in 2:d
        while !isempty(I)
            j = last(I)
            # Robustness fix for near-equal slopes
            if abs(σ_sorted[i] - σ_sorted[j]) < 1e-9
                if μ_sorted[i] <= μ_sorted[j]
                    @goto next_i
                else
                    pop!(I); pop!(Z_tilde); continue
                end
            end
            z = (μ_sorted[j] - μ_sorted[i]) / (σ_sorted[i] - σ_sorted[j])
            if z > last(Z_tilde)
                push!(I, i); push!(Z_tilde, z); break
            else
                pop!(I); pop!(Z_tilde)
            end
        end
        if isempty(I)
            push!(I, i); push!(Z_tilde, -Inf)
        end
        @label next_i
    end
    
    push!(Z_tilde, Inf)
    norm_dist = Normal(0, 1)
    z_upper, z_lower = Z_tilde[2:end], Z_tilde[1:end-1]
    
    A_vec = pdf.(norm_dist, z_lower) - pdf.(norm_dist, z_upper)
    B_vec = cdf.(norm_dist, z_upper) - cdf.(norm_dist, z_lower)
    
    μ_I, σ_I = μ_sorted[I], σ_sorted[I]
    
    expected_max = (B_vec' * μ_I) + (A_vec' * σ_I)
    
    return expected_max
end


"""
    posterior_cov(gp::GPE, X1::AbstractMatrix, X2::AbstractMatrix)

Computes the posterior covariance matrix `k_n(X1, X2)` between two sets of
points `X1` and `X2`, given the GP's training data.

The calculation uses the formula `k(X1, X2) - k(X1, X) * K_inv * k(X, X2)`,
using the pre-computed Cholesky factorization of the GP's kernel matrix
for high efficiency.

# Arguments
- `gp::GPE`: The trained Gaussian Process model.
- `X1::AbstractMatrix`: A `d x n1` matrix of `n1` points.
- `X2::AbstractMatrix`: A `d x n2` matrix of `n2` points.

# Returns
- `Matrix{Float64}`: The `n1 x n2` posterior covariance matrix.
"""
function posterior_cov(gp::GPE, X1::AbstractMatrix, X2::AbstractMatrix)
    # Formeln är: k_n(X1, X2) = k(X1, X2) - k(X1, X) * K_inv * k(X, X2)
    # där K = k(X,X) + σ_n²*I
    
    # Beräkna de priora kovarianstermerna
    prior_cov_12 = cov(gp.kernel, X1, X2)
    prior_cov_1X = cov(gp.kernel, X1, gp.x)
    prior_cov_X2 = cov(gp.kernel, gp.x, X2)
    
    # Lös systemet K * M = k(X, X2) för att effektivt få K_inv * k(X, X2)
    # GaussianProcesses.jl lagrar den Cholesky-faktoriserade matrisen i gp.cK
    # vilket gör detta mycket effektivt.
    # L*L' * M = prior_cov_X2  =>  L' * M = L \ prior_cov_X2  =>  M = L' \ (L \ prior_cov_X2)
    K_inv_k_X2 = gp.cK \ prior_cov_X2
  
    # Beräkna korrektionstermen
    correction = prior_cov_1X * K_inv_k_X2
    
    return prior_cov_12 - correction
end




"""
    knowledgeGradientDiscrete(gp, xnew, domain_points)

Computes the Knowledge Gradient (KG) acquisition function where the future
maximum is constrained to occur on a fixed, discrete set of points.

This function is the analytical engine for the `knowledgeGradientHybrid` heuristic.

# Arguments
- `gp`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the KG.
- `domain_points`: A `d x M` matrix of `M` discrete points in the
  scaled `[-1, 1]` space where the future maximum is sought.

# Returns
- `Float64`: The discrete Knowledge Gradient value.
"""
function knowledgeGradientDiscrete(gp, xnew, domain_points)
    xvec = xnew isa Number ? [xnew] : xnew
    xnew_mat = reshape(xvec, :, 1)

    # Get current posterior mean over the discrete domain points.
    μ_current, _ = predict_f(gp, domain_points)
    
    # Get the predictive distribution of a noisy observation y at xnew.
    _, σ²_y_new = predict_y(gp, xnew_mat)
    σ_y_new = sqrt(max(σ²_y_new[1], -1e-8))

    # Calculate the posterior covariance vector between the domain points and xnew.
     post_cov_vec = posterior_cov(gp, domain_points, xnew_mat)
     σ̃  = post_cov_vec ./ σ_y_new

    # Construct μ and σ for the core algorithm
    μ = vec(μ_current)
   
    
    # 1. Call the pure function to get ONLY the expected future maximum.
    expected_max_future = ExpectedMaxGaussian(μ, vec(σ̃ ))
    
    # 2. Calculate the current maximum (the baseline) from the μ vector.
    max_μ_current = maximum(μ)
    
    # 3. Perform the final subtraction to get the true KG value.
    kg_value = expected_max_future - max_μ_current
    
    return kg_value
end

# Helper method for convenience to handle scalar xnew
function knowledgeGradientDiscrete(gp::GPE, xnew::Number, domain_points::Matrix{Float64})
    return knowledgeGradientDiscrete(gp, [xnew], domain_points)
end


###################################################
"""
    knowledgeGradientHybrid(gp, xnew, lower, upper; n_z=5)

Calculates the Hybrid Knowledge Gradient (KGh) acquisition function for a candidate point `xnew`.

The Hybrid KG combines the strengths of the Monte-Carlo and Discrete KG methods. It uses
a small, deterministic set of `n_z` fantasy scenarios to identify a high-potential
set of future maximizers (`X_MC`), and then uses the fast, analytical Discrete KG
algorithm on that small set.

# Arguments
- `gp::GPE`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the KG.
- `lower`, `upper`: The bounds of the search domain for finding the fantasy maximizers.
- `n_z::Int`: The number of deterministic Z-samples to use (default is 5, as in the paper).

# Returns
- `Float64`: The positive Hybrid Knowledge Gradient value.
"""
function knowledgeGradientHybrid(gp::GPE, xnew; n_z::Int=5)
    d = gp.dim
    xvec = xnew isa Number ? [xnew] : xnew
    xnew_scaled = reshape(xvec, :, 1)

    # --- Stage 1: Find the set of fantasy maximizers, X_MC ---
    
    probabilities = range(0.5/n_z, 1 - 0.5/n_z, length=n_z)
    Z_h = quantile.(Normal(), probabilities)

    lower= -1.0 * ones(d)
    upper = 1.0 * ones(d)

    X_MC_cols = []
    for z_val in Z_h
        # Create the fantasy posterior mean function for this specific Z
        function μ_fantasy(x_scaled)
            x_scaled_mat = reshape(x_scaled, d, 1)
            
            # This is the reparameterization trick: μ_future = μ_current + σ * Z
            μ_current, _ = predict_f(gp, x_scaled_mat)
            _, σ²_y_new = predict_y(gp, xnew_scaled)
            σ_y_new = sqrt(max(σ²_y_new[1], 1e-8))
            
            # This is the covariance between the current point and the domain points
            post_cov_vec = posterior_cov(gp, x_scaled_mat, xnew_scaled)
            σ̃  = post_cov_vec ./ σ_y_new
            
            fantasy_mean = vec(μ_current) .+ σ̃  .* z_val
            return fantasy_mean[1]
        end

        # Find the maximizer of this fantasy posterior
        _, x_star_j = multi_start_maximize(μ_fantasy, lower, upper; n_starts=10)
        push!(X_MC_cols, x_star_j)
    end

    # Create the X_MC matrix from the collected maximizers
    X_MC = hcat(unique(X_MC_cols)...)

    # --- Stage 2: Calculate the final KG value analytically (Corrected) ---
    
    # The Hybrid KG is defined as the Discrete KG evaluated using the smart set X_MC.
    # Your knowledgeGradientDiscrete function already correctly calculates:
    # E[max(μ_future(X_MC))] - max(μ_current(X_MC))
    # This is the correct and final fix.
    kgh_value = knowledgeGradientDiscrete(gp, xnew_scaled, X_MC)
    
    return kgh_value
end


# Fortfarande ett experiment, istället för montecarlo så gör man kvadratur.
"""
    knowledgeGradientQuadrature(gp, xnew; n_z=20, alpha=0.5, n_starts=15)

Computes the Knowledge Gradient (KG) using a direct and deterministic quadrature
approximation of the expectation integral.

This is faster and smoother to optimize relative to Monte Carlo and more robust than Hybrid methods. It
uses a tail-focused quadrature scheme based on a Beta distribution transformation to
increase exploration, while compensating with non-uniform weights to maintain an
unbiased estimate. The resulting acquisition surface is deterministic (not noisy)
but not necessarily smooth.

# Arguments
- `gp`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the KG.
- `n_z::Int`: The number of nodes (fantasy scenarios) used for the quadrature approximation.
- `alpha::Float64`: Controls the tail-focus of the quadrature nodes. `alpha < 1.0`
  emphasizes the tails (more exploration), while `alpha = 1.0` reverts to uniform spacing (on a cdf).
- `n_starts::Int`: The number of restarts for the inner optimization that finds the
  maximum of each fantasy posterior.

# Returns
- `Float64`: The Knowledge Gradient value, guaranteed to be non-negative.
"""
#function knowledgeGradientQuadrature(gp::GPE, xnew; n_z::Int=20, alpha::Float64=0.5, n_starts::Int=15)
#    d = gp.dim
#    xvec = xnew isa Number ? [xnew] : xnew
#    xnew_scaled = reshape(xvec, :, 1)
#
#    # Step 1: find current maximum of the posterior mean
#    μ_current_func(x) = predict_f(gp, reshape(x, :, 1))[1][1]
#    lower = -1.0 * ones(d)
#    upper = 1.0 * ones(d)
#    max_μ_current, _ = multi_start_maximize(μ_current_func, lower, upper; n_starts=n_starts*2)
#
#    # Step 2: set up quadrature points and weights (Beta-trick)
#    linear_probabilities = range(1e-6, 1.0 - 1e-6, length=n_z)
#    tail_focused_probabilities = quantile.(Beta(alpha, alpha), linear_probabilities)
#    Z_h = quantile.(Normal(), tail_focused_probabilities)
#    
#    p_ext = [0.0; tail_focused_probabilities; 1.0]
#    quadrature_weights = [(p_ext[i+1] - p_ext[i-1]) / 2 for i in 2:(length(p_ext)-1)]
#    
#    # Step 3: compute maximum for each fantasy
#    fantasy_max_values = zeros(n_z)
#    for (i, z_val) in enumerate(Z_h)
#        function μ_fantasy(x_scaled)
#            x_scaled_mat = reshape(x_scaled, d, 1)
#            μ_current, _ = predict_f(gp, x_scaled_mat)
#            _, σ²_y_new = predict_y(gp, xnew_scaled)
#            σ_y_new = sqrt(max(σ²_y_new[1], 1e-6))
#            post_cov_vec = posterior_cov(gp, x_scaled_mat, xnew_scaled)
#            σ̃ = post_cov_vec ./ σ_y_new
#            fantasy_mean = vec(μ_current) .+ σ̃ .* z_val
#            return fantasy_mean[1]
#        end
#        max_val, _ = multi_start_maximize(μ_fantasy, lower, upper; n_starts=n_starts)
#        fantasy_max_values[i] = max_val
#    end
#
#    # --- Steg 4: Beräkna det viktade förväntningsvärdet ---
#    expected_max_future = sum(fantasy_max_values .* quadrature_weights)
#
#    # --- Steg 5: Slutgiltigt KG-värde ---
#    kg_value = expected_max_future - max_μ_current
#    
#    return max(kg_value, 0.0) # KG ska inte vara negativ
#end


# Snabbare version än den ovan då vi kan förberäkna saker utanför den innersta loopen.
function knowledgeGradientQuadrature(gp::GPE, xnew; n_z::Int=20, alpha::Float64=0.5, n_starts::Int=15)
    d = gp.dim
    xvec = xnew isa Number ? [xnew] : xnew
    xnew_scaled = reshape(xvec, :, 1)

    # Step 1: find current maximum of the posterior mean
    μ_current_func(x) = predict_f(gp, reshape(x, :, 1))[1][1]
    lower = -1.0 * ones(d)
    upper = 1.0 * ones(d)
    max_μ_current, _ = multi_start_maximize(μ_current_func, lower, upper; n_starts=n_starts*2)

#    # Step 2: set up quadrature points and weights (Beta-trick)
    linear_probabilities = range(1e-6, 1.0 - 1e-6, length=n_z)
    tail_focused_probabilities = quantile.(Beta(alpha, alpha), linear_probabilities)
    Z_h = quantile.(Normal(), tail_focused_probabilities)
    
    p_ext = [0.0; tail_focused_probabilities; 1.0]
    quadrature_weights = [(p_ext[i+1] - p_ext[i-1]) / 2 for i in 2:(length(p_ext)-1)]

    # Precompute quantities that are constant over all z-values
    _, σ²_y_new = predict_y(gp, xnew_scaled)
    σ_y_new = sqrt(max(σ²_y_new[1], 1e-6))
    k_X_xnew = cov(gp.kernel, gp.x, xnew_scaled)
    K_inv_k_X_xnew = gp.cK \ k_X_xnew

#    # Step 3: compute maximum for each fantasy
    fantasy_max_values = zeros(n_z)
    for (i, z_val) in enumerate(Z_h)
        
        # Precompute quantities that are constant for this particular z-value.
        alpha_fantasy = gp.alpha .- (K_inv_k_X_xnew .* z_val) ./ σ_y_new
        c_fantasy = z_val / σ_y_new

        # Cheaper fantacies
        function μ_fantasy_fast(x_scaled)
            # These are the two expensive operations that are left.
            k_x_X = cov(gp.kernel, reshape(x_scaled, d, 1), gp.x)
            k_x_xnew = cov(gp.kernel, reshape(x_scaled, d, 1), xnew_scaled)
            
            # These ar cheaper.
            pred = (k_x_X * alpha_fantasy)[1] + k_x_xnew[1] * c_fantasy
            return pred
        end

        max_val, _ = multi_start_maximize(μ_fantasy_fast, lower, upper; n_starts=n_starts)
        fantasy_max_values[i] = max_val
    end

    # Weighted expected maximum.
    expected_max_future = sum(fantasy_max_values .* quadrature_weights)

    #  Final KG value
    kg_value = expected_max_future - max_μ_current
    
    return max(kg_value, 0.0)
end


###########################
# Experimenting:
"""
    posterior_variance(gp, xnew)

Computes the posterior variance of the Gaussian Process `gp` at a new point `xnew`.

This acquisition function directs sampling to regions of highest uncertainty in the
model. It is the primary acquisition function for standard Bayesian Quadrature
methods like WSABI, but can also be used for pure exploration in Bayesian Optimization.

# Arguments
- `gp`: The trained Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the posterior variance.

# Returns
- `Float64`: The posterior variance at `xnew`.

# Examples
```julia-repl
julia> X_train = [1.0, 5.0];
julia> y_train = [sin(x) for x in X_train];
julia> gp = GP(X_train', y_train, MeanZero(), SE(0.0, 0.0));
julia> optimize!(gp);

julia> # The point of highest uncertainty is halfway between the samples
julia> x_candidate = 3.0;
julia> posterior_variance(gp, x_candidate)
1.0000010000000002
"""
function posterior_variance(gp, xnew)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)

    # predict_f returns (mean, variance)
    _, σ² = predict_f(gp, xvec)

    # We want to maximize the variance
    return σ²[1]
end


###################

"""
    estimate_integral_wsabi(gp, bounds; n_samples=100_000, y_mean=0.0, y_std=1.0)

Estimates the integral of the original function f(x) using the final GP model
trained on warped data g(x) = log(f(x)).

It uses Monte Carlo integration on the posterior expectation of f(x).
E[f(x)] = exp(μ_g(x) + σ²_g(x)/2), where μ_g and σ²_g are the posterior mean
and variance of the GP fitted to the log-transformed data.

# Arguments
- `gp`: The final trained GP model.
- `bounds`: A tuple (lo, hi) defining the integration domain.
- `n_samples`: Number of Monte Carlo samples for the approximation.
- `y_mean`, `y_std`: The mean and std dev used to standardize the warped y-values,
                     needed to un-scale the GP's predictions.

# Returns
- `Float64`: The estimated value of the integral.
"""
function estimate_integral_wsabi(gp, bounds; n_samples=100_000, y_mean=0.0, y_std=1.0)
    lo, hi = bounds
    d = gp.dim

    # Calculate the volume of the integration domain
    domain_volume = prod(hi .- lo)

    # Generate a large number of random points within the original domain
    X_mc_orig = rand(d, n_samples) .* (hi .- lo) .+ lo

    # Rescale points to [-1, 1] for the GP
    X_mc_scaled = rescale(X_mc_orig', lo, hi)'

    # Get posterior mean and variance from the GP on the scaled points
    μ_scaled, σ²_scaled = predict_f(gp, X_mc_scaled)

    # --- Un-standardize the GP's predictions ---
    # The GP was trained on z = (y_warped - μ_y) / σ_y
    # So, y_warped = z * σ_y + μ_y
    μ_unstandardized = μ_scaled .* y_std .+ y_mean
    σ²_unstandardized = σ²_scaled .* (y_std^2)

    # Calculate the expected value of the original (un-warped) function f(x)
    # E[f(x)] = exp(μ_g + σ²_g/2)
    integrand_values = exp.(μ_unstandardized .+ 0.5 .* σ²_unstandardized)

    # The Monte Carlo estimate of the integral is the mean of these values times the domain volume
    integral_estimate = mean(integrand_values) * domain_volume

    return integral_estimate
end