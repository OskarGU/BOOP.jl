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
    knowledgeGradientMonteCarlo(gp, xnew, lower, upper; n_samples=20)

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
function knowledgeGradientMonteCarlo(gp, xnew; n_samples::Int=200)
    xvec = xnew isa Number ? [xnew] : xnew
    xnew = reshape(xvec, :, 1)

    # 1. Find the MAXIMUM of the *current* posterior mean. This is our baseline.
    μ_current(x) = predict_f(gp, reshape(x, :, 1))[1][1]
    d = gp.dim
    lower= fill(-1.0, d)
    upper= fill(1.0, d)
    max_μ_current, _ = multi_start_maximize(μ_current, lower, upper; n_starts=40)

    # 2. Get the predictive distribution at the candidate point `xnew`. (No change needed)
    μ_new_point, σ²_new_point = predict_y(gp, xnew)
    predictive_dist = Normal(μ_new_point[1], sqrt(max(σ²_new_point[1], 1e-6)))

    # 3. Run Monte Carlo simulation to estimate the expected future MAXIMUM.
    future_maximums = zeros(n_samples)
    for i in 1:n_samples
        # a. Draw one potential future observation `y_sample` at `xnew`.
        y_sample = [rand(predictive_dist)]

        # b. Create a temporary, "fantasy" GP model.
        x_updated = hcat(gp.x, xnew)
        y_updated = vcat(gp.y, y_sample)
        gp_fantasy = GP(x_updated, y_updated, gp.mean, gp.kernel, gp.logNoise)

        # c. Find the MAXIMUM of the posterior mean of this new fantasy GP.
        μ_fantasy(x) = predict_f(gp_fantasy, reshape(x, :, 1))[1][1]
        future_maximums[i], _ = multi_start_maximize(μ_fantasy, lower, upper; n_starts=10)
    end

    # 4. Calculate the expected value of the future maximum.
    expected_max_μ_future = mean(future_maximums)

    # 5. The KG is the expected INCREASE in the MAXIMUM.
    kg = expected_max_μ_future - max_μ_current

    # 6. Return the natural, positive KG value.
    #    The optimization loop is responsible for negating this if it uses a minimizer.
    return kg
end





"""
Performs multi-start minimization and returns both the minimum value
and the location (argmin) where it was found.
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


# 2. Modify the maximize helper to pass through both results
"""
Performs multi-start maximization and returns both the maximum value
and the location (argmax) where it was found.
"""
function multi_start_maximize(f, lower, upper; n_starts=20)
    objective_to_minimize = x -> -f(x)
    
    # Get both the minimum value and the argmin from the helper
    min_val, argmin_x = multi_start_minimize(objective_to_minimize, lower, upper; n_starts=n_starts)
    
    # The maximum is -min_val, and the argmax is the same as the argmin of the negative function
    return (-min_val, argmin_x)
end



##########################
"""
Calculates E[max(μ + σZ)] where Z ~ N(0,1).
This is a robust, "pure" version that ONLY returns the expected maximum.
It handles cases where slopes (σ) are equal or nearly equal.
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
Computes the posterior covariance between points in the GP. 
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
Computes the Knowledge Gradient acquisition function for a multi-dimensional GP
using a fixed discrete set of points.
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
This version is designed for MAXIMIZATION problems.

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





###########################
# Experimenting:
"""
    posterior_variance(gp, xnew)

Computes the posterior variance of the Gaussian Process `gp` at a new point `xnew`.
This is the primary acquisition function for standard Bayesian Quadrature methods
like WSABI, as it directs sampling to regions of highest uncertainty in the model.

Returns the posterior variance at `xnew`.
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