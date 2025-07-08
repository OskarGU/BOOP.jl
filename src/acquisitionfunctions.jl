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
