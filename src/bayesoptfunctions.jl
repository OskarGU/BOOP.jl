# Propose next point using multiple random restarts
function propose_next(gp, ybest; bounds=(-1.0, 1.0), d=1, n_restarts=20, acq=expected_improvement, ξ = 0.10)
    best_val = Inf
    best_x = zeros(d)
    acqIn(x) = -acq(gp, x, ybest; ξ=ξ)

    for _ in 1:n_restarts
        x0 = rand(Uniform(bounds[1], bounds[2]), d)
        res = optimize(acqIn,
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