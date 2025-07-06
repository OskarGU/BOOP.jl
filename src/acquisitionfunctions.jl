# This file contain acquisition functions that can be used in the Bayesian optimization.

# Expected Improvement (EI) acquisition function
function expected_improvement(gp, xnew, ybest; ξ = 0.10)
    xvec = xnew isa Number ? [xnew] : xnew
    xvec = reshape(xvec, :, 1)
    μ, σ² = predict_f(gp, xvec)
    σ = max(sqrt.(σ²[1]), 0.1^6)  # Ensure positive std deviation
    Δ = ybest .- μ .- ξ
    Z = Δ ./ σ
    Φ = cdf.(Normal(), Z)
    ϕ = pdf.(Normal(), Z)
    ei = Δ .* Φ .+ σ .* ϕ
    return ei[1]
end