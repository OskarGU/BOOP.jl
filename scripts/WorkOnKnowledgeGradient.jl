using BOOP
using GaussianProcesses, Optim, Distributions, Random, LinearAlgebra
# --- Example Usage ---

# 1. Define a problem
f(x) = (x[1]-2)^2 + 4*sin.(x[1]) + randn() * 0.5 # Noisy 1D objective
lb = -2.0 # Lower bound
ub = 6.0   # Upper bound

# 2. Initial Data
n_initial = 6
X_train = reshape(range(lb, ub, length=n_initial), 1, :)
Y_train = [f(x) for x in eachcol(X_train)]

# 3. Create and fit GP model
mZero = MeanZero()
kern = SE(0.41, 2.0) # Kernel with trainable hyperparameters
logNoise = -2.6 # log(sqrt(noise))
gp = GPE(X_train, Y_train, mZero, kern, logNoise)
optimize!(gp; noisebounds=(-6.0, 3.0)) # Optimize hyperparameters
plot(gp)


knowledge_gradient(gp, 0.5, -2., 6., n_samples=500)

##########
# I think the function works but need to be optimized in clever way. It is noisy 
# due to montecarlo sampling, and a little bit expensive. Should we use BO for this?
# A good way i think is the hybrid one-shot approach. The classical approach would 
# be on a fairly dense grid, but that is not very efficient.

# We start with evaluating the KG at a dense grid of points.
# 4. Define domain for finding posterior minimum
domain = reshape(collect(range(lb, ub, length=200)), 1, :)

# 5. Find the next point to sample using KG
# We need to wrap the acquisition function for the optimizer
acquisition_func(xnew) = knowledge_gradient(gp, xnew[1], -2., 6., n_samples=500)

@time acquisition_func(1.2)

domainEval = -2:0.1:6
af = acquisition_func.(domainEval) 
paf = plot(domainEval, af)
pgp = plot(gp)
plot(paf, pgp, layout=(2,1))

# Use an optimizer (e.g., from Optim.jl) to find the point that maximizes KG
# (minimizes -KG)
result = optimize(acquisition_func, [lb], [ub], [3.0], Fminbox(BFGS()))
x_next = Optim.minimizer(result)

println("Current GP state optimized.")
println("Next point to sample according to KG: ", x_next)
println("KG value at that point: ", -Optim.minimum(result))





############################################
# Discrete knowledge Gradient
using GaussianProcesses
using Distributions
using LinearAlgebra
using Statistics

# Ensure you have a standard Normal distribution object available
const Normal_0_1 = Normal(0.0, 1.0)

"""
Computes the Knowledge Gradient analytically for a candidate point `xnew`
over a discretized domain.

This version avoids Monte Carlo simulation by using the closed-form expression
for the expected value of the minimum of a set of Gaussian variables.

# Arguments
- `gp::GPE`: The current Gaussian Process model.
- `xnew`: The candidate point at which to evaluate the KG.
- `domain_points`: A matrix representing the discretized domain.

# Returns
- A `Float64` representing the negative Knowledge Gradient value.
"""
function knowledge_gradient_analytic_discrete(gp::GPE, xnew, domain_points)
    # Ensure xnew and domain_points are correctly shaped matrices
    xnew_mat = reshape(xnew isa Number ? [xnew] : xnew, :, 1)
    domain_mat = reshape(domain_points, :, size(gp.x, 1))
    
    # 1. Find the minimum of the *current* posterior mean over the discretized domain.
    μ_current, Σ_current = predict_f(gp, domain_mat'; full_cov=true)
    min_μ_current = minimum(μ_current)

    # 2. Get the predictive distribution at the candidate point `xnew`.
    # This is the distribution of a future *noisy* observation y.
    μ_new_point, σ²_new_point = predict_y(gp, xnew_mat)
    σ_new_point = sqrt(max(σ²_new_point[1], 1e-9))

    # 3. Calculate the update vector. This vector determines how the posterior mean
    #    at all domain points changes in response to an observation at xnew.
    #    It's derived from the GP posterior update equations.
    cov_dx = vcat([cov(gp.kernel, [domain_mat[i];;], xnew_mat) for i in 1:length(domain_mat)]...) # Covariance between domain and xnew
    # The update vector v = K(X, x_*) / (K(x_*, x_*) + σ_n²)
    update_vector = cov_dx ./ (σ²_new_point[1]) # Note: predict_y's variance includes noise

    # The fantasy mean is μ_fantasy = μ_current + update_vector * (y_sample - μ_new_point)
    # Let y_sample = μ_new_point + σ_new_point * Z, where Z ~ N(0,1)
    # Then μ_fantasy = μ_current + update_vector * (σ_new_point * Z)
    # This is a set of Gaussians whose minimum we need to find the expectation of.
    
    # 4. Compute the expected future minimum analytically.
    # Let u_i = μ_current[i] and v_i = update_vector[i] * σ_new_point
    # We want E[min_i(u_i + v_i*Z)]
    u = vec(μ_current)
    v = vec(update_vector) .* σ_new_point
    
    # Sort by slopes v_i to efficiently calculate the expectation
    p = sortperm(v)
    u_sorted, v_sorted = u[p], v[p]
    
    expected_min_μ_future = 0.0
    # The formula involves summing over segments where each line is the minimum.
    # The crossing points of lines u_i + v_i*z and u_j + v_j*z are z_ij = (u_i - u_j) / (v_j - v_i)
    z_vals = -Inf
    
    for i in 1:length(u)-1
        z_next = (u_sorted[i] - u_sorted[i+1]) / (v_sorted[i+1] - v_sorted[i])
        
        # Integrate E[u_i + v_i*Z] from z_vals to z_next
        # Integral of (u+v*z)*phi(z)dz = u*(cdf(b)-cdf(a)) - v*(pdf(b)-pdf(a))
        cdf_z_next = cdf(Normal_0_1, z_next)
        cdf_z_vals = cdf(Normal_0_1, z_vals)

        pdf_z_next = pdf(Normal_0_1, z_next)
        pdf_z_vals = pdf(Normal_0_1, z_vals)
        
        term1 = u_sorted[i] * (cdf_z_next - cdf_z_vals)
        term2 = v_sorted[i] * (pdf_z_vals - pdf_z_next) # Note the order
        
        expected_min_μ_future += term1 + term2
        z_vals = z_next
    end
    
    # Add the final interval from the last z_vals to +Inf
    cdf_z_vals = cdf(Normal_0_1, z_vals)
    pdf_z_vals = pdf(Normal_0_1, z_vals)
    
    term1_last = u_sorted[end] * (1.0 - cdf_z_vals)
    term2_last = v_sorted[end] * (pdf_z_vals) # pdf(+inf) is 0
    expected_min_μ_future += term1_last + term2_last

    # 5. The Knowledge Gradient is the improvement.
    kg = min_μ_current - expected_min_μ_future

    # 6. Return negative for maximization.
    return -kg
end


domain_points = range(-2.0, 6.0, length=600) # Discretized domain for KG evaluation
knowledge_gradient_analytic_discrete(gp::GPE, 2.1, domain_points)