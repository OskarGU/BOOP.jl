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
acquisition_func(xnew) = knowledge_gradient(gp, xnew[1], -2., 6., n_samples=450)

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







#################################
################################
using Distributions
using GaussianProcesses
using LinearAlgebra
using Statistics

# ==============================================================================
#  1. The Robust, Standalone KG Algorithm (from our previous conversation)
# ==============================================================================
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


# ==============================================================================
#  2. The New High-Level Wrapper Function (your plan)
# ==============================================================================

"""
Computes the Knowledge Gradient acquisition function for a multi-dimensional GP.

This version correctly handles multi-dimensional inputs by expecting `domain_points`
to be a D x d matrix.
"""
function kg_acquisition_function_multid(gp::GPE, xnew::Vector{Float64}, domain_points::Matrix{Float64})
    # --- Step A: Get GP-specific values ---
    
    # Get the current posterior mean over the discrete domain points.
    # FIX: Removed incorrect transpose. `domain_points` is already in the correct D x d format.
    μ_current, _ = predict_f(gp, domain_points)
    
    # Get the predictive distribution of a noisy observation y at xnew.
    xnew_mat = reshape(xnew, :, 1)
    _, σ²_y_new = predict_y(gp, xnew_mat)
    σ_y_new = sqrt(max(σ²_y_new[1], 1e-12))

    # Calculate the covariance vector between the domain points and xnew.
    # FIX: Removed incorrect transpose here as well.
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

# 1. Example Setup for a 2D GP
# Let's assume your GP takes 2D inputs.
# The observed data would have dimensions: x_obs (2 x N), y_obs (N,)
D = 2 # Number of dimensions
num_obs = 10
x_obs_2d = rand(D, num_obs) * 10
y_obs_2d = [sin(x[1]) + cos(x[2]) for x in eachcol(x_obs_2d)]

# Create and optimize a 2D GP model
# Note the use of `SEArd` for Automatic Relevance Determination per dimension.
kernel_2d = SEArd(zeros(D), 0.0) 
gp_2d = GP(x_obs_2d, y_obs_2d, MeanConst(0.0), kernel_2d, -2.0)
# optimize!(gp_2d) # You would normally optimize the hyperparameters

# 2. Create a 2D grid for the `domain_points`
pts_per_dim = 10
x1_range = range(0, 10, length=pts_per_dim);
x2_range = range(0, 10, length=pts_per_dim);

# Create the grid by collecting all combinations
grid_points = hcat([[x1, x2] for x1 in x1_range for x2 in x2_range]...);
# `grid_points` is now a 2 x 400 matrix (D x d), which is the required format.

# 3. Use the multi-dimensional acquisition function
# Let's test a candidate point
x_candidate_2d = [2.5, 7.5];

# Calculate the KG value
@time kg = kg_acquisition_function_multid(gp_2d, x_candidate_2d, grid_points);

println("KG value at $(x_candidate_2d): ", kg)


KGSug = 1:0.2:10
KGEval=[]
for i in 1:length(KGSug)
    push!(KGEval, kg_acquisition_function_multid(gp_2d, [KGSug[i], 7.5], grid_points))
    print(i)
end

plot!(KGSug, KGEval)