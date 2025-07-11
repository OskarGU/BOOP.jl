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
optimize!(gp) # Optimize hyperparameters
plot(gp)


knowledge_gradient(gp, xnew, -2., 6., n_samples=500)

##########
# I think the function works but need to be optimized in clever way. It is noisy 
# due to montecarlo sampling, and a little bit expensive. Should we use BO for this?
# A good way i think is the hybrid one-shot approach. The classical approach would 
# be on a fairly dense grid, but that is not very efficient.


# 4. Define domain for finding posterior minimum
domain = reshape(collect(range(lb, ub, length=200)), 1, :)

# 5. Find the next point to sample using KG
# We need to wrap the acquisition function for the optimizer
acquisition_func(x) = knowledge_gradient(gp, x, domain)

acquisition_func(1.2) 
# Use an optimizer (e.g., from Optim.jl) to find the point that maximizes KG
# (minimizes -KG)
result = optimize(acquisition_func, [lb], [ub], [3.0], Fminbox(BFGS()))
x_next = Optim.minimizer(result)

println("Current GP state optimized.")
println("Next point to sample according to KG: ", x_next)
println("KG value at that point: ", -Optim.minimum(result))





